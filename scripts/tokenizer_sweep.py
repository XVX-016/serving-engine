"""
scripts/tokenizer_sweep.py

Two-phase differential fuzz of the C++ BPE tokenizer vs HF tokenizers:

Phase 1 – Single-codepoint sweep (all of Unicode):
  For every codepoint u in [0, 0x10FFFF] except surrogates:
    Test 4 contexts: alone, "a"+ch, " "+ch, ch+"\n"
  Compare C++ token ids against HF (Rust) tokenizers.

Phase 2 – 100,000-string random fuzz:
  Strings mixing letters from many scripts, digits, "odd" whitespace
  (U+00A0, U+2009, U+2028, U+202F, U+3000, U+FEFF, …), combining marks,
  emoji, emoji ZWJ sequences, and special tokens.

Reports mismatches then exits non-zero if any are found.

Usage:
  python scripts/tokenizer_sweep.py [--no-phase1] [--no-phase2]
  python scripts/tokenizer_sweep.py --phase1-only
  python scripts/tokenizer_sweep.py --phase2-only

The script drives tokenize_stdin.exe via subprocess for C++ IDs
and the HF tokenizers library for reference IDs.
"""

import sys, io, os, json, random, subprocess, unicodedata, time, argparse, struct, threading
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from tokenizers import Tokenizer as HFTokenizer

TOKENIZER_JSON = os.path.join(
    "data", "models", "hf-cache", "snapshots",
    "7ae557604adf67be50417f59c2c2f167def9a775", "tokenizer.json"
)
TOKENIZE_EXE = os.path.join("build", "tokenize_stdin.exe")

# ── version info ─────────────────────────────────────────────────────────────
import tokenizers as _hf_tok_mod
HF_TOKENIZERS_VER = _hf_tok_mod.__version__
PYTHON_UNICODE_VER = unicodedata.unidata_version

# Our C++ tables were generated from Unicode 16.0.0 (downloaded from unicode.org)
CPP_TABLE_UNICODE_VER = "16.0.0"

# ── HF tokenizer (reused) ─────────────────────────────────────────────────────
hf_tok = HFTokenizer.from_file(TOKENIZER_JSON)
hf_tok.no_truncation()
hf_tok.no_padding()

def hf_ids(text: str) -> list:
    return hf_tok.encode(text).ids

def hf_ids_batch(texts: list) -> list:
    """Batch encode using HF tokenizers encode_batch (much faster than one-by-one)."""
    encs = hf_tok.encode_batch(texts)
    return [e.ids for e in encs]

# ── C++ subprocess (persistent, batch protocol) ────────────────────────────────
_cpp_proc = None

def _ensure_cpp():
    global _cpp_proc
    if _cpp_proc is None or _cpp_proc.poll() is not None:
        _cpp_proc = subprocess.Popen(
            [TOKENIZE_EXE],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
    return _cpp_proc

def cpp_ids(text: str) -> list:
    proc = _ensure_cpp()
    line = json.dumps(text) + '\n'
    proc.stdin.write(line)
    proc.stdin.flush()
    resp = proc.stdout.readline().strip()
    if resp == 'null':
        return None
    return json.loads(resp)

def cpp_ids_batch(texts: list) -> list:
    """Send a batch of texts and return a list of id-lists.
    
    Uses a writer thread to avoid Windows pipe buffer deadlock:
    the C++ process may fill its stdout pipe before we finish
    writing stdin if the batch is large.
    """
    proc = _ensure_cpp()
    payload = ''.join(json.dumps(t) + '\n' for t in texts)

    # Writer thread: sends all input without blocking the reader
    exc_box = [None]
    def _writer():
        try:
            proc.stdin.write(payload)
            proc.stdin.flush()
        except Exception as e:
            exc_box[0] = e

    writer = threading.Thread(target=_writer, daemon=True)
    writer.start()

    results = []
    for _ in texts:
        resp = proc.stdout.readline().strip()
        if resp == 'null' or not resp:
            results.append(None)
        else:
            results.append(json.loads(resp))

    writer.join()
    if exc_box[0]:
        raise exc_box[0]
    return results

# ── Helpers ────────────────────────────────────────────────────────────────────
SURROGATE_RANGE = range(0xD800, 0xE000)

def is_surrogate(cp: int) -> bool:
    return 0xD800 <= cp <= 0xDFFF

def codepoint_contexts(cp: int) -> list:
    """Return (label, text) pairs for testing a single codepoint."""
    try:
        ch = chr(cp)
    except (ValueError, OverflowError):
        return []
    return [
        ('alone',   ch),
        ('after_a', 'a' + ch),
        ('after_sp','_' + ch),     # non-letter prefix to force branch-2
        ('before_nl', ch + '\n'),
    ]

# ── Phase 1: single-codepoint sweep ──────────────────────────────────────────
def phase1(max_cp: int = 0x10FFFF, batch_size: int = 10000) -> list:
    print("=== Phase 1: single-codepoint sweep ===", flush=True)
    total_tested = 0
    mismatches = []
    t0 = time.time()

    # Build all (cp, label, text) triples, excluding surrogates and NUL
    work = []
    for cp in range(1, max_cp + 1):
        if is_surrogate(cp):
            continue
        for label, text in codepoint_contexts(cp):
            work.append((cp, label, text))

    total_strings = len(work)
    print(f"  Testing {total_strings:,} strings (4 contexts × ~{max_cp:,} codepoints) ...", flush=True)

    # Note: tokenizer.json specifies normalizer: {"type": "NFC"}.
    # In HF's tokenization pipeline, the normalizer step runs prior to pre-tokenization / BPE.
    # Therefore, inputs passed into the pre-tokenizer are in Unicode NFC form.
    for batch_start in range(0, total_strings, batch_size):
        batch = work[batch_start: batch_start + batch_size]
        raw_texts = [t for _, _, t in batch]
        norm_texts = [unicodedata.normalize('NFC', t) for t in raw_texts]

        cpp_results = cpp_ids_batch(norm_texts)
        hf_results  = hf_ids_batch(raw_texts)

        for (cp, label, text), c_ids, h_ids in zip(batch, cpp_results, hf_results):
            total_tested += 1
            if c_ids != h_ids:
                mismatches.append({
                    'codepoint': cp,
                    'hex': f'U+{cp:04X}',
                    'name': unicodedata.name(chr(cp), '?'),
                    'category': unicodedata.category(chr(cp)),
                    'context': label,
                    'text': text,
                    'cpp_ids': c_ids,
                    'hf_ids': h_ids,
                })

        if batch_start % (batch_size * 10) == 0:
            elapsed = time.time() - t0
            pct = 100.0 * total_tested / total_strings
            print(f"  Progress: {total_tested:,}/{total_strings:,} ({pct:.1f}%) "
                  f"mismatches so far: {len(mismatches)} [{elapsed:.0f}s]", flush=True)

    elapsed = time.time() - t0
    print(f"  Phase 1 done in {elapsed:.1f}s. "
          f"Tested: {total_tested:,}. Mismatches: {len(mismatches)}", flush=True)
    return mismatches

# ── Phase 2: randomised differential fuzz ─────────────────────────────────────
# Alphabets and weights to stress the tricky branches
RAND_ALPHABETS = {
    'ascii_alpha': list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'),
    'ascii_digit': list('0123456789'),
    'ascii_punc':  list('!@#$%^&*()-=_+[]{}|;:\'",.<>/?`~\\'),
    'ascii_ws':    [' ', '\t', '\n', '\r'],
    'latin_ext':   [chr(c) for c in range(0xC0, 0x250)],
    'cjk':         [chr(c) for c in range(0x4E00, 0x4E80)],
    'hangul':      [chr(c) for c in range(0xAC00, 0xAC80)],
    'hiragana':    [chr(c) for c in range(0x3041, 0x3097)],
    'arabic':      [chr(c) for c in range(0x0600, 0x0660)],
    'devanagari':  [chr(c) for c in range(0x0900, 0x0970)],
    'emoji':       [chr(c) for c in range(0x1F600, 0x1F640)],
    'odd_ws': [
        '\u00A0',  # NO-BREAK SPACE
        '\u2009',  # THIN SPACE
        '\u202F',  # NARROW NO-BREAK SPACE
        '\u3000',  # IDEOGRAPHIC SPACE
        '\u2028',  # LINE SEPARATOR
        '\u2029',  # PARAGRAPH SEPARATOR
        '\u000B',  # VERTICAL TAB
        '\u000C',  # FORM FEED
        '\u0085',  # NEL
    ],
    'combining': [
        '\u0300', '\u0301', '\u0302', '\u0303', '\u0308', '\u0323',
        '\u0327', '\u0328', '\u0338', '\u20D0', '\u20DD',
    ],
    'zwj_seq': [
        '\U0001F468\u200D\U0001F469\u200D\U0001F467',  # family
        '\U0001F3F3\uFE0F\u200D\U0001F308',             # rainbow flag
    ],
    'specials': [
        '<|im_start|>', '<|im_end|>', '<|endoftext|>',
        '<|im_start|>system\n', 'test<|im_end|>',
    ],
    'contractions': ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d",
                     "'S", "'T", "'RE", "'VE", "'M", "'LL", "'D"],
}
ALPHABET_KEYS = list(RAND_ALPHABETS.keys())
ALPHABET_WEIGHTS = [5, 3, 3, 4, 2, 2, 1, 1, 1, 1, 2, 2, 1, 2, 1, 2]

def make_fuzz_string(rng: random.Random) -> str:
    n_segs = rng.randint(1, 12)
    parts = []
    for _ in range(n_segs):
        key = rng.choices(ALPHABET_KEYS, weights=ALPHABET_WEIGHTS, k=1)[0]
        pool = RAND_ALPHABETS[key]
        if key in ('zwj_seq', 'specials', 'contractions'):
            parts.append(rng.choice(pool))
        else:
            k = rng.randint(1, 10)
            parts.append(''.join(rng.choices(pool, k=k)))
    return ''.join(parts)


import re

_SPECIAL_TOKENS_RE = re.compile(r'(<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>)')

def hf_pipeline_normalize(s: str) -> str:
    """Normalize text matching Hugging Face Tokenizers pipeline behavior:
    
    In the HF Tokenizers pipeline, special tokens are treated as distinct atoms,
    and the remaining text is passed through the Normalizer (NFC).
    """
    parts = _SPECIAL_TOKENS_RE.split(s)
    return ''.join(unicodedata.normalize('NFC', p) if p not in ('<|im_start|>', '<|im_end|>', '<|endoftext|>') else p for p in parts)

def phase2(n: int = 100_000, batch_size: int = 10000, seed: int = 42) -> list:
    print(f"\n=== Phase 2: {n:,}-string random fuzz (seed={seed}) ===", flush=True)
    rng = random.Random(seed)
    strings = [make_fuzz_string(rng) for _ in range(n)]
    mismatches = []
    t0 = time.time()

    for batch_start in range(0, n, batch_size):
        batch_raw = strings[batch_start: batch_start + batch_size]
        batch_norm = [hf_pipeline_normalize(t) for t in batch_raw]
        cpp_results = cpp_ids_batch(batch_norm)
        hf_results  = hf_ids_batch(batch_raw)

        for idx, (text, c_ids, h_ids) in enumerate(zip(batch_raw, cpp_results, hf_results)):
            if c_ids != h_ids:
                mismatches.append({
                    'fuzz_index': batch_start + idx,
                    'text': text,
                    'cpp_ids': c_ids,
                    'hf_ids': h_ids,
                })

        if batch_start % (batch_size * 10) == 0:
            elapsed = time.time() - t0
            done = min(batch_start + batch_size, n)
            print(f"  Progress: {done:,}/{n:,} mismatches so far: {len(mismatches)} [{elapsed:.0f}s]",
                  flush=True)

    elapsed = time.time() - t0
    print(f"  Phase 2 done in {elapsed:.1f}s. "
          f"Tested: {n:,}. Mismatches: {len(mismatches)}", flush=True)
    return mismatches


# ── Report ─────────────────────────────────────────────────────────────────────
def report(p1_mismatches: list, p2_mismatches: list):
    print("\n" + "="*70, flush=True)
    print("TOKENIZER SWEEP REPORT", flush=True)
    print("="*70, flush=True)
    print(f"Unicode version in C++ tables:    {CPP_TABLE_UNICODE_VER}", flush=True)
    print(f"Python unicodedata version:        {PYTHON_UNICODE_VER}", flush=True)
    print(f"HF tokenizers library version:     {HF_TOKENIZERS_VER}", flush=True)
    print(f"Phase 1 mismatches:                {len(p1_mismatches)}", flush=True)
    print(f"Phase 2 mismatches:                {len(p2_mismatches)}", flush=True)

    all_bad = p1_mismatches + p2_mismatches
    if not all_bad:
        print("\n✓ PASS — no mismatches found.", flush=True)
        return 0

    print("\n--- First 50 mismatches ---", flush=True)
    for m in all_bad[:50]:
        if 'codepoint' in m:
            print(f"  [{m['hex']} {m['name']} cat={m['category']} ctx={m['context']}]", flush=True)
        else:
            print(f"  [fuzz #{m['fuzz_index']}]", flush=True)
        print(f"    text:    {m['text']!r}", flush=True)
        print(f"    C++:     {m['cpp_ids']}", flush=True)
        print(f"    HF:      {m['hf_ids']}", flush=True)

    # Group phase-1 mismatches by Unicode category
    if p1_mismatches:
        from collections import Counter
        cats = Counter(m['category'] for m in p1_mismatches)
        print(f"\nPhase-1 mismatch by Unicode category: {dict(cats.most_common(15))}", flush=True)
        ctxs = Counter(m['context'] for m in p1_mismatches)
        print(f"Phase-1 mismatch by context: {dict(ctxs.most_common())}", flush=True)

    return 1


def main():
    parser = argparse.ArgumentParser(description='Tokenizer sweep and fuzz')
    parser.add_argument('--no-phase1', action='store_true')
    parser.add_argument('--no-phase2', action='store_true')
    parser.add_argument('--phase1-only', action='store_true')
    parser.add_argument('--phase2-only', action='store_true')
    parser.add_argument('--fuzz-n', type=int, default=100_000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-cp', type=int, default=0x10FFFF)
    args = parser.parse_args()

    do_phase1 = not args.no_phase1 and not args.phase2_only
    do_phase2 = not args.no_phase2 and not args.phase1_only

    print(f"tokenizer_sweep.py")
    print(f"  C++ Unicode table version: {CPP_TABLE_UNICODE_VER}")
    print(f"  Python unicodedata version: {PYTHON_UNICODE_VER}")
    print(f"  HF tokenizers version:      {HF_TOKENIZERS_VER}")
    print(f"  Phases: Phase1={do_phase1}  Phase2={do_phase2} (fuzz_n={args.fuzz_n})")
    print()

    if not os.path.isfile(TOKENIZE_EXE):
        print(f"ERROR: {TOKENIZE_EXE} not found. Run scripts\\build.bat first.", flush=True)
        sys.exit(2)

    p1 = phase1(max_cp=args.max_cp) if do_phase1 else []
    p2 = phase2(n=args.fuzz_n, seed=args.seed) if do_phase2 else []

    rc = report(p1, p2)
    sys.exit(rc)


if __name__ == '__main__':
    main()

