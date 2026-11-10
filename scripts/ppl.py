#!/usr/bin/env python3
"""
scripts/ppl.py

Computes perplexity on the WikiText-2 test split:
- Context length: 1024
- Stride: 512
- Evaluation mode: identical sliding-window negative log likelihood (NLL) calculation
  across HF FP16, Engine FP16, Engine INT8, and Engine INT4.
"""

import os
import sys
import argparse
import math
import subprocess
import json
import struct
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "data/models/hf-cache/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"

def evaluate_hf_fp16(model_path, encodings, max_length=1024, stride=512, device="cuda"):
    print(f"[HF FP16] Loading model from {model_path} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device
    )
    model.eval()

    seq_len = encodings.input_ids.size(1)
    nlls = []
    total_tokens = 0

    print(f"[HF FP16] Evaluating over {seq_len} tokens (context={max_length}, stride={stride})...")
    prev_end_loc = 0
    with torch.no_grad():
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # number of tokens to predict
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            outputs = model(input_ids, labels=target_ids)
            # loss is averaged over the non -100 tokens in target_ids
            neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood)
            total_tokens += trg_len
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

    ppl = torch.exp(torch.stack(nlls).sum() / total_tokens).item()
    print(f"[HF FP16] Perplexity: {ppl:.4f} (over {total_tokens} tokens)")
    return ppl

def evaluate_engine(engine_bin, model_path, encodings_tokens, max_length=1024, stride=512):
    tmp_file = "data/ppl_tokens.bin"
    os.makedirs(os.path.dirname(tmp_file), exist_ok=True)
    tokens_np = np.array(encodings_tokens, dtype=np.int32)
    with open(tmp_file, "wb") as f:
        # Write uint32 length header followed by int32 tokens
        f.write(struct.pack("<I", len(tokens_np)))
        f.write(tokens_np.tobytes())

    cmd = [
        engine_bin,
        "--model", model_path,
        "--ppl-tokens", tmp_file,
        "--ppl-max-len", str(max_length),
        "--ppl-stride", str(stride)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("Engine run failed:")
        print(res.stderr)
        return None
    for line in res.stdout.splitlines():
        if "PPL_RESULT:" in line:
            parts = line.split()
            return float(parts[1])
    print("Engine output did not contain PPL_RESULT:\n", res.stdout)
    return None

def main():
    parser = argparse.ArgumentParser(description="Evaluate perplexity on WikiText-2 test")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--engine-bin", type=str, default="build/generate.exe")
    parser.add_argument("--mode", type=str, choices=["hf", "engine", "all"], default="hf")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--num-tokens", type=int, default=0, help="Limit number of tokens for quick check (0 = full)")
    args = parser.parse_args()

    print("Loading WikiText-2 test split...")
    test_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    test_text = "\n\n".join(test_ds["text"])

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    encodings = tokenizer(test_text, return_tensors="pt")

    if args.num_tokens > 0:
        encodings.input_ids = encodings.input_ids[:, :args.num_tokens]

    print(f"Total tokens in test split: {encodings.input_ids.size(1)}")

    if args.mode in ["hf", "all"]:
        hf_ppl = evaluate_hf_fp16(args.model_path, encodings, max_length=args.max_length, stride=args.stride, device=args.device)

    tokens_list = encodings.input_ids[0].tolist()

    if args.mode in ["engine", "all"]:
        print(f"\n[Engine FP16] Evaluating...")
        e_fp16 = evaluate_engine(args.engine_bin, args.model_path, tokens_list, max_length=args.max_length, stride=args.stride)
        print(f"[Engine FP16] Perplexity: {e_fp16:.4f}")

        print(f"\n[Engine INT8] Evaluating...")
        e_int8 = evaluate_engine(args.engine_bin, "data/models/qwen2.5-0.5b-int8", tokens_list, max_length=args.max_length, stride=args.stride)
        print(f"[Engine INT8] Perplexity: {e_int8:.4f}")

        print(f"\n[Engine INT4 G128] Evaluating...")
        e_int4 = evaluate_engine(args.engine_bin, "data/models/qwen2.5-0.5b-int4-g128", tokens_list, max_length=args.max_length, stride=args.stride)
        print(f"[Engine INT4 G128] Perplexity: {e_int4:.4f}")

if __name__ == "__main__":
    main()

