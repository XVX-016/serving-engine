import subprocess
import re
import json

LLAMA_BENCH = r"third_party\llama.cpp\build\bin\llama-bench.exe"
GENERATE = r"build\generate.exe"

MODELS = {
    "our_fp16": ("FP16 (fused, graph)", [GENERATE, "--model", "data/models/qwen2.5-0.5b-fp16-fused", "--bench-decode", "--cuda-graph"]),
    "our_int8_legacy": ("INT8 Legacy (fused, graph)", [GENERATE, "--model", "data/models/qwen2.5-0.5b-int8-fused", "--bench-decode", "--cuda-graph", "--legacy-gemv"]),
    "our_int8_dp4a": ("INT8 dp4a (fused, graph)", [GENERATE, "--model", "data/models/qwen2.5-0.5b-int8-fused", "--bench-decode", "--cuda-graph", "--dp4a-gemv"]),
    "our_int4_legacy": ("INT4 Legacy (fused, graph)", [GENERATE, "--model", "data/models/qwen2.5-0.5b-int4-g128-head-int8-fused", "--bench-decode", "--cuda-graph", "--legacy-gemv"]),
    "our_int4_dp4a": ("INT4 dp4a (fused, graph)", [GENERATE, "--model", "data/models/qwen2.5-0.5b-int4-g128-head-int8-fused", "--bench-decode", "--cuda-graph", "--dp4a-gemv"]),
    "llama_f16": ("llama.cpp F16", [LLAMA_BENCH, "-m", "data/models/qwen2.5-0.5b-instruct-fp16.gguf", "-n", "32", "-p", "128", "-ngl", "99", "-r", "5"]),
    "llama_q8": ("llama.cpp Q8_0", [LLAMA_BENCH, "-m", "data/models/qwen2.5-0.5b-instruct-q8_0.gguf", "-n", "32", "-p", "128", "-ngl", "99", "-r", "5"]),
    "llama_q4": ("llama.cpp Q4_K_M", [LLAMA_BENCH, "-m", "data/models/qwen2.5-0.5b-instruct-q4_k_m.gguf", "-n", "32", "-p", "128", "-ngl", "99", "-r", "5"]),
}

results = {k: [] for k in MODELS}

def run_cmd(cmd):
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = res.stdout
    # Check if our engine
    m = re.search(r"Decode Ctx=128:\s*([0-9.]+)\s*ms/step", out)
    if m:
        return float(m.group(1))
    # Check if llama-bench
    # Output table line: | model | ... | tg 32 | <t/s> ± <std> |
    for line in out.splitlines():
        if "| qwen2.5-0.5b" in line or "| Qwen2.5-0.5B" in line:
            parts = [p.strip() for p in line.split("|")]
            # find tg column (usually 6th or 7th)
            # Example format: | model | size | params | backend | ngl | test | t/s |
            for p in parts:
                if "±" in p:
                    # e.g. "374.82 ± 1.23"
                    tok_s = float(p.split("±")[0].strip())
                    ms_step = 1000.0 / tok_s
                    return ms_step
    print("Could not parse output:")
    print(out)
    return 0.0

print("Starting 3-round interleaved benchmark at ctx=128...")
for round_idx in range(1, 4):
    print(f"\n--- Round {round_idx} / 3 ---")
    for key, (label, cmd) in MODELS.items():
        ms = run_cmd(cmd)
        results[key].append(ms)
        print(f"  {label:<30}: {ms:.4f} ms/step")

with open("interleaved_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nFinished 3-round interleaved benchmark! Summary saved to interleaved_results.json.")
