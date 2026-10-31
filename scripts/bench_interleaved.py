#!/usr/bin/env python3
"""
scripts/bench_interleaved.py

Interleaved re-run benchmark for Task P0b:
Runs py-engine and llama.cpp F16 interleaved in order A B A B A B (3 reps each).
Records ms/step, tok/s, median SM clock, memory clock, power, and NVML throttle reasons.
"""

import os
import sys
import time
import subprocess
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

PY_ENGINE_DIR = os.path.join(ROOT_DIR, "reference", "py-engine")
if PY_ENGINE_DIR not in sys.path:
    sys.path.insert(0, PY_ENGINE_DIR)

from scripts.nvml_logger import NVMLLogger

PROFILES_DIR = os.path.join(ROOT_DIR, "docs", "profiles")
os.makedirs(PROFILES_DIR, exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LLAMA_BENCH_EXE = os.path.join(ROOT_DIR, "third_party", "llama.cpp", "build", "bin", "llama-bench.exe")
FP16_GGUF = os.path.join(ROOT_DIR, "data", "models", "qwen2.5-0.5b-instruct-fp16.gguf")

def run_py_engine(rep_idx: int, runner, prompt_text: str, gen_tokens: int = 128):
    csv_path = os.path.join(PROFILES_DIR, f"interleaved_pyengine_rep{rep_idx}.csv")
    with NVMLLogger(csv_path) as nvml_log:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        generated = runner.generate(prompt_text, max_new_tokens=gen_tokens)
        torch.cuda.synchronize()
        total_time = time.perf_counter() - t0

    actual_gen = len(generated)
    avg_step_ms = (total_time / (actual_gen + 1)) * 1000.0
    decode_tok_s = 1000.0 / avg_step_ms if avg_step_ms > 0 else 0.0

    # Summarize NVML
    sm_clocks = [s["sm_clock_mhz"] for s in nvml_log._samples] if nvml_log._samples else []
    median_sm = int(np.median(sm_clocks)) if sm_clocks else 0
    throttles = set()
    for s in nvml_log._samples:
        for r in s["throttle_reasons"].split("|"):
            if r != "NONE":
                throttles.add(r)
    throttle_str = "|".join(sorted(throttles)) if throttles else "NONE"

    print(f"[Run A{rep_idx}] py-engine: total={total_time*1000.0:.2f}ms, step={avg_step_ms:.2f}ms/step, tok/s={decode_tok_s:.1f}, SM_clock={median_sm}MHz, throttle={throttle_str}")
    return {
        "step_ms": avg_step_ms,
        "tok_s": decode_tok_s,
        "median_sm_mhz": median_sm,
        "throttle": throttle_str,
        "csv": csv_path
    }

def run_llama_cpp(rep_idx: int, gen_tokens: int = 128):
    csv_path = os.path.join(PROFILES_DIR, f"interleaved_llamacpp_rep{rep_idx}.csv")
    cmd = [
        LLAMA_BENCH_EXE,
        "-m", FP16_GGUF,
        "-p", "512",
        "-n", str(gen_tokens),
        "-ngl", "99",
        "-r", "1"
    ]
    with NVMLLogger(csv_path) as nvml_log:
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", check=True)
        total_time = time.perf_counter() - t0

    # Parse tg128 from output
    import re
    tg_tok_s = 0.0
    for line in proc.stdout.splitlines():
        if "tg128" in line:
            m = re.search(r"tg128\s*\|\s*([0-9.]+)", line)
            if m:
                tg_tok_s = float(m.group(1))

    step_ms = 1000.0 / tg_tok_s if tg_tok_s > 0 else 0.0
    sm_clocks = [s["sm_clock_mhz"] for s in nvml_log._samples] if nvml_log._samples else []
    median_sm = int(np.median(sm_clocks)) if sm_clocks else 0
    throttles = set()
    for s in nvml_log._samples:
        for r in s["throttle_reasons"].split("|"):
            if r != "NONE":
                throttles.add(r)
    throttle_str = "|".join(sorted(throttles)) if throttles else "NONE"

    print(f"[Run B{rep_idx}] llama.cpp: tg128={tg_tok_s:.2f} tok/s ({step_ms:.2f} ms/step), SM_clock={median_sm}MHz, throttle={throttle_str}")
    return {
        "step_ms": step_ms,
        "tok_s": tg_tok_s,
        "median_sm_mhz": median_sm,
        "throttle": throttle_str,
        "csv": csv_path
    }

def main():
    print("=== Initializing py-engine for interleaved benchmark ===")
    from engine.kernels.paged_attention_integration import KernelPagedManualDecode

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="cuda")
    hf_model.eval()

    runner = KernelPagedManualDecode(
        model=hf_model,
        tokenizer=tokenizer,
        block_size=16,
        num_blocks=256,
        device="cuda",
        dtype=torch.float16,
    )

    base_text = "The optimization of deep neural network inference on graphics hardware requires careful tuning of memory layouts, register allocation, and thread parallelism. "
    tokens = tokenizer.encode(base_text)
    mult = (512 // len(tokens)) + 1
    input_ids = (tokens * mult)[:512]
    prompt_text = tokenizer.decode(input_ids, skip_special_tokens=True)

    # Warmup both
    print("Warming up py-engine...")
    _ = runner.generate("Warmup test", max_new_tokens=5)
    print("Warming up llama.cpp...")
    subprocess.run([LLAMA_BENCH_EXE, "-m", FP16_GGUF, "-p", "64", "-n", "8", "-ngl", "99", "-r", "1"],
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    py_results = []
    llama_results = []

    print("\n=== Starting Interleaved A B A B A B Runs ===")
    for rep in range(1, 4):
        print(f"\n--- Iteration {rep}/3 ---")
        # Run A: py-engine
        res_a = run_py_engine(rep, runner, prompt_text, 128)
        py_results.append(res_a)
        time.sleep(2)  # Cooldown between runs

        # Run B: llama.cpp
        res_b = run_llama_cpp(rep, 128)
        llama_results.append(res_b)
        time.sleep(2)

    print("\n=== Summary of Interleaved Benchmark ===")
    print("System | Rep 1 tok/s | Rep 2 tok/s | Rep 3 tok/s | Median SM Clock | Throttling Observed")
    print("-" * 85)
    py_toks = [r["tok_s"] for r in py_results]
    py_sms = [r["median_sm_mhz"] for r in py_results]
    py_throttles = set(r["throttle"] for r in py_results)
    print(f"py-engine | {py_toks[0]:.1f} | {py_toks[1]:.1f} | {py_toks[2]:.1f} | {int(np.median(py_sms))} MHz | {', '.join(py_throttles)}")

    ll_toks = [r["tok_s"] for r in llama_results]
    ll_sms = [r["median_sm_mhz"] for r in llama_results]
    ll_throttles = set(r["throttle"] for r in llama_results)
    print(f"llama.cpp | {ll_toks[0]:.1f} | {ll_toks[1]:.1f} | {ll_toks[2]:.1f} | {int(np.median(ll_sms))} MHz | {', '.join(ll_throttles)}")

if __name__ == "__main__":
    main()
