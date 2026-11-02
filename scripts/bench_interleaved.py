#!/usr/bin/env python3
"""
scripts/bench_interleaved.py

Interleaved benchmark for Task P0c:
Runs py-engine, llama.cpp F16, llama.cpp Q8_0, and llama.cpp Q4_K_M
in interleaved order across 5 repetitions on AC power.
Logs NVML clocks, power, temperature, and throttle reasons.
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
Q8_0_GGUF = os.path.join(ROOT_DIR, "data", "models", "qwen2.5-0.5b-instruct-q8_0.gguf")
Q4_K_M_GGUF = os.path.join(ROOT_DIR, "data", "models", "qwen2.5-0.5b-instruct-q4_k_m.gguf")

def run_py_engine(rep_idx: int, runner, prompt_text: str, gen_tokens: int = 128):
    csv_path = os.path.join(PROFILES_DIR, f"p0c_interleaved_pyengine_rep{rep_idx}.csv")
    with NVMLLogger(csv_path) as nvml_log:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        generated = runner.generate(prompt_text, max_new_tokens=gen_tokens)
        torch.cuda.synchronize()
        total_time = time.perf_counter() - t0

    actual_gen = len(generated)
    avg_step_ms = (total_time / (actual_gen + 1)) * 1000.0
    decode_tok_s = 1000.0 / avg_step_ms if avg_step_ms > 0 else 0.0

    sm_clocks = [s["sm_clock_mhz"] for s in nvml_log._samples] if nvml_log._samples else []
    median_sm = int(np.median(sm_clocks)) if sm_clocks else 0
    powers = [s["power_w"] for s in nvml_log._samples] if nvml_log._samples else []
    median_power = float(np.median(powers)) if powers else 0.0
    throttles = set()
    for s in nvml_log._samples:
        for r in s["throttle_reasons"].split("|"):
            if r != "NONE":
                throttles.add(r)
    throttle_str = "|".join(sorted(throttles)) if throttles else "NONE"

    print(f"[Rep {rep_idx}] py-engine: total={total_time*1000.0:.2f}ms, step={avg_step_ms:.2f}ms/step, tok/s={decode_tok_s:.1f}, SM={median_sm}MHz, Power={median_power:.1f}W, throttle={throttle_str}")
    return {
        "step_ms": avg_step_ms,
        "tok_s": decode_tok_s,
        "median_sm_mhz": median_sm,
        "median_power_w": median_power,
        "throttle": throttle_str,
        "csv": csv_path
    }

def run_llama_cpp(name: str, model_path: str, rep_idx: int, gen_tokens: int = 128):
    csv_path = os.path.join(PROFILES_DIR, f"p0c_interleaved_{name}_rep{rep_idx}.csv")
    cmd = [
        LLAMA_BENCH_EXE,
        "-m", model_path,
        "-p", "512",
        "-n", str(gen_tokens),
        "-ngl", "99",
        "-r", "1"
    ]
    with NVMLLogger(csv_path) as nvml_log:
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", check=True)
        total_time = time.perf_counter() - t0

    import re
    tg_tok_s = 0.0
    pp_tok_s = 0.0
    for line in proc.stdout.splitlines():
        if "tg128" in line:
            m = re.search(r"tg128\s*\|\s*([0-9.]+)", line)
            if m:
                tg_tok_s = float(m.group(1))
        if "pp512" in line:
            m = re.search(r"pp512\s*\|\s*([0-9.]+)", line)
            if m:
                pp_tok_s = float(m.group(1))

    step_ms = 1000.0 / tg_tok_s if tg_tok_s > 0 else 0.0
    sm_clocks = [s["sm_clock_mhz"] for s in nvml_log._samples] if nvml_log._samples else []
    median_sm = int(np.median(sm_clocks)) if sm_clocks else 0
    powers = [s["power_w"] for s in nvml_log._samples] if nvml_log._samples else []
    median_power = float(np.median(powers)) if powers else 0.0
    throttles = set()
    for s in nvml_log._samples:
        for r in s["throttle_reasons"].split("|"):
            if r != "NONE":
                throttles.add(r)
    throttle_str = "|".join(sorted(throttles)) if throttles else "NONE"

    print(f"[Rep {rep_idx}] llama.cpp {name}: tg128={tg_tok_s:.2f} tok/s ({step_ms:.2f} ms/step), pp512={pp_tok_s:.1f} tok/s, SM={median_sm}MHz, Power={median_power:.1f}W, throttle={throttle_str}")
    return {
        "step_ms": step_ms,
        "tok_s": tg_tok_s,
        "pp_tok_s": pp_tok_s,
        "median_sm_mhz": median_sm,
        "median_power_w": median_power,
        "throttle": throttle_str,
        "csv": csv_path
    }

def main():
    print("=== Initializing py-engine for extended interleaved benchmark ===")
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

    print("Warming up py-engine...")
    _ = runner.generate("Warmup test", max_new_tokens=5)
    print("Warming up llama.cpp...")
    for mpath in [FP16_GGUF, Q8_0_GGUF, Q4_K_M_GGUF]:
        subprocess.run([LLAMA_BENCH_EXE, "-m", mpath, "-p", "64", "-n", "8", "-ngl", "99", "-r", "1"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    results = {
        "py_engine": [],
        "llama_f16": [],
        "llama_q8_0": [],
        "llama_q4_k_m": []
    }

    n_reps = 5
    print(f"\n=== Starting Interleaved 4-Config Benchmark across {n_reps} Repetitions ===")
    for rep in range(1, n_reps + 1):
        print(f"\n---------------------------------------------")
        print(f"--- Repetition {rep}/{n_reps} ---")
        print(f"---------------------------------------------")

        # 1. py-engine
        r_py = run_py_engine(rep, runner, prompt_text, 128)
        results["py_engine"].append(r_py)
        time.sleep(2)

        # 2. llama.cpp F16
        r_f16 = run_llama_cpp("F16", FP16_GGUF, rep, 128)
        results["llama_f16"].append(r_f16)
        time.sleep(2)

        # 3. llama.cpp Q8_0
        r_q8 = run_llama_cpp("Q8_0", Q8_0_GGUF, rep, 128)
        results["llama_q8_0"].append(r_q8)
        time.sleep(2)

        # 4. llama.cpp Q4_K_M
        r_q4 = run_llama_cpp("Q4_K_M", Q4_K_M_GGUF, rep, 128)
        results["llama_q4_k_m"].append(r_q4)
        time.sleep(2)

    print("\n" + "="*90)
    print("=== SUMMARY OF INTERLEAVED BENCHMARK (5 Reps) ===")
    print("="*90)
    print(f"{'Config':<16} | {'Median tok/s':<12} | {'Min..Max tok/s':<18} | {'Median ms':<10} | {'Median SM':<10} | {'Power (W)':<10} | {'Throttle':<15}")
    print("-" * 90)

    for cfg_key, cfg_label in [
        ("py_engine", "py-engine"),
        ("llama_f16", "llama.cpp F16"),
        ("llama_q8_0", "llama.cpp Q8_0"),
        ("llama_q4_k_m", "llama.cpp Q4_K_M")
    ]:
        toks = [r["tok_s"] for r in results[cfg_key]]
        steps = [r["step_ms"] for r in results[cfg_key]]
        sms = [r["median_sm_mhz"] for r in results[cfg_key]]
        powers = [r["median_power_w"] for r in results[cfg_key]]
        throttles = set(r["throttle"] for r in results[cfg_key])
        med_tok = float(np.median(toks))
        min_tok = min(toks)
        max_tok = max(toks)
        med_step = float(np.median(steps))
        med_sm = int(np.median(sms))
        med_pwr = float(np.median(powers))
        thr_str = "|".join(sorted(throttles)) if throttles else "NONE"
        print(f"{cfg_label:<16} | {med_tok:<12.2f} | [{min_tok:.1f} .. {max_tok:.1f}]      | {med_step:<10.2f} | {med_sm:<10} | {med_pwr:<10.1f} | {thr_str:<15}")

if __name__ == "__main__":
    main()
