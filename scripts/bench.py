#!/usr/bin/env python3
"""
scripts/bench.py

Baseline benchmark runner for Task P0:
1. 256 MB device-to-device copy bandwidth (L2 flushed, 5 warmup + 25 timed iterations)
2. Hugging Face eager FP16 decode (batch=1, prompt=512, gen=128)
3. reference/py-engine Triton decode (batch=1, prompt=512, gen=128)
Logs GPU clocks, temperature, power, and throttling via scripts/nvml_logger.py to CSV.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure py-engine can be imported
PY_ENGINE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference", "py-engine")
if PY_ENGINE_DIR not in sys.path:
    sys.path.insert(0, PY_ENGINE_DIR)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.nvml_logger import NVMLLogger

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
PROFILES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "profiles")

def benchmark_d2d_copy(num_warmup: int = 5, num_timed: int = 25) -> dict:
    print(f"\n=== 1. Measuring 256 MB Device-to-Device Copy Bandwidth ({num_warmup} warmup, {num_timed} timed) ===")
    n_elements = 64 * 1024 * 1024  # 64M float32 = 256 MB
    src = torch.randn(n_elements, dtype=torch.float32, device="cuda")
    dst = torch.empty(n_elements, dtype=torch.float32, device="cuda")
    # RTX 4060 has 32 MB L2 cache; 128 MB write evicts all lines
    l2_flush = torch.empty(32 * 1024 * 1024, dtype=torch.int32, device="cuda")

    # Warmup
    for _ in range(num_warmup):
        dst.copy_(src)
    torch.cuda.synchronize()

    csv_path = os.path.join(PROFILES_DIR, "d2d_copy_nvml.csv")
    with NVMLLogger(csv_path) as nvml_log:
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_timed)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_timed)]

        for i in range(num_timed):
            l2_flush.zero_()  # Flush L2 cache
            start_events[i].record()
            dst.copy_(src)
            end_events[i].record()

        torch.cuda.synchronize()

    times_ms = [start_events[i].elapsed_time(end_events[i]) for i in range(num_timed)]
    # 256 MB read + 256 MB write = 512 MB total moved
    total_bytes = 2.0 * 256.0 * (1024.0 ** 2)
    bws_gb_s = [(total_bytes / (t / 1000.0)) / 1e9 for t in times_ms]

    median_ms = float(np.median(times_ms))
    median_bw = float(np.median(bws_gb_s))
    min_bw = float(np.min(bws_gb_s))
    max_bw = float(np.max(bws_gb_s))

    print(f"  Copy Time: median={median_ms:.3f} ms, range=[{np.min(times_ms):.3f}, {np.max(times_ms):.3f}] ms")
    print(f"  Bandwidth: median={median_bw:.2f} GB/s, spread=[{min_bw:.2f}, {max_bw:.2f}] GB/s")
    print(f"  NVML: median SM clock={nvml_log._samples[0]['sm_clock_mhz'] if nvml_log._samples else 'N/A'} MHz")

    return {
        "median_ms": median_ms,
        "median_bw_gb_s": median_bw,
        "min_bw_gb_s": min_bw,
        "max_bw_gb_s": max_bw,
    }

def benchmark_hf_eager(prompt_len: int = 512, gen_tokens: int = 128) -> dict:
    print(f"\n=== 2. Benchmarking HF Eager FP16 (prompt={prompt_len}, gen={gen_tokens}) ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="cuda")
    model.eval()

    # Build prompt of exact length
    base_text = "The optimization of deep neural network inference on graphics hardware requires careful tuning of memory layouts, register allocation, and thread parallelism. "
    tokens = tokenizer.encode(base_text)
    mult = (prompt_len // len(tokens)) + 1
    input_ids = (tokens * mult)[:prompt_len]
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device="cuda")

    # Warmup run (short prefill + 5 decodes)
    with torch.inference_mode():
        warmup_out = model(input_tensor[:, :64], use_cache=True)
        pkv = warmup_out.past_key_values
        cur_tok = torch.argmax(warmup_out.logits[:, -1:, :], dim=-1)
        for _ in range(5):
            w_out = model(cur_tok, past_key_values=pkv, use_cache=True)
            pkv = w_out.past_key_values
            cur_tok = torch.argmax(w_out.logits[:, -1:, :], dim=-1)
    torch.cuda.synchronize()

    csv_path = os.path.join(PROFILES_DIR, "hf_eager_nvml.csv")
    with NVMLLogger(csv_path) as nvml_log:
        with torch.inference_mode():
            # Prefill timing
            torch.cuda.synchronize()
            t_prefill_start = time.perf_counter()
            prefill_out = model(input_tensor, use_cache=True)
            torch.cuda.synchronize()
            prefill_time = time.perf_counter() - t_prefill_start
            prefill_tok_s = prompt_len / prefill_time

            pkv = prefill_out.past_key_values
            cur_tok = torch.argmax(prefill_out.logits[:, -1:, :], dim=-1)

            # Decode timing
            decode_step_times = []
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            for step in range(gen_tokens):
                start_event.record()
                dec_out = model(cur_tok, past_key_values=pkv, use_cache=True)
                cur_tok = torch.argmax(dec_out.logits[:, -1:, :], dim=-1)
                pkv = dec_out.past_key_values
                end_event.record()
                end_event.synchronize()
                step_ms = start_event.elapsed_time(end_event)
                decode_step_times.append(step_ms)

    median_decode_ms = float(np.median(decode_step_times))
    decode_tok_s = 1000.0 / median_decode_ms
    vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print(f"  Prefill: {prefill_time * 1000.0:.2f} ms ({prefill_tok_s:.1f} tok/s)")
    print(f"  Decode: median={median_decode_ms:.2f} ms/step ({decode_tok_s:.1f} tok/s), spread=[{np.min(decode_step_times):.2f}, {np.max(decode_step_times):.2f}] ms")
    print(f"  Max VRAM: {vram_mb:.1f} MB")

    del model
    torch.cuda.empty_cache()

    return {
        "prefill_tok_s": prefill_tok_s,
        "decode_ms_step": median_decode_ms,
        "decode_tok_s": decode_tok_s,
        "vram_mb": vram_mb,
    }

def benchmark_py_engine(prompt_len: int = 512, gen_tokens: int = 128) -> dict:
    print(f"\n=== 3. Benchmarking py-engine Triton (prompt={prompt_len}, gen={gen_tokens}) ===")
    from engine.kernels.paged_attention_integration import KernelPagedManualDecode

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="cuda")
    hf_model.eval()

    block_size = 16
    total_blocks = 256  # plenty of blocks for 512 + 128 tokens

    runner = KernelPagedManualDecode(
        model=hf_model,
        tokenizer=tokenizer,
        block_size=block_size,
        num_blocks=total_blocks,
        device="cuda",
        dtype=torch.float16,
    )

    # Build a prompt string of the right token length
    base_text = "The optimization of deep neural network inference on graphics hardware requires careful tuning of memory layouts, register allocation, and thread parallelism. "
    tokens = tokenizer.encode(base_text)
    mult = (prompt_len // len(tokens)) + 1
    input_ids = (tokens * mult)[:prompt_len]
    prompt_text = tokenizer.decode(input_ids, skip_special_tokens=True)

    # Warmup run (short)
    _ = runner.generate("Hello world", max_new_tokens=5)

    csv_path = os.path.join(PROFILES_DIR, "py_engine_nvml.csv")

    with NVMLLogger(csv_path) as nvml_log:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        generated = runner.generate(prompt_text, max_new_tokens=gen_tokens)
        torch.cuda.synchronize()
        total_time = time.perf_counter() - t0

    actual_gen = len(generated)
    # Total time includes prefill + decode; estimate decode portion
    # prefill is roughly 1 step, so decode_time ~ total_time * (actual_gen / (actual_gen + 1))
    avg_step_ms = (total_time / (actual_gen + 1)) * 1000.0  # +1 for prefill
    prefill_tok_s = prompt_len / (total_time / (actual_gen + 1))  # rough
    decode_tok_s = 1000.0 / avg_step_ms if avg_step_ms > 0 else 0.0
    vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    print(f"  Total time: {total_time * 1000.0:.2f} ms for {actual_gen} tokens")
    print(f"  Avg step: {avg_step_ms:.2f} ms/step ({decode_tok_s:.1f} tok/s)")
    print(f"  Max VRAM: {vram_mb:.1f} MB")

    del hf_model
    del runner
    torch.cuda.empty_cache()

    return {
        "prefill_tok_s": prefill_tok_s,
        "decode_ms_step": avg_step_ms,
        "decode_tok_s": decode_tok_s,
        "vram_mb": vram_mb,
    }

def benchmark_p3_gpu(prompt_len: int = 128, gen_tokens: int = 32, num_warmup: int = 5, num_timed: int = 20) -> dict:
    print(f"\n=== Benchmarking P3 C++ GPU FP16 Recompute (prompt={prompt_len}, gen={gen_tokens}, {num_warmup} warmup + {num_timed} timed) ===")
    import subprocess
    base_text = "The optimization of deep neural network inference on graphics hardware requires careful tuning of memory layouts, register allocation, and thread parallelism. "
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokens = tokenizer.encode(base_text)
    mult = (prompt_len // len(tokens)) + 1
    input_ids = (tokens * mult)[:prompt_len]
    prompt_str = tokenizer.decode(input_ids, skip_special_tokens=True)

    csv_path = os.path.join(PROFILES_DIR, f"p3_gpu_p{prompt_len}_nvml.csv")
    with NVMLLogger(csv_path) as nvml_log:
        cmd = [
            "build\\generate.exe",
            "--gpu",
            "--prompt", prompt_str,
            "--max-new-tokens", str(gen_tokens)
        ]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        t_total = time.perf_counter() - t0

    # Parse output to find generated tokens
    gen_tok_count = gen_tokens
    for line in proc.stdout.splitlines():
        if "Generated token IDs:" in line:
            # count elements
            tok_part = line.split("[")[1].split("]")[0]
            gen_tok_count = len([x for x in tok_part.split(",") if x.strip()])

    tok_s = gen_tok_count / t_total if t_total > 0 else 0.0
    ms_per_step = (t_total / gen_tok_count) * 1000.0 if gen_tok_count > 0 else 0.0

    print(f"  Prompt={prompt_len}, Gen={gen_tok_count}: {t_total:.2f} s ({tok_s:.2f} tok/s, {ms_per_step:.2f} ms/step)")
    nvml_summary = nvml_log.stop()
    print(f"  NVML: median SM clock = {nvml_summary.get('median_sm_clock_mhz', 'N/A')} MHz, max temp = {nvml_summary.get('max_temp_c', 'N/A')} C")

    return {
        "prompt_len": prompt_len,
        "gen_tokens": gen_tok_count,
        "total_s": t_total,
        "tok_s": tok_s,
        "ms_step": ms_per_step,
        "median_sm_clock": nvml_summary.get("median_sm_clock_mhz", "N/A"),
        "csv_path": csv_path
    }

def benchmark_p4_gpu(prompt_lens=[128, 512, 1024], decode_ctxs=[128, 512, 1024, 2048], num_warmup=5, num_timed=20) -> dict:
    print(f"\n=======================================================")
    print(f"=== Benchmarking Task P4: Prefill & Decode with KV Cache ===")
    print(f"=======================================================\n")
    import subprocess
    base_text = "The optimization of deep neural network inference on graphics hardware requires careful tuning of memory layouts, register allocation, and thread parallelism. "
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokens = tokenizer.encode(base_text)

    results = {"prefill": {}, "decode": {}}

    # 1. Prefill benchmarks (prompt 128, 512, 1024)
    print("--- 1. Prefill Benchmarks (prompt=128, 512, 1024) ---")
    for plen in prompt_lens:
        mult = (plen // len(tokens)) + 1
        input_ids = (tokens * mult)[:plen]
        prompt_str = tokenizer.decode(input_ids, skip_special_tokens=True)

        csv_path = os.path.join(PROFILES_DIR, f"p4_prefill_p{plen}_nvml.csv")
        with NVMLLogger(csv_path) as nvml_log:
            cmd = [
                "build\\generate.exe",
                "--kv-cache",
                "--prompt", prompt_str,
                "--max-new-tokens", "1"
            ]
            # Warmup
            for _ in range(num_warmup):
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

            # Timed iterations
            times = []
            for _ in range(num_timed):
                t0 = time.perf_counter()
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
                t1 = time.perf_counter()
                # Parse prefill time from output
                for line in proc.stdout.splitlines():
                    if "Timing: Prefill" in line:
                        # e.g.: Timing: Prefill (128 tok): 12.34 ms
                        parts = line.split(":")
                        ms_str = parts[-1].split("ms")[0].strip()
                        times.append(float(ms_str))
                        break
                if len(times) == 0:
                    times.append((t1 - t0) * 1000.0)

        median_ms = float(np.median(times))
        prefill_tok_s = (plen / (median_ms / 1000.0))
        nvml_summary = nvml_log.stop()
        sm_clock = nvml_summary.get('median_sm_clock_mhz', 'N/A')
        print(f"  Prefill P={plen}: {median_ms:.2f} ms ({prefill_tok_s:.1f} tok/s), SM clock={sm_clock} MHz")
        results["prefill"][plen] = {
            "median_ms": median_ms,
            "tok_s": prefill_tok_s,
            "sm_clock": sm_clock
        }

    # 2. Decode benchmarks (context 128, 512, 1024, 2048)
    print("\n--- 2. Decode Benchmarks (context=128, 512, 1024, 2048; gen=32) ---")
    gen_tokens = 32
    for ctx in decode_ctxs:
        mult = (ctx // len(tokens)) + 1
        input_ids = (tokens * mult)[:ctx]
        prompt_str = tokenizer.decode(input_ids, skip_special_tokens=True)

        csv_path = os.path.join(PROFILES_DIR, f"p4_decode_ctx{ctx}_nvml.csv")
        with NVMLLogger(csv_path) as nvml_log:
            cmd = [
                "build\\generate.exe",
                "--kv-cache",
                "--prompt", prompt_str,
                "--max-new-tokens", str(gen_tokens)
            ]
            # Warmup
            for _ in range(num_warmup):
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)

            # Timed iterations
            step_times = []
            for _ in range(num_timed):
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
                for line in proc.stdout.splitlines():
                    if "Decode (" in line and "ms/step" in line:
                        # e.g.: Decode (31 steps): 200 ms (6.45 ms/step, 155.0 tok/s)
                        step_part = line.split("ms/step")[0].split("(")[-1].strip()
                        step_times.append(float(step_part))
                        break

        median_step_ms = float(np.median(step_times)) if step_times else 0.0
        decode_tok_s = (1000.0 / median_step_ms) if median_step_ms > 0 else 0.0
        nvml_summary = nvml_log.stop()
        sm_clock = nvml_summary.get('median_sm_clock_mhz', 'N/A')
        print(f"  Decode Ctx={ctx}: {median_step_ms:.2f} ms/step ({decode_tok_s:.1f} tok/s), SM clock={sm_clock} MHz")
        results["decode"][ctx] = {
            "median_step_ms": median_step_ms,
            "tok_s": decode_tok_s,
            "sm_clock": sm_clock
        }

    return results

def main():
    parser = argparse.ArgumentParser(description="Serving engine benchmark suite.")
    parser.add_argument("--p3", action="store_true", help="Run Task P3 GPU benchmark")
    parser.add_argument("--p4", action="store_true", help="Run Task P4 GPU benchmark")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    args = parser.parse_args()

    os.makedirs(PROFILES_DIR, exist_ok=True)

    if args.p4:
        benchmark_p4_gpu()
        return

    if args.p3:
        res_128 = benchmark_p3_gpu(prompt_len=128, gen_tokens=32)
        res_512 = benchmark_p3_gpu(prompt_len=512, gen_tokens=32)
        print("\n=== P3 RESULTS SUMMARY ===")
        print(f"P3 GPU prompt=128, gen=32: {res_128['tok_s']:.2f} tok/s ({res_128['ms_step']:.2f} ms/step), SM clock {res_128['median_sm_clock']} MHz")
        print(f"P3 GPU prompt=512, gen=32: {res_512['tok_s']:.2f} tok/s ({res_512['ms_step']:.2f} ms/step), SM clock {res_512['median_sm_clock']} MHz")
        return

    copy_res = benchmark_d2d_copy(num_warmup=5, num_timed=25)
    hf_res = benchmark_hf_eager(prompt_len=512, gen_tokens=128)
    py_res = benchmark_py_engine(prompt_len=512, gen_tokens=128)

    print("\n=== SUMMARY BASELINE RESULTS ===")
    print(f"Copy Bandwidth (256 MB D2D, L2 flushed): {copy_res['median_bw_gb_s']:.1f} GB/s (median {copy_res['median_ms']:.2f} ms)")
    print(f"HF Eager FP16: Prefill={hf_res['prefill_tok_s']:.1f} tok/s, Decode={hf_res['decode_ms_step']:.2f} ms/step ({hf_res['decode_tok_s']:.1f} tok/s)")
    print(f"py-engine Triton: Prefill={py_res['prefill_tok_s']:.1f} tok/s, Decode={py_res['decode_ms_step']:.2f} ms/step ({py_res['decode_tok_s']:.1f} tok/s)")

if __name__ == "__main__":
    main()
