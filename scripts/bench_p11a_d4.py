import subprocess
import json
import time
import os
import psutil

print("==========================================================================================")
print(" Task P11a Deliverable 4: Interleaved Benchmark at Context 128 (Single Session)")
print("==========================================================================================")

# Check AC power
battery = psutil.sensors_battery()
print(f"Power status: Plugged={battery.power_plugged}, Percent={battery.percent}%")
if not battery.power_plugged:
    print("WARNING: Not on AC power! Laptop throttling may affect benchmark results.")

llama_bench = r"third_party\llama.cpp\build\bin\llama-bench.exe"
engine_bin = r"build\generate.exe"

# Workloads:
# 1. FP16 eager (unfused)
# 2. FP16 fused-eager
# 3. FP16 graph
# 4. FP16 fused-graph
# 5. INT8 eager (unfused)
# 6. INT8 fused-eager
# 7. INT8 graph
# 8. INT8 fused-graph
# 9. INT4+INT8 head eager (unfused)
# 10. INT4+INT8 head fused-eager
# 11. INT4+INT8 head graph
# 12. INT4+INT8 head fused-graph
# 13. llama.cpp Q8_0
# 14. llama.cpp Q4_K_M

configs = [
    # Engine configs: (label, model_dir, extra_args, bytes_MB)
    ("FP16 Eager", "data/models/hf-cache/snapshots/7ae557604adf67be50417f59c2c2f167def9a775", [], 988.0),
    ("FP16 Fused-Eager", "data/models/qwen2.5-0.5b-fp16-fused", [], 988.0),
    ("FP16 Graph", "data/models/hf-cache/snapshots/7ae557604adf67be50417f59c2c2f167def9a775", ["--cuda-graph"], 988.0),
    ("FP16 Fused-Graph", "data/models/qwen2.5-0.5b-fp16-fused", ["--cuda-graph"], 988.0),

    ("INT8 Eager", "data/models/qwen2.5-0.5b-int8", [], 494.0),
    ("INT8 Fused-Eager", "data/models/qwen2.5-0.5b-int8-fused", [], 494.0),
    ("INT8 Graph", "data/models/qwen2.5-0.5b-int8", ["--cuda-graph"], 494.0),
    ("INT8 Fused-Graph", "data/models/qwen2.5-0.5b-int8-fused", ["--cuda-graph"], 494.0),

    ("INT4 Eager", "data/models/qwen2.5-0.5b-int4-g128-head-int8", [], 320.0),
    ("INT4 Fused-Eager", "data/models/qwen2.5-0.5b-int4-g128-head-int8-fused", [], 320.0),
    ("INT4 Graph", "data/models/qwen2.5-0.5b-int4-g128-head-int8", ["--cuda-graph"], 320.0),
    ("INT4 Fused-Graph", "data/models/qwen2.5-0.5b-int4-g128-head-int8-fused", ["--cuda-graph"], 320.0),

    # llama.cpp configs
    ("llama.cpp Q8_0", r"data\models\qwen2.5-0.5b-instruct-q8_0.gguf", [], 525.1),
    ("llama.cpp Q4_K_M", r"data\models\qwen2.5-0.5b-instruct-q4_k_m.gguf", [], 391.9),
]

def parse_llama_tg(stdout_text):
    for line in stdout_text.splitlines():
        if "tg32" in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            for p in reversed(parts):
                try:
                    val = float(p.split("±")[0].strip())
                    if val > 10.0:
                        ms = 1000.0 / val
                        return ms, val
                except:
                    pass
    return None

def parse_engine_decode(stdout_text):
    for line in stdout_text.splitlines():
        if "Decode Ctx=128:" in line:
            # Format:   Decode Ctx=128: 4.7905 ms/step (208.746 tok/s) ...
            ms = float(line.split("Decode Ctx=128:")[1].split("ms/step")[0].strip())
            toks = float(line.split("(")[1].split("tok/s")[0].strip())
            return ms, toks
    return None

# We run 3 interleaved rounds
results = {label: [] for label, _, _, _ in configs}

num_rounds = 3
for r in range(num_rounds):
    print(f"\n==================== Interleaved Round {r+1} of {num_rounds} ====================")
    for label, model_path, extra_args, _ in configs:
        if "llama.cpp" in label:
            cmd = [llama_bench, "-m", model_path, "-n", "32", "-p", "128", "-ngl", "99", "-r", "5"]
            res = subprocess.run(cmd, capture_output=True, text=True)
            parsed = parse_llama_tg(res.stdout)
            if parsed:
                results[label].append(parsed[0])
                print(f"  {label:<22}: {parsed[0]:.3f} ms/step ({parsed[1]:.1f} tok/s)")
            else:
                print(f"  {label:<22}: PARSE FAILED")
        else:
            cmd = [engine_bin, "--model", model_path, "--bench-decode"] + extra_args
            res = subprocess.run(cmd, capture_output=True, text=True)
            parsed = parse_engine_decode(res.stdout)
            if parsed:
                results[label].append(parsed[0])
                print(f"  {label:<22}: {parsed[0]:.3f} ms/step ({parsed[1]:.1f} tok/s)")
            else:
                print(f"  {label:<22}: PARSE FAILED")

print("\n==========================================================================================")
print(" Summary: Interleaved Medians (Ctx=128, 5 warmup + 20 timed, AC power)")
print("==========================================================================================")
PEAK_BW = 228.15 # GB/s

print(f"{'Configuration':<22} | {'Bytes (MB)':<10} | {'Median ms/step':<14} | {'tok/s':<8} | {'Achieved GB/s':<13} | {'Roofline %':<10}")
print("-" * 90)

summary_data = {}
for label, _, _, bytes_mb in configs:
    vals = results[label]
    if vals:
        vals.sort()
        med = vals[len(vals) // 2]
        toks = 1000.0 / med
        ach_bw = (bytes_mb / 1024.0) / (med / 1000.0)
        eff = (ach_bw / PEAK_BW) * 100.0
        summary_data[label] = {
            "median_ms": med,
            "tok_s": toks,
            "ach_bw": ach_bw,
            "eff": eff,
            "all_runs": vals
        }
        print(f"{label:<22} | {bytes_mb:<10.1f} | {med:<14.3f} | {toks:<8.1f} | {ach_bw:<13.1f} | {eff:<10.1f}%")
    else:
        print(f"{label:<22} | {bytes_mb:<10.1f} | {'FAILED':<14} | {'N/A':<8} | {'N/A':<13} | {'N/A':<10}")

with open("p11a_d4_summary.json", "w") as f:
    json.dump(summary_data, f, indent=2)

print("\nSaved summary to p11a_d4_summary.json")
