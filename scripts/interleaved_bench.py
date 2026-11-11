import subprocess
import json
import time

llama_bench = r"third_party\llama.cpp\build\bin\llama-bench.exe"
engine_bin = r"build\generate.exe"

print("==========================================================================================")
print(" Task P8c Deliverable 5: Interleaved Comparison at Context 128 (Single Session)")
print("==========================================================================================")

# llama.cpp configs: (name, path, bytes_read_MB)
# As specified: Q8_0 = 525.1 MB, Q4_K_M = 391.9 MB
llama_runs = [
    ("llama.cpp Q8_0", r"data\models\qwen2.5-0.5b-instruct-q8_0.gguf", 525.1),
    ("llama.cpp Q4_K_M", r"data\models\qwen2.5-0.5b-instruct-q4_k_m.gguf", 391.9)
]

# Engine configs:
engine_runs = [
    ("Engine INT8 (INT8 head)", r"data/models/qwen2.5-0.5b-int8", 494.0),
    ("Engine INT4 g128 (INT8 head)", r"data/models/qwen2.5-0.5b-int4-g128-head-int8", 320.0)
]

results = {name: [] for name, _, _ in llama_runs}
for name, _, _ in engine_runs:
    results[name] = []

PEAK_BW = 228.15 # GB/s

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

for round_idx in range(3):
    print(f"\n--- Round {round_idx + 1} of 3 ---")
    
    # 1. llama.cpp Q8_0
    cmd_q8 = [llama_bench, "-m", llama_runs[0][1], "-n", "32", "-p", "128", "-ngl", "99", "-r", "5"]
    res = subprocess.run(cmd_q8, capture_output=True, text=True)
    parsed = parse_llama_tg(res.stdout)
    if parsed:
        results["llama.cpp Q8_0"].append(parsed)
        print(f"  llama.cpp Q8_0: {parsed[0]:.3f} ms/step ({parsed[1]:.1f} tok/s)")
    else:
        print("  llama.cpp Q8_0: PARSE FAILED")
            
    # 2. Engine INT8 (dp4a)
    cmd_e_int8 = [engine_bin, "--model", engine_runs[0][1], "--bench-decode"]
    res = subprocess.run(cmd_e_int8, capture_output=True, text=True)
    parsed = parse_engine_decode(res.stdout)
    if parsed:
        results["Engine INT8 (INT8 head)"].append(parsed)
        print(f"  Engine INT8 (dp4a): {parsed[0]:.3f} ms/step ({parsed[1]:.1f} tok/s)")
    else:
        print("  Engine INT8: PARSE FAILED")
            
    # 3. llama.cpp Q4_K_M
    cmd_q4 = [llama_bench, "-m", llama_runs[1][1], "-n", "32", "-p", "128", "-ngl", "99", "-r", "5"]
    res = subprocess.run(cmd_q4, capture_output=True, text=True)
    parsed = parse_llama_tg(res.stdout)
    if parsed:
        results["llama.cpp Q4_K_M"].append(parsed)
        print(f"  llama.cpp Q4_K_M: {parsed[0]:.3f} ms/step ({parsed[1]:.1f} tok/s)")
    else:
        print("  llama.cpp Q4_K_M: PARSE FAILED")

    # 4. Engine INT4 + INT8 head (dp4a)
    cmd_e_int4 = [engine_bin, "--model", engine_runs[1][1], "--bench-decode"]
    res = subprocess.run(cmd_e_int4, capture_output=True, text=True)
    parsed = parse_engine_decode(res.stdout)
    if parsed:
        results["Engine INT4 g128 (INT8 head)"].append(parsed)
        print(f"  Engine INT4 g128 (dp4a): {parsed[0]:.3f} ms/step ({parsed[1]:.1f} tok/s)")
    else:
        print("  Engine INT4: PARSE FAILED")

print("\n==========================================================================================")
print(" Summary: Interleaved Medians and Efficiency vs Read-Bandwidth Roofline (228.15 GB/s)")
print("==========================================================================================")
traffic_map = {
    "llama.cpp Q8_0": 525.1,
    "llama.cpp Q4_K_M": 391.9,
    "Engine INT8 (INT8 head)": 494.0,
    "Engine INT4 g128 (INT8 head)": 320.0
}

for name, vals in results.items():
    if not vals:
        print(f"{name:<30}: NO DATA")
        continue
    ms_list = sorted([v[0] for v in vals])
    tok_list = sorted([v[1] for v in vals])
    med_ms = ms_list[len(ms_list)//2]
    med_tok = tok_list[len(tok_list)//2]
    
    traffic_mb = traffic_map[name]
    ceiling_ms = traffic_mb / PEAK_BW
    achieved_bw = traffic_mb / med_ms
    pct_roofline = (ceiling_ms / med_ms) * 100.0
    
    print(f"{name:<30} | {med_ms:5.2f} ms/step ({med_tok:5.1f} tok/s) | Read: {traffic_mb:5.1f} MB | Ceiling: {ceiling_ms:4.2f} ms | Achieved: {achieved_bw:5.1f} GB/s ({pct_roofline:4.1f}% of roofline)")
