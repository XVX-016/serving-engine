import subprocess
import csv
import sys
import os

NCU_PATH = r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.0\ncu.bat"

configs = [
    ("fp16", "data/models/qwen2.5-0.5b-fp16-fused", "fp16_ncu_uncontrolled.csv", "fp16_ncu_locked.csv"),
    ("int8", "data/models/qwen2.5-0.5b-int8-fused", "int8_ncu_uncontrolled.csv", "int8_ncu_locked.csv"),
    ("int4", "data/models/qwen2.5-0.5b-int4-g128-head-int8-fused", "int4_ncu_uncontrolled.csv", "int4_ncu_locked.csv"),
]

for name, model_path, unctrl_csv, locked_csv in configs:
    print(f"==================================================")
    print(f"Profiling {name}: Uncontrolled clocks & caches")
    print(f"==================================================")
    cmd_unctrl = [
        NCU_PATH,
        "--profile-from-start", "0",
        "--graph-profiling", "node",
        "--clock-control", "none",
        "--cache-control", "none",
        "--metrics", "gpu__time_duration.sum",
        "--csv",
        r"build\generate.exe",
        "--model", model_path,
        "--profile-graph-step"
    ]
    res_unctrl = subprocess.run(cmd_unctrl, capture_output=True, text=True, shell=True)
    with open(unctrl_csv, "w", encoding="utf-8") as f:
        f.write(res_unctrl.stdout)
        f.write("\n=== STDERR ===\n")
        f.write(res_unctrl.stderr)
    print(f"Saved {unctrl_csv}")
    
    print(f"==================================================")
    print(f"Profiling {name}: Default locked clocks & reset caches")
    print(f"==================================================")
    cmd_locked = [
        NCU_PATH,
        "--profile-from-start", "0",
        "--graph-profiling", "node",
        "--metrics", "gpu__time_duration.sum",
        "--csv",
        r"build\generate.exe",
        "--model", model_path,
        "--profile-graph-step"
    ]
    res_locked = subprocess.run(cmd_locked, capture_output=True, text=True, shell=True)
    with open(locked_csv, "w", encoding="utf-8") as f:
        f.write(res_locked.stdout)
        f.write("\n=== STDERR ===\n")
        f.write(res_locked.stderr)
    print(f"Saved {locked_csv}")

print("All profiling complete!")
