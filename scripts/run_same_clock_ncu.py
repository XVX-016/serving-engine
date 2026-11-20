import subprocess
import csv
import re
import os

NCU_PATH = r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.0\ncu.bat"

models = [
    ("fp16", r"data/models/qwen2.5-0.5b-fp16-fused", "p11c_fp16_ncu.csv", 989728000),
    ("int8", r"data/models/qwen2.5-0.5b-int8-fused", "p11c_int8_ncu.csv", 496678018),
    ("int4", r"data/models/qwen2.5-0.5b-int4-g128-head-int8-fused", "p11c_int4_ncu.csv", 322747522)
]

for name, model_path, out_csv, total_bytes in models:
    print(f"==================================================")
    print(f"Running Same-Clock NCU Profiling for {name.upper()}")
    print(f"==================================================")
    cmd = [
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
    res = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(res.stdout)
        f.write("\n=== STDERR ===\n")
        f.write(res.stderr)
    print(f"Saved {out_csv} ({len(res.stdout)} chars)")

print("\nDone same-clock NCU profiling!")
