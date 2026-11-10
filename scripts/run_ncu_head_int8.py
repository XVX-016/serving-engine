import subprocess
import os

ncu_path = r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.0\target\windows-desktop-win7-x64\ncu.exe"
out_csv = "docs/profiles/p8b_ncu_int4_head_int8.csv"

cmd = [
    ncu_path,
    "--metrics", "gpu__time_duration.sum",
    "--launch-skip", "2298",
    "--launch-count", "339",
    "--csv",
    r"build\generate.exe",
    "--model", "data/models/qwen2.5-0.5b-int4-g128-head-int8",
    "--profile-decode"
]

print("Profiling INT4 with true INT8 head using NCU...")
res = subprocess.run(cmd, capture_output=True, text=True)
with open(out_csv, "w") as f:
    f.write(res.stdout)
lines = [l for l in res.stdout.splitlines() if l.startswith('"') and not l.startswith('"ID"')]
print(f"Captured {len(lines)} kernels to {out_csv}")

