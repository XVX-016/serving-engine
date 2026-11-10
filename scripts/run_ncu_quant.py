import subprocess
import os

ncu_path = r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.0\target\windows-desktop-win7-x64\ncu.exe"
os.makedirs("docs/profiles", exist_ok=True)

models = [
    ("int8", "data/models/qwen2.5-0.5b-int8", "docs/profiles/p8b_ncu_int8.csv"),
    ("int4", "data/models/qwen2.5-0.5b-int4-g128", "docs/profiles/p8b_ncu_int4.csv"),
]

for name, model_path, out_csv in models:
    print(f"=== Profiling {name} with NCU (skip 2298, count 339) ===")
    cmd = [
        ncu_path,
        "--metrics", "gpu__time_duration.sum",
        "--launch-skip", "2298",
        "--launch-count", "339",
        "--csv",
        r"build\generate.exe",
        "--model", model_path,
        "--profile-decode"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    with open(out_csv, "w") as f:
        f.write(res.stdout)
    lines = [l for l in res.stdout.splitlines() if l.startswith('"') and not l.startswith('"ID"')]
    print(f"Captured {len(lines)} kernels to {out_csv}")

print("Both NCU runs finished successfully.")

