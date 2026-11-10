import subprocess
import os

ncu_path = r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.0\target\windows-desktop-win7-x64\ncu.exe"
os.makedirs("docs/profiles", exist_ok=True)

models = [
    ("fp16", "data/models/hf-cache/snapshots/7ae557604adf67be50417f59c2c2f167def9a775", "docs/profiles/p8b_ncu_fp16.csv"),
    ("int8", "data/models/qwen2.5-0.5b-int8", "docs/profiles/p8b_ncu_int8.csv"),
    ("int4", "data/models/qwen2.5-0.5b-int4-g128", "docs/profiles/p8b_ncu_int4.csv"),
    ("int4_head_int8", "data/models/qwen2.5-0.5b-int4-g128-head-int8", "docs/profiles/p8b_ncu_int4_head_int8.csv"),
]

for name, model_path, out_csv in models:
    print(f"=== Profiling {name} with NCU ===")
    cmd = [
        ncu_path,
        "--metrics", "gpu__time_duration.sum",
        "--launch-skip", "2130",
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

print("All NCU runs finished successfully.")

