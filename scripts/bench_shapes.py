import subprocess
import csv

ncu_path = r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.0\target\windows-desktop-win7-x64\ncu.exe"
engine_bin = r"build\generate.exe"

shapes = [
    ("q_proj", 896, 896),
    ("k_proj", 128, 896),
    ("v_proj", 128, 896),
    ("o_proj", 896, 896),
    ("gate_proj", 4864, 896),
    ("up_proj", 4864, 896),
    ("down_proj", 896, 4864),
    ("lm_head", 151936, 896)
]

print("=== Benchmarking Per-Shape GEMV Latencies (dp4a vs Legacy) ===")
# We will run a script or profile to measure the time for each shape directly

