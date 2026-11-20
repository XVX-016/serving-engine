import subprocess
import csv
import json

ncu_path = r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.0\target\windows-desktop-win7-x64\ncu.exe"
engine_bin = r"build\generate.exe"

# Models to test
configs = [
    ("INT8", r"data/models/qwen2.5-0.5b-int8", 1.0),
    ("INT4 g128", r"data/models/qwen2.5-0.5b-int4-g128-head-int8", 0.515625)
]

print("=== Task P8c: Legacy vs dp4a Per-Shape Evaluation (Same Session) ===")

for mode_name, model_path, b_per_wt in configs:
    for gemv_type in ["dp4a", "legacy"]:
        cmd = [engine_bin, "--model", model_path, "--profile-decode"]
        if gemv_type == "legacy":
            cmd.append("--legacy-gemv")
            
        res = subprocess.run(cmd, capture_output=True, text=True)
        # Parse profile output
        print(f"\n--- {mode_name} ({gemv_type}) ---")
        lines = res.stdout.splitlines()
        for l in lines:
            if "Unprofiled GPU Step:" in l or "Instrumented GPU Time:" in l or "Total Kernels Launched:" in l:
                print(" ", l.strip())
            if any(op in l for op in ["gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj", "lm_head"]):
                print(" ", l.strip())

