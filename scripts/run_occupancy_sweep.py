import subprocess
import csv
import json
import io

shapes = [
    ("qkv", 0, 1152, 896),
    ("o_proj", 1, 896, 896),
    ("gate_up", 2, 9728, 896),
    ("down_proj", 3, 896, 4864),
    ("lm_head", 4, 151936, 896),
]

kernels = [
    ("Legacy", 0),
    ("Stripped", 1),
    ("Split-K", 2),
]

metrics = [
    "launch__waves_per_multiprocessor",
    "launch__registers_per_thread",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__time_duration.sum",
    "smsp__inst_executed.sum",
]

metrics_str = ",".join(metrics)

NCU_PATH = r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.0\ncu.bat"

results = []

for s_name, s_idx, N, K in shapes:
    for k_name, k_idx in kernels:
        cmd = [
            NCU_PATH, "--csv", "--metrics", metrics_str,
            "build/tests/profile_single_kernel.exe", str(s_idx), str(k_idx)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        # Parse CSV from output
        lines = res.stdout.splitlines()
        csv_lines = [l for l in lines if l.startswith('"')]
        reader = csv.DictReader(io.StringIO("\n".join(csv_lines)))
        row_dict = {"shape": s_name, "N": N, "K": K, "kernel": k_name}
        for row in reader:
            m_name = row.get("Metric Name")
            m_val = row.get("Metric Value", "").replace(",", "")
            if m_name in metrics:
                row_dict[m_name] = float(m_val)
        
        # Resident blocks per SM on sm_89 (1024 threads/block = 32 warps/block; max 48 warps/SM -> resident blocks = 1)
        row_dict["resident_blocks_per_sm"] = 1
        
        # Calculate full waves and last wave utilization
        # Grid size
        if k_name in ["Legacy", "Stripped"]:
            grid = (N + 31) // 32
        else:
            grid = (N + 15) // 16
        waves = grid / 24.0
        full_waves = int(grid // 24)
        rem = grid % 24
        last_wave_util = (rem / 24.0 * 100.0) if rem > 0 else (100.0 if grid > 0 else 0.0)
        row_dict["grid"] = grid
        row_dict["waves"] = waves
        row_dict["full_waves"] = full_waves
        row_dict["last_wave_util_pct"] = last_wave_util
        
        results.append(row_dict)
        print(f"Done {s_name} {k_name}: time={row_dict.get('gpu__time_duration.sum', 0)/1000.0:.2f} us, waves={waves:.2f}, dram={row_dict.get('dram__throughput.avg.pct_of_peak_sustained_elapsed', 0):.1f}%, inst={row_dict.get('smsp__inst_executed.sum', 0):.0f}")

with open("scripts/occupancy_and_timer_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved scripts/occupancy_and_timer_results.json")
