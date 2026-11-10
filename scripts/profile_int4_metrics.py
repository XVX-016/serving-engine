import subprocess

ncu_path = r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.0\target\windows-desktop-win7-x64\ncu.exe"

# Launch 2298 + 11 is gate_proj for layer 0 (grid 152, block 1024)
# Let's profile launch 2309 with DRAM throughput and issue active metrics
metrics = [
    "gpu__time_duration.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__cycles_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active"
]

cmd = [
    ncu_path,
    "--metrics", ",".join(metrics),
    "--launch-skip", "2309",
    "--launch-count", "1",
    "--csv",
    r"build\generate.exe",
    "--model", "data/models/qwen2.5-0.5b-int4-g128",
    "--profile-decode"
]

print("Profiling INT4 gate_proj kernel metrics...")
res = subprocess.run(cmd, capture_output=True, text=True)
lines = [l for l in res.stdout.splitlines() if l.startswith('"') and not l.startswith('"ID"')]
for l in lines:
    parts = l.split('","')
    print(f"Metric {parts[12]}: {parts[14]} {parts[13]}")

