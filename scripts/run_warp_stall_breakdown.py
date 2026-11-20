import subprocess
import csv

ncu = r'C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.0\target\windows-desktop-win7-x64\ncu.exe'

# Warp state stall reasons as percentage of cycles or average warps
stall_reasons = [
    "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    "smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct",
    "smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct",
    "smsp__warp_issue_stalled_drain_per_warp_active.pct",
    "smsp__warp_issue_stalled_imc_miss_per_warp_active.pct",
    "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_membar_per_warp_active.pct",
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_misc_per_warp_active.pct",
    "smsp__warp_issue_stalled_no_instruction_per_warp_active.pct",
    "smsp__warp_issue_stalled_not_selected_per_warp_active.pct",
    "smsp__warp_issue_stalled_selected_per_warp_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_sleeping_per_warp_active.pct",
    "smsp__warp_issue_stalled_tex_throttle_per_warp_active.pct",
    "smsp__warp_issue_stalled_wait_per_warp_active.pct"
]

print("Profiling INT4 gate_proj warp-state stalls...")
cmd = [
    ncu,
    '--kernel-name-base', 'function',
    '--kernel-name', 'regex:gemv_int4',
    '--launch-count', '1',
    '--metrics', ','.join(stall_reasons),
    '--csv',
    r'build\generate.exe',
    '--model', 'data/models/qwen2.5-0.5b-int4-g128-head-int8',
    '--profile-decode'
]
res = subprocess.run(cmd, capture_output=True, text=True)
lines = [l for l in res.stdout.splitlines() if l.startswith('"') and not l.startswith('"ID"')]

print("\n=== Warp State Breakdown for INT4 gate_proj (__dp4a) ===")
stalls = {}
for l in lines:
    parts = list(csv.reader([l]))[0]
    metric_name = parts[12]
    metric_val = parts[14]
    stalls[metric_name] = float(metric_val.replace(',', ''))

# Sort by percentage descending
for k, v in sorted(stalls.items(), key=lambda x: x[1], reverse=True):
    short_k = k.replace("smsp__warp_issue_stalled_", "").replace("_per_warp_active.pct", "")
    print(f"  {short_k:<25}: {v:6.2f}%")

