import subprocess

ncu = r'C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.0\target\windows-desktop-win7-x64\ncu.exe'
cmd = [ncu, '--query-metrics']
res = subprocess.run(cmd, capture_output=True, text=True)
all_metrics = set(m.strip() for m in res.stdout.splitlines() if m.strip())

targets = [
    'dram__bytes_read.sum',
    'dram__bytes.sum',
    'sm__throughput.avg_pct_of_peak_sustained_elapsed',
    'sm__warps_active.avg_pct_of_peak_sustained_elapsed',
    'smsp__warps_eligible.avg_per_cycle_active',
    'smsp__warp_issue_stalled_barrier_per_warp_active.pct',
    'smsp__warp_issue_stalled_drain_per_warp_active.pct',
    'smsp__warp_issue_stalled_imc_per_warp_active.pct',
    'smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct',
    'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct',
    'smsp__warp_issue_stalled_math_throttle_per_warp_active.pct',
    'smsp__warp_issue_stalled_membar_per_warp_active.pct',
    'smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct',
    'smsp__warp_issue_stalled_misc_per_warp_active.pct',
    'smsp__warp_issue_stalled_no_instruction_per_warp_active.pct',
    'smsp__warp_issue_stalled_not_selected_per_warp_active.pct',
    'smsp__warp_issue_stalled_selected_per_warp_active.pct',
    'smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct',
    'smsp__warp_issue_stalled_sleeping_per_warp_active.pct',
    'smsp__warp_issue_stalled_tex_throttle_per_warp_active.pct',
    'smsp__warp_issue_stalled_wait_per_warp_active.pct',
    'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum',
    'l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum',
    'lts__t_sectors_op_read.sum',
    'lts__t_requests_op_read.sum',
    'smsp__inst_executed.avg_per_cycle_active',
    'smsp__inst_executed.sum'
]

valid = []
for t in targets:
    if t in all_metrics:
        print(f"VALID: {t}")
        valid.append(t)
    else:
        print(f"MISSING: {t}")

with open("scripts/valid_d2_metrics.txt", "w") as f:
    f.write(",".join(valid))
print(f"Saved {len(valid)} valid metrics to scripts/valid_d2_metrics.txt")
