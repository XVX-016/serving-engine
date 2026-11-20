import subprocess
import csv

ncu_path = r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.0\target\windows-desktop-win7-x64\ncu.exe"
metrics = [
    "gpu__time_duration.sum",
    "smsp__thread_inst_executed.sum",
    "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed"
]

cases = [
    ("INT8 dp4a gate_proj", r"data/models/qwen2.5-0.5b-int8", 2309, 4864, 896, 1.0, False),
    ("INT8 legacy gate_proj", r"data/models/qwen2.5-0.5b-int8", 2309, 4864, 896, 1.0, True),
    ("INT4 dp4a gate_proj", r"data/models/qwen2.5-0.5b-int4-g128", 2309, 4864, 896, 0.515625, False),
    ("INT4 legacy gate_proj", r"data/models/qwen2.5-0.5b-int4-g128", 2309, 4864, 896, 0.515625, True),
    ("INT4+INT8head dp4a head", r"data/models/qwen2.5-0.5b-int4-g128-head-int8", 2637, 151936, 896, 1.0, False),
    ("INT4+INT8head legacy head", r"data/models/qwen2.5-0.5b-int4-g128-head-int8", 2637, 151936, 896, 1.0, True),
]

for label, mod_path, skip, N, K, b_per_w, leg in cases:
    wt_bytes = N * K * b_per_w
    cmd = [
        ncu_path,
        "--metrics", ",".join(metrics),
        "--launch-skip", str(skip),
        "--launch-count", "1",
        "--csv",
        r"build\generate.exe",
        "--model", mod_path,
        "--profile-decode"
    ]
    if leg:
        cmd.append("--legacy-gemv")
    res = subprocess.run(cmd, capture_output=True, text=True)
    lines = [l for l in res.stdout.splitlines() if l.startswith('"') and not l.startswith('"ID"')]
    m_vals = {}
    for l in lines:
        parts = list(csv.reader([l]))[0]
        m_vals[parts[12]] = parts[14]
    dur_ns = float(m_vals.get("gpu__time_duration.sum", "0").replace(',', ''))
    dur_us = dur_ns / 1e3
    th_inst = float(m_vals.get("smsp__thread_inst_executed.sum", "0").replace(',', ''))
    alu_pct = float(m_vals.get("sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active", "0").replace(',', ''))
    dram_pct = float(m_vals.get("dram__throughput.avg.pct_of_peak_sustained_elapsed", "0").replace(',', ''))
    gb_s = (wt_bytes / (dur_ns * 1e-9)) / 1e9 if dur_ns > 0 else 0
    inst_per_wt = th_inst / (N * K) if (N * K) > 0 else 0
    print(f"{label:<26} | {dur_us:7.2f} us | {gb_s:6.2f} GB/s | DRAM {dram_pct:4.1f}% | ALU {alu_pct:4.1f}% | Inst/wt: {inst_per_wt:5.2f}")

