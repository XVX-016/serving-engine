import subprocess
import csv

ncu_path = r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.0\target\windows-desktop-win7-x64\ncu.exe"

metrics = [
    "gpu__time_duration.sum",
    "sm__inst_executed.sum",
    "smsp__thread_inst_executed.sum",
    "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed"
]

shapes = [
    ("q_proj (896->896)", 2300, 896, 896),
    ("k_proj (896->128)", 2301, 128, 896),
    ("v_proj (896->128)", 2302, 128, 896),
    ("o_proj (896->896)", 2307, 896, 896),
    ("gate_proj (896->4864)", 2309, 4864, 896),
    ("up_proj (896->4864)", 2310, 4864, 896),
    ("down_proj (4864->896)", 2312, 896, 4864),
    ("lm_head (896->151936)", 2636, 151936, 896)
]

models = [
    ("INT8", r"data/models/qwen2.5-0.5b-int8", 1.0),
    ("INT4 g128", r"data/models/qwen2.5-0.5b-int4-g128", 0.515625),
    ("INT4 + INT8 head", r"data/models/qwen2.5-0.5b-int4-g128-head-int8", 0.515625)
]

print("=== Task P8c Deliverable 1: Baseline NCU Instruction & Bandwidth Profiling ===")

for mod_name, mod_path, bytes_per_wt in models:
    print(f"\n=================== Model: {mod_name} ===================")
    print(f"{'Shape':<25} | {'Duration (us)':<14} | {'Traffic (KB)':<14} | {'GB/s':<10} | {'DRAM %':<8} | {'ALU %':<8} | {'Thread Inst':<14} | {'Inst/Weight':<12}")
    print("-" * 120)
    
    for s_name, skip, N, K in shapes:
        if "lm_head" in s_name:
            if mod_name == "INT8":
                wt_bytes = N * K * 2.0 # FP16 head
            elif "head" in mod_name:
                wt_bytes = N * K * 1.0 # INT8 head
            else:
                wt_bytes = N * K * 2.0 # FP16 head
        else:
            wt_bytes = N * K * bytes_per_wt
            
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
        total_weights = N * K
        inst_per_wt = th_inst / total_weights if total_weights > 0 else 0
        
        print(f"{s_name:<25} | {dur_us:<14.2f} | {wt_bytes/1024:<14.1f} | {gb_s:<10.2f} | {dram_pct:<8.1f} | {alu_pct:<8.1f} | {int(th_inst):<14} | {inst_per_wt:<12.2f}")

