import subprocess
import csv

ncu_path = r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.0\target\windows-desktop-win7-x64\ncu.exe"
metrics = [
    "gpu__time_duration.sum",
    "smsp__thread_inst_executed.sum",
    "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed"
]

# For INT8 dp4a model:
# Layer 0 starts at launch 2296:
# 2296: quant_act_norm
# 2297: gemv q_proj (896->896)
# 2298: rmsnorm k_bias? wait, gemv_k_proj is 2300, gemv_v_proj is 2301
# Let's inspect shapes systematically

