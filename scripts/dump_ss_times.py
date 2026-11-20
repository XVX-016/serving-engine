import re

for name, p in [("FP16", "p11c_fp16_ncu.csv"), ("INT8", "p11c_int8_ncu.csv"), ("INT4", "p11c_int4_ncu.csv")]:
    with open(p, "r", encoding="utf-8") as f:
        text = f.read().split("=== STDERR ===")[0]
    m_ss = re.search(r"Graph step steady-state 2-event time:\s*([0-9.]+)\s*ms", text)
    print(f"{name}: Steady-state 2-event time = {m_ss.group(1)} ms")
