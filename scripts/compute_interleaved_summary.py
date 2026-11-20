import numpy as np

our_fp16 = [4.8589, 4.8594, 4.8608]
our_int8_leg = [3.4423, 3.4427, 3.4464]
our_int8_dp4a = [3.5837, 3.5861, 3.5862]
our_int4_leg = [3.6075, 3.6037, 3.6128]
our_int4_dp4a = [3.7592, 3.7587, 3.7601]

llama_f16_toks = [196.88, 199.06, 198.95]
llama_f16 = [1000.0 / t for t in llama_f16_toks]

llama_q8_toks = [327.47, 331.21, 331.20]
llama_q8 = [1000.0 / t for t in llama_q8_toks]

llama_q4_toks = [369.87, 372.74, 377.10]
llama_q4 = [1000.0 / t for t in llama_q4_toks]

bytes_map = {
    'FP16 (fused, graph)': 989728000,
    'INT8 Legacy (fused, graph)': 496678018,
    'INT8 dp4a (fused, graph)': 496678018,
    'INT4 Legacy (fused, graph)': 322747522,
    'INT4 dp4a (fused, graph)': 322747522,
    'llama.cpp F16': 989781504,
    'llama.cpp Q8_0': 526692864,
    'llama.cpp Q4_K_M': 393432576
}

data = [
    ('FP16 (fused, graph)', our_fp16),
    ('INT8 Legacy (fused, graph)', our_int8_leg),
    ('INT8 dp4a (fused, graph)', our_int8_dp4a),
    ('INT4 Legacy (fused, graph)', our_int4_leg),
    ('INT4 dp4a (fused, graph)', our_int4_dp4a),
    ('llama.cpp F16', llama_f16),
    ('llama.cpp Q8_0', llama_q8),
    ('llama.cpp Q4_K_M', llama_q4),
]

print(f"{'Configuration':<28} | {'R1 (ms)':<8} | {'R2 (ms)':<8} | {'R3 (ms)':<8} | {'Median (ms)':<11} | {'Traffic (MB)':<12} | {'Achieved GB/s':<14} | {'Roofline %':<10}")
print('-' * 115)
for name, runs in data:
    med = float(np.median(runs))
    b = bytes_map[name]
    bw = (b / 1e9) / (med / 1000.0)
    rf = (bw / 228.15) * 100.0
    print(f"{name:<28} | {runs[0]:<8.4f} | {runs[1]:<8.4f} | {runs[2]:<8.4f} | {med:<11.4f} | {b/1e6:<12.3f} | {bw:<14.2f} | {rf:<10.1f}%")

print("\nPercent difference relative to llama.cpp baselines:")
f16_diff = (np.median(our_fp16) - np.median(llama_f16)) / np.median(llama_f16) * 100.0
q8_diff = (np.median(our_int8_leg) - np.median(llama_q8)) / np.median(llama_q8) * 100.0
q4_diff = (np.median(our_int4_leg) - np.median(llama_q4)) / np.median(llama_q4) * 100.0
print(f"FP16 vs llama.cpp F16:    {f16_diff:+.2f}% ({np.median(our_fp16):.4f} vs {np.median(llama_f16):.4f} ms)")
print(f"INT8 vs llama.cpp Q8_0:   {q8_diff:+.2f}% ({np.median(our_int8_leg):.4f} vs {np.median(llama_q8):.4f} ms)")
print(f"INT4 vs llama.cpp Q4_K_M: {q4_diff:+.2f}% ({np.median(our_int4_leg):.4f} vs {np.median(llama_q4):.4f} ms)")
