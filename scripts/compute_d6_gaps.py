BW_CEILING = 228.15 # GB/s

configs = [
    ("FP16 (fused, graph)", 4.8594, 5.105, 989728000),
    ("INT8 (fused, graph)", 3.4427, 3.694, 496678018),
    ("INT4 (fused, graph)", 3.6075, 3.928, 322747522)
]

print("=== Task P11b Deliverable 6: P12 Megakernel Gap Analysis ===")
print("Definitions:")
print("  (a) Addressable by megakernel: (graph_step - ncu_kernel_sum) / graph_step")
print("      Measures inter-kernel bubbles, launch overheads, and GPU tail idle times.")
print("  (b) Kernel-efficiency gap:    (ncu_kernel_sum - bytes / 228.15 GB/s) / ncu_kernel_sum")
print("      Measures what better intra-kernel memory streaming and unrolling could recover.\n")

print(f"{'Config':22s} | {'Graph Step':10s} | {'NCU Sum':8s} | {'Ideal Time':10s} | {'Gap (a) [Megakernel]':20s} | {'Gap (b) [Kernels]':18s}")
print("-" * 105)

for name, step, ncu_sum, b in configs:
    ideal = (b / 228.15e9) * 1000.0
    gap_a = (step - ncu_sum) / step * 100.0
    gap_b = (ncu_sum - ideal) / ncu_sum * 100.0
    print(f"{name:22s} | {step:8.3f} ms | {ncu_sum:6.3f} ms | {ideal:8.3f} ms | {gap_a:18.1f}% | {gap_b:16.1f}%")

print("\nP12 Stop Rule Evaluation:")
print("  Rule: If Gap (a) is under 15% for BOTH FP16 and INT4, recommend stopping P12 with a negative result.")
gap_a_fp16 = (4.8594 - 5.105) / 4.8594 * 100.0
gap_a_int4 = (3.6075 - 3.928) / 3.6075 * 100.0
print(f"  FP16 Gap (a): {gap_a_fp16:.1f}%")
print(f"  INT4 Gap (a): {gap_a_int4:.1f}%")
