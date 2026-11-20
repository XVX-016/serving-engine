"""
Fit: t = a + c*rows + bytes/B  for INT4 GEMV nodes only (no lm_head).
Uses gpu__time_duration.sum from occupancy_and_timer_results.json (Legacy kernel only).
Checks whether rows and bytes are separable. Reports residuals and B.
Retracts 428.2 GB/s if B > 228 GB/s.
"""
import json
import numpy as np

with open("scripts/occupancy_and_timer_results.json") as f:
    data = json.load(f)

# INT4 GEMV shapes, legacy kernel only, exclude lm_head
# Weight bytes = N * K/2  (INT4 nibble packing)
# Scale bytes  = N * (K//128) * 2  (fp16 per group)
# X bytes      = K * 2  (fp16 activations)
shapes = []
for row in data:
    if row["kernel"] != "Legacy":
        continue
    if row["shape"] == "lm_head":
        continue
    N, K = row["N"], row["K"]
    t_us = row["gpu__time_duration.sum"] / 1000.0  # ns -> us
    bytes_w = N * (K // 2)
    bytes_s = N * (K // 128) * 2
    bytes_x = K * 2
    total_bytes = bytes_w + bytes_s + bytes_x
    shapes.append({
        "shape": row["shape"],
        "N": N, "K": K,
        "rows": N,
        "bytes": total_bytes,
        "t_us": t_us
    })
    print(f"  {row['shape']:10s}  N={N:6d}  K={K:5d}  bytes={total_bytes:>10d}  t={t_us:.3f} us")

# Build design matrix  [1, rows, bytes] for t = a + c*rows + bytes/B
t = np.array([s["t_us"] for s in shapes])
rows = np.array([s["rows"] for s in shapes], dtype=float)
byt  = np.array([s["bytes"] for s in shapes], dtype=float)

A = np.column_stack([np.ones(len(shapes)), rows, byt])

# Least squares
result = np.linalg.lstsq(A, t, rcond=None)
coeff = result[0]
a, c, inv_B = coeff
B_GBs = 1.0 / inv_B  # bytes/us -> GB/s

t_pred = A @ coeff
residuals = t - t_pred

print("\n--- Fit: t = a + c*rows + bytes/B ---")
print(f"  a       = {a:.4f} us  (intercept)")
print(f"  c       = {c*1e6:.4f} us/row  ({c:.6e} us/row)")
print(f"  1/B     = {inv_B:.6e} us/byte")
print(f"  B       = {B_GBs:.2f} GB/s")
print(f"  Roofline ceiling: 228 GB/s")
if B_GBs > 228.0:
    print(f"  WARNING: B={B_GBs:.1f} GB/s > 228 GB/s roofline ceiling — INVALID fit")
    print(f"  RETRACT: 428.2 GB/s figure from P11d")
else:
    print(f"  B within roofline ceiling")

print("\nResiduals per shape:")
for i, s in enumerate(shapes):
    print(f"  {s['shape']:10s}  t={t[i]:.3f}  pred={t_pred[i]:.3f}  resid={residuals[i]:+.3f} us")
print(f"\n  Max |residual| = {np.abs(residuals).max():.3f} us")
print(f"  RMSE = {np.sqrt((residuals**2).mean()):.3f} us")

# Test separability: fit rows-only and bytes-only models
Ar = np.column_stack([np.ones(len(shapes)), rows])
Ab = np.column_stack([np.ones(len(shapes)), byt])
coeff_r = np.linalg.lstsq(Ar, t, rcond=None)[0]
coeff_b = np.linalg.lstsq(Ab, t, rcond=None)[0]
resid_r = t - Ar @ coeff_r
resid_b = t - Ab @ coeff_b
print(f"\nSeparability test:")
print(f"  Rows-only model: RMSE = {np.sqrt((resid_r**2).mean()):.3f} us  (a={coeff_r[0]:.4f}, c={coeff_r[1]*1e6:.4f} us/row)")
print(f"  Bytes-only model: RMSE = {np.sqrt((resid_b**2).mean()):.3f} us  (a={coeff_b[0]:.4f}, B={1.0/coeff_b[1]:.1f} GB/s)")
print(f"  Joint model: RMSE = {np.sqrt((residuals**2).mean()):.3f} us")
print(f"  Conclusion: ", end="")
rmse_joint = np.sqrt((residuals**2).mean())
rmse_rows = np.sqrt((resid_r**2).mean())
rmse_bytes = np.sqrt((resid_b**2).mean())
if rmse_joint < 0.3 * min(rmse_rows, rmse_bytes):
    print("rows and bytes are NOT separable (joint model substantially better)")
elif min(rmse_rows, rmse_bytes) < 1.5 * rmse_joint:
    print("rows and bytes are approximately separable (single-variable nearly as good)")
else:
    print("partial separability — joint model is better but single-variable usable")
