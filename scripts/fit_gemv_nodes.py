import csv
import io
import numpy as np

def fit_gemv(csv_path, model_type):
    with open(csv_path, 'r', encoding='utf-8') as f:
        text = f.read().split('=== STDERR ===')[0]

    csv_idx = text.find('"ID","Process ID"')
    reader = csv.DictReader(io.StringIO(text[csv_idx:]))

    gemv_data = [] # (shape_name, bytes, time_us)
    rows = list(reader)
    for i, r in enumerate(rows):
        name = r.get('Kernel Name', '')
        dur_us = float(r.get('Metric Value', 0).replace(',', '')) / 1e3
        grid = r.get('Grid Size', '')
        if '(36, 1, 1)' in grid or '(4, 36, 1)' in grid:
            b = 2067072 if model_type == 'fp16' else (1036800 if model_type == 'int8' else 538560)
            gemv_data.append(('qkv', b, dur_us))
        elif '(304, 1, 1)' in grid or '(4, 304, 1)' in grid:
            b = 17432576 if model_type == 'fp16' else (8735744 if model_type == 'int8' else 4528384)
            gemv_data.append(('gate_up', b, dur_us))
        elif '(28, 1, 1)' in grid:
            idx_in_layer = (i - 1) % 11
            if idx_in_layer == 6:
                b = 1605632 if model_type == 'fp16' else (804608 if model_type == 'int8' else 417088)
                gemv_data.append(('o_proj', b, dur_us))
            else:
                b = 8716288 if model_type == 'fp16' else (4360000 if model_type == 'int8' else 2264192)
                gemv_data.append(('down_proj', b, dur_us))
        elif 'cutlass' in name or '(8, 594, 1)' in grid or '(4748, 1, 1)' in grid:
            b = 272269312 if model_type == 'fp16' else 136438528
            gemv_data.append(('head', b, dur_us))

    by_shape = {}
    for s, b, t in gemv_data:
        by_shape.setdefault(s, {'bytes': b, 'times': []})['times'].append(t)

    print(f"\n=================== {model_type.upper()} GEMV Node Analysis ({len(gemv_data)} nodes) ===================")
    print(f"{'Shape':<12} | {'Bytes':<12} | {'Count':<6} | {'Mean Time (us)':<15} | {'Achieved GB/s':<14}")
    print("-" * 68)
    for s, d in by_shape.items():
        mean_t = np.mean(d['times'])
        bw = (d['bytes'] / 1e9) / (mean_t * 1e-6) # B / 1e9 / (us * 1e-6) = GB/s
        print(f"{s:<12} | {d['bytes']:<12,d} | {len(d['times']):<6d} | {mean_t:<15.2f} | {bw:<14.2f}")

    # X in bytes, Y in us
    # t_us = t0_us + (bytes / B_bytes_per_us)
    # B in GB/s = B_bytes_per_us / 1000.0
    # Let Y = m * X + c.
    # m is us/byte. So 1/m is bytes/us.
    # In GB/s: (1/m bytes/us) * (1e6 us / 1 s) / 1e9 = 1 / (m * 1000.0)
    X = np.array([d[1] for d in gemv_data], dtype=np.float64)
    Y = np.array([d[2] for d in gemv_data], dtype=np.float64)

    m, c = np.polyfit(X, Y, 1)
    B_gbps = 1.0 / (m * 1000.0) if m > 0 else 0.0
    residuals = Y - (m * X + c)
    rms_res = np.sqrt(np.mean(residuals**2))
    t0_us = c
    t0_total_ms = (t0_us * len(gemv_data)) / 1e3

    print(f"\nLinear Fit: t = t0 + bytes / B (Across all {len(gemv_data)} GEMV nodes):")
    print(f"  t0 (launch/fixed intercept) : {t0_us:.3f} us per GEMV")
    print(f"  B (streaming slope)        : {B_gbps:.2f} GB/s")
    print(f"  RMS Residual               : {rms_res:.3f} us")
    print(f"  Total t0 x node count      : {t0_total_ms:.3f} ms (out of {np.sum(Y)/1e3:.3f} ms total GEMV time, {t0_total_ms/(np.sum(Y)/1e3)*100:.1f}%)")

    # Fit across the 96 layer projections (excluding head)
    layer_data = [d for d in gemv_data if d[0] != 'head']
    X_lay = np.array([d[1] for d in layer_data], dtype=np.float64)
    Y_lay = np.array([d[2] for d in layer_data], dtype=np.float64)
    m_lay, c_lay = np.polyfit(X_lay, Y_lay, 1)
    B_lay_gbps = 1.0 / (m_lay * 1000.0) if m_lay > 0 else 0.0
    res_lay = Y_lay - (m_lay * X_lay + c_lay)
    rms_lay = np.sqrt(np.mean(res_lay**2))
    t0_lay_ms = (c_lay * len(layer_data)) / 1e3

    print(f"\nLinear Fit (Layer Projections Only, 96 nodes, excluding LM Head):")
    print(f"  t0                         : {c_lay:.3f} us per GEMV")
    print(f"  B                          : {B_lay_gbps:.2f} GB/s")
    print(f"  RMS Residual               : {rms_lay:.3f} us")
    print(f"  Total t0 x 96 nodes        : {t0_lay_ms:.3f} ms (out of {np.sum(Y_lay)/1e3:.3f} ms, {t0_lay_ms/(np.sum(Y_lay)/1e3)*100:.1f}%)")

for m, p in [('fp16', 'fp16_ncu_uncontrolled.csv'),
            ('int8', 'int8_ncu_uncontrolled.csv'),
            ('int4', 'int4_ncu_uncontrolled.csv')]:
    fit_gemv(p, m)
