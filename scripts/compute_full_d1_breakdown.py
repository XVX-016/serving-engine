import csv
import io

files = [
    ('FP16 Uncontrolled', 'fp16_ncu_uncontrolled.csv', 989.728),
    ('INT8 Uncontrolled', 'int8_ncu_uncontrolled.csv', 496.678),
    ('INT4 Uncontrolled', 'int4_ncu_uncontrolled.csv', 322.748),
]

for title, fname, traffic_mb in files:
    with open(fname, 'r', encoding='utf-8') as f:
        text = f.read().split('=== STDERR ===')[0]
    idx = text.find('"ID","Process ID"')
    reader = csv.DictReader(io.StringIO(text[idx:]))
    cats = {}
    tot_ns = 0
    for i, r in enumerate(reader):
        name = r.get('Kernel Name', '')
        dur_ns = float(r.get('Metric Value', 0).replace(',', ''))
        grid = r.get('Grid Size', '')
        tot_ns += dur_ns
        
        c = 'Other'
        if 'embedding_lookup' in name: c = 'Embedding'
        elif 'rmsnorm' in name: c = 'RMSNorm'
        elif 'rope' in name: c = 'RoPE'
        elif 'append_kv' in name: c = 'KV Append'
        elif 'decode_splitk' in name: c = 'Attention'
        elif 'swiglu' in name: c = 'SwiGLU'
        elif 'cutlass' in name or '(8, 594, 1)' in grid or '(4748, 1, 1)' in grid: c = 'LM Head GEMV'
        elif '(36, 1, 1)' in grid or '(4, 36, 1)' in grid: c = 'QKV GEMV'
        elif '(304, 1, 1)' in grid or '(4, 304, 1)' in grid: c = 'Gate/Up GEMV'
        elif '(28, 1, 1)' in grid:
            idx_in_layer = (i - 1) % 11
            if idx_in_layer == 6: c = 'O-proj GEMV'
            else: c = 'Down GEMV'
        
        cats.setdefault(c, [0, 0.0])
        cats[c][0] += 1
        cats[c][1] += dur_ns
    
    print(f'=== {title} ===')
    print(f'Total kernel duration: {tot_ns/1e6:.3f} ms')
    for c, (cnt, ns) in sorted(cats.items(), key=lambda x: x[1][1], reverse=True):
        print(f'  {c:<15}: {cnt:2d} launches, {ns/1e6:6.3f} ms ({ns/tot_ns*100:4.1f}%)')
