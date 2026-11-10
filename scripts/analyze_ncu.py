import csv
import collections

profiles = [
    ("FP16", "docs/profiles/p8b_ncu_fp16.csv"),
    ("INT8", "docs/profiles/p8b_ncu_int8.csv"),
    ("INT4 g128", "docs/profiles/p8b_ncu_int4.csv"),
]

for name, path in profiles:
    with open(path, 'r') as f:
        lines = f.readlines()
    data = [l for l in lines if l.startswith('"') and not l.startswith('"ID"')]
    
    categories = collections.defaultdict(float)
    counts = collections.defaultdict(int)
    shapes = collections.defaultdict(float)
    shape_counts = collections.defaultdict(int)
    
    total_ns = 0.0
    for l in data:
        parts = list(csv.reader([l]))[0]
        kname = parts[4]
        ns = float(parts[14].replace(',', ''))
        total_ns += ns
        
        # Categorize
        if 'gemv' in kname or 'gemm' in kname or 'cublas' in kname:
            # Check grid size or demangled name to differentiate shapes if possible
            grid = parts[8]
            block = parts[7]
            if '151936' in kname or '64x64' in kname or '(151936' in grid:
                categories['lm_head'] += ns
                counts['lm_head'] += 1
            else:
                categories['linear_gemv'] += ns
                counts['linear_gemv'] += 1
                shapes[f"{kname[:40]} grid={grid}"] += ns
                shape_counts[f"{kname[:40]} grid={grid}"] += 1
        elif 'decode_splitk' in kname or 'attention' in kname:
            categories['attention'] += ns
            counts['attention'] += 1
        elif 'rmsnorm' in kname:
            categories['rmsnorm'] += ns
            counts['rmsnorm'] += 1
        elif 'rope' in kname:
            categories['rope'] += ns
            counts['rope'] += 1
        elif 'swiglu' in kname:
            categories['swiglu'] += ns
            counts['swiglu'] += 1
        elif 'append_kv' in kname:
            categories['cache_append'] += ns
            counts['cache_append'] += 1
        elif 'embedding' in kname:
            categories['embedding'] += ns
            counts['embedding'] += 1
        else:
            categories['other'] += ns
            counts['other'] += 1
            
    print(f"\n=================== {name} (Total profiled: {len(data)} kernels, {total_ns/1e6:.4f} ms) ===================")
    for cat, ns in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat:<18}: {counts[cat]:>3} launches, {ns/1e6:>8.4f} ms ({(ns/total_ns)*100:>5.1f}%)")

