import csv

with open('docs/profiles/p4b_ncu_raw.csv', 'r') as f:
    lines = f.readlines()

data_lines = [l for l in lines if l.startswith('"') and not l.startswith('"ID"')]
print(f"Total profiled kernel launches: {len(data_lines)}")

categories = {
    'linear_cublas': 0.0,
    'decode_gqa': 0.0,
    'rmsnorm': 0.0,
    'bias_add': 0.0,
    'residual_add': 0.0,
    'rope': 0.0,
    'swiglu': 0.0,
    'cache_append': 0.0,
    'embedding': 0.0,
    'other': 0.0
}
cat_counts = {k: 0 for k in categories}

total_ns = 0.0
for l in data_lines:
    parts = list(csv.reader([l]))[0]
    kernel_name = parts[4]
    val_str = parts[14].replace(',', '')
    ns = float(val_str)
    total_ns += ns
    
    if 'gemv' in kernel_name or 'gemm' in kernel_name or 'cublas' in kernel_name:
        categories['linear_cublas'] += ns
        cat_counts['linear_cublas'] += 1
    elif 'decode_gqa' in kernel_name:
        categories['decode_gqa'] += ns
        cat_counts['decode_gqa'] += 1
    elif 'rmsnorm' in kernel_name:
        categories['rmsnorm'] += ns
        cat_counts['rmsnorm'] += 1
    elif 'bias_add' in kernel_name:
        categories['bias_add'] += ns
        cat_counts['bias_add'] += 1
    elif 'residual_add' in kernel_name:
        categories['residual_add'] += ns
        cat_counts['residual_add'] += 1
    elif 'rope' in kernel_name:
        categories['rope'] += ns
        cat_counts['rope'] += 1
    elif 'swiglu' in kernel_name:
        categories['swiglu'] += ns
        cat_counts['swiglu'] += 1
    elif 'append_kv' in kernel_name or 'cache_append' in kernel_name:
        categories['cache_append'] += ns
        cat_counts['cache_append'] += 1
    elif 'embedding' in kernel_name:
        categories['embedding'] += ns
        cat_counts['embedding'] += 1
    else:
        categories['other'] += ns
        cat_counts['other'] += 1

total_ms = total_ns / 1e6
print(f"Total NCU Kernel Time: {total_ms:.4f} ms")
print(f"{'Category':<20} | {'Count':<6} | {'NCU Total (ms)':<15} | {'% of Total':<10}")
print('-'*60)
for k, v in sorted(categories.items(), key=lambda x: x[1], reverse=True):
    ms = v / 1e6
    pct = (v / total_ns) * 100 if total_ns > 0 else 0
    print(f"{k:<20} | {cat_counts[k]:<6} | {ms:<15.4f} | {pct:<10.2f}%")

