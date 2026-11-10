import csv
import collections

# Map of layer GEMV order:
# Each layer has 7 GEMVs:
# 0: q_proj (896 -> 896, grid=28)
# 1: k_proj (896 -> 128, grid=4)
# 2: v_proj (896 -> 128, grid=4)
# 3: o_proj (896 -> 896, grid=28)
# 4: gate_proj (896 -> 4864, grid=152)
# 5: up_proj (896 -> 4864, grid=152)
# 6: down_proj (4864 -> 896, grid=28)
# Plus lm_head (896 -> 151936, cutlass wmma, grid=(8, 594, 1))

# Byte sizes (read bytes per step for each shape across 24 layers):
# q_proj: 24 * (896 * 896 * bytes_per_elem + overhead)
# k_proj: 24 * (896 * 128 * bytes_per_elem)
# v_proj: 24 * (896 * 128 * bytes_per_elem)
# o_proj: 24 * (896 * 896 * bytes_per_elem)
# gate_proj: 24 * (896 * 4864 * bytes_per_elem)
# up_proj: 24 * (896 * 4864 * bytes_per_elem)
# down_proj: 24 * (4864 * 896 * bytes_per_elem)
# lm_head: 151936 * 896 * 2 (FP16) = 272,269,312 bytes

shape_names = ["q_proj (896->896)", "k_proj (896->128)", "v_proj (896->128)", 
               "o_proj (896->896)", "gate_proj (896->4864)", "up_proj (896->4864)", "down_proj (4864->896)"]

# Weight elements per layer
shape_elems = [896*896, 896*128, 896*128, 896*896, 896*4864, 896*4864, 4864*896]

models = [
    ("FP16", "docs/profiles/p8b_ncu_fp16.csv", 2.0),
    ("INT8", "docs/profiles/p8b_ncu_int8.csv", 1.0),
    ("INT4 g128", "docs/profiles/p8b_ncu_int4.csv", 0.5 + 2.0/128), # 0.515625 bytes/weight
]

for name, path, bytes_per_wt in models:
    with open(path, 'r') as f:
        lines = f.readlines()
    data = [l for l in lines if l.startswith('"') and not l.startswith('"ID"')]
    
    gemvs = []
    attention_ms = 0.0
    rmsnorm_ms = 0.0
    other_ms = 0.0
    
    for l in data:
        parts = list(csv.reader([l]))[0]
        kname = parts[4]
        ms = float(parts[14].replace(',', '')) / 1e6
        if 'gemv' in kname or 'gemm' in kname or 'cublas' in kname or 'cutlass' in kname:
            gemvs.append(ms)
        elif 'decode_splitk' in kname or 'attention' in kname:
            attention_ms += ms
        elif 'rmsnorm' in kname:
            rmsnorm_ms += ms
        else:
            other_ms += ms
            
    assert len(gemvs) == 169
    layer_gemvs = gemvs[:168]
    lm_head_ms = gemvs[168]
    
    # Aggregate by shape
    shape_totals = [0.0]*7
    for i in range(168):
        shape_idx = i % 7
        shape_totals[shape_idx] += layer_gemvs[i]
        
    print(f"\n================================================================================")
    print(f"Profile: {name} (Kernel Sum: {sum(gemvs) + attention_ms + rmsnorm_ms + other_ms:.4f} ms)")
    print(f"================================================================================")
    print(f"{'Shape / Operation':<25} | {'Count':<5} | {'Time (ms)':<10} | {'Traffic (MB)':<12} | {'Achieved GB/s':<14} | {'% Peak (228)':<12}")
    print("-" * 88)
    
    tot_layer_traffic = 0.0
    tot_layer_time = sum(shape_totals)
    for s_idx in range(7):
        t_ms = shape_totals[s_idx]
        traffic_bytes = 24 * shape_elems[s_idx] * bytes_per_wt
        traffic_mb = traffic_bytes / 1e6
        tot_layer_traffic += traffic_mb
        gb_s = (traffic_bytes / (t_ms * 1e-3)) / 1e9
        pct_peak = (gb_s / 228.15) * 100
        print(f"{shape_names[s_idx]:<25} | {24:<5} | {t_ms:<10.4f} | {traffic_mb:<12.2f} | {gb_s:<14.2f} | {pct_peak:<12.1f}%")
        
    # LM head
    lm_traffic_bytes = 151936 * 896 * 2.0 # FP16 head for all three
    lm_traffic_mb = lm_traffic_bytes / 1e6
    lm_gb_s = (lm_traffic_bytes / (lm_head_ms * 1e-3)) / 1e9
    lm_pct = (lm_gb_s / 228.15) * 100
    print(f"{'LM Head (896->151936)':<25} | {1:<5} | {lm_head_ms:<10.4f} | {lm_traffic_mb:<12.2f} | {lm_gb_s:<14.2f} | {lm_pct:<12.1f}%")
    
    print("-" * 88)
    print(f"{'Total Linear GEMV':<25} | {169:<5} | {sum(gemvs):<10.4f} | {tot_layer_traffic + lm_traffic_mb:<12.2f} | {((tot_layer_traffic + lm_traffic_mb)*1e6/(sum(gemvs)*1e-3)/1e9):<14.2f} | {(((tot_layer_traffic + lm_traffic_mb)*1e6/(sum(gemvs)*1e-3)/1e9)/228.15)*100:<12.1f}%")
    print(f"{'Attention (Split-K)':<25} | {48:<5} | {attention_ms:<10.4f} | {'-':<12} | {'-':<14} | {'-':<12}")
    print(f"{'RMSNorm':<25} | {49:<5} | {rmsnorm_ms:<10.4f} | {'-':<12} | {'-':<14} | {'-':<12}")
    print(f"{'Small Kernels (RoPE/etc)':<25} | {73:<5} | {other_ms:<10.4f} | {'-':<12} | {'-':<14} | {'-':<12}")
    tot_k = sum(gemvs) + attention_ms + rmsnorm_ms + other_ms
    print(f"{'Total Profiled Kernels':<25} | {len(data):<5} | {tot_k:<10.4f} | {'-':<12} | {'-':<14} | {'-':<12}")

