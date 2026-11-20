import csv
import io
import re

def analyze_csv(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    parts = text.split('=== STDERR ===')
    stdout_part = parts[0]
    
    m = re.search(r'Graph step steady-state 2-event time:\s*([0-9.]+)\s*ms', stdout_part)
    step_time_ms = float(m.group(1)) if m else None
    
    csv_header = '"ID","Process ID"'
    csv_idx = stdout_part.find(csv_header)
    if csv_idx == -1:
        print(f'{path}: CSV not found!')
        return
    csv_text = stdout_part[csv_idx:]
    reader = csv.DictReader(io.StringIO(csv_text))
    
    total_ns = 0
    kernel_count = 0
    categories = {}
    
    for row in reader:
        name = row.get('Kernel Name', '')
        dur_str = row.get('Metric Value', '').replace(',', '')
        if not dur_str:
            continue
        dur_ns = float(dur_str)
        total_ns += dur_ns
        kernel_count += 1
        
        # Categorize kernel
        cat = "Other"
        if "embedding_lookup" in name:
            cat = "Embedding"
        elif "rmsnorm" in name:
            cat = "RMSNorm"
        elif "rope" in name:
            cat = "RoPE"
        elif "append_kv" in name:
            cat = "KV Append"
        elif "decode_splitk" in name:
            cat = "Attention"
        elif "swiglu" in name:
            cat = "SwiGLU"
        elif "gemv" in name or "cutlass" in name or "gemm" in name:
            # Check grid/block or name
            grid = row.get('Grid Size', '')
            if "cutlass" in name or "(8, 594, 1)" in grid: # 896 -> 151936
                cat = "LM Head GEMV"
            elif "(36, 1, 1)" in grid or "(4, 36, 1)" in grid: # 896 -> 1152 (QKV)
                cat = "QKV GEMV"
            elif "(28, 1, 1)" in grid and "EpiResidual" in name and dur_ns < 25000: # 896 -> 896 (O-proj)
                cat = "O-proj GEMV"
            elif "(304, 1, 1)" in grid or "(4, 304, 1)" in grid: # 896 -> 9728 (GateUp)
                cat = "Gate/Up GEMV"
            elif "(28, 1, 1)" in grid and dur_ns > 25000: # 4864 -> 896 (Down)
                cat = "Down GEMV"
            else:
                cat = "Other GEMV"
        
        if cat not in categories:
            categories[cat] = {"count": 0, "total_ns": 0}
        categories[cat]["count"] += 1
        categories[cat]["total_ns"] += dur_ns
            
    total_ms = total_ns / 1e6
    bubble_ms = (step_time_ms - total_ms) if step_time_ms else 0.0
    bubble_pct = (bubble_ms / step_time_ms * 100.0) if step_time_ms else 0.0
    print(f'==================================================')
    print(f'File: {path}')
    print(f'Kernels: {kernel_count}, Total Kernel Duration Sum: {total_ms:.3f} ms')
    print(f'Graph Step (2-event total): {step_time_ms:.3f} ms')
    print(f'Inter-kernel Bubble / Overhead: {bubble_ms:.3f} ms ({bubble_pct:.1f}% of step)')
    print('Category breakdown:')
    for c, data in sorted(categories.items(), key=lambda x: x[1]['total_ns'], reverse=True):
        c_ms = data['total_ns'] / 1e6
        c_pct = (c_ms / total_ms) * 100.0
        print(f'  {c:15s}: {data["count"]:3d} launches, {c_ms:6.3f} ms ({c_pct:5.1f}%)')

for f in ['fp16_ncu_uncontrolled.csv', 'fp16_ncu_locked.csv', 
          'int8_ncu_uncontrolled.csv', 'int8_ncu_locked.csv',
          'int4_ncu_uncontrolled.csv', 'int4_ncu_locked.csv']:
    analyze_csv(f)
