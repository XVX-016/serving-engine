import csv
import io
import re

BANDWIDTH_CEILING = 228.15 # GB/s

def parse_accounting(csv_path, model_type):
    with open(csv_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    parts = text.split('=== STDERR ===')
    stdout_part = parts[0]
    
    m = re.search(r'Graph step steady-state 2-event time:\s*([0-9.]+)\s*ms', stdout_part)
    step_time_ms = float(m.group(1)) if m else 0.0
    
    csv_header = '"ID","Process ID"'
    csv_idx = stdout_part.find(csv_header)
    reader = csv.DictReader(io.StringIO(stdout_part[csv_idx:]))
    
    # Categories:
    # 0: Embedding (1)
    # 1: RMSNorm (24*2 + 1 = 49)
    # 2: QKV GEMV (24)
    # 3: RoPE (24)
    # 4: KV Append (24)
    # 5: Attention (24*2 = 48)
    # 6: O-proj GEMV (24)
    # 7: Gate/Up GEMV (24)
    # 8: SwiGLU (24)
    # 9: Down GEMV (24)
    # 10: LM Head GEMV (1)
    
    cats = {
        "Embedding": {"count": 0, "ns": 0.0, "bytes": 0},
        "RMSNorm": {"count": 0, "ns": 0.0, "bytes": 0},
        "QKV GEMV": {"count": 0, "ns": 0.0, "bytes": 0},
        "RoPE": {"count": 0, "ns": 0.0, "bytes": 0},
        "KV Append": {"count": 0, "ns": 0.0, "bytes": 0},
        "Attention": {"count": 0, "ns": 0.0, "bytes": 0},
        "O-proj GEMV": {"count": 0, "ns": 0.0, "bytes": 0},
        "Gate/Up GEMV": {"count": 0, "ns": 0.0, "bytes": 0},
        "SwiGLU": {"count": 0, "ns": 0.0, "bytes": 0},
        "Down GEMV": {"count": 0, "ns": 0.0, "bytes": 0},
        "LM Head GEMV": {"count": 0, "ns": 0.0, "bytes": 0},
        "Other": {"count": 0, "ns": 0.0, "bytes": 0}
    }
    
    # Bytes per op per layer:
    # FP16:
    #   QKV (896 -> 1152): 1152*896*2 + 1152*2 = 2,067,072 B
    #   O-proj (896 -> 896): 896*896*2 = 1,605,632 B
    #   Gate/Up (896 -> 9728): 9728*896*2 = 17,432,576 B
    #   Down (4864 -> 896): 896*4864*2 = 8,716,288 B
    #   LM Head: 151936 * 896 * 2 = 272,269,312 B
    #   Norms (49): 49 * 896 * 2 = 87,808 B
    #   Embedding: 896 * 2 = 1,792 B
    #   RoPE: 24 * (14 * 64 * 2 + 2 * 64 * 2) = 49,152 B
    #   KV Append: 24 * (2 * 2 * 64 * 2) = 12,288 B
    #   Attention: KV reads at ctx 128 = 24 * 2 * 2 * 128 * 64 * 2 = 1,572,864 B
    #   SwiGLU: 24 * (2 * 4864 * 2 + 4864 * 2) = 700,416 B
    
    # INT8:
    #   QKV (896 -> 1152): 1152*896*1 + 1152*2(bias) + 1152*2(scales) = 1,036,800 B
    #   O-proj (896 -> 896): 896*896*1 + 896*2 = 804,608 B
    #   Gate/Up (896 -> 9728): 9728*896*1 + 9728*2 = 8,735,744 B
    #   Down (4864 -> 896): 896*4864*1 + 896*2 = 4,360,000 B
    #   LM Head (INT8): 151936 * 896 * 1 + 151936 * 2 = 136,438,528 B
    
    # INT4 (g128):
    #   QKV (896 -> 1152): 1152*896/2 + 1152*(896/128)*2(scales) + 1152*(896/128)/2(zeros) + 1152*2(bias) = 516096 + 16128 + 4032 + 2304 = 538,560 B
    #   O-proj (896 -> 896): 896*896/2 + 896*(896/128)*2 + 896*(896/128)/2 = 401408 + 12544 + 3136 = 417,088 B
    #   Gate/Up (896 -> 9728): 9728*896/2 + 9728*7*2 + 9728*7/2 = 4358144 + 136192 + 34048 = 4,528,384 B
    #   Down (4864 -> 896): 896*4864/2 + 896*(4864/128)*2 + 896*38/2 = 2179072 + 68096 + 17024 = 2,264,192 B
    #   LM Head (INT8): 151936 * 896 * 1 + 151936 * 2 = 136,438,528 B

    rows = list(reader)
    for i, r in enumerate(rows):
        name = r.get('Kernel Name', '')
        dur_ns = float(r.get('Metric Value', 0).replace(',', ''))
        grid = r.get('Grid Size', '')
        
        if "embedding_lookup" in name:
            c = "Embedding"
            b = 1792 if model_type == "fp16" else 898
        elif "rmsnorm" in name:
            c = "RMSNorm"
            b = 896 * 2
        elif "rope" in name:
            c = "RoPE"
            b = (14 + 2) * 64 * 2
        elif "append_kv" in name:
            c = "KV Append"
            b = 2 * 2 * 64 * 2
        elif "decode_splitk" in name:
            c = "Attention"
            b = (2 * 2 * 128 * 64 * 2) // 2 # split across partial and reduce
        elif "swiglu" in name:
            c = "SwiGLU"
            b = 3 * 4864 * 2
        elif "cutlass" in name or "(4748, 1, 1)" in grid or "(8, 594, 1)" in grid:
            c = "LM Head GEMV"
            b = 272269312 if model_type == "fp16" else 136438528
        elif "(36, 1, 1)" in grid or "(4, 36, 1)" in grid:
            c = "QKV GEMV"
            b = 2067072 if model_type == "fp16" else (1036800 if model_type == "int8" else 538560)
        elif "(304, 1, 1)" in grid or "(4, 304, 1)" in grid:
            c = "Gate/Up GEMV"
            b = 17432576 if model_type == "fp16" else (8735744 if model_type == "int8" else 4528384)
        elif "(28, 1, 1)" in grid:
            # Could be O-proj or Down
            # Look at kernel index in layer:
            # Layer structure: 0:RMSNorm, 1:QKV, 2:RoPE, 3:Append, 4:AttnPart, 5:AttnRed, 6:O-proj, 7:RMSNorm, 8:GateUp, 9:SwiGLU, 10:Down
            idx_in_layer = (i - 1) % 11
            if idx_in_layer == 6:
                c = "O-proj GEMV"
                b = 1605632 if model_type == "fp16" else (804608 if model_type == "int8" else 417088)
            else:
                c = "Down GEMV"
                b = 8716288 if model_type == "fp16" else (4360000 if model_type == "int8" else 2264192)
        else:
            c = "Other"
            b = 0
            
        cats[c]["count"] += 1
        cats[c]["ns"] += dur_ns
        cats[c]["bytes"] += b

    print(f"=== {model_type.upper()} ({csv_path}) ===")
    total_ns = sum(v["ns"] for v in cats.values())
    total_ms = total_ns / 1e6
    bubble_ms = step_time_ms - total_ms
    bubble_pct = (bubble_ms / step_time_ms) * 100.0
    
    print(f"Total Kernels: {len(rows)}, Kernel Sum: {total_ms:.3f} ms, Step Time: {step_time_ms:.3f} ms")
    print(f"Inter-kernel Bubble: {bubble_ms:.3f} ms ({bubble_pct:.1f}% of step)\n")
    print(f"{'Category':15s} | {'Count':5s} | {'Time (ms)':9s} | {'% Step':6s} | {'Bytes (MB)':10s} | {'GB/s':7s} | {'% Roofline':10s}")
    print("-" * 75)
    for c, d in cats.items():
        if d["count"] == 0: continue
        ms = d["ns"] / 1e6
        pct_step = (ms / step_time_ms) * 100.0
        mb = d["bytes"] / 1e6
        gbps = (d["bytes"] / 1e9) / (ms / 1000.0) if ms > 0 else 0.0
        roof_pct = (gbps / BANDWIDTH_CEILING) * 100.0
        print(f"{c:15s} | {d['count']:5d} | {ms:9.3f} | {pct_step:5.1f}% | {mb:10.3f} | {gbps:7.1f} | {roof_pct:9.1f}%")
    print("\n")

for m, p in [("fp16", "fp16_ncu_uncontrolled.csv"), 
            ("int8", "int8_ncu_uncontrolled.csv"), 
            ("int4", "int4_ncu_uncontrolled.csv")]:
    parse_accounting(p, m)
