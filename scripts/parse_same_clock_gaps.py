import csv
import io
import re

BW_CEILING = 228.15 # GB/s

def parse_same_clock(csv_path, total_bytes):
    with open(csv_path, 'r', encoding='utf-8') as f:
        text = f.read()

    parts = text.split('=== STDERR ===')
    stdout_part = parts[0]

    # Steady state 2-event time
    m_ss = re.search(r'Graph step steady-state 2-event time:\s*([0-9.]+)\s*ms', stdout_part)
    steady_state_ms = float(m_ss.group(1)) if m_ss else 0.0

    # Profiled replay exact time
    m_prof = re.search(r'Graph step exact profiled replay time:\s*([0-9.]+)\s*ms', stdout_part)
    profiled_replay_ms = float(m_prof.group(1)) if m_prof else 0.0

    csv_header = '"ID","Process ID"'
    csv_idx = stdout_part.find(csv_header)
    reader = csv.DictReader(io.StringIO(stdout_part[csv_idx:]))

    total_ns = 0.0
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

        grid = row.get('Grid Size', '')
        # Categorize
        if "embedding_lookup" in name:
            c = "Embedding"
        elif "rmsnorm" in name:
            c = "RMSNorm"
        elif "rope" in name:
            c = "RoPE"
        elif "append_kv" in name:
            c = "KV Append"
        elif "decode_splitk" in name:
            c = "Attention"
        elif "swiglu" in name:
            c = "SwiGLU"
        elif "cutlass" in name or "(4748, 1, 1)" in grid or "(8, 594, 1)" in grid:
            c = "LM Head GEMV"
        elif "(36, 1, 1)" in grid or "(4, 36, 1)" in grid:
            c = "QKV GEMV"
        elif "(304, 1, 1)" in grid or "(4, 304, 1)" in grid:
            c = "Gate/Up GEMV"
        elif "(28, 1, 1)" in grid:
            # O-proj vs Down
            if "EpiResidual" in name and dur_ns < 25000:
                c = "O-proj GEMV"
            else:
                c = "Down GEMV"
        else:
            c = "Other"

        categories.setdefault(c, [0, 0.0])
        categories[c][0] += 1
        categories[c][1] += dur_ns

    kernel_sum_ms = total_ns / 1e6
    bubble_prof_ms = profiled_replay_ms - kernel_sum_ms
    bubble_prof_pct = (bubble_prof_ms / profiled_replay_ms) * 100.0 if profiled_replay_ms > 0 else 0.0

    bubble_ss_ms = steady_state_ms - kernel_sum_ms
    bubble_ss_pct = (bubble_ss_ms / steady_state_ms) * 100.0 if steady_state_ms > 0 else 0.0

    ideal_roofline_ms = (total_bytes / BW_CEILING / 1e9) * 1000.0

    # Gap (a) addressable: (graph_step - kernel_sum) / graph_step
    gap_a_prof = (profiled_replay_ms - kernel_sum_ms) / profiled_replay_ms * 100.0
    gap_a_ss = (steady_state_ms - kernel_sum_ms) / steady_state_ms * 100.0

    # Gap (b) kernel efficiency: (kernel_sum - ideal_roofline) / kernel_sum
    gap_b = (kernel_sum_ms - ideal_roofline_ms) / kernel_sum_ms * 100.0

    return {
        'kernels': kernel_count,
        'kernel_sum_ms': kernel_sum_ms,
        'profiled_replay_ms': profiled_replay_ms,
        'steady_state_ms': steady_state_ms,
        'bubble_prof_ms': bubble_prof_ms,
        'bubble_prof_pct': bubble_prof_pct,
        'bubble_ss_ms': bubble_ss_ms,
        'bubble_ss_pct': bubble_ss_pct,
        'ideal_roofline_ms': ideal_roofline_ms,
        'gap_a_prof': gap_a_prof,
        'gap_a_ss': gap_a_ss,
        'gap_b': gap_b,
        'categories': categories
    }

models = [
    ("FP16", "p11c_fp16_ncu.csv", 989728000),
    ("INT8", "p11c_int8_ncu.csv", 496678018),
    ("INT4", "p11c_int4_ncu.csv", 322747522)
]

print("==========================================================================================")
print("SAME-CLOCK / SAME-PROCESS NCU REPLAY & GRAPH STEP GAP ANALYSIS")
print("==========================================================================================")
print(f"{'Config':<8} | {'Nodes':<5} | {'Kernel Sum':<11} | {'Profiled Step':<14} | {'Bubble (ms)':<12} | {'Gap (a) [Megakernel]':<22} | {'Gap (b) [Kernels]':<18}")
print("-" * 105)

all_res = {}
for name, p, b in models:
    res = parse_same_clock(p, b)
    all_res[name] = res
    print(f"{name:<8} | {res['kernels']:<5d} | {res['kernel_sum_ms']:<8.3f} ms | {res['profiled_replay_ms']:<11.3f} ms | {res['bubble_prof_ms']:<6.3f} ms ({res['bubble_prof_pct']:<4.1f}%) | {res['gap_a_prof']:<21.1f}% | {res['gap_b']:<17.1f}%")

print("\nAlso compared against steady-state (unprofiled) step in same process:")
for name, res in all_res.items():
    print(f"  {name}: Steady-State Step={res['steady_state_ms']:.3f} ms, Kernel Sum={res['kernel_sum_ms']:.3f} ms, Bubble={res['bubble_ss_ms']:.3f} ms ({res['bubble_ss_pct']:.1f}%), Gap (a)={res['gap_a_ss']:.1f}%")
