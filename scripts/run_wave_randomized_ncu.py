import subprocess, random, re

exe = r'build\tests\bench_wave_single.exe'
ncu = r'C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.0\ncu.bat'

points = [3072, 3840, 4608, 5376]
runs = []
for r in range(3):
    for p in points:
        runs.append((p, r))
random.seed(42)
random.shuffle(runs)

print("Starting wave sweep NCU check: %d runs..." % len(runs))
results = {}

for N, rep in runs:
    cmd = [ncu, '--csv', '--metrics', 'gpu__time_duration.sum', exe, str(N), '0']
    out = subprocess.run(cmd, capture_output=True, text=True)
    durs = []
    for line in out.stdout.splitlines():
        if 'gemv_legacy' in line and 'gpu__time_duration.sum' in line:
            parts = line.split(',')
            val_str = parts[-1].strip().strip('\"').replace(',', '')
            try:
                dur = float(val_str) / 1000.0 # us
                durs.append(dur)
            except:
                pass
    if durs:
        # durs contains all 6 invocations in that process run; take the median of them or the last one
        med_dur = sorted(durs)[len(durs)//2]
        results.setdefault(N, []).append(med_dur)
        print("N=%d (rep %d): %.2f us (all 6 calls: min=%.2f, max=%.2f)" % (N, rep, med_dur, min(durs), max(durs)))
    else:
        print("N=%d (rep %d): failed to parse" % (N, rep))

print("\nSummary of NCU Durations across 3 independent process runs (us):")
for N in points:
    vals = results.get(N, [])
    if vals:
        s_vals = sorted(vals)
        med = s_vals[len(s_vals)//2]
        print("N=%d (m=%d): min=%.2f, median=%.2f, max=%.2f, reps=%s" % (N, N//768, min(vals), med, max(vals), str(vals)))
