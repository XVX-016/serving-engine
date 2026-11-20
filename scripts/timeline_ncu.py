import subprocess
import csv

ncu_path = r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.0\target\windows-desktop-win7-x64\ncu.exe"

cmd = [ncu_path, '--launch-skip', '2300', '--launch-count', '25',
       '--metrics', 'gpu__time_duration.sum,smsp__thread_inst_executed.sum,sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed',
       '--csv', r'build\generate.exe', '--model', r'data/models/qwen2.5-0.5b-int4-g128', '--profile-decode']
res = subprocess.run(cmd, capture_output=True, text=True)
lines = [l for l in res.stdout.splitlines() if l.startswith('"') and not l.startswith('"ID"')]

by_id = {}
for l in lines:
    parts = list(csv.reader([l]))[0]
    kid = parts[0]
    kname = parts[3]
    metric = parts[12]
    val = parts[14]
    if kid not in by_id:
        by_id[kid] = {'name': kname}
    by_id[kid][metric] = val

for kid, data in sorted(by_id.items(), key=lambda x: int(x[0])):
    dur = float(data.get('gpu__time_duration.sum', '0').replace(',', '')) / 1e3
    alu = data.get('sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active', '0')
    dram = data.get('dram__throughput.avg.pct_of_peak_sustained_elapsed', '0')
    inst = data.get('smsp__thread_inst_executed.sum', '0')
    name = data['name']
    print(f"{kid:<4} | {name:<40} | {dur:6.2f} us | DRAM {dram:>5}% | ALU {alu:>5}% | Inst {inst}")

