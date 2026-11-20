import subprocess
import csv

ncu_path = r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.0\target\windows-desktop-win7-x64\ncu.exe"
cmd = [ncu_path, '--launch-skip', '2300', '--launch-count', '1',
       '--metrics', 'smsp__inst_executed.sum,smsp__thread_inst_executed.sum,sm__inst_executed.sum',
       '--csv', r'build\generate.exe', '--model', r'data/models/qwen2.5-0.5b-int4-g128', '--profile-decode']
res = subprocess.run(cmd, capture_output=True, text=True)
for l in res.stdout.splitlines():
    if l.startswith('"') and not l.startswith('"ID"'):
        p = list(csv.reader([l]))[0]
        print(p[12], ":", p[14])

