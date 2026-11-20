import subprocess

ncu_path = r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.0\target\windows-desktop-win7-x64\ncu.exe"

# For legacy INT8 (no act quant)
cmd = [ncu_path, '--launch-skip', '2295', '--launch-count', '30',
       r'build\generate.exe', '--model', r'data/models/qwen2.5-0.5b-int8', '--profile-decode', '--legacy-gemv']
res = subprocess.run(cmd, capture_output=True, text=True)

idx = 2295
for line in res.stdout.splitlines():
    if 'Profiling "' in line:
        kname = line.split('Profiling "')[1].split('"')[0]
        print(f"{idx}: {kname}")
        idx += 1

