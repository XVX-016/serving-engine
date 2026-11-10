import subprocess

cmd = [
    r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.0\target\windows-desktop-win7-x64\ncu.exe",
    "--metrics", "gpu__time_duration.sum",
    "--launch-skip", "2636",
    "--launch-count", "2",
    "--csv",
    r"build\generate.exe",
    "--model", "data/models/qwen2.5-0.5b-int4-g128-head-int8",
    "--profile-decode"
]
res = subprocess.run(cmd, capture_output=True, text=True)
for l in res.stdout.splitlines():
    if l.startswith('"') and not l.startswith('"ID"'):
        parts = l.split('","')
        print(parts[4][:70], parts[8], parts[14])

