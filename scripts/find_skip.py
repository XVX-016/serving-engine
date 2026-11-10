import subprocess

ncu_path = r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.0\target\windows-desktop-win7-x64\ncu.exe"
# Let's run with --launch-skip 1800 --launch-count 400
cmd = [ncu_path, "--metrics", "gpu__time_duration.sum", "--launch-skip", "1800", "--launch-count", "100", "--csv", r"build\generate.exe", "--profile-decode"]
print("Running NCU...")
res = subprocess.run(cmd, capture_output=True, text=True)
lines = [l for l in res.stdout.splitlines() if l.startswith('"') and not l.startswith('"ID"')]
print(f"Captured {len(lines)} kernels:")
for i, l in enumerate(lines[:15]):
    parts = l.split('","')
    print(f"Launch {1800+i}: {parts[4]}")

