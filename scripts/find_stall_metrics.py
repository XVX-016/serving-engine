import subprocess

ncu = r'C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.0\target\windows-desktop-win7-x64\ncu.exe'
cmd = [ncu, '--query-metrics']
res = subprocess.run(cmd, capture_output=True, text=True)
all_metrics = [m.strip() for m in res.stdout.splitlines() if m.strip()]
stall_metrics = [m for m in all_metrics if 'stall' in m.lower() or 'warp_state' in m.lower()]
print(f"Total metrics found: {len(all_metrics)}, stall metrics found: {len(stall_metrics)}")
for m in stall_metrics[:40]:
    print(m)

