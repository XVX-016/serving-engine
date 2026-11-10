import subprocess

cmd = [
    r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.0\target\windows-desktop-win7-x64\ncu.exe",
    "--metrics", "gpu__time_duration.sum",
    "--launch-skip", "2299",
    "--launch-count", "339",
    "--csv",
    r"build\generate.exe",
    "--model", "data/models/qwen2.5-0.5b-int4-g128-head-int8",
    "--profile-decode"
]
print("Profiling full step for int4_head_int8...")
res = subprocess.run(cmd, capture_output=True, text=True)
with open("docs/profiles/p8b_ncu_int4_head_int8.csv", "w") as f:
    f.write(res.stdout)
print("Saved to docs/profiles/p8b_ncu_int4_head_int8.csv")

