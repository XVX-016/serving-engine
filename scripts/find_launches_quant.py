import subprocess

ncu_path = r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.0\target\windows-desktop-win7-x64\ncu.exe"

for name, model_path in [("INT8", r"data/models/qwen2.5-0.5b-int8"), ("INT4", r"data/models/qwen2.5-0.5b-int4-g128")]:
    cmd = [ncu_path, "--metrics", "gpu__time_duration.sum", "--launch-count", "2600", "--csv", r"build\generate.exe", "--model", model_path, "--profile-decode"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    lines = [l for l in res.stdout.splitlines() if l.startswith('"') and not l.startswith('"ID"')]
    
    embed_launches = []
    cutlass_launches = []
    for i, l in enumerate(lines):
        parts = l.split('","')
        if "embedding" in parts[4]:
            embed_launches.append(i)
        if "cutlass" in parts[4] or "151936" in parts[4] or "gemm" in parts[4]:
            cutlass_launches.append((i, parts[4]))
            
    print(f"\n{name}: Total launches = {len(lines)}")
    print(f"Embedding launches: {embed_launches}")
    print(f"LM head / GEMM launches: {cutlass_launches[:10]}")

