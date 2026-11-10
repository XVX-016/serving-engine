import subprocess

ncu_path = r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.1.0\target\windows-desktop-win7-x64\ncu.exe"
# Let's search across launches 0 to 2500
cmd = [ncu_path, "--metrics", "gpu__time_duration.sum", "--launch-count", "2600", "--csv", r"build\generate.exe", "--profile-decode"]
print("Running NCU to capture all launches...")
res = subprocess.run(cmd, capture_output=True, text=True)
lines = [l for l in res.stdout.splitlines() if l.startswith('"') and not l.startswith('"ID"')]
print(f"Total launches: {len(lines)}")

# Let's find embedding_lookup_kernel launches
embedding_indices = []
for i, l in enumerate(lines):
    parts = l.split('","')
    if "embedding_lookup" in parts[4]:
        embedding_indices.append(i)

print(f"Embedding lookup launches at indices: {embedding_indices}")
for idx in embedding_indices:
    print(f"Index {idx}")

