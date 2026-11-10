import subprocess
import os

models = [
    ("FP16", "data/models/hf-cache/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"),
    ("INT8", "data/models/qwen2.5-0.5b-int8"),
    ("INT4 g128", "data/models/qwen2.5-0.5b-int4-g128"),
]

print("=== Interleaved Prefill Benchmark (3 Rounds, 5 warmup + 20 timed each) ===")
results = {m[0]: [] for m in models}

for r in range(3):
    print(f"\n--- Round {r+1} ---")
    for name, path in models:
        cmd = [r"build\generate.exe", "--model", path, "--bench-prefill"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        # Parse P=512 line
        for line in res.stdout.splitlines():
            if "Prefill P=512:" in line:
                print(f"  {name:10}: {line.strip()}")
                # Extract ms
                parts = line.split("Prefill P=512:")[1].split("ms")[0].strip()
                results[name].append(float(parts))

print("\n=== Summary Medians across rounds (P=512) ===")
for name in results:
    vals = sorted(results[name])
    med = vals[len(vals)//2]
    print(f"  {name:10}: {med:.2f} ms (runs: {results[name]})")

