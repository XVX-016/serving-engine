import csv
import collections

with open("docs/profiles/p8b_ncu_fp16.csv", 'r') as f:
    lines = f.readlines()
data = [l for l in lines if l.startswith('"') and not l.startswith('"ID"')]

print("Total launches:", len(data))
gemvs = []
for i, l in enumerate(data):
    parts = list(csv.reader([l]))[0]
    kname = parts[4]
    ns = float(parts[14].replace(',', ''))
    grid = parts[8]
    block = parts[7]
    if 'gemv' in kname or 'gemm' in kname or 'cublas' in kname:
        gemvs.append((i, kname, grid, block, ns/1e6))

print(f"Total gemvs found: {len(gemvs)}")
# Print first 10 and last 5
for idx, kname, grid, block, ms in gemvs[:10]:
    print(f"[{idx}] {kname[:50]} grid={grid} block={block} ms={ms:.4f}")
for idx, kname, grid, block, ms in gemvs[-5:]:
    print(f"[{idx}] {kname[:50]} grid={grid} block={block} ms={ms:.4f}")

