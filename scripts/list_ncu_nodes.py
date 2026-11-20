import csv
import io

with open('fp16_ncu_uncontrolled.csv', 'r', encoding='utf-8') as f:
    text = f.read().split('=== STDERR ===')[0]

csv_idx = text.find('"ID","Process ID"')
reader = csv.DictReader(io.StringIO(text[csv_idx:]))

kernels = []
for row in reader:
    kernels.append((row.get('Kernel Name'), row.get('Grid Size'), row.get('Block Size')))

print(f"Total profiled kernels: {len(kernels)}")
for i, (k, grid, block) in enumerate(kernels[:25]):
    print(f"{i:2d}: grid={grid:<15} block={block:<15} name={k[:70]}")

print("...")
for i, (k, grid, block) in enumerate(kernels[-10:], start=len(kernels)-10):
    print(f"{i:2d}: grid={grid:<15} block={block:<15} name={k[:70]}")
