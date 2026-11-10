import csv

with open('docs/profiles/p8b_ncu_int8.csv', 'r') as f:
    lines = [l for l in f if l.startswith('"') and not l.startswith('"ID"')]

print(f"Total lines: {len(lines)}")
for i, l in enumerate(lines[-15:]):
    parts = list(csv.reader([l]))[0]
    ms = float(parts[14].replace(',', '')) / 1e6
    print(f"[{len(lines)-15+i}] {parts[4][:60]} grid={parts[8]} ms={ms:.4f}")

