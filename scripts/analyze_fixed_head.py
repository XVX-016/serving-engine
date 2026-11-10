import csv

with open("docs/profiles/p8b_ncu_int4_head_int8.csv", 'r') as f:
    lines = [l for l in f if l.startswith('"') and not l.startswith('"ID"')]

print(f"Total lines: {len(lines)}")
tot_ns = 0
for i, l in enumerate(lines):
    parts = list(csv.reader([l]))[0]
    ns = float(parts[14].replace(',', ''))
    tot_ns += ns
    if 'gemv_int8_f32' in parts[4]:
        print(f"Found LM Head: idx={i}, name={parts[4][:50]}, grid={parts[8]}, ms={ns/1e6:.4f}")

print(f"Total step kernel sum: {tot_ns/1e6:.4f} ms")

