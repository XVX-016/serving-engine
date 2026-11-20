import csv
import io

with open('int4_ncu_uncontrolled.csv', 'r', encoding='utf-8') as f:
    text = f.read().split('=== STDERR ===')[0]

csv_idx = text.find('"ID","Process ID"')
reader = csv.DictReader(io.StringIO(text[csv_idx:]))

for i, row in enumerate(reader):
    if i < 15 or i > 255:
        dur = float(row.get('Metric Value', 0).replace(',', '')) / 1e3
        print(f"{i:3d}: grid={row.get('Grid Size')} block={row.get('Block Size')} dur={dur:6.1f}us name={row.get('Kernel Name')[:60]}")
