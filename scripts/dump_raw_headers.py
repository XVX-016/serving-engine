with open('scripts/gate_proj_ncu_raw.csv') as f:
    text = f.read()

lines = [l for l in text.splitlines() if l.startswith('"ID"') or (l.startswith('"') and not l.startswith('"ID"'))]
print(f'Lines: {len(lines)}')
if len(lines) >= 2:
    import csv
    headers = list(csv.reader([lines[0]]))[0]
    vals = list(csv.reader([lines[1]]))[0]
    print(f'Total metrics collected: {len(headers)}')
    for h in headers:
        if any(k in h.lower() for k in ['dram', 'throughput', 'occupancy', 'warp', 'l1', 'l2', 'inst', 'sector', 'stall']):
            print(h)
