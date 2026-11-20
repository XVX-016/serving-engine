import csv

with open('scripts/gate_proj_ncu_raw.csv', 'r') as f:
    lines = [l for l in f if l.startswith('"ID"') or (l.startswith('"') and not l.startswith('"ID"'))]

headers = list(csv.reader([lines[0]]))[0]
values = list(csv.reader([lines[1]]))[0]

print("=== Warp Stall & Execution Breakdown for INT4 gate_proj ===")
for h, v in zip(headers, values):
    hl = h.lower()
    if any(k in hl for k in ['stall', 'warp_issue', 'warp_state', 'active_warps', 'eligible_warps', 'smsp__warp_cycles']):
        try:
            val = float(v.replace(',', ''))
            if val > 0.01:
                print(f"{h:<65} = {v}")
        except:
            pass

