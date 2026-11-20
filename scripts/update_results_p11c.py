with open('RESULTS.md', 'r', encoding='utf-8') as f:
    text = f.read()

target = '| P11b | Deliverable 6 (P12 Megakernel Gap Analysis) | - | - | - | - | - | - | - | - | Gap (a) addressable: FP16 -5.1%, INT4 -8.9% (<15% for both). Gap (b) kernel efficiency: FP16 15.0%, INT4 64.0%. Megakernel stopped; kernel efficiency dominates. |\n'

repl = '''| P11b | Deliverable 6 (P12 Megakernel Gap Analysis) | - | - | - | - | - | - | - | - | Retracted (method: mixed clock states: benchmark graph steps were compared against clock-locked NCU kernel sums, producing negative gaps -5.1% and -8.9% and misleading bubble claims; replaced by P11c same-clock analysis). |
| P11c | Same-Clock Kernel Accounting & Gap Analysis | - | - | - | - | - | - | - | - | Evaluated with --clock-control none --cache-control none in same process. FP16: step 5.581 ms, sum 5.111 ms (gap a: 8.4%, gap b: 15.1%). INT8: step 4.063 ms, sum 3.693 ms (gap a: 9.1%, gap b: 41.1%). INT4: step 4.284 ms, sum 3.928 ms (gap a: 8.3%, gap b: 64.0%). Both (a) < 15%. Kernel efficiency (b) dominates. |
| P11c | Interleaved Decode: INT4 Split-K Fused-Graph (Ctx 128) | 1 | 128 | 32 | - | 3.28 ms | 304.8 | 322.7 MB | - | 1.41 ms roofline ceiling (43.1% of 228.15 GB/s, 98.38 GB/s). 2-warp Split-K with shared reduction improves step from 3.61 ms to 3.28 ms (+9.06% speedup). Bit-exact with baseline. |
'''

text = text.replace(target, repl, 1)

with open('RESULTS.md', 'w', encoding='utf-8') as f:
    f.write(text)

print('Updated RESULTS.md with P11c entries.')
