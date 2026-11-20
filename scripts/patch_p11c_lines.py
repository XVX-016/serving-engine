with open('docs/reports/P11c.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()

lines[187] = '1. **Which kernel has 304 blocks and why?**:\n'
lines[188] = '   - The **fused Gate/Up projection** (`gate_up`) has $N = 9728$.\n'
lines[189] = '   - The legacy GEMV kernel assigns 1 warp per row and 32 warps (1024 threads) per thread block.\n'
lines[190] = '   - $\\text{Grid Size} = \\lceil 9728 / 32 \\rceil = \\mathbf{304\\text{ blocks}}$. Across 24 SMs, this yields $\\mathbf{12.67\\text{ blocks per SM}}$ ($304 \\times 32 / 24 = \\mathbf{405.3\\text{ warps per SM}}$). [Corrected in P11d from earlier typo "304 warps / 12.6 warps per SM"].\n'

with open('docs/reports/P11c.md', 'w', encoding='utf-8') as f:
    f.writelines(lines)
print('P11c.md lines updated')
