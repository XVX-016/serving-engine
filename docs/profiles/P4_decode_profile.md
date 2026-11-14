# Task P4 Decode Profile: In-Engine CUDA Event Instrumentation

## Decode Step Breakdown (Context = 517, Batch = 1, KV Cache Mode)

- **Total Kernels Launched**: 315
- **Host Wall Time**: 8.5272 ms
- **GPU Busy Time**: 5.83533 ms
- **Idle Gap Time**: 2.69187 ms (31.5681%)
- **Is Decode Launch-Bound?**: **YES (Launch Overhead Dominant)**

## Top Operations by Accumulated GPU Time

| Rank | Kernel / Operation | Total Invocations | Total Time (ms) | % of GPU Time |
|---|---|---|---|---|
| 1 | `gate_proj` | 24 | 0.93696 | 16.0567% |
| 2 | `up_proj` | 24 | 0.922368 | 15.8066% |
| 3 | `lm_head` | 1 | 0.615488 | 10.5476% |
| 4 | `down_proj_residual` | 24 | 0.599136 | 10.2674% |
| 5 | `decode_attention` | 24 | 0.536096 | 9.18708% |
| 6 | `q_proj` | 24 | 0.363904 | 6.23622% |
| 7 | `o_proj_residual` | 24 | 0.362016 | 6.20387% |
| 8 | `v_proj` | 24 | 0.317504 | 5.44107% |
| 9 | `cache_append` | 24 | 0.243264 | 4.16881% |
| 10 | `k_proj` | 24 | 0.23968 | 4.1074% |
| 11 | `input_rmsnorm` | 24 | 0.18704 | 3.2053% |
| 12 | `post_rmsnorm` | 24 | 0.177696 | 3.04518% |
| 13 | `swiglu` | 24 | 0.164032 | 2.81102% |
| 14 | `rope` | 24 | 0.151168 | 2.59057% |
| 15 | `embedding` | 1 | 0.012 | 0.205644% |
| 16 | `final_rmsnorm` | 1 | 0.006976 | 0.119548% |
