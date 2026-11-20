# Task P4 Decode Profile: In-Engine CUDA Event Instrumentation

## Decode Step Breakdown (Context = 517, Batch = 1, KV Cache Mode)

- **Total Kernels Launched**: 315
- **Host Wall Time**: 26.4676 ms
- **GPU Busy Time**: 17.6504 ms
- **Idle Gap Time**: 8.8172 ms (33.3132%)
- **Is Decode Launch-Bound?**: **YES (Launch Overhead Dominant)**

## Top Operations by Accumulated GPU Time

| Rank | Kernel / Operation | Total Invocations | Total Time (ms) | % of GPU Time |
|---|---|---|---|---|
| 1 | `decode_attention` | 24 | 1.94602 | 11.0253% |
| 2 | `up_proj` | 24 | 1.41968 | 8.04333% |
| 3 | `lm_head` | 1 | 1.3096 | 7.41966% |
| 4 | `gate_proj` | 24 | 1.30499 | 7.39355% |
| 5 | `down_proj_residual` | 24 | 1.2607 | 7.14264% |
| 6 | `cache_append` | 24 | 1.21312 | 6.87305% |
| 7 | `v_proj` | 24 | 1.18848 | 6.73345% |
| 8 | `q_proj` | 24 | 1.1673 | 6.61343% |
| 9 | `swiglu` | 24 | 1.12886 | 6.39569% |
| 10 | `o_proj_residual` | 24 | 1.11805 | 6.33441% |
| 11 | `post_rmsnorm` | 24 | 1.11494 | 6.31682% |
| 12 | `input_rmsnorm` | 24 | 1.11277 | 6.30449% |
| 13 | `rope` | 24 | 1.0905 | 6.17831% |
| 14 | `k_proj` | 24 | 1.0729 | 6.07859% |
| 15 | `embedding` | 1 | 0.15952 | 0.903776% |
| 16 | `final_rmsnorm` | 1 | 0.042976 | 0.243485% |
