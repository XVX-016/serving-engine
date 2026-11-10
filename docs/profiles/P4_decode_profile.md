# Task P4 Decode Profile: In-Engine CUDA Event Instrumentation

## Decode Step Breakdown (Context = 517, Batch = 1, KV Cache Mode)

- **Total Kernels Launched**: 315
- **Host Wall Time**: 19.7257 ms
- **GPU Busy Time**: 14.4906 ms
- **Idle Gap Time**: 5.23508 ms (26.5394%)
- **Is Decode Launch-Bound?**: **YES (Launch Overhead Dominant)**

## Top Operations by Accumulated GPU Time

| Rank | Kernel / Operation | Total Invocations | Total Time (ms) | % of GPU Time |
|---|---|---|---|---|
| 1 | `up_proj` | 24 | 1.53453 | 10.5898% |
| 2 | `gate_proj` | 24 | 1.4711 | 10.1521% |
| 3 | `decode_attention` | 24 | 1.39923 | 9.65612% |
| 4 | `down_proj_residual` | 24 | 1.13187 | 7.81106% |
| 5 | `swiglu` | 24 | 1.04371 | 7.20267% |
| 6 | `rope` | 24 | 0.956864 | 6.60333% |
| 7 | `input_rmsnorm` | 24 | 0.910496 | 6.28335% |
| 8 | `post_rmsnorm` | 24 | 0.904896 | 6.2447% |
| 9 | `v_proj` | 24 | 0.89568 | 6.1811% |
| 10 | `q_proj` | 24 | 0.887264 | 6.12302% |
| 11 | `o_proj_residual` | 24 | 0.842208 | 5.81209% |
| 12 | `cache_append` | 24 | 0.841696 | 5.80856% |
| 13 | `k_proj` | 24 | 0.84032 | 5.79906% |
| 14 | `lm_head` | 1 | 0.655712 | 4.52508% |
| 15 | `embedding` | 1 | 0.141888 | 0.979171% |
| 16 | `final_rmsnorm` | 1 | 0.033152 | 0.228782% |
