# Task P4 Decode Profile: In-Engine CUDA Event Instrumentation

## Decode Step Breakdown (Context = 517, Batch = 1, KV Cache Mode)

- **Total Kernels Launched**: 435
- **Host Wall Time**: 33.6535 ms
- **GPU Busy Time**: 25.2981 ms
- **Idle Gap Time**: 8.35536 ms (24.8276%)
- **Is Decode Launch-Bound?**: **YES (Launch Overhead Dominant)**

## Top Operations by Accumulated GPU Time

| Rank | Kernel / Operation | Total Invocations | Total Time (ms) | % of GPU Time |
|---|---|---|---|---|
| 1 | `decode_attention` | 24 | 10.4809 | 41.4296% |
| 2 | `gate_proj` | 24 | 1.90179 | 7.51752% |
| 3 | `up_proj` | 24 | 1.57645 | 6.23148% |
| 4 | `down_proj` | 24 | 1.36176 | 5.38285% |
| 5 | `o_proj` | 24 | 1.35245 | 5.34604% |
| 6 | `lm_head` | 1 | 1.13018 | 4.46743% |
| 7 | `q_proj` | 24 | 1.09821 | 4.34106% |
| 8 | `k_proj` | 24 | 0.97664 | 3.86052% |
| 9 | `v_proj` | 24 | 0.719168 | 2.84277% |
| 10 | `v_bias` | 24 | 0.652192 | 2.57802% |
| 11 | `k_bias` | 24 | 0.51936 | 2.05296% |
| 12 | `cache_append` | 24 | 0.512096 | 2.02424% |
| 13 | `rope` | 24 | 0.463968 | 1.834% |
| 14 | `attn_residual` | 24 | 0.457504 | 1.80845% |
| 15 | `post_rmsnorm` | 24 | 0.446528 | 1.76506% |
| 16 | `q_bias` | 24 | 0.417184 | 1.64907% |
| 17 | `mlp_residual` | 24 | 0.411552 | 1.62681% |
| 18 | `swiglu` | 24 | 0.395008 | 1.56141% |
| 19 | `input_rmsnorm` | 24 | 0.378272 | 1.49526% |
| 20 | `embedding` | 1 | 0.0336 | 0.132816% |
| 21 | `final_rmsnorm` | 1 | 0.013312 | 0.0526205% |
