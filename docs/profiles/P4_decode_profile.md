# Task P4 Decode Profile: In-Engine CUDA Event Instrumentation

## Decode Step Breakdown (Context = 517, Batch = 1, KV Cache Mode)

- **Total Kernels Launched**: 435
- **Host Wall Time**: 23.6101 ms
- **GPU Busy Time**: 18.2694 ms
- **Idle Gap Time**: 5.34066 ms (22.6202%)
- **Is Decode Launch-Bound?**: **YES (Launch Overhead Dominant)**

## Top Operations by Accumulated GPU Time

| Rank | Kernel / Operation | Total Invocations | Total Time (ms) | % of GPU Time |
|---|---|---|---|---|
| 1 | `decode_attention` | 24 | 10.066 | 55.0977% |
| 2 | `gate_proj` | 24 | 1.43402 | 7.84926% |
| 3 | `up_proj` | 24 | 1.16979 | 6.403% |
| 4 | `lm_head` | 1 | 1.12122 | 6.13711% |
| 5 | `down_proj` | 24 | 1.05648 | 5.78277% |
| 6 | `o_proj` | 24 | 0.542976 | 2.97205% |
| 7 | `q_proj` | 24 | 0.477248 | 2.61227% |
| 8 | `v_proj` | 24 | 0.36048 | 1.97313% |
| 9 | `q_bias` | 24 | 0.33456 | 1.83125% |
| 10 | `rope` | 24 | 0.289216 | 1.58306% |
| 11 | `cache_append` | 24 | 0.2544 | 1.39249% |
| 12 | `k_proj` | 24 | 0.22112 | 1.21033% |
| 13 | `input_rmsnorm` | 24 | 0.160928 | 0.880859% |
| 14 | `post_rmsnorm` | 24 | 0.149888 | 0.82043% |
| 15 | `attn_residual` | 24 | 0.139072 | 0.761227% |
| 16 | `k_bias` | 24 | 0.13552 | 0.741785% |
| 17 | `swiglu` | 24 | 0.115104 | 0.630036% |
| 18 | `mlp_residual` | 24 | 0.112064 | 0.613396% |
| 19 | `v_bias` | 24 | 0.107264 | 0.587123% |
| 20 | `embedding` | 1 | 0.016384 | 0.0896798% |
| 21 | `final_rmsnorm` | 1 | 0.005664 | 0.0310026% |
