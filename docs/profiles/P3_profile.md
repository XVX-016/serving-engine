# Task P3 Profile: In-Engine CUDA Event Instrumentation

> [!NOTE]
> This profile was generated using **in-engine CUDA event instrumentation** (`cudaEventElapsedTime` around every kernel/cuBLAS launch), NOT Nsight Systems CLI. The elevated Nsight Systems profiling command is documented in `docs/ENV.md`.

## Forward Step Breakdown (Seq Len = 512, Batch = 1, Recompute Mode)

- **Total Kernels / API Calls per Step**: 411
- **Host Wall Time**: 395.242 ms
- **Total GPU Busy Time**: 118.783 ms
- **Idle Gap Total**: 276.459 ms (69.9468%)
- **Is Decode Launch-Bound?**: **YES (Launch Overhead Dominant)**

## Top Operations by Accumulated GPU Time

| Rank | Kernel / Operation | Total Invocations | Total Time (ms) | % of GPU Time |
|---|---|---|---|---|
| 1 | `attention` | 24 | 89.0542 | 74.9723% |
| 2 | `lm_head` | 1 | 4.80461 | 4.04487% |
| 3 | `down_proj` | 24 | 4.70352 | 3.95976% |
| 4 | `gate_proj` | 24 | 4.60317 | 3.87528% |
| 5 | `up_proj` | 24 | 4.59603 | 3.86927% |
| 6 | `o_proj` | 24 | 2.12464 | 1.78868% |
| 7 | `q_proj` | 24 | 1.87456 | 1.57814% |
| 8 | `swiglu` | 24 | 0.95232 | 0.801732% |
| 9 | `k_proj` | 24 | 0.726784 | 0.611859% |
| 10 | `v_proj` | 24 | 0.687808 | 0.579046% |
| 11 | `rope` | 24 | 0.670304 | 0.56431% |
| 12 | `input_rmsnorm` | 24 | 0.6288 | 0.529369% |
| 13 | `v_bias` | 24 | 0.624672 | 0.525894% |
| 14 | `k_bias` | 24 | 0.571008 | 0.480716% |
| 15 | `mlp_residual` | 24 | 0.566048 | 0.47654% |
| 16 | `q_bias` | 24 | 0.521728 | 0.439228% |
| 17 | `attn_residual` | 24 | 0.521568 | 0.439094% |
| 18 | `post_rmsnorm` | 24 | 0.5016 | 0.422283% |
| 19 | `final_rmsnorm` | 1 | 0.03616 | 0.0304421% |
| 20 | `embedding` | 1 | 0.013312 | 0.011207% |

## Analysis & Takeaways
1. **Launch Overhead**: At sequence length 512, 411 kernel launches take 118.783 ms of compute time while host overhead is 276.459 ms.
2. **cuBLAS Dominance**: Linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `lm_head`) account for the majority of execution time.
3. **Next Steps (P4 / P5)**: KV-caching eliminates $O(S^2)$ attention and recomputes only sequence length 1 per token; fused kernels and CUDA graphs will eliminate launch bubbles completely.
