# Task P3 Profile: In-Engine CUDA Event Instrumentation

> [!NOTE]
> This profile was generated using **in-engine CUDA event instrumentation** (`cudaEventElapsedTime` around every kernel/cuBLAS launch), NOT Nsight Systems CLI. The elevated Nsight Systems profiling command is documented in `docs/ENV.md`.

## Forward Step Breakdown (Seq Len = 512, Batch = 1, Recompute Mode)

- **Total Kernels / API Calls per Step**: 411
- **Host Wall Time**: 395.242 ms
- **Total GPU Busy Time (Kernel Execution)**: 118.783 ms
- **Non-GPU Host / Overhead Gap**: 276.459 ms
- **Is Decode Launch-Bound?**: **NOT YET MEASURED (Confounded in P3)**
  - *Analysis of the gap*: True CUDA kernel launches take ~5–20 µs each on modern GPUs, which accounts for at most ~2–8 ms across 411 launches. The remaining ~270 ms gap is dominated by:
    1. **Synchronous profiling instrumentation**: The in-engine timer calls cudaEventSynchronize around every single kernel/cuBLAS launch, causing host stalls and preventing queue overlap.
    2. **Massive Host-Device Memory Transfer**: Recompute mode at 512 tokens produces and transfers logits for all 512 positions (512 * 151,936 * 2 ~= 155.6 MB D2H copy).
    3. **Host Argmax**: The CPU iterates over all 512 * 151,936 ~= 77.8M float elements to select the next token.
    4. **Unrepresentative Workload**: Recompute mode at 512 tokens is prefill-like, not the single-token cached decode step the launch-bound question addresses.
  - *Resolution*: Recalibrate the profiling instrument in P4 by removing per-kernel syncs, extracting logits for the last token only, and evaluating a true single-token decode step with KV cache.

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
