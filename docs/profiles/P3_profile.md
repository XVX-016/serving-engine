# Task P3 Profile: In-Engine CUDA Event Instrumentation

> [!NOTE]
> This profile was generated using **in-engine CUDA event instrumentation** (`cudaEventElapsedTime` around every kernel/cuBLAS launch), NOT Nsight Systems CLI. The elevated Nsight Systems profiling command is documented in `docs/ENV.md`.

## Forward Step Breakdown (Seq Len = 512, Batch = 1, Recompute Mode)

- **Total Kernels / API Calls per Step**: 411
- **Host Wall Time**: 352.852 ms
- **Total GPU Busy Time (Kernel Execution)**: 163.828 ms
- **Non-GPU Host / Overhead Gap**: 189.024 ms
- **Is Decode Launch-Bound?**: **NOT YET MEASURED (Confounded in P3)**
  - *Analysis of the gap*: True CUDA kernel launches take ~5–20 µs each on modern GPUs, which accounts for at most ~2–8 ms across 411 launches. The remaining ~180+ ms gap is dominated by:
    1. **Synchronous profiling instrumentation**: The in-engine timer calls cudaEventSynchronize around every single kernel/cuBLAS launch, causing host stalls and preventing queue overlap.
    2. **Massive Host-Device Memory Transfer**: Recompute mode at 512 tokens produces and transfers logits for all 512 positions (512 * 151,936 * 2 ~= 155.6 MB D2H copy).
    3. **Host Argmax**: The CPU iterates over all 512 * 151,936 ~= 77.8M float elements to select the next token.
    4. **Unrepresentative Workload**: Recompute mode at 512 tokens is prefill-like, not the single-token cached decode step the launch-bound question addresses.
  - *Resolution*: Recalibrate the profiling instrument in P4 by removing per-kernel syncs, extracting logits for the last token only, and evaluating a true single-token decode step with KV cache.

## Top Operations by Accumulated GPU Time

| Rank | Kernel / Operation | Total Invocations | Total Time (ms) | % of GPU Time |
|---|---|---|---|---|
| 1 | `attention` | 24 | 90.1625 | 55.0348% |
| 2 | `up_proj` | 24 | 7.36557 | 4.49591% |
| 3 | `down_proj` | 24 | 7.35338 | 4.48847% |
| 4 | `gate_proj` | 24 | 7.3008 | 4.45638% |
| 5 | `o_proj` | 24 | 5.57235 | 3.40134% |
| 6 | `lm_head` | 1 | 4.74294 | 2.89507% |
| 7 | `q_proj` | 24 | 4.52669 | 2.76307% |
| 8 | `k_proj` | 24 | 3.57808 | 2.18405% |
| 9 | `swiglu` | 24 | 3.52512 | 2.15172% |
| 10 | `v_proj` | 24 | 3.51994 | 2.14855% |
| 11 | `post_rmsnorm` | 24 | 3.45475 | 2.10877% |
| 12 | `attn_residual` | 24 | 3.40973 | 2.08128% |
| 13 | `mlp_residual` | 24 | 3.35776 | 2.04956% |
| 14 | `input_rmsnorm` | 24 | 3.23686 | 1.97577% |
| 15 | `q_bias` | 24 | 3.16486 | 1.93182% |
| 16 | `rope` | 24 | 3.11546 | 1.90166% |
| 17 | `k_bias` | 24 | 3.06922 | 1.87344% |
| 18 | `v_bias` | 24 | 2.97891 | 1.81832% |
| 19 | `embedding` | 1 | 0.25616 | 0.156359% |
| 20 | `final_rmsnorm` | 1 | 0.136992 | 0.0836194% |

## Analysis & Takeaways
1. **Launch Overhead**: At sequence length 512, 411 kernel launches take 163.828 ms of compute time while host overhead is 189.024 ms.
2. **cuBLAS Dominance**: Linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `lm_head`) account for the majority of execution time.
3. **Next Steps (P4 / P5)**: KV-caching eliminates $O(S^2)$ attention and recomputes only sequence length 1 per token; fused kernels and CUDA graphs will eliminate launch bubbles completely.
