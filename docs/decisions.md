# Architectural Decisions (docs/decisions.md)

## 1. KV Cache Layout: `[layer][2 (K or V)][kv_head][max_seq][head_dim]` in FP16

### Decision:
The contiguous KV cache is allocated per sequence with the layout:
`[num_layers][2 (0=K, 1=V)][num_kv_heads][max_seq][head_dim]` in FP16.

### Justification:
- Along the fastest-moving dimension, each head vector of length `head_dim` (64 half elements = 128 bytes) is stored contiguously in memory. A 128-byte chunk matches the exact transaction size of a GPU L2 cache line and global memory bus transaction.
- When an attention threadblock reads past keys and values for a specific `kv_head`, each token's key/value vector is loaded with fully coalesced memory transactions.
- In Task P7, split-KV decode and flash attention will partition chunks of tokens along `max_seq`, where head-major layout ensures each thread block processes a contiguous memory slice `[chunk_size, head_dim]`.

---

## 2. FP32 Logits — Implemented (P4b)

### Decision:
The LM head projection operates on FP16 activations and FP16 tied embedding weights using `cublasGemmEx` with **FP32 output on device** (`CUDA_R_32F` output type). The logits buffer (`DeviceBuffer<float>`) and the sampler (`sample(const float*, ...)`, `argmax_kernel_f32`, `apply_repetition_penalty_kernel_f32`) all operate natively in FP32. No FP16-to-FP32 conversion is needed in the sampling path.

### Implementation Status:
- **Implemented** in commit `90f97b2` (P4b). The `matmul_transB_f32` function in `forward_gpu.cu` calls `cublasGemmEx` with `CUDA_R_32F` for the C/D matrices.
- **P3 teacher-forced gate result after FP32 LM head**:
  - `03_long/pos12` mismatch **resolved**: The two candidates (token 13 vs 11) had a margin of only 0.003 in the CPU FP32 reference. FP16 output truncation rounded both to 22.265625, creating a spurious tie. FP32 output preserves the 0.0044 gap.
  - `05_unicode/pos37` mismatch **remains — valid near-tie**: Token 11884 vs 9612 have only 0.0094 margin in the FP32 reference. FP16 accumulation across the preceding 24 transformer layers shifts the hidden state enough to reverse the order by 0.0018. This is a genuine near-tie originating from accumulated FP16 rounding in the transformer layers, not from the LM head output format.

### Justification:
- Qwen2.5-0.5B-Instruct has a model output vocabulary dimension of 151,936 tokens (padded from the tokenizer's 151,665 tokens up to the nearest multiple of 64 for optimal tensor core alignment).
- Softmax and temperature scaling on 151,936 logits require high numerical dynamic range to avoid underflow/overflow during exponentiation.
- FP32 accumulation in cuBLAS (`CUBLAS_COMPUTE_32F`) prevents catastrophic cancellation across the large hidden dimension ($H=896$).
- Unpadded/padded vocabulary: The tokenizer's valid token vocabulary ends at token ID 151,664 (size = 151,665). Any token indices from 151,665 up to 151,935 are padding entries. In the sampler, these indices are strictly masked to $-\infty$ (and excluded from argmax reduction) so they can never be sampled or generated.

---

## 3. GPU vs Host Sampling Roadmap

### Current Status (Task P4 / P4b):
- Sampling currently executes on the host after device-to-host transfer of the single-token logits vector ($1 \times 151,936$), using `std::nth_element` (partial sort in $\sim 0.2$ ms) for top-$k$, followed by temperature scaling, top-$p$ filtering, and seeded RNG inversion.
- Repetition penalty and greedy argmax reduction kernels are available on GPU.

### Roadmap to P10 / P11:
- For batch-1 decode with host control, downloading $1 \times 151,936$ FP32 logits takes $\sim 0.3$ ms across PCIe, which is negligible compared to current decode latencies ($\sim 15$ ms).
- However, for continuous batching (Task P10) and CUDA Graphs (Task P11), CPU synchronization on the decode loop is strictly prohibited.
- Full GPU-native sampling (GPU top-$k$ via radix select / parallel bitonic select, GPU top-$p$ scan, and GPU RNG kernel without host round-trip) will be integrated directly into the CUDA Graph runner prior to P10/P11.

