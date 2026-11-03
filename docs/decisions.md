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

## 2. FP32 Logits Evaluation & Precision Format

### Decision:
The LM head projection operates on FP16 activations and FP16 tied embedding weights using `cublasGemmEx` with FP32 accumulation (`CUDA_R_32F`). Logits returned to host and processed by the sampler are represented in FP32 (or FP16 on device during fused sampling).

### Justification:
- Qwen2.5-0.5B-Instruct has a vocabulary of 151,936 tokens.
- Softmax and temperature scaling on 151,936 logits require high numerical dynamic range to avoid underflow/overflow during exponentiation.
- FP32 accumulation in cuBLAS prevents catastrophic cancellation across the large hidden dimension ($H=896$).
- Unpadded/padded vocabulary: The model's true vocabulary size is 151,936. Any token indices $\ge 151,936$ (or unused padding tokens up to power-of-two boundaries) are strictly masked out ($-\infty$) before softmax and sampling.

---

## 3. GPU vs Host Sampling Roadmap

### Current Status (Task P4 / P4b):
- Sampling currently executes on the host after device-to-host transfer of the single-token logits vector ($1 \times 151,936$), using `std::nth_element` (partial sort in $\sim 0.2$ ms) for top-$k$, followed by temperature scaling, top-$p$ filtering, and seeded RNG inversion.
- Repetition penalty and greedy argmax reduction kernels are available on GPU.

### Roadmap to P10 / P11:
- For batch-1 decode with host control, downloading $1 \times 151,936$ FP16 logits takes $\sim 0.15$ ms across PCIe, which is negligible compared to current decode latencies ($\sim 15$ ms).
- However, for continuous batching (Task P10) and CUDA Graphs (Task P11), CPU synchronization on the decode loop is strictly prohibited.
- Full GPU-native sampling (GPU top-$k$ via radix select / parallel bitonic select, GPU top-$p$ scan, and GPU RNG kernel without host round-trip) will be integrated directly into the CUDA Graph runner prior to P10/P11.

