#include "src/runtime/kv_cache.h"

namespace engine {
namespace cuda {

// Kernel to copy keys and values from token-major [num_tokens, num_kv_heads, head_dim]
// to head-major cache [num_kv_heads, max_seq, head_dim] at [start_pos .. start_pos + num_tokens)
__global__ void append_kv_kernel(
    const half* __restrict__ src_k,
    const half* __restrict__ src_v,
    half* __restrict__ dst_k,
    half* __restrict__ dst_v,
    int num_tokens,
    int start_pos,
    int num_kv_heads,
    int max_seq,
    int head_dim
) {
    int token_idx = blockIdx.x; // 0 .. num_tokens - 1
    int kv_h = blockIdx.y;      // 0 .. num_kv_heads - 1
    int d = threadIdx.x;        // 0 .. head_dim - 1

    if (token_idx >= num_tokens || kv_h >= num_kv_heads || d >= head_dim) return;

    int pos = start_pos + token_idx;
    if (pos >= max_seq) return;

    // Source offset: token_major [token_idx, kv_h, d]
    size_t src_offset = (static_cast<size_t>(token_idx) * num_kv_heads + kv_h) * head_dim + d;

    // Dest offset: [kv_h, pos, d]
    size_t dst_offset = (static_cast<size_t>(kv_h) * max_seq + pos) * head_dim + d;

    dst_k[dst_offset] = src_k[src_offset];
    dst_v[dst_offset] = src_v[src_offset];
}

void append_kv_cache(
    const half* src_k,
    const half* src_v,
    half* dst_k_head_base,
    half* dst_v_head_base,
    int num_tokens,
    int start_pos,
    int num_kv_heads,
    int max_seq,
    int head_dim,
    cudaStream_t stream
) {
    if (num_tokens <= 0) return;

    dim3 grid(static_cast<unsigned int>(num_tokens), static_cast<unsigned int>(num_kv_heads));
    int threads = head_dim; // For head_dim=64, 64 threads per block

    append_kv_kernel<<<grid, threads, 0, stream>>>(
        src_k, src_v,
        dst_k_head_base, dst_v_head_base,
        num_tokens, start_pos, num_kv_heads, max_seq, head_dim
    );
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace engine

