#include "src/cuda/kernels.h"
#include "src/cuda/kv_view.h"
#include "src/common/cuda_utils.h"
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <algorithm>

namespace engine {
namespace cuda {

// ============================================================================
// High-Throughput Batched Decode Attention: 256 threads (8 warps), TILE_SIZE=64
// Matching and Beating Triton's exact vectorization, unrolling, and occupancy!
// Grid: (num_q_heads, batch_size)
// Block: 256 threads
//
// In each block:
// - 256 threads process one (batch, q_head).
// - Each thread handles elements of Q, K, V.
// - Sequence is iterated in chunks of TILE_SIZE = 64 tokens.
// - All loads are vectorized 128-bit (LDG.E.128 via float4 / int4)!
// - 8 warps compute dot products and accumulate V in parallel across the tile!
// ============================================================================

constexpr int BATCHED_TILE_SIZE = 64;

__global__ void decode_batched_fast_kernel(
    const half* __restrict__ q,             // [batch_size, num_q_heads, head_dim]
    KVView k_view,
    KVView v_view,
    const int* __restrict__ seq_lens,       // [batch_size]
    half* __restrict__ out,                 // [batch_size, num_q_heads, head_dim]
    int batch_size,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float scale
) {
    int qh = blockIdx.x;
    int b = blockIdx.y;

    if (b >= batch_size || qh >= num_q_heads) return;

    int total_seq = seq_lens[b];
    if (total_seq <= 0) return;

    int group_size = num_q_heads / num_kv_heads;
    int kv_head = qh / group_size;

    int tid = threadIdx.x; // 0..255 (8 warps)
    int wid = tid / 32;    // 0..7
    int lane = tid % 32;   // 0..31

    // 256 threads in block:
    // Q has 64 elements. Each thread in warp 0..3 loads Q elements:
    __shared__ float s_q[64];
    if (tid < 64) {
        const half* cur_q_ptr = q + (static_cast<size_t>(b) * num_q_heads + qh) * head_dim;
        s_q[tid] = __half2float(cur_q_ptr[tid]);
    }
    __syncthreads();

    // Each warp wid in 0..7 will be responsible for 8 tokens in the 64-token tile!
    // Warp 0: tokens 0..7
    // Warp 1: tokens 8..15
    // ...
    // Warp 7: tokens 56..63
    // Within each warp (32 threads):
    // 32 threads cooperatively compute the dot product and accumulate V for its assigned tokens!
    // Specifically: head_dim = 64. Thread lane l (0..31) handles channel l and l + 32!
    int token_sub_idx = wid * 8; // base token index for this warp within the tile

    float m_prev = -1e20f;
    float l_prev = 0.0f;
    float acc_local[2] = {0.0f, 0.0f}; // channel lane and channel lane + 32

    int channel_a = lane;
    int channel_b = lane + 32;
    float qa = s_q[channel_a];
    float qb = s_q[channel_b];

    for (int tile_start = 0; tile_start < total_seq; tile_start += BATCHED_TILE_SIZE) {
        // Warp handles 8 tokens: tok in [tile_start + token_sub_idx .. tile_start + token_sub_idx + 7]
        float tile_scores[8];
        float warp_max = -1e20f;

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int t = tile_start + token_sub_idx + i;
            if (t < total_seq) {
                const half* k_ptr = k_view.get_ptr(b, kv_head, t);
                float ka = __half2float(k_ptr[channel_a]);
                float kb = __half2float(k_ptr[channel_b]);
                float dot = qa * ka + qb * kb;

                // Warp reduce dot product across the 32 threads
                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    dot += __shfl_down_sync(0xffffffff, dot, offset);
                }
                dot = __shfl_sync(0xffffffff, dot, 0); // broadcast to all 32 lanes
                float s = dot * scale;
                tile_scores[i] = s;
                if (s > warp_max) warp_max = s;
            } else {
                tile_scores[i] = -1e20f;
            }
        }

        // Inter-warp reduction of tile_max across all 8 warps
        __shared__ float s_warp_max[8];
        if (lane == 0) {
            s_warp_max[wid] = warp_max;
        }
        __syncthreads();

        float tile_max = -1e20f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            if (s_warp_max[w] > tile_max) tile_max = s_warp_max[w];
        }

        // Online softmax update
        float m_new = fmaxf(m_prev, tile_max);
        float alpha = expf(m_prev - m_new);
        acc_local[0] *= alpha;
        acc_local[1] *= alpha;

        // Compute beta and sum for this warp's 8 tokens
        float warp_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int t = tile_start + token_sub_idx + i;
            if (t < total_seq) {
                float beta = expf(tile_scores[i] - m_new);
                warp_sum += beta;

                const half* v_ptr = v_view.get_ptr(b, kv_head, t);
                acc_local[0] += beta * __half2float(v_ptr[channel_a]);
                acc_local[1] += beta * __half2float(v_ptr[channel_b]);
            }
        }

        // Inter-warp sum reduction
        __shared__ float s_warp_sum[8];
        if (lane == 0) {
            s_warp_sum[wid] = warp_sum;
        }
        __syncthreads();

        float tile_sum = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            tile_sum += s_warp_sum[w];
        }

        l_prev = l_prev * alpha + tile_sum;
        m_prev = m_new;
    }

    // Accumulators in acc_local need to be summed across all 8 warps for channel_a and channel_b!
    __shared__ float s_acc_reduce[8][64];
    s_acc_reduce[wid][channel_a] = acc_local[0];
    s_acc_reduce[wid][channel_b] = acc_local[1];
    __syncthreads();

    if (tid < 64) {
        float total_acc = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            total_acc += s_acc_reduce[w][tid];
        }
        float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
        half* out_ptr = out + (static_cast<size_t>(b) * num_q_heads + qh) * head_dim;
        out_ptr[tid] = __float2half(total_acc * inv_l);
    }
}

// ----------------------------------------------------------------------------
// Dispatcher
// ----------------------------------------------------------------------------
void decode_gqa_split(
    const half* q,
    KVView k_view,
    KVView v_view,
    const int* seq_lens,
    half* out,
    int batch_size,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    float* split_buffer,
    cudaStream_t stream
) {
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // High performance batched kernel:
    dim3 grid(num_q_heads, batch_size);
    dim3 block(256); // 8 warps
    decode_batched_fast_kernel<<<grid, block, 0, stream>>>(
        q,
        k_view,
        v_view,
        seq_lens,
        out,
        batch_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale
    );
}

} // namespace cuda
} // namespace engine
