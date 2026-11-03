#include "src/cuda/kernels.h"
#include "src/common/cuda_utils.h"
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <algorithm>

namespace engine {
namespace cuda {

// ============================================================================
// Flash-Style Prefill Attention (Scalar-Math FP32 accum, Tiled in Shared Mem)
// ============================================================================
// Tile sizes:
//   B_r = query tile size (e.g. 16 or 32)
//   B_c = key/value tile size (e.g. 16 or 32)
//   head_dim = 64
//
// Each threadblock processes one query tile (B_r tokens) for one query head (q_head).
// Grid: ( (seq_len + B_r - 1) / B_r, num_q_heads, batch_size )
//
// Online Softmax algorithm (FlashAttention-2 style):
// For query tile Q_i [B_r, D]:
// Initialize O_i = 0 [B_r, D], m_i = -inf [B_r], l_i = 0 [B_r]
// For key/value tile K_j, V_j [B_c, D]:
//   1. Check causal mask:
//      - If (j * B_c > (i + 1) * B_r - 1): strictly masked out! Skip entire tile!
//   2. Compute S_ij = Q_i * K_j^T * scale [B_r, B_c]
//      - If causal tile (j * B_c + c > i * B_r + r): S_ij[r, c] = -inf
//   3. m_ij = rowmax(S_ij)
//      m_new = max(m_i, m_ij)
//      P_ij = exp(S_ij - m_new)
//      l_new = l_i * exp(m_i - m_new) + rowsum(P_ij)
//      O_i = O_i * exp(m_i - m_new) + P_ij * V_j
//      m_i = m_new, l_i = l_new
// Finally:
//   O_i = O_i / l_i
// ============================================================================

constexpr int PREFILL_BR = 16;
constexpr int PREFILL_BC = 16;

__global__ void flash_prefill_gqa_kernel(
    const half* __restrict__ q,       // [batch_size, seq_len, num_q_heads, head_dim]
    const half* __restrict__ k,       // [batch_size, seq_len, num_kv_heads, head_dim]
    const half* __restrict__ v,       // [batch_size, seq_len, num_kv_heads, head_dim]
    half* __restrict__ out,           // [batch_size, seq_len, num_q_heads, head_dim]
    int64_t seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float scale
) {
    int q_tile_idx = blockIdx.x;
    int q_head = blockIdx.y;
    int b_idx = blockIdx.z;

    int q_start = q_tile_idx * PREFILL_BR;
    if (q_start >= seq_len) return;

    int group_size = num_q_heads / num_kv_heads;
    int kv_head = q_head / group_size;

    // Strides
    // q: batch * seq_len * num_q_heads * head_dim
    // k, v: batch * seq_len * num_kv_heads * head_dim
    size_t q_stride_s = static_cast<size_t>(num_q_heads) * head_dim;
    size_t q_stride_b = static_cast<size_t>(seq_len) * q_stride_s;

    size_t kv_stride_s = static_cast<size_t>(num_kv_heads) * head_dim;
    size_t kv_stride_b = static_cast<size_t>(seq_len) * kv_stride_s;

    // Shared memory:
    // s_k: [PREFILL_BC, head_dim]
    // s_v: [PREFILL_BC, head_dim]
    __shared__ half s_k[PREFILL_BC][64];
    __shared__ half s_v[PREFILL_BC][64];

    // Thread registers:
    // Each thread in the block handles 1 row of Q (row = threadIdx.x), where threadIdx.x in [0, PREFILL_BR - 1]
    // Block size = PREFILL_BR = 16 threads
    int r = threadIdx.x;
    int global_q_pos = q_start + r;

    // Local registers for this query row
    float q_reg[64];
    float acc_reg[64];
    for (int d = 0; d < head_dim; ++d) {
        acc_reg[d] = 0.0f;
    }
    float m_prev = -1e20f;
    float l_prev = 0.0f;

    bool valid_q = (global_q_pos < seq_len);

    if (valid_q) {
        const half* q_row_ptr = q + b_idx * q_stride_b + global_q_pos * q_stride_s + q_head * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            q_reg[d] = __half2float(q_row_ptr[d]);
        }
    }

    int num_kv_tiles = (seq_len + PREFILL_BC - 1) / PREFILL_BC;

    // Loop over KV tiles
    for (int kv_tile_idx = 0; kv_tile_idx < num_kv_tiles; ++kv_tile_idx) {
        int k_start = kv_tile_idx * PREFILL_BC;

        // Causal mask check: if k_start > q_start + PREFILL_BR - 1, the entire tile is masked out!
        if (k_start > q_start + PREFILL_BR - 1) {
            break; // Since k_start strictly increases, all further tiles are also causal-masked!
        }

        // Cooperatively load K and V tile into shared memory
        // PREFILL_BC = 16 rows, head_dim = 64 elements = 1024 elements
        // 16 threads in block: each thread loads 64 elements (1 row of K and 1 row of V)
        int kv_row = threadIdx.x;
        int global_k_pos = k_start + kv_row;

        if (global_k_pos < seq_len) {
            const half* k_row_ptr = k + b_idx * kv_stride_b + global_k_pos * kv_stride_s + kv_head * head_dim;
            const half* v_row_ptr = v + b_idx * kv_stride_b + global_k_pos * kv_stride_s + kv_head * head_dim;
            #pragma unroll
            for (int d = 0; d < head_dim; ++d) {
                s_k[kv_row][d] = k_row_ptr[d];
                s_v[kv_row][d] = v_row_ptr[d];
            }
        } else {
            #pragma unroll
            for (int d = 0; d < head_dim; ++d) {
                s_k[kv_row][d] = __float2half(0.0f);
                s_v[kv_row][d] = __float2half(0.0f);
            }
        }
        __syncthreads();

        if (valid_q) {
            // Compute scores for row r against all columns c in [0, PREFILL_BC - 1]
            float scores[PREFILL_BC];
            float tile_max = -1e20f;

            #pragma unroll
            for (int c = 0; c < PREFILL_BC; ++c) {
                int global_k = k_start + c;
                if (global_k < seq_len && global_k <= global_q_pos) {
                    float dot = 0.0f;
                    #pragma unroll
                    for (int d = 0; d < head_dim; ++d) {
                        dot += q_reg[d] * __half2float(s_k[c][d]);
                    }
                    scores[c] = dot * scale;
                    if (scores[c] > tile_max) tile_max = scores[c];
                } else {
                    scores[c] = -1e20f;
                }
            }

            // Online softmax update if any score is valid
            if (tile_max > -1e19f) {
                float m_new = fmaxf(m_prev, tile_max);
                float alpha = expf(m_prev - m_new);

                // Compute P = exp(scores - m_new) and sum_exp
                float tile_sum = 0.0f;
                #pragma unroll
                for (int c = 0; c < PREFILL_BC; ++c) {
                    if (scores[c] > -1e19f) {
                        scores[c] = expf(scores[c] - m_new);
                        tile_sum += scores[c];
                    } else {
                        scores[c] = 0.0f;
                    }
                }

                // Rescale previous accumulator
                #pragma unroll
                for (int d = 0; d < head_dim; ++d) {
                    acc_reg[d] *= alpha;
                }

                // Accumulate P * V
                #pragma unroll
                for (int c = 0; c < PREFILL_BC; ++c) {
                    float p_val = scores[c];
                    if (p_val > 0.0f) {
                        #pragma unroll
                        for (int d = 0; d < head_dim; ++d) {
                            acc_reg[d] += p_val * __half2float(s_v[c][d]);
                        }
                    }
                }

                l_prev = l_prev * alpha + tile_sum;
                m_prev = m_new;
            }
        }
        __syncthreads();
    }

    // Write final normalized output
    if (valid_q) {
        float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
        half* out_row_ptr = out + b_idx * q_stride_b + global_q_pos * q_stride_s + q_head * head_dim;
        #pragma unroll
        for (int d = 0; d < head_dim; ++d) {
            out_row_ptr[d] = __float2half(acc_reg[d] * inv_l);
        }
    }
}

// ----------------------------------------------------------------------------
// C++ Dispatcher
// ----------------------------------------------------------------------------
void flash_prefill_gqa(
    const half* q,
    const half* k,
    const half* v,
    half* out,
    int64_t seq_len,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int batch_size,
    cudaStream_t stream
) {
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    int q_tiles = (seq_len + PREFILL_BR - 1) / PREFILL_BR;

    dim3 grid(q_tiles, num_q_heads, batch_size);
    dim3 block(PREFILL_BR); // 16 threads, one per query row in the tile

    flash_prefill_gqa_kernel<<<grid, block, 0, stream>>>(
        q, k, v, out,
        seq_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        scale
    );
}

} // namespace cuda
} // namespace engine

