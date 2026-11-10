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
// Multi-Token Prefetch Split-KV Batched Decode Attention
//
// Grid: (num_q_heads, batch_size, num_splits)
// Block: 128 threads (4 warps)
//
// Adaptive chunk size:
// At batch >= 16, fewer splits (SPLIT_CHUNK_SIZE=256) maintain high warp
// workload and reduce reduction overhead; at batch < 16, SPLIT_CHUNK_SIZE=128
// provides ample grid parallelism across SMs.
// ============================================================================

__device__ __forceinline__ int get_split_chunk_size(int batch_size) {
    return (batch_size >= 16) ? 256 : 128;
}

__global__ void decode_splitk_partial_kernel(
    const half* __restrict__ q,             // [batch_size, num_q_heads, head_dim]
    KVView k_view,
    KVView v_view,
    const int* __restrict__ seq_lens,       // [batch_size]
    float* __restrict__ split_buffer,       // [batch_size, num_q_heads, max_splits, head_dim + 2]
    int batch_size,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int max_splits,
    int split_chunk_size,
    float scale
) {
    int qh = blockIdx.x;
    int b = blockIdx.y;
    int split_id = blockIdx.z;

    if (b >= batch_size || qh >= num_q_heads || split_id >= max_splits) return;

    int total_seq = seq_lens[b];
    int split_start = split_id * split_chunk_size;
    if (split_start >= total_seq) {
        int tid = threadIdx.x;
        size_t entry_offset = ((static_cast<size_t>(b) * num_q_heads + qh) * max_splits + split_id) * (head_dim + 2);
        if (tid < head_dim) {
            split_buffer[entry_offset + tid] = 0.0f;
        } else if (tid == head_dim) {
            split_buffer[entry_offset + head_dim] = -1e20f;
        } else if (tid == head_dim + 1) {
            split_buffer[entry_offset + head_dim + 1] = 0.0f;
        }
        return;
    }

    int split_end = min(split_start + split_chunk_size, total_seq);

    int group_size = num_q_heads / num_kv_heads;
    int kv_head = qh / group_size;

    int tid = threadIdx.x; // 0..127 (4 warps)
    int wid = tid / 32;    // 0..3
    int lane = tid % 32;   // 0..31

    __shared__ float s_q[64];
    if (tid < 64) {
        const half* cur_q_ptr = q + (static_cast<size_t>(b) * num_q_heads + qh) * head_dim;
        s_q[tid] = __half2float(cur_q_ptr[tid]);
    }
    __syncthreads();

    int channel_a = lane;
    int channel_b = lane + 32;
    float qa = s_q[channel_a];
    float qb = s_q[channel_b];

    float m_prev = -1e20f;
    float l_prev = 0.0f;
    float acc_local[2] = {0.0f, 0.0f};

    constexpr int SUB_TILE = 32;
    int token_sub_idx = wid * 8; // warp wid handles 8 tokens per sub-tile

    for (int tile_start = split_start; tile_start < split_end; tile_start += SUB_TILE) {
        float tile_scores[8];
        float warp_max = -1e20f;

        // Multi-token prefetch registers for K
        float k_reg_a[8];
        float k_reg_b[8];

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int t = tile_start + token_sub_idx + i;
            if (t < split_end) {
                const half* k_ptr = k_view.get_ptr(b, kv_head, t);
                k_reg_a[i] = __half2float(k_ptr[channel_a]);
                k_reg_b[i] = __half2float(k_ptr[channel_b]);
            } else {
                k_reg_a[i] = 0.0f;
                k_reg_b[i] = 0.0f;
            }
        }

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int t = tile_start + token_sub_idx + i;
            if (t < split_end) {
                float dot = qa * k_reg_a[i] + qb * k_reg_b[i];

                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    dot += __shfl_down_sync(0xffffffff, dot, offset);
                }
                dot = __shfl_sync(0xffffffff, dot, 0);
                float s = dot * scale;
                tile_scores[i] = s;
                if (s > warp_max) warp_max = s;
            } else {
                tile_scores[i] = -1e20f;
            }
        }

        __shared__ float s_warp_max[4];
        if (lane == 0) {
            s_warp_max[wid] = warp_max;
        }
        __syncthreads();

        float tile_max = -1e20f;
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            if (s_warp_max[w] > tile_max) tile_max = s_warp_max[w];
        }

        float m_new = fmaxf(m_prev, tile_max);
        float alpha = expf(m_prev - m_new);
        acc_local[0] *= alpha;
        acc_local[1] *= alpha;

        // Multi-token prefetch registers for V
        float v_reg_a[8];
        float v_reg_b[8];

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int t = tile_start + token_sub_idx + i;
            if (t < split_end) {
                const half* v_ptr = v_view.get_ptr(b, kv_head, t);
                v_reg_a[i] = __half2float(v_ptr[channel_a]);
                v_reg_b[i] = __half2float(v_ptr[channel_b]);
            } else {
                v_reg_a[i] = 0.0f;
                v_reg_b[i] = 0.0f;
            }
        }

        float warp_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int t = tile_start + token_sub_idx + i;
            if (t < split_end) {
                float beta = expf(tile_scores[i] - m_new);
                warp_sum += beta;

                acc_local[0] += beta * v_reg_a[i];
                acc_local[1] += beta * v_reg_b[i];
            }
        }

        __shared__ float s_warp_sum[4];
        if (lane == 0) {
            s_warp_sum[wid] = warp_sum;
        }
        __syncthreads();

        float tile_sum = 0.0f;
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            tile_sum += s_warp_sum[w];
        }

        l_prev = l_prev * alpha + tile_sum;
        m_prev = m_new;
    }

    __shared__ float s_acc_reduce[4][64];
    s_acc_reduce[wid][channel_a] = acc_local[0];
    s_acc_reduce[wid][channel_b] = acc_local[1];
    __syncthreads();

    size_t entry_offset = ((static_cast<size_t>(b) * num_q_heads + qh) * max_splits + split_id) * (head_dim + 2);
    if (tid < 64) {
        float total_acc = 0.0f;
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            total_acc += s_acc_reduce[w][tid];
        }
        split_buffer[entry_offset + tid] = total_acc;
    } else if (tid == 64) {
        split_buffer[entry_offset + head_dim] = m_prev;
    } else if (tid == 65) {
        split_buffer[entry_offset + head_dim + 1] = l_prev;
    }
}

// ============================================================================
// Reduction Kernel: Combines partial split accumulators into final output
// Grid: (num_q_heads, batch_size)
// Block: 64 threads (one thread per head_dim channel)
// ============================================================================

__global__ void decode_splitk_reduce_kernel(
    const float* __restrict__ split_buffer, // [batch_size, num_q_heads, max_splits, head_dim + 2]
    half* __restrict__ out,                 // [batch_size, num_q_heads, head_dim]
    int batch_size,
    int num_q_heads,
    int head_dim,
    int num_splits
) {
    int qh = blockIdx.x;
    int b = blockIdx.y;
    int tid = threadIdx.x; // 0..63

    if (b >= batch_size || qh >= num_q_heads || tid >= head_dim) return;

    size_t base_split_offset = ((static_cast<size_t>(b) * num_q_heads + qh) * num_splits) * (head_dim + 2);

    float m_global = -1e20f;
    for (int s = 0; s < num_splits; ++s) {
        size_t entry_offset = base_split_offset + static_cast<size_t>(s) * (head_dim + 2);
        float m_s = split_buffer[entry_offset + head_dim];
        if (m_s > m_global) m_global = m_s;
    }

    float l_global = 0.0f;
    float acc_channel = 0.0f;

    for (int s = 0; s < num_splits; ++s) {
        size_t entry_offset = base_split_offset + static_cast<size_t>(s) * (head_dim + 2);
        float m_s = split_buffer[entry_offset + head_dim];
        float l_s = split_buffer[entry_offset + head_dim + 1];
        float acc_s = split_buffer[entry_offset + tid];

        float alpha = expf(m_s - m_global);
        l_global += l_s * alpha;
        acc_channel += acc_s * alpha;
    }

    float inv_l = (l_global > 0.0f) ? (1.0f / l_global) : 0.0f;
    half* out_ptr = out + (static_cast<size_t>(b) * num_q_heads + qh) * head_dim;
    out_ptr[tid] = __float2half(acc_channel * inv_l);
}

// ----------------------------------------------------------------------------
// Dispatcher: Split-KV path forced for all configurations
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

    int split_chunk_size = (batch_size >= 16) ? 256 : 128;
    int num_splits = (max_seq_len + split_chunk_size - 1) / split_chunk_size;
    if (num_splits < 1) num_splits = 1;

    dim3 grid_partial(num_q_heads, batch_size, num_splits);
    dim3 block_partial(128); // 4 warps
    decode_splitk_partial_kernel<<<grid_partial, block_partial, 0, stream>>>(
        q,
        k_view,
        v_view,
        seq_lens,
        split_buffer,
        batch_size,
        num_q_heads,
        num_kv_heads,
        head_dim,
        num_splits,
        split_chunk_size,
        scale
    );

    dim3 grid_reduce(num_q_heads, batch_size);
    dim3 block_reduce(64);
    decode_splitk_reduce_kernel<<<grid_reduce, block_reduce, 0, stream>>>(
        split_buffer,
        out,
        batch_size,
        num_q_heads,
        head_dim,
        num_splits
    );
}

} // namespace cuda
} // namespace engine
