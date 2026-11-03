#include "src/cuda/kernels.h"
#include "src/common/cuda_utils.h"
#include <cmath>

namespace engine {
namespace cuda {

// Single-token decode attention:
// q: single token query [num_q_heads, head_dim]
// k_cache: [num_kv_heads, max_seq, head_dim]
// v_cache: [num_kv_heads, max_seq, head_dim]
// out: single token out [num_q_heads, head_dim]
// total_seq_len: number of valid tokens in cache (current_pos + 1)
// Each block handles 1 query head: grid = (num_q_heads)
__global__ void decode_gqa_kernel(
    const half* __restrict__ q,
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    half* __restrict__ out,
    int total_seq_len,
    int num_q_heads,
    int num_kv_heads,
    int max_seq,
    int head_dim,
    float scale
) {
    int q_head = blockIdx.x;
    if (q_head >= num_q_heads) return;

    int group_size = num_q_heads / num_kv_heads;
    int kv_head = q_head / group_size;

    const half* cur_q = q + q_head * head_dim;
    const half* cur_k_head = k_cache + static_cast<size_t>(kv_head) * max_seq * head_dim;
    const half* cur_v_head = v_cache + static_cast<size_t>(kv_head) * max_seq * head_dim;

    extern __shared__ float s_mem[];
    float* s_acc = s_mem; // size head_dim

    // Initialize accumulator in shared memory
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        s_acc[d] = 0.0f;
    }
    __syncthreads();

    float m_prev = -1e20f;
    float d_prev = 0.0f;

    // Iterate through past tokens 0 .. total_seq_len - 1
    for (int k_pos = 0; k_pos < total_seq_len; ++k_pos) {
        const half* cur_k = cur_k_head + static_cast<size_t>(k_pos) * head_dim;
        const half* cur_v = cur_v_head + static_cast<size_t>(k_pos) * head_dim;

        // Compute dot product q . k in FP32
        float dot = 0.0f;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            dot += __half2float(cur_q[d]) * __half2float(cur_k[d]);
        }

        // Warp reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        }

        __shared__ float s_dot[32];
        int lane = threadIdx.x % 32;
        int wid = threadIdx.x / 32;
        if (lane == 0) {
            s_dot[wid] = dot;
        }
        __syncthreads();

        float score = 0.0f;
        if (wid == 0) {
            float val = (lane < (blockDim.x + 31) / 32) ? s_dot[lane] : 0.0f;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                val += __shfl_down_sync(0xffffffff, val, offset);
            }
            if (lane == 0) {
                s_dot[0] = val * scale;
            }
        }
        __syncthreads();

        score = s_dot[0];

        // Online softmax update:
        float m_new = (score > m_prev) ? score : m_prev;
        float alpha = expf(m_prev - m_new);
        float beta = expf(score - m_new);
        float d_new = d_prev * alpha + beta;

        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float v_val = __half2float(cur_v[d]);
            s_acc[d] = s_acc[d] * alpha + beta * v_val;
        }

        m_prev = m_new;
        d_prev = d_new;
        __syncthreads();
    }

    // Final normalization
    half* cur_out = out + q_head * head_dim;
    float inv_d = (d_prev > 0.0f) ? (1.0f / d_prev) : 0.0f;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        cur_out[d] = __float2half(s_acc[d] * inv_d);
    }
}

void decode_gqa(
    const half* q,
    const half* k_cache,
    const half* v_cache,
    half* out,
    int total_seq_len,
    int num_q_heads,
    int num_kv_heads,
    int max_seq,
    int head_dim,
    cudaStream_t stream
) {
    if (total_seq_len <= 0) return;
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    dim3 grid(static_cast<unsigned int>(num_q_heads));
    int threads = 128;
    size_t smem = head_dim * sizeof(float);

    decode_gqa_kernel<<<grid, threads, smem, stream>>>(
        q, k_cache, v_cache, out,
        total_seq_len, num_q_heads, num_kv_heads, max_seq, head_dim, scale
    );
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace engine

