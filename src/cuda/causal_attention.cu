#include "src/cuda/kernels.h"
#include "src/common/cuda_utils.h"
#include <cmath>
#include <cfloat>

namespace engine {
namespace cuda {

// Each block computes attention for one (query_pos, query_head)
// Grid: (seq_len, num_q_heads)
// Threads: 128 or 256
__global__ void causal_gqa_kernel(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    half* __restrict__ out,
    int64_t seq_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    float scale
) {
    int64_t q_pos = blockIdx.x;
    int64_t q_head = blockIdx.y;
    if (q_pos >= seq_len || q_head >= num_q_heads) return;

    int64_t group_size = num_q_heads / num_kv_heads;
    int64_t kv_head = q_head / group_size;

    const half* cur_q = q + (q_pos * num_q_heads + q_head) * head_dim;

    // Dynamically sized shared memory or fixed shared memory for scores up to max context
    // For general shapes, we compute online softmax!
    // Online Softmax tracking: m = max score, d = sum of exp(score - m)
    // And accumulated output accumulator: acc[d] in float32.

    // Each thread in the block can cooperatively compute dot-products
    // But since one block handles one (q_pos, q_head), and threads = 128:
    // Let's use warp shuffle / shared memory to compute dot product for each key token j in [0, q_pos].

    extern __shared__ float s_mem[]; // Used for reduction within block
    float* s_acc = s_mem; // size head_dim

    // Initialize output accumulator for this thread
    for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        s_acc[d] = 0.0f;
    }
    __syncthreads();

    float m_prev = -1e20f;
    float d_prev = 0.0f;

    for (int64_t k_pos = 0; k_pos <= q_pos; ++k_pos) {
        const half* cur_k = k + (k_pos * num_kv_heads + kv_head) * head_dim;
        const half* cur_v = v + (k_pos * num_kv_heads + kv_head) * head_dim;

        // Compute dot product q . k in FP32
        float dot = 0.0f;
        for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            dot += __half2float(cur_q[d]) * __half2float(cur_k[d]);
        }

        // Warp reduction for dot
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
        // m_new = max(m_prev, score)
        // alpha = exp(m_prev - m_new)
        // beta = exp(score - m_new)
        // d_new = d_prev * alpha + beta
        // acc = acc * alpha + beta * v
        float m_new = (score > m_prev) ? score : m_prev;
        float alpha = expf(m_prev - m_new);
        float beta = expf(score - m_new);
        float d_new = d_prev * alpha + beta;

        for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
            float v_val = __half2float(cur_v[d]);
            s_acc[d] = s_acc[d] * alpha + beta * v_val;
        }

        m_prev = m_new;
        d_prev = d_new;
        __syncthreads();
    }

    // Final normalization by d_prev and write to out
    half* cur_out = out + (q_pos * num_q_heads + q_head) * head_dim;
    float inv_d = (d_prev > 0.0f) ? (1.0f / d_prev) : 0.0f;

    for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
        cur_out[d] = __float2half(s_acc[d] * inv_d);
    }
}

void causal_gqa(
    const half* q,
    const half* k,
    const half* v,
    half* out,
    int64_t seq_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    cudaStream_t stream
) {
    if (seq_len <= 0) return;
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    dim3 grid(static_cast<unsigned int>(seq_len), static_cast<unsigned int>(num_q_heads));
    int threads = 128;
    size_t smem = head_dim * sizeof(float);

    causal_gqa_kernel<<<grid, threads, smem, stream>>>(
        q, k, v, out, seq_len, num_q_heads, num_kv_heads, head_dim, scale
    );
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace engine

