#include "src/cuda/kernels.h"
#include "src/common/cuda_utils.h"
#include <cmath>

namespace engine {
namespace cuda {

__global__ void rope_kernel(
    half* __restrict__ q,
    half* __restrict__ k,
    int64_t seq_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t start_pos,
    float rope_theta
) {
    int64_t half_dim = head_dim / 2;
    int64_t i = threadIdx.x; // index in [0, half_dim)
    if (i >= half_dim) return;

    int64_t s = blockIdx.x; // token position in sequence
    int64_t h = blockIdx.y; // head index

    int64_t pos = start_pos + s;
    float freq = 1.0f / powf(rope_theta, (2.0f * static_cast<float>(i)) / static_cast<float>(head_dim));
    float val = static_cast<float>(pos) * freq;
    float cos_val = cosf(val);
    float sin_val = sinf(val);

    // Apply to Q if h < num_q_heads
    if (h < num_q_heads) {
        int64_t q_offset = (s * num_q_heads + h) * head_dim;
        float x1 = __half2float(q[q_offset + i]);
        float x2 = __half2float(q[q_offset + i + half_dim]);

        // rotate_half: [-x2, x1]
        // x_rot = x * cos + rotate_half(x) * sin
        float out1 = x1 * cos_val - x2 * sin_val;
        float out2 = x2 * cos_val + x1 * sin_val;

        q[q_offset + i] = __float2half(out1);
        q[q_offset + i + half_dim] = __float2half(out2);
    }

    // Apply to K if h < num_kv_heads
    if (h < num_kv_heads) {
        int64_t k_offset = (s * num_kv_heads + h) * head_dim;
        float x1 = __half2float(k[k_offset + i]);
        float x2 = __half2float(k[k_offset + i + half_dim]);

        float out1 = x1 * cos_val - x2 * sin_val;
        float out2 = x2 * cos_val + x1 * sin_val;

        k[k_offset + i] = __float2half(out1);
        k[k_offset + i + half_dim] = __float2half(out2);
    }
}

void apply_rope(
    half* q,
    half* k,
    int64_t seq_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t start_pos,
    float rope_theta,
    cudaStream_t stream
) {
    if (seq_len <= 0) return;
    int64_t max_heads = (num_q_heads > num_kv_heads) ? num_q_heads : num_kv_heads;
    dim3 grid(static_cast<unsigned int>(seq_len), static_cast<unsigned int>(max_heads));
    int threads = static_cast<int>(head_dim / 2);
    if (threads < 1) threads = 1;

    rope_kernel<<<grid, threads, 0, stream>>>(
        q, k, seq_len, num_q_heads, num_kv_heads, head_dim, start_pos, rope_theta
    );
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace engine

