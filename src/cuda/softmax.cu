#include "src/cuda/kernels.h"
#include "src/common/cuda_utils.h"
#include <cmath>
#include <cfloat>

namespace engine {
namespace cuda {

__inline__ __device__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__inline__ __device__ float warp_reduce_sum_f(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void softmax_kernel(
    const half* __restrict__ x,
    half* __restrict__ out,
    int64_t rows,
    int64_t cols
) {
    int64_t row = blockIdx.x;
    if (row >= rows) return;

    const half* row_x = x + row * cols;
    half* row_out = out + row * cols;

    // 1. Find max in FP32
    float thread_max = -1e20f;
    for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
        thread_max = fmaxf(thread_max, __half2float(row_x[i]));
    }
    thread_max = warp_reduce_max(thread_max);

    __shared__ float s_val[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    if (lane == 0) s_val[wid] = thread_max;
    __syncthreads();

    float row_max = -1e20f;
    if (wid == 0) {
        float val = (lane < (blockDim.x + 31) / 32) ? s_val[lane] : -1e20f;
        row_max = warp_reduce_max(val);
        if (lane == 0) s_val[0] = row_max;
    }
    __syncthreads();
    row_max = s_val[0];

    // 2. Sum exp(x - max) in FP32
    float thread_sum = 0.0f;
    for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
        thread_sum += expf(__half2float(row_x[i]) - row_max);
    }
    thread_sum = warp_reduce_sum_f(thread_sum);
    if (lane == 0) s_val[wid] = thread_sum;
    __syncthreads();

    float row_sum = 0.0f;
    if (wid == 0) {
        float val = (lane < (blockDim.x + 31) / 32) ? s_val[lane] : 0.0f;
        row_sum = warp_reduce_sum_f(val);
        if (lane == 0) s_val[0] = row_sum;
    }
    __syncthreads();
    row_sum = s_val[0];

    // 3. Normalize
    float inv_sum = 1.0f / row_sum;
    for (int64_t i = threadIdx.x; i < cols; i += blockDim.x) {
        float e = expf(__half2float(row_x[i]) - row_max);
        row_out[i] = __float2half(e * inv_sum);
    }
}

void softmax(
    const half* x,
    half* out,
    int64_t rows,
    int64_t cols,
    cudaStream_t stream
) {
    if (rows <= 0 || cols <= 0) return;
    int threads = 256;
    if (cols < 256) threads = static_cast<int>(cols);
    softmax_kernel<<<static_cast<unsigned int>(rows), threads, 0, stream>>>(
        x, out, rows, cols
    );
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace engine

