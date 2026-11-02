#include "src/cuda/kernels.h"
#include "src/common/cuda_utils.h"
#include <cmath>

namespace engine {
namespace cuda {

__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void rmsnorm_kernel(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    half* __restrict__ out,
    int64_t rows,
    int64_t dim,
    float eps
) {
    int64_t row = blockIdx.x;
    if (row >= rows) return;

    const half* row_x = x + row * dim;
    half* row_out = out + row * dim;

    // 1. Compute sum of squares in FP32
    float thread_sum = 0.0f;
    for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = __half2float(row_x[i]);
        thread_sum += val * val;
    }

    // Warp reduction
    thread_sum = warp_reduce_sum(thread_sum);

    // Block reduction across warps
    __shared__ float shared_sum[32]; // Max 1024 threads = 32 warps
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    if (lane == 0) {
        shared_sum[wid] = thread_sum;
    }
    __syncthreads();

    // First warp reduces the shared sums
    float total_sum = 0.0f;
    if (wid == 0) {
        float val = (lane < (blockDim.x + 31) / 32) ? shared_sum[lane] : 0.0f;
        total_sum = warp_reduce_sum(val);
        if (lane == 0) {
            shared_sum[0] = total_sum;
        }
    }
    __syncthreads();

    total_sum = shared_sum[0];
    float rsqrt_val = rsqrtf(total_sum / static_cast<float>(dim) + eps);

    // 2. Normalize and scale by weight
    for (int64_t i = threadIdx.x; i < dim; i += blockDim.x) {
        float val = __half2float(row_x[i]);
        float w = __half2float(weight[i]);
        row_out[i] = __float2half(val * rsqrt_val * w);
    }
}

void rmsnorm(
    const half* x,
    const half* weight,
    half* out,
    int64_t rows,
    int64_t dim,
    float eps,
    cudaStream_t stream
) {
    if (rows <= 0) return;
    int threads = 256;
    if (dim < 256) threads = static_cast<int>(dim);
    rmsnorm_kernel<<<static_cast<unsigned int>(rows), threads, 0, stream>>>(
        x, weight, out, rows, dim, eps
    );
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace engine

