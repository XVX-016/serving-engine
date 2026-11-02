#include "src/cuda/kernels.h"
#include "src/common/cuda_utils.h"

namespace engine {
namespace cuda {

__global__ void bias_add_kernel(
    const half* __restrict__ x,
    const half* __restrict__ bias,
    half* __restrict__ out,
    int64_t M,
    int64_t N
) {
    int64_t row = blockIdx.y;
    int64_t col = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    int64_t idx = row * N + col;
    float val_x = __half2float(x[idx]);
    float val_b = __half2float(bias[col]);
    out[idx] = __float2half(val_x + val_b);
}

void bias_add(
    const half* x,
    const half* bias,
    half* out,
    int64_t M,
    int64_t N,
    cudaStream_t stream
) {
    if (M <= 0 || N <= 0) return;
    int threads = 256;
    int blocks_x = static_cast<int>((N + threads - 1) / threads);
    dim3 grid(blocks_x, static_cast<unsigned int>(M));
    bias_add_kernel<<<grid, threads, 0, stream>>>(x, bias, out, M, N);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace engine

