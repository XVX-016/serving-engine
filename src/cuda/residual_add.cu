#include "src/cuda/kernels.h"
#include "src/common/cuda_utils.h"

namespace engine {
namespace cuda {

__global__ void residual_add_kernel(
    const half* __restrict__ a,
    const half* __restrict__ b,
    half* __restrict__ out,
    int64_t count
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    float val_a = __half2float(a[idx]);
    float val_b = __half2float(b[idx]);
    out[idx] = __float2half(val_a + val_b);
}

void residual_add(
    const half* a,
    const half* b,
    half* out,
    int64_t count,
    cudaStream_t stream
) {
    if (count <= 0) return;
    int threads = 256;
    int blocks = static_cast<int>((count + threads - 1) / threads);
    residual_add_kernel<<<blocks, threads, 0, stream>>>(a, b, out, count);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace engine

