#include "src/cuda/kernels.h"
#include "src/common/cuda_utils.h"
#include <cmath>

namespace engine {
namespace cuda {

__global__ void swiglu_kernel(
    const half* __restrict__ gate,
    const half* __restrict__ up,
    half* __restrict__ out,
    int64_t count
) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    float g = __half2float(gate[idx]);
    float u = __half2float(up[idx]);

    // silu(g) = g / (1.0 + exp(-g))
    float silu_g = g / (1.0f + expf(-g));
    float res = silu_g * u;

    out[idx] = __float2half(res);
}

void swiglu(
    const half* gate,
    const half* up,
    half* out,
    int64_t count,
    cudaStream_t stream
) {
    if (count <= 0) return;
    int threads = 256;
    int blocks = static_cast<int>((count + threads - 1) / threads);
    swiglu_kernel<<<blocks, threads, 0, stream>>>(gate, up, out, count);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace engine

