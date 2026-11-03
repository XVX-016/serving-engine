// gemv.cu  --  Custom FP16 GEMV for decode (M=1), one warp per output neuron.
//
// Strategy: each output neuron n is computed by one warp (32 threads).
//   - Thread t computes partial dot product over k = t, t+32, t+64, ...
//   - Warp-reduces with __shfl_down_sync.
//   - Thread 0 applies epilogue (bias or residual) and writes FP16 result.
//
// Grid: (ceil(N/WARPS_PER_BLOCK), 1)
// Block: (WARPS_PER_BLOCK * 32, 1)  WARPS_PER_BLOCK = 32 → 1024 threads

#include "src/cuda/kernels.h"
#include "src/common/cuda_utils.h"
#include <cuda_fp16.h>

namespace engine {
namespace cuda {

// --------------------------------------------------------------------------
// Epilogue tag types (compile-time dispatch via if constexpr; no runtime branch)
// --------------------------------------------------------------------------
struct EpiNone    {};  // y[n] = dot
struct EpiBias    {};  // y[n] = dot + b[n]
struct EpiResidual{}; // y[n] = dot + r[n]

// --------------------------------------------------------------------------
// Core kernel  (one warp per output neuron)
// --------------------------------------------------------------------------
template <typename Epi>
__global__ void gemv_kernel(
    const half* __restrict__ W,     // [N, K]
    const half* __restrict__ x,     // [K]
    const half* __restrict__ extra, // bias or residual (ignored for EpiNone)
    half* __restrict__ y,           // [N]
    int N,
    int K
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane    = threadIdx.x % 32;

    if (warp_id >= N) return;

    const half* row = W + static_cast<size_t>(warp_id) * K;

    // Compute partial dot in FP32
    float acc = 0.0f;
    for (int k = lane; k < K; k += 32) {
        acc += __half2float(row[k]) * __half2float(x[k]);
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    // Lane 0 writes result
    if (lane == 0) {
        if constexpr (!std::is_same_v<Epi, EpiNone>) {
            acc += __half2float(extra[warp_id]);
        }
        y[warp_id] = __float2half(acc);
    }
}

// --------------------------------------------------------------------------
// Launch helper: 32 warps (1024 threads) per block
// --------------------------------------------------------------------------
static constexpr int WARPS_PER_BLOCK = 32;
static constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;

// --------------------------------------------------------------------------
// Public API
// --------------------------------------------------------------------------
void gemv(
    const half* W,
    const half* x,
    half* y,
    int N,
    int K,
    cudaStream_t stream
) {
    if (N <= 0 || K <= 0) return;
    int blocks = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    gemv_kernel<EpiNone><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(W, x, nullptr, y, N, K);
    CUDA_CHECK(cudaGetLastError());
}

void gemv_bias(
    const half* W,
    const half* x,
    const half* b,
    half* y,
    int N,
    int K,
    cudaStream_t stream
) {
    if (N <= 0 || K <= 0) return;
    int blocks = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    gemv_kernel<EpiBias><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(W, x, b, y, N, K);
    CUDA_CHECK(cudaGetLastError());
}

void gemv_residual(
    const half* W,
    const half* x,
    const half* r,
    half* y,
    int N,
    int K,
    cudaStream_t stream
) {
    if (N <= 0 || K <= 0) return;
    int blocks = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    gemv_kernel<EpiResidual><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(W, x, r, y, N, K);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace engine

