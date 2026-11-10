// gemv_quant.cu  --  Custom INT8 and INT4 GEMV kernels with fused dequant + epilogue
//
// Designed for sm_89 (Ada Lovelace):
// - Vectorized 128-bit loads (int4 / uint4) for maximum global memory throughput.
// - In-register dequantization.
// - Fused epilogues: None, Bias, Residual.
//
// Strategy: One warp (32 threads) per output neuron n.
// Thread lane computes dot product over stride 32.

#include "src/cuda/kernels.h"
#include "src/common/cuda_utils.h"
#include <cuda_fp16.h>
#include <cstdint>

namespace engine {
namespace cuda {

// Epilogue tags
struct EpiNone     {};
struct EpiBias     {};
struct EpiResidual {};

// ============================================================================
// 1. INT8 Symmetric GEMV (Per-channel scale)
// W_q: [N, K] int8_t
// scales: [N] half (one scale per output neuron n)
// x: [K] half
// ============================================================================
template <typename Epi>
__global__ void gemv_int8_kernel(
    const int8_t* __restrict__ W_q,
    const half* __restrict__ scales,
    const half* __restrict__ x,
    const half* __restrict__ extra, // bias or residual
    half* __restrict__ y,
    int N,
    int K
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane    = threadIdx.x % 32;

    if (warp_id >= N) return;

    const int8_t* row = W_q + static_cast<size_t>(warp_id) * K;
    float scale = __half2float(scales[warp_id]);

    float acc = 0.0f;

    // Vectorized 16-byte loads: 16 int8 elements per thread
    // 32 threads * 16 elements = 512 elements per iteration
    int k_tile = 0;
    for (; k_tile + 32 * 16 <= K; k_tile += 32 * 16) {
        int k = k_tile + lane * 16;
        // Load 16 bytes = 16 int8
        int4 raw = *reinterpret_cast<const int4*>(row + k);
        const int8_t* q = reinterpret_cast<const int8_t*>(&raw);

        // Load 16 halfs from x in two 128-bit transactions
        float4 x_vec0 = *reinterpret_cast<const float4*>(x + k);
        float4 x_vec1 = *reinterpret_cast<const float4*>(x + k + 8);
        const half* x_h0 = reinterpret_cast<const half*>(&x_vec0);
        const half* x_h1 = reinterpret_cast<const half*>(&x_vec1);

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc += (static_cast<float>(q[i]) * scale) * __half2float(x_h0[i]);
        }
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc += (static_cast<float>(q[i + 8]) * scale) * __half2float(x_h1[i]);
        }
    }

    // Remainder loop
    for (int r = k_tile + lane; r < K; r += 32) {
        acc += (static_cast<float>(row[r]) * scale) * __half2float(x[r]);
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    // Lane 0 writes output
    if (lane == 0) {
        if constexpr (!std::is_same_v<Epi, EpiNone>) {
            acc += __half2float(extra[warp_id]);
        }
        y[warp_id] = __float2half(acc);
    }
}

// ============================================================================
// 1b. INT8 Symmetric GEMV with FP32 output (for LM Head logits)
// Same structure as gemv_int8_kernel but writes float* output directly.
// ============================================================================
__global__ void gemv_int8_f32_kernel(
    const int8_t* __restrict__ W_q,
    const half* __restrict__ scales,
    const half* __restrict__ x,
    float* __restrict__ y_f32,
    int N,
    int K
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane    = threadIdx.x % 32;

    if (warp_id >= N) return;

    const int8_t* row = W_q + static_cast<size_t>(warp_id) * K;
    float scale = __half2float(scales[warp_id]);

    float acc = 0.0f;

    // Vectorized 16-byte loads: 16 int8 elements per thread
    int k_tile = 0;
    for (; k_tile + 32 * 16 <= K; k_tile += 32 * 16) {
        int k = k_tile + lane * 16;
        int4 raw = *reinterpret_cast<const int4*>(row + k);
        const int8_t* q = reinterpret_cast<const int8_t*>(&raw);

        float4 x_vec0 = *reinterpret_cast<const float4*>(x + k);
        float4 x_vec1 = *reinterpret_cast<const float4*>(x + k + 8);
        const half* x_h0 = reinterpret_cast<const half*>(&x_vec0);
        const half* x_h1 = reinterpret_cast<const half*>(&x_vec1);

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc += (static_cast<float>(q[i]) * scale) * __half2float(x_h0[i]);
        }
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc += (static_cast<float>(q[i + 8]) * scale) * __half2float(x_h1[i]);
        }
    }

    // Remainder loop
    for (int r = k_tile + lane; r < K; r += 32) {
        acc += (static_cast<float>(row[r]) * scale) * __half2float(x[r]);
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    // Lane 0 writes FP32 output directly
    if (lane == 0) {
        y_f32[warp_id] = acc;
    }
}

// ============================================================================
// 2. INT4 Grouped GEMV (Group Size 128 or 64, Symmetric)
// W_q: [N, K/2] uint8_t (packed two nibbles per byte)
// scales: [N, K / group_size] half
// x: [K] half
//
// Uses the "magic number" FP16 bit trick for fast nibble-to-half conversion:
//   XOR nibble with 8 to convert two's complement to unsigned offset, then
//   embed into FP16 mantissa with exponent 2^10 = 1024: val = 1024 + (nibble^8).
//   Subtract 1032 (= 1024 + 8) to recover the signed integer value.
//   This avoids integer shift/branch per element.
// ============================================================================
template <typename Epi, int GROUP_SIZE>
__global__ void gemv_int4_kernel(
    const uint8_t* __restrict__ W_q,
    const half* __restrict__ scales,
    const half* __restrict__ x,
    const half* __restrict__ extra,
    half* __restrict__ y,
    int N,
    int K
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane    = threadIdx.x % 32;

    if (warp_id >= N) return;

    int num_groups = K / GROUP_SIZE;
    const uint8_t* row_bytes = W_q + static_cast<size_t>(warp_id) * (K / 2);
    const half* row_scales = scales + static_cast<size_t>(warp_id) * num_groups;

    float acc = 0.0f;

    // Vectorized load: 16 bytes = 32 nibbles (32 INT4 elements) per thread
    // 32 threads * 32 elements = 1024 elements per iteration
    int k_tile = 0;
    for (; k_tile + 32 * 32 <= K; k_tile += 32 * 32) {
        int k = k_tile + lane * 32;
        // Load 16 packed bytes = 32 elements
        uint4 raw = *reinterpret_cast<const uint4*>(row_bytes + k / 2);
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&raw);

        // Load 32 halfs from x in four 128-bit float4 loads
        float4 x_vec0 = *reinterpret_cast<const float4*>(x + k);
        float4 x_vec1 = *reinterpret_cast<const float4*>(x + k + 8);
        float4 x_vec2 = *reinterpret_cast<const float4*>(x + k + 16);
        float4 x_vec3 = *reinterpret_cast<const float4*>(x + k + 24);

        const half* x_h[4] = {
            reinterpret_cast<const half*>(&x_vec0),
            reinterpret_cast<const half*>(&x_vec1),
            reinterpret_cast<const half*>(&x_vec2),
            reinterpret_cast<const half*>(&x_vec3)
        };

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            uint8_t b = bytes[i];
            int elem_idx = k + i * 2;
            int g_idx = elem_idx / GROUP_SIZE;
            float scale = __half2float(row_scales[g_idx]);

            // Magic number dequant: XOR with 8 converts two's complement to offset
            // Then embed in FP16 mantissa at exponent 2^10, subtract 1032
            uint16_t lo16 = 0x6400u | static_cast<uint16_t>((b & 0x0Fu) ^ 0x08u);
            uint16_t hi16 = 0x6400u | static_cast<uint16_t>(((b >> 4) & 0x0Fu) ^ 0x08u);
            float dq0 = (__half2float(*reinterpret_cast<half*>(&lo16)) - 1032.0f) * scale;
            float dq1 = (__half2float(*reinterpret_cast<half*>(&hi16)) - 1032.0f) * scale;

            int h_chunk0 = (i * 2) / 8;
            int h_sub0   = (i * 2) % 8;
            acc += dq0 * __half2float(x_h[h_chunk0][h_sub0]);

            int h_chunk1 = (i * 2 + 1) / 8;
            int h_sub1   = (i * 2 + 1) % 8;
            acc += dq1 * __half2float(x_h[h_chunk1][h_sub1]);
        }
    }

    // Remainder loop (scalar fallback)
    for (int r = k_tile + lane * 2; r + 1 < K; r += 32 * 2) {
        uint8_t b = row_bytes[r / 2];
        int g_idx0 = r / GROUP_SIZE;
        float scale0 = __half2float(row_scales[g_idx0]);
        int v0 = b & 0x0F;
        int q0 = (v0 >= 8) ? (v0 - 16) : v0;
        acc += (static_cast<float>(q0) * scale0) * __half2float(x[r]);

        int g_idx1 = (r + 1) / GROUP_SIZE;
        float scale1 = __half2float(row_scales[g_idx1]);
        int v1 = (b >> 4) & 0x0F;
        int q1 = (v1 >= 8) ? (v1 - 16) : v1;
        acc += (static_cast<float>(q1) * scale1) * __half2float(x[r + 1]);
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    if (lane == 0) {
        if constexpr (!std::is_same_v<Epi, EpiNone>) {
            acc += __half2float(extra[warp_id]);
        }
        y[warp_id] = __float2half(acc);
    }
}

// ============================================================================
// 3. Tile / Buffer Dequantization for Prefill Path
// Dequantizes packed weights into an FP16 buffer prior to WMMA / cuBLAS GEMM.
// ============================================================================
__global__ void dequantize_int8_to_fp16_kernel(
    const int8_t* __restrict__ W_q,
    const half* __restrict__ scales,
    half* __restrict__ W_fp16,
    int N,
    int K
) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total = static_cast<size_t>(N) * K;
    if (idx >= total) return;

    int row = static_cast<int>(idx / K);
    int col = static_cast<int>(idx % K);

    float s = __half2float(scales[row]);
    int8_t q = W_q[idx];
    W_fp16[idx] = __float2half(static_cast<float>(q) * s);
}

template <int GROUP_SIZE>
__global__ void dequantize_int4_to_fp16_kernel(
    const uint8_t* __restrict__ W_q,
    const half* __restrict__ scales,
    half* __restrict__ W_fp16,
    int N,
    int K
) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t total_bytes = static_cast<size_t>(N) * (K / 2);
    if (idx >= total_bytes) return;

    int row = static_cast<int>(idx / (K / 2));
    int col_byte = static_cast<int>(idx % (K / 2));

    int k0 = col_byte * 2;
    int k1 = col_byte * 2 + 1;
    uint8_t b = W_q[idx];

    int g0 = k0 / GROUP_SIZE;
    int g1 = k1 / GROUP_SIZE;
    float s0 = __half2float(scales[static_cast<size_t>(row) * (K / GROUP_SIZE) + g0]);
    float s1 = __half2float(scales[static_cast<size_t>(row) * (K / GROUP_SIZE) + g1]);

    int v0 = b & 0x0F;
    int q0 = (v0 >= 8) ? (v0 - 16) : v0;
    int v1 = (b >> 4) & 0x0F;
    int q1 = (v1 >= 8) ? (v1 - 16) : v1;

    W_fp16[static_cast<size_t>(row) * K + k0] = __float2half(static_cast<float>(q0) * s0);
    W_fp16[static_cast<size_t>(row) * K + k1] = __float2half(static_cast<float>(q1) * s1);
}

// ============================================================================
// Public Dispatch APIs
// ============================================================================
static constexpr int WARPS_PER_BLOCK = 32;
static constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;

void gemv_int8(
    const int8_t* W_q,
    const half* scales,
    const half* x,
    half* y,
    int N,
    int K,
    cudaStream_t stream
) {
    if (N <= 0 || K <= 0) return;
    int blocks = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    gemv_int8_kernel<EpiNone><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(W_q, scales, x, nullptr, y, N, K);
    CUDA_CHECK(cudaGetLastError());
}

void gemv_int8_bias(
    const int8_t* W_q,
    const half* scales,
    const half* x,
    const half* b,
    half* y,
    int N,
    int K,
    cudaStream_t stream
) {
    if (N <= 0 || K <= 0) return;
    int blocks = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    gemv_int8_kernel<EpiBias><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(W_q, scales, x, b, y, N, K);
    CUDA_CHECK(cudaGetLastError());
}

void gemv_int8_residual(
    const int8_t* W_q,
    const half* scales,
    const half* x,
    const half* r,
    half* y,
    int N,
    int K,
    cudaStream_t stream
) {
    if (N <= 0 || K <= 0) return;
    int blocks = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    gemv_int8_kernel<EpiResidual><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(W_q, scales, x, r, y, N, K);
    CUDA_CHECK(cudaGetLastError());
}

void gemv_int8_f32(
    const int8_t* W_q,
    const half* scales,
    const half* x,
    float* y_f32,
    int N,
    int K,
    cudaStream_t stream
) {
    if (N <= 0 || K <= 0) return;
    int blocks = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    gemv_int8_f32_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(W_q, scales, x, y_f32, N, K);
    CUDA_CHECK(cudaGetLastError());
}

void gemv_int4_g128(
    const uint8_t* W_q,
    const half* scales,
    const half* x,
    half* y,
    int N,
    int K,
    cudaStream_t stream
) {
    if (N <= 0 || K <= 0) return;
    int blocks = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    gemv_int4_kernel<EpiNone, 128><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(W_q, scales, x, nullptr, y, N, K);
    CUDA_CHECK(cudaGetLastError());
}

void gemv_int4_g128_bias(
    const uint8_t* W_q,
    const half* scales,
    const half* x,
    const half* b,
    half* y,
    int N,
    int K,
    cudaStream_t stream
) {
    if (N <= 0 || K <= 0) return;
    int blocks = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    gemv_int4_kernel<EpiBias, 128><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(W_q, scales, x, b, y, N, K);
    CUDA_CHECK(cudaGetLastError());
}

void gemv_int4_g128_residual(
    const uint8_t* W_q,
    const half* scales,
    const half* x,
    const half* r,
    half* y,
    int N,
    int K,
    cudaStream_t stream
) {
    if (N <= 0 || K <= 0) return;
    int blocks = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    gemv_int4_kernel<EpiResidual, 128><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(W_q, scales, x, r, y, N, K);
    CUDA_CHECK(cudaGetLastError());
}

void dequantize_int8(
    const int8_t* W_q,
    const half* scales,
    half* W_fp16,
    int N,
    int K,
    cudaStream_t stream
) {
    if (N <= 0 || K <= 0) return;
    size_t total = static_cast<size_t>(N) * K;
    int block = 256;
    int grid = static_cast<int>((total + block - 1) / block);
    dequantize_int8_to_fp16_kernel<<<grid, block, 0, stream>>>(W_q, scales, W_fp16, N, K);
    CUDA_CHECK(cudaGetLastError());
}

void dequantize_int4_g128(
    const uint8_t* W_q,
    const half* scales,
    half* W_fp16,
    int N,
    int K,
    cudaStream_t stream
) {
    if (N <= 0 || K <= 0) return;
    size_t total_bytes = static_cast<size_t>(N) * (K / 2);
    int block = 256;
    int grid = static_cast<int>((total_bytes + block - 1) / block);
    dequantize_int4_to_fp16_kernel<128><<<grid, block, 0, stream>>>(W_q, scales, W_fp16, N, K);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace engine

