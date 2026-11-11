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
// 0. Activation Quantization Kernel (Task P8c)
// Quantizes FP16 activation vector x [K] to INT8 symmetric with per-vector scale:
// scale = max(|x|) / 127.0f, x_q = round(x / scale)
// Single block with 256 threads reduces max(|x|) via warp-shuffles and shared mem.
// ============================================================================
__global__ void quantize_activation_int8_kernel(
    const half* __restrict__ x,
    int8_t* __restrict__ x_q,
    float* __restrict__ scale,
    int K
) {
    __shared__ float s_amax[32];
    int tid = threadIdx.x;
    int lane = tid % 32;
    int wid = tid / 32;

    // Step 1: Thread-local max(|x|)
    float local_max = 0.0f;
    for (int i = tid; i < K; i += blockDim.x) {
        float val = fabsf(__half2float(x[i]));
        local_max = fmaxf(local_max, val);
    }

    // Step 2: Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
    }

    if (lane == 0) {
        s_amax[wid] = local_max;
    }
    __syncthreads();

    // Step 3: Block reduction by first warp
    if (wid == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        float warp_val = (lane < num_warps) ? s_amax[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            warp_val = fmaxf(warp_val, __shfl_down_sync(0xffffffff, warp_val, offset));
        }
        if (lane == 0) {
            float s = (warp_val > 1e-8f) ? (warp_val / 127.0f) : 1.0f;
            *scale = s;
            s_amax[0] = s;
        }
    }
    __syncthreads();

    float s = s_amax[0];
    float inv_s = 1.0f / s;

    // Step 4: Quantize and write out
    for (int i = tid; i < K; i += blockDim.x) {
        float val = __half2float(x[i]);
        int q = __float2int_rn(val * inv_s);
        q = max(-127, min(127, q));
        x_q[i] = static_cast<int8_t>(q);
    }
}

void quantize_activation_int8(
    const half* x,
    int8_t* x_q,
    float* scale,
    int K,
    cudaStream_t stream
) {
    if (K <= 0) return;
    int threads = 256;
    quantize_activation_int8_kernel<<<1, threads, 0, stream>>>(x, x_q, scale, K);
    CUDA_CHECK(cudaGetLastError());
}

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
// 1c. Fast INT8 GEMV using __dp4a SIMD Dot Product (Task P8c)
// W_q: [N, K] int8_t
// scales: [N] half (one scale per output neuron n)
// x_q: [K] int8_t (quantized activation vector)
// x_scale: [1] float (activation scale)
// Vectorized 16-byte loads: 16 int8 weights per thread -> 4x __dp4a instructions.
// Warp reduction and single scale multiply: dot = sumi * (w_scale * x_scale)
// ============================================================================
template <typename Epi>
__global__ void gemv_int8_dp4a_kernel(
    const int8_t* __restrict__ W_q,
    const half* __restrict__ scales,
    const int8_t* __restrict__ x_q,
    const float* __restrict__ x_scale_ptr,
    const half* __restrict__ extra,
    half* __restrict__ y,
    int N,
    int K
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane    = threadIdx.x % 32;

    if (warp_id >= N) return;

    const int8_t* row = W_q + static_cast<size_t>(warp_id) * K;
    float w_scale = __half2float(scales[warp_id]);
    float x_scale = *x_scale_ptr;

    int sumi = 0;

    // Vectorized 16-byte loads: 16 elements per thread = 512 elements per warp iteration
    int k_tile = 0;
    for (; k_tile + 32 * 16 <= K; k_tile += 32 * 16) {
        int k = k_tile + lane * 16;
        int4 w_raw = *reinterpret_cast<const int4*>(row + k);
        int4 x_raw = *reinterpret_cast<const int4*>(x_q + k);

        sumi = __dp4a(w_raw.x, x_raw.x, sumi);
        sumi = __dp4a(w_raw.y, x_raw.y, sumi);
        sumi = __dp4a(w_raw.z, x_raw.z, sumi);
        sumi = __dp4a(w_raw.w, x_raw.w, sumi);
    }

    // Remainder loop
    for (int r = k_tile + lane; r < K; r += 32) {
        sumi += static_cast<int>(row[r]) * static_cast<int>(x_q[r]);
    }

    // Warp-level reduction of integer sum
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sumi += __shfl_down_sync(0xffffffff, sumi, offset);
    }

    if (lane == 0) {
        float acc = static_cast<float>(sumi) * (w_scale * x_scale);
        if constexpr (!std::is_same_v<Epi, EpiNone>) {
            acc += __half2float(extra[warp_id]);
        }
        y[warp_id] = __float2half(acc);
    }
}

__global__ void gemv_int8_dp4a_f32_kernel(
    const int8_t* __restrict__ W_q,
    const half* __restrict__ scales,
    const int8_t* __restrict__ x_q,
    const float* __restrict__ x_scale_ptr,
    float* __restrict__ y_f32,
    int N,
    int K
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane    = threadIdx.x % 32;

    if (warp_id >= N) return;

    const int8_t* row = W_q + static_cast<size_t>(warp_id) * K;
    float w_scale = __half2float(scales[warp_id]);
    float x_scale = *x_scale_ptr;

    int sumi = 0;

    int k_tile = 0;
    for (; k_tile + 32 * 16 <= K; k_tile += 32 * 16) {
        int k = k_tile + lane * 16;
        int4 w_raw = *reinterpret_cast<const int4*>(row + k);
        int4 x_raw = *reinterpret_cast<const int4*>(x_q + k);

        sumi = __dp4a(w_raw.x, x_raw.x, sumi);
        sumi = __dp4a(w_raw.y, x_raw.y, sumi);
        sumi = __dp4a(w_raw.z, x_raw.z, sumi);
        sumi = __dp4a(w_raw.w, x_raw.w, sumi);
    }

    for (int r = k_tile + lane; r < K; r += 32) {
        sumi += static_cast<int>(row[r]) * static_cast<int>(x_q[r]);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sumi += __shfl_down_sync(0xffffffff, sumi, offset);
    }

    if (lane == 0) {
        y_f32[warp_id] = static_cast<float>(sumi) * (w_scale * x_scale);
    }
}

// ============================================================================
// 2b. Fast INT4 Grouped GEMV using __dp4a with Masked Nibbles (Task P8c)
// W_q: [N, K/2] uint8_t (packed two nibbles per byte)
// scales: [N, K / GROUP_SIZE] half
// x_q: [K] int8_t
// Vectorized 16-byte load = 32 packed nibbles = 32 INT4 elements.
// Unpacks via bitmask: low = w & 0x0F0F0F0F, high = (w >> 4) & 0x0F0F0F0F.
// Signed two's complement recovery: subtract 8 if nibble >= 8.
// With __dp4a:
//   sum_lo = __dp4a(vi0, x_even, sum_lo) - 16 * sum(x_even if negative) ...
// Notice: for unsigned nibbles u in [0, 15], signed weight q = u - 16*(u >= 8) or symmetric q = u - 8.
// Since our quant format is symmetric round-to-nearest in [-8, 7]:
//   q = (u >= 8) ? (u - 16) : u.
// In bit representation: sign bit is at bit 3 (mask 0x08).
// When we sign-extend 4-bit to 8-bit:
//   b0_8 = (u0 & 0x08) ? (u0 | 0xF0) : u0.
// We can compute the signed byte packed directly via bitwise operations:
//   mask_sign = (u & 0x08080808) >> 3; // 1 where negative
//   signed_bytes = u | (mask_sign * 0xF0); // sign extend to 8-bit!
// Then signed_bytes is a standard int8x4, and __dp4a(signed_bytes, x_q, sum) is EXACT!
// This requires ONLY 2 bitwise instructions to unpack 4 signed INT4s into an int32!
// ============================================================================
template <typename Epi, int GROUP_SIZE>
__global__ void gemv_int4_dp4a_kernel(
    const uint8_t* __restrict__ W_q,
    const half* __restrict__ scales,
    const int8_t* __restrict__ x_q,
    const float* __restrict__ x_scale_ptr,
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
    float x_scale = *x_scale_ptr;

    float acc = 0.0f;

    // Vectorized load: 16 bytes = 32 packed nibbles per thread
    // 32 threads * 32 elements = 1024 elements per warp iteration
    int k_tile = 0;
    for (; k_tile + 32 * 32 <= K; k_tile += 32 * 32) {
        int k = k_tile + lane * 32;

        // Load 16 packed bytes = 4 uint32 words
        uint4 raw = *reinterpret_cast<const uint4*>(row_bytes + k / 2);
        const uint32_t* w_words = reinterpret_cast<const uint32_t*>(&raw);

        // Load 32 activation bytes = 8 uint32 words
        const int4* x_ptr = reinterpret_cast<const int4*>(x_q + k);
        int4 x_vec0 = x_ptr[0]; // elements 0..15
        int4 x_vec1 = x_ptr[1]; // elements 16..31

        const int32_t* x_words = reinterpret_cast<const int32_t*>(&x_vec0);

        // Process 4 words (each has 4 bytes = 8 nibbles)
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            uint32_t packed = w_words[w];
            int elem_idx = k + w * 8;
            int g_idx = elem_idx / GROUP_SIZE;
            float group_scale = __half2float(row_scales[g_idx]);

            // Unpack low nibbles of 4 bytes
            uint32_t lo = packed & 0x0F0F0F0Fu;
            // Sign-extend 4-bit to 8-bit: if bit 3 is set, set upper nibble to 0xF
            uint32_t sign_lo = (lo & 0x08080808u) >> 3;
            uint32_t q_lo = lo | (sign_lo * 0xF0u);

            // Unpack high nibbles of 4 bytes
            uint32_t hi = (packed >> 4) & 0x0F0F0F0Fu;
            uint32_t sign_hi = (hi & 0x08080808u) >> 3;
            uint32_t q_hi = hi | (sign_hi * 0xF0u);

            // Note: in our packing, byte b has low nibble at elem 0, high nibble at elem 1.
            // But x_q elements are contiguous: [x0, x1, x2, x3, x4, x5, x6, x7].
            // To do SIMD dot product with __dp4a:
            // lo has [q0, q2, q4, q6], and hi has [q1, q3, q5, q7].
            // We can interleave or simply permute x, or load x as bytes!
            // Let's unpack the two uint32 x words into even and odd bytes using __byte_perm:
            int32_t x0 = (w < 2) ? reinterpret_cast<const int32_t*>(&x_vec0)[w*2] : reinterpret_cast<const int32_t*>(&x_vec1)[(w-2)*2];
            int32_t x1 = (w < 2) ? reinterpret_cast<const int32_t*>(&x_vec0)[w*2+1] : reinterpret_cast<const int32_t*>(&x_vec1)[(w-2)*2+1];

            // x0 has bytes [x0, x1, x2, x3]
            // x1 has bytes [x4, x5, x6, x7]
            // We need x_even = [x0, x2, x4, x6] and x_odd = [x1, x3, x5, x7]
            int32_t x_even = __byte_perm(x0, x1, 0x6420);
            int32_t x_odd  = __byte_perm(x0, x1, 0x7531);

            int group_sum = 0;
            group_sum = __dp4a(static_cast<int32_t>(q_lo), x_even, group_sum);
            group_sum = __dp4a(static_cast<int32_t>(q_hi), x_odd,  group_sum);

            acc += static_cast<float>(group_sum) * (group_scale * x_scale);
        }
    }

    // Remainder loop
    for (int r = k_tile + lane * 2; r + 1 < K; r += 32 * 2) {
        uint8_t b = row_bytes[r / 2];
        int g_idx0 = r / GROUP_SIZE;
        float scale0 = __half2float(row_scales[g_idx0]);
        int v0 = b & 0x0F;
        int q0 = (v0 >= 8) ? (v0 - 16) : v0;
        acc += (static_cast<float>(q0) * (scale0 * x_scale)) * static_cast<float>(x_q[r]);

        int g_idx1 = (r + 1) / GROUP_SIZE;
        float scale1 = __half2float(row_scales[g_idx1]);
        int v1 = (b >> 4) & 0x0F;
        int q1 = (v1 >= 8) ? (v1 - 16) : v1;
        acc += (static_cast<float>(q1) * (scale1 * x_scale)) * static_cast<float>(x_q[r + 1]);
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

// SIMD dp4a Dispatch functions
void gemv_int8_dp4a(
    const int8_t* W_q,
    const half* scales,
    const int8_t* x_q,
    const float* x_scale,
    half* y,
    int N,
    int K,
    cudaStream_t stream
) {
    if (N <= 0 || K <= 0) return;
    int blocks = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    gemv_int8_dp4a_kernel<EpiNone><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(W_q, scales, x_q, x_scale, nullptr, y, N, K);
    CUDA_CHECK(cudaGetLastError());
}

void gemv_int8_dp4a_bias(
    const int8_t* W_q,
    const half* scales,
    const int8_t* x_q,
    const float* x_scale,
    const half* b,
    half* y,
    int N,
    int K,
    cudaStream_t stream
) {
    if (N <= 0 || K <= 0) return;
    int blocks = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    gemv_int8_dp4a_kernel<EpiBias><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(W_q, scales, x_q, x_scale, b, y, N, K);
    CUDA_CHECK(cudaGetLastError());
}

void gemv_int8_dp4a_residual(
    const int8_t* W_q,
    const half* scales,
    const int8_t* x_q,
    const float* x_scale,
    const half* r,
    half* y,
    int N,
    int K,
    cudaStream_t stream
) {
    if (N <= 0 || K <= 0) return;
    int blocks = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    gemv_int8_dp4a_kernel<EpiResidual><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(W_q, scales, x_q, x_scale, r, y, N, K);
    CUDA_CHECK(cudaGetLastError());
}

void gemv_int8_dp4a_f32(
    const int8_t* W_q,
    const half* scales,
    const int8_t* x_q,
    const float* x_scale,
    float* y_f32,
    int N,
    int K,
    cudaStream_t stream
) {
    if (N <= 0 || K <= 0) return;
    int blocks = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    gemv_int8_dp4a_f32_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(W_q, scales, x_q, x_scale, y_f32, N, K);
    CUDA_CHECK(cudaGetLastError());
}

void gemv_int4_g128_dp4a(
    const uint8_t* W_q,
    const half* scales,
    const int8_t* x_q,
    const float* x_scale,
    half* y,
    int N,
    int K,
    cudaStream_t stream
) {
    if (N <= 0 || K <= 0) return;
    int blocks = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    gemv_int4_dp4a_kernel<EpiNone, 128><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(W_q, scales, x_q, x_scale, nullptr, y, N, K);
    CUDA_CHECK(cudaGetLastError());
}

void gemv_int4_g128_dp4a_bias(
    const uint8_t* W_q,
    const half* scales,
    const int8_t* x_q,
    const float* x_scale,
    const half* b,
    half* y,
    int N,
    int K,
    cudaStream_t stream
) {
    if (N <= 0 || K <= 0) return;
    int blocks = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    gemv_int4_dp4a_kernel<EpiBias, 128><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(W_q, scales, x_q, x_scale, b, y, N, K);
    CUDA_CHECK(cudaGetLastError());
}

void gemv_int4_g128_dp4a_residual(
    const uint8_t* W_q,
    const half* scales,
    const int8_t* x_q,
    const float* x_scale,
    const half* r,
    half* y,
    int N,
    int K,
    cudaStream_t stream
) {
    if (N <= 0 || K <= 0) return;
    int blocks = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    gemv_int4_dp4a_kernel<EpiResidual, 128><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(W_q, scales, x_q, x_scale, r, y, N, K);
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

