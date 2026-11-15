#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

// Original INT4 kernel logic
__global__ void gemv_int4_baseline(
    const uint8_t* __restrict__ W_q,
    const half* __restrict__ scales,
    const half* __restrict__ x,
    half* __restrict__ y,
    int N,
    int K
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane    = threadIdx.x % 32;
    if (warp_id >= N) return;

    int num_groups = K / 128;
    const uint8_t* row_bytes = W_q + static_cast<size_t>(warp_id) * (K / 2);
    const half* row_scales = scales + static_cast<size_t>(warp_id) * num_groups;

    float acc = 0.0f;
    int k_tile = 0;
    for (; k_tile + 32 * 32 <= K; k_tile += 32 * 32) {
        int k = k_tile + lane * 32;
        uint4 raw = *reinterpret_cast<const uint4*>(row_bytes + k / 2);
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&raw);

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
            int g_idx = elem_idx / 128;
            float scale = __half2float(row_scales[g_idx]);

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

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) {
        y[warp_id] = __float2half(acc);
    }
}

// Variant A: Stripped dequant math (Identical memory traffic: loads W_q, scales, x; forces result use)
__global__ void gemv_int4_stripped_dequant(
    const uint8_t* __restrict__ W_q,
    const half* __restrict__ scales,
    const half* __restrict__ x,
    half* __restrict__ y,
    int N,
    int K
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane    = threadIdx.x % 32;
    if (warp_id >= N) return;

    int num_groups = K / 128;
    const uint8_t* row_bytes = W_q + static_cast<size_t>(warp_id) * (K / 2);
    const half* row_scales = scales + static_cast<size_t>(warp_id) * num_groups;

    float acc = 0.0f;
    int k_tile = 0;
    for (; k_tile + 32 * 32 <= K; k_tile += 32 * 32) {
        int k = k_tile + lane * 32;
        uint4 raw = *reinterpret_cast<const uint4*>(row_bytes + k / 2);
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&raw);

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
            int g_idx = elem_idx / 128;
            float scale = __half2float(row_scales[g_idx]);

            float dq0 = static_cast<float>(b & 0x0Fu) * scale;
            float dq1 = static_cast<float>((b >> 4) & 0x0Fu) * scale;

            int h_chunk0 = (i * 2) / 8;
            int h_sub0   = (i * 2) % 8;
            acc += dq0 * __half2float(x_h[h_chunk0][h_sub0]);

            int h_chunk1 = (i * 2 + 1) / 8;
            int h_sub1   = (i * 2 + 1) % 8;
            acc += dq1 * __half2float(x_h[h_chunk1][h_sub1]);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) {
        y[warp_id] = __float2half(acc);
    }
}

// Variant B: Fast Integer SIMD Dequant with 32-bit unpacked math
__global__ void gemv_int4_fast_unpack(
    const uint8_t* __restrict__ W_q,
    const half* __restrict__ scales,
    const half* __restrict__ x,
    half* __restrict__ y,
    int N,
    int K
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane    = threadIdx.x % 32;
    if (warp_id >= N) return;

    int num_groups = K / 128;
    const uint8_t* row_bytes = W_q + static_cast<size_t>(warp_id) * (K / 2);
    const half* row_scales = scales + static_cast<size_t>(warp_id) * num_groups;

    float acc = 0.0f;
    int k_tile = 0;
    for (; k_tile + 32 * 32 <= K; k_tile += 32 * 32) {
        int k = k_tile + lane * 32;
        uint4 raw = *reinterpret_cast<const uint4*>(row_bytes + k / 2);
        const uint32_t* u32_vals = reinterpret_cast<const uint32_t*>(&raw);

        float4 x_vec0 = *reinterpret_cast<const float4*>(x + k);
        float4 x_vec1 = *reinterpret_cast<const float4*>(x + k + 8);
        float4 x_vec2 = *reinterpret_cast<const float4*>(x + k + 16);
        float4 x_vec3 = *reinterpret_cast<const float4*>(x + k + 24);

        const half2* x_h2 = reinterpret_cast<const half2*>(&x_vec0); // 16 half2 pairs

        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            uint32_t val = u32_vals[w];
            #pragma unroll
            for (int b_idx = 0; b_idx < 4; ++b_idx) {
                uint8_t byte_val = (val >> (b_idx * 8)) & 0xFF;
                int elem_idx = k + w * 8 + b_idx * 2;
                int g_idx = elem_idx / 128;
                float scale = __half2float(row_scales[g_idx]);

                int s0 = static_cast<int>(byte_val & 0x0F) - 8;
                int s1 = static_cast<int>((byte_val >> 4) & 0x0F) - 8;

                half2 x_pair = x_h2[w * 4 + b_idx];
                float x0 = __half2float(x_pair.x);
                float x1 = __half2float(x_pair.y);

                acc += (s0 * x0 + s1 * x1) * scale;
            }
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) {
        y[warp_id] = __float2half(acc);
    }
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    size_t l2_size = prop.l2CacheSize;
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "L2 Cache Size: " << l2_size << " bytes (" << (l2_size / 1e6) << " MB)\n";

    // Working set: N=70000, K=4096 -> 147.84 MB (> 4.4x L2 cache)
    int K = 4096;
    int N = 70000;
    size_t wt_bytes = static_cast<size_t>(N) * (K / 2);
    size_t scales_bytes = static_cast<size_t>(N) * (K / 128) * sizeof(half);
    size_t total_traffic_bytes = wt_bytes + scales_bytes;
    std::cout << "Working Set: N=" << N << ", K=" << K << "\n";
    std::cout << "Weight Bytes: " << wt_bytes << " B (" << (wt_bytes / 1e6) << " MB)\n";
    std::cout << "Total Working Set Read: " << total_traffic_bytes << " B (" << (total_traffic_bytes / 1e6) << " MB, " 
              << (total_traffic_bytes / static_cast<double>(l2_size)) << "x L2 Cache)\n\n";

    uint8_t* d_W;
    half* d_scales;
    half* d_x;
    half* d_y;
    CUDA_CHECK(cudaMalloc(&d_W, wt_bytes));
    CUDA_CHECK(cudaMalloc(&d_scales, scales_bytes));
    CUDA_CHECK(cudaMalloc(&d_x, K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(half)));

    CUDA_CHECK(cudaMemset(d_W, 0x55, wt_bytes));
    CUDA_CHECK(cudaMemset(d_scales, 0x3c, scales_bytes));
    CUDA_CHECK(cudaMemset(d_x, 0x3c, K * sizeof(half)));

    // 128 MB L2 cache flush buffer
    const size_t flush_bytes = 128 * 1024 * 1024;
    int* d_flush = nullptr;
    CUDA_CHECK(cudaMalloc(&d_flush, flush_bytes));

    int threads = 1024;
    int warps_per_block = threads / 32;
    int blocks = (N + warps_per_block - 1) / warps_per_block;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto bench_kernel = [&](const char* name, auto kernel_func) {
        // 5 warmups
        for (int w = 0; w < 5; ++w) {
            kernel_func<<<blocks, threads>>>(d_W, d_scales, d_x, d_y, N, K);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // 20 timed runs with L2 flush
        std::vector<float> times(20);
        for (int i = 0; i < 20; ++i) {
            CUDA_CHECK(cudaMemset(d_flush, 0, flush_bytes));
            CUDA_CHECK(cudaEventRecord(start));
            kernel_func<<<blocks, threads>>>(d_W, d_scales, d_x, d_y, N, K);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
        }

        std::sort(times.begin(), times.end());
        float median_ms = times[10];
        double bw_gbps = (total_traffic_bytes / 1e9) / (median_ms / 1000.0);
        std::cout << "Kernel: " << name << "\n";
        std::cout << "  Median Time: " << median_ms << " ms (min=" << times[0] << ", max=" << times[19] << ")\n";
        std::cout << "  Achieved Bandwidth: " << bw_gbps << " GB/s (" 
                  << (bw_gbps / 228.15 * 100.0) << "% of 228.15 GB/s ceiling)\n\n";
    };

    bench_kernel("Baseline INT4 GEMV (Magic Float Mantissa Dequant)", gemv_int4_baseline);
    bench_kernel("Variant A (Stripped Dequant Math, Same Memory Traffic)", gemv_int4_stripped_dequant);
    bench_kernel("Variant B (Fast Integer SIMD Unpack)", gemv_int4_fast_unpack);

    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_scales));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_flush));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
