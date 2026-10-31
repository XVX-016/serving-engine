#include "src/cuda/read_bandwidth.h"
#include "src/common/cuda_utils.h"
#include <vector>
#include <algorithm>
#include <iostream>

// Vectorized 128-bit load kernel that reads 256 MB and accumulates locally to force streaming from DRAM
__global__ void read_256mb_kernel(const float4* __restrict__ data, size_t num_float4, float* __restrict__ out) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    for (size_t i = idx; i < num_float4; i += stride) {
        float4 val = data[i];
        acc0 += val.x;
        acc1 += val.y;
        acc2 += val.z;
        acc3 += val.w;
    }

    float thread_sum = acc0 + acc1 + acc2 + acc3;
    // Warp shuffle reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    __shared__ float warp_sums[32];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    if (lane == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_sum = (lane < (blockDim.x / 32)) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (lane == 0) {
            atomicAdd(out, block_sum);
        }
    }
}

ReadBandwidthResult run_read_bandwidth_benchmark(int num_warmup, int num_timed) {
    // 256 MB = 256 * 1024 * 1024 bytes = 16,777,216 float4 elements
    const size_t total_bytes = 256 * 1024 * 1024;
    const size_t num_float4 = total_bytes / sizeof(float4);

    float4* d_data = nullptr;
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, total_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_data, 1, total_bytes));
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));

    // 128 MB L2 cache flush buffer (RTX 4060 has 32 MB L2 cache)
    const size_t flush_bytes = 128 * 1024 * 1024;
    int* d_flush = nullptr;
    CUDA_CHECK(cudaMalloc(&d_flush, flush_bytes));

    int block_size = 256;
    // 24 SMs on RTX 4060 Laptop -> grid of 96 or 192 blocks keeps high occupancy
    int grid_size = 192;

    // Warmup
    for (int i = 0; i < num_warmup; ++i) {
        read_256mb_kernel<<<grid_size, block_size>>>(d_data, num_float4, d_out);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times_ms(num_timed);

    for (int i = 0; i < num_timed; ++i) {
        // Evict L2 cache
        CUDA_CHECK(cudaMemset(d_flush, 0, flush_bytes));

        CUDA_CHECK(cudaEventRecord(start));
        read_256mb_kernel<<<grid_size, block_size>>>(d_data, num_float4, d_out);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times_ms[i] = ms;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_flush));

    std::vector<float> bws_gb_s(num_timed);
    for (int i = 0; i < num_timed; ++i) {
        // Read-only: 256 MB read = 0.268435456 GB transferred
        bws_gb_s[i] = (static_cast<double>(total_bytes) / (times_ms[i] / 1000.0)) / 1e9;
    }

    std::sort(times_ms.begin(), times_ms.end());
    std::sort(bws_gb_s.begin(), bws_gb_s.end());

    ReadBandwidthResult res;
    res.median_ms = times_ms[num_timed / 2];
    res.median_gb_s = bws_gb_s[num_timed / 2];
    res.min_gb_s = bws_gb_s.front();
    res.max_gb_s = bws_gb_s.back();
    return res;
}

