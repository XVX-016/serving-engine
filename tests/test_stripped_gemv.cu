#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>

__global__ void gemv_int4_stripped_kernel(
    const uint8_t* __restrict__ W_q,
    const half* __restrict__ scales,
    const int8_t* __restrict__ x_q,
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

    uint32_t dummy_accum = 0;

    int k_tile = 0;
    for (; k_tile + 32 * 32 <= K; k_tile += 32 * 32) {
        int k = k_tile + lane * 32;

        uint4 raw = *reinterpret_cast<const uint4*>(row_bytes + k / 2);
        const int4* x_ptr = reinterpret_cast<const int4*>(x_q + k);
        int4 x_vec0 = x_ptr[0];
        int4 x_vec1 = x_ptr[1];

        half s = row_scales[(k) / 128];

        dummy_accum ^= raw.x ^ raw.y ^ raw.z ^ raw.w ^ x_vec0.x ^ x_vec1.w ^ (*reinterpret_cast<const uint16_t*>(&s));
    }

    if (dummy_accum == 0x12345678) {
        y[warp_id] = __float2half(1.0f);
    }
}

int main() {
    int N = 4864; // gate_proj
    int K = 896;

    size_t w_bytes = static_cast<size_t>(N) * (K / 2);
    size_t scale_bytes = static_cast<size_t>(N) * (K / 128) * sizeof(half);
    size_t x_bytes = K * sizeof(int8_t);
    size_t y_bytes = N * sizeof(half);

    // Number of weight buffers to cycle through to flush L2 (RTX 4060 Laptop has 32MB L2 cache)
    // 2.13 MB per buffer, 32 buffers = ~68 MB (> 2x L2 size)
    const int NUM_BUFS = 32;
    std::vector<uint8_t*> d_w(NUM_BUFS);
    std::vector<half*> d_scales(NUM_BUFS);
    for (int b = 0; b < NUM_BUFS; ++b) {
        cudaMalloc(&d_w[b], w_bytes);
        cudaMalloc(&d_scales[b], scale_bytes);
    }

    int8_t *d_x;
    half *d_y;
    cudaMalloc(&d_x, x_bytes);
    cudaMalloc(&d_y, y_bytes);

    int threads_per_block = 256;
    int warps_per_block = threads_per_block / 32;
    int blocks = (N + warps_per_block - 1) / warps_per_block;

    // Warmup
    for (int i = 0; i < NUM_BUFS; ++i) {
        gemv_int4_stripped_kernel<<<blocks, threads_per_block>>>(d_w[i], d_scales[i], d_x, d_y, N, K);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int iters = 64;
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        int buf_idx = i % NUM_BUFS;
        gemv_int4_stripped_kernel<<<blocks, threads_per_block>>>(d_w[buf_idx], d_scales[buf_idx], d_x, d_y, N, K);
    }
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_us = (ms * 1000.0f) / iters;

    double total_read_bytes = w_bytes + scale_bytes + x_bytes;
    double gb_s = (total_read_bytes / (avg_us * 1e-6)) / 1e9;

    std::cout << "=== Stripped-Compute Test with L2 Cache Flushed (Cold DRAM Read) ===\n";
    std::cout << "INT4 gate_proj: N=4864, K=896 (Cycling 32 buffers = 68.1 MB across 32MB L2)\n";
    std::cout << "Weights: " << w_bytes / 1024.0 << " KB, Scales: " << scale_bytes / 1024.0 << " KB\n";
    std::cout << "Average Cold Latency: " << avg_us << " us\n";
    std::cout << "Achieved DRAM Read Bandwidth: " << gb_s << " GB/s (" << (gb_s / 228.15) * 100.0 << "% of 228.15 GB/s roofline)\n";

    for (int b = 0; b < NUM_BUFS; ++b) {
        cudaFree(d_w[b]);
        cudaFree(d_scales[b]);
    }
    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}
