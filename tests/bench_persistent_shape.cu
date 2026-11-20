// bench_persistent_shape.cu
// Benchmark gemv_int4_persistent for a given shape index.
// Warmup 5 + 20 timed iterations, CUDA events, L2 flush between runs.
// Usage: bench_persistent_shape.exe [shape_idx]  (0=qkv, 1=o_proj, 2=gate_up, 3=down_proj, 4=lm_head)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdint>
#include "src/cuda/gemv_persistent.cuh"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " : " << cudaGetErrorString(err) << "\n"; \
        exit(1); \
    } \
} while(0)

static void flush_l2(char* buf, size_t sz) {
    // Write-then-read to flush L2
    CUDA_CHECK(cudaMemset(buf, 0, sz));
    CUDA_CHECK(cudaDeviceSynchronize());
}

int main(int argc, char** argv) {
    int target_shape = 0;
    if (argc > 1) target_shape = std::stoi(argv[1]);

    struct Shape { int N, K; const char* name; };
    Shape shapes[] = {
        {1152,   896,  "qkv"},
        { 896,   896,  "o_proj"},
        {9728,   896,  "gate_up"},
        { 896,  4864,  "down_proj"},
        {151936, 896,  "lm_head"},
    };

    int N = shapes[target_shape].N;
    int K = shapes[target_shape].K;
    const char* name = shapes[target_shape].name;

    // Persistent kernel: 24 blocks (one per SM on sm_89), 1024 threads
    // 32 warps per block; 24*32 = 768 warps total; each loops over rows.
    int num_blocks   = 24;
    int block_size   = 1024;

    std::cerr << "Shape: " << name << "  N=" << N << " K=" << K
              << "  grid=" << num_blocks << " block=" << block_size << "\n";

    uint8_t* d_W; half* d_s; half* d_x; half* d_y;
    CUDA_CHECK(cudaMalloc(&d_W, static_cast<size_t>(N) * (K / 2)));
    CUDA_CHECK(cudaMalloc(&d_s, static_cast<size_t>(N) * (K / 128) * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_x, K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(half)));

    CUDA_CHECK(cudaMemset(d_W, 0x55, static_cast<size_t>(N) * (K / 2)));
    CUDA_CHECK(cudaMemset(d_s, 0x3c, static_cast<size_t>(N) * (K / 128) * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_x, 0x3c, K * sizeof(half)));

    // L2 flush buffer (8MB)
    const size_t flush_sz = 8*1024*1024;
    char* d_flush;
    CUDA_CHECK(cudaMalloc(&d_flush, flush_sz));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // Warmup
    for (int i = 0; i < 5; ++i) {
        gemv_int4_persistent<<<num_blocks, block_size>>>(d_W, d_s, d_x, d_y, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed
    std::vector<float> times;
    for (int i = 0; i < 20; ++i) {
        flush_l2(d_flush, flush_sz);
        CUDA_CHECK(cudaEventRecord(ev_start));
        gemv_int4_persistent<<<num_blocks, block_size>>>(d_W, d_s, d_x, d_y, N, K);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
        times.push_back(ms * 1000.0f); // us
    }

    std::sort(times.begin(), times.end());
    float med = times[times.size()/2];
    float mn = times.front(), mx = times.back();

    size_t bytes_w = static_cast<size_t>(N) * (K / 2);
    size_t bytes_s = static_cast<size_t>(N) * (K / 128) * sizeof(half);
    size_t bytes_x = K * sizeof(half);
    size_t total_bytes = bytes_w + bytes_s + bytes_x;
    double bw_GBs = static_cast<double>(total_bytes) / (med * 1e-6) / 1e9;

    std::cout << "PERSISTENT " << name
              << "  N=" << N << " K=" << K
              << "  median=" << med << " us"
              << "  min=" << mn << " us"
              << "  max=" << mx << " us"
              << "  bytes=" << total_bytes
              << "  BW=" << bw_GBs << " GB/s\n";

    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_s));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_flush));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    return 0;
}
