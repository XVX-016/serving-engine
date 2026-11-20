// bench_wave_single.cu
// Benchmarks one N value (fresh allocation each run) for NCU profiling.
// Usage: bench_wave_single.exe N [kernel]
//   kernel: 0=legacy_1024 (default), 1=splitk_1024
// Runs 5 warmup + 1 NCU-profiled iteration (NCU will replay as needed).
// Fresh alloc per run avoids aliasing with prior test points.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <cstdlib>
#include <cstdint>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << "\n"; \
        exit(1); \
    } \
} while(0)

// Legacy 1024-thread warp-per-row GEMV
__global__ void gemv_legacy(
    const uint8_t* __restrict__ W_q, const half* __restrict__ scales,
    const half* __restrict__ x, half* __restrict__ y, int N, int K)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane    = threadIdx.x % 32;
    if (warp_id >= N) return;
    const uint8_t* row = W_q + (size_t)warp_id * (K/2);
    const half*    sc  = scales + (size_t)warp_id * (K/128);
    float acc = 0.f;
    for (int k = lane*2; k < K; k += 64) {
        uint8_t b = row[k/2];
        float s = __half2float(sc[k/128]);
        uint16_t lo = 0x6400u | (uint16_t)((b & 0x0F) ^ 0x08);
        uint16_t hi = 0x6400u | (uint16_t)((b >> 4)  ^ 0x08);
        float d0 = (__half2float(*(half*)&lo) - 1032.f) * s;
        float d1 = (__half2float(*(half*)&hi) - 1032.f) * s;
        acc += d0 * __half2float(x[k]) + d1 * __half2float(x[k+1]);
    }
    for (int o=16; o>0; o>>=1) acc += __shfl_down_sync(0xffffffff, acc, o);
    if (lane==0) y[warp_id] = __float2half(acc);
}

// Split-K 1024-thread 2-warps-per-row GEMV
__global__ void gemv_splitk(
    const uint8_t* __restrict__ W_q, const half* __restrict__ scales,
    const half* __restrict__ x, half* __restrict__ y, int N, int K)
{
    constexpr int RPB = 16;
    int bwid = threadIdx.x / 32, lane = threadIdx.x % 32;
    int row  = blockIdx.x * RPB + bwid/2;
    int slic = bwid % 2;
    if (row >= N) return;
    const uint8_t* rw = W_q + (size_t)row*(K/2);
    const half*    sc = scales + (size_t)row*(K/128);
    int sk = (K+1)/2, ss = slic*sk, se = min(ss+sk, K);
    float acc = 0.f;
    for (int k = ss+lane*2; k < se; k += 64) {
        uint8_t b = rw[k/2];
        float s = __half2float(sc[k/128]);
        uint16_t lo = 0x6400u | (uint16_t)((b & 0x0F) ^ 0x08);
        uint16_t hi = 0x6400u | (uint16_t)((b >> 4)  ^ 0x08);
        acc += (__half2float(*(half*)&lo)-1032.f)*s*__half2float(x[k])
             + (__half2float(*(half*)&hi)-1032.f)*s*__half2float(x[k+1]);
    }
    for (int o=16; o>0; o>>=1) acc += __shfl_down_sync(0xffffffff, acc, o);
    __shared__ float sp[32];
    if (lane==0) sp[bwid]=acc;
    __syncthreads();
    if (slic==0 && lane==0) y[row]=__float2half(sp[bwid*2]+sp[bwid*2+1]);
}

int main(int argc, char** argv) {
    int N = 3840, kernel = 0;
    int K = 896;
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) kernel = atoi(argv[2]);

    // Fresh allocation (not reused from prior run)
    uint8_t* d_W; half* d_s; half* d_x; half* d_y;
    CUDA_CHECK(cudaMalloc(&d_W, (size_t)N*(K/2)));
    CUDA_CHECK(cudaMalloc(&d_s, (size_t)N*(K/128)*sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_x, K*sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_y, N*sizeof(half)));
    CUDA_CHECK(cudaMemset(d_W, 0x55, (size_t)N*(K/2)));
    CUDA_CHECK(cudaMemset(d_s, 0x3c, (size_t)N*(K/128)*sizeof(half)));
    CUDA_CHECK(cudaMemset(d_x, 0x3c, K*sizeof(half)));

    // L2 flush buffer
    char* d_fl; CUDA_CHECK(cudaMalloc(&d_fl, 8<<20));
    CUDA_CHECK(cudaMemset(d_fl, 0, 8<<20));

    std::cerr << "bench_wave_single N=" << N << " K=" << K
              << " kernel=" << (kernel==0?"legacy":"splitk") << "\n";

    auto run = [&]() {
        CUDA_CHECK(cudaMemset(d_fl, 0, 8<<20)); // flush L2
        if (kernel == 0) {
            int g = (N+31)/32;
            gemv_legacy<<<g, 1024>>>(d_W, d_s, d_x, d_y, N, K);
        } else {
            int g = (N+15)/16;
            gemv_splitk<<<g, 1024>>>(d_W, d_s, d_x, d_y, N, K);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    };

    // 5 warmup + 1 profiled (NCU replays the profiled call)
    for (int i=0; i<5; i++) run();
    run(); // NCU profiles this call

    CUDA_CHECK(cudaFree(d_W)); CUDA_CHECK(cudaFree(d_s));
    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_fl));
    return 0;
}
