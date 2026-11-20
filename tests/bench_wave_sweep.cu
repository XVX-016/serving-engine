#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

#define WARPS_PER_ROW 2

// Legacy (1024 threads, 32 warps/block -> 1 warp/row -> 32 rows/block)
__global__ void gemv_legacy_1024(
    const uint8_t* __restrict__ W_q,
    const half* __restrict__ scales,
    const half* __restrict__ x,
    half* __restrict__ y,
    int N, int K
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane    = threadIdx.x % 32;
    if (warp_id >= N) return;
    int num_groups = K / 128;
    const uint8_t* row_bytes = W_q + static_cast<size_t>(warp_id) * (K / 2);
    const half* row_scales = scales + static_cast<size_t>(warp_id) * num_groups;
    float acc = 0.0f;
    for (int k = lane * 2; k < K; k += 64) {
        uint8_t b = row_bytes[k / 2];
        float scale = __half2float(row_scales[k / 128]);
        uint16_t lo16 = 0x6400u | static_cast<uint16_t>((b & 0x0Fu) ^ 0x08u);
        uint16_t hi16 = 0x6400u | static_cast<uint16_t>(((b >> 4) & 0x0Fu) ^ 0x08u);
        float dq0 = (__half2float(*reinterpret_cast<half*>(&lo16)) - 1032.0f) * scale;
        float dq1 = (__half2float(*reinterpret_cast<half*>(&hi16)) - 1032.0f) * scale;
        acc += dq0 * __half2float(x[k]) + dq1 * __half2float(x[k + 1]);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) y[warp_id] = __float2half(acc);
}

// Split-K 1024 (1024 threads, 32 warps/block -> 2 warps/row -> 16 rows/block)
__global__ void gemv_splitk_1024(
    const uint8_t* __restrict__ W_q,
    const half* __restrict__ scales,
    const half* __restrict__ x,
    half* __restrict__ y,
    int N, int K
) {
    constexpr int ROWS_PER_BLOCK = 16;
    int block_warp_id = threadIdx.x / 32;
    int lane          = threadIdx.x % 32;
    int row_in_block = block_warp_id / 2;
    int slice_idx    = block_warp_id % 2;
    int row_id = blockIdx.x * ROWS_PER_BLOCK + row_in_block;
    if (row_id >= N) return;
    int num_groups = K / 128;
    const uint8_t* row_bytes = W_q + static_cast<size_t>(row_id) * (K / 2);
    const half* row_scales = scales + static_cast<size_t>(row_id) * num_groups;
    int slice_K = (K + 1) / 2;
    int slice_start = slice_idx * slice_K;
    int slice_end   = min(slice_start + slice_K, K);
    float acc = 0.0f;
    for (int k = slice_start + lane * 2; k < slice_end; k += 64) {
        uint8_t b = row_bytes[k / 2];
        float scale = __half2float(row_scales[k / 128]);
        uint16_t lo16 = 0x6400u | static_cast<uint16_t>((b & 0x0Fu) ^ 0x08u);
        uint16_t hi16 = 0x6400u | static_cast<uint16_t>(((b >> 4) & 0x0Fu) ^ 0x08u);
        float dq0 = (__half2float(*reinterpret_cast<half*>(&lo16)) - 1032.0f) * scale;
        float dq1 = (__half2float(*reinterpret_cast<half*>(&hi16)) - 1032.0f) * scale;
        acc += dq0 * __half2float(x[k]) + dq1 * __half2float(x[k + 1]);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    __shared__ float s_partial[32];
    if (lane == 0) s_partial[block_warp_id] = acc;
    __syncthreads();
    if (slice_idx == 0 && lane == 0) {
        float total_acc = s_partial[row_in_block * 2] + s_partial[row_in_block * 2 + 1];
        y[row_id] = __float2half(total_acc);
    }
}

// 256-thread block variant (8 warps/block -> 8 rows/block in legacy)
__global__ void gemv_legacy_256(
    const uint8_t* __restrict__ W_q,
    const half* __restrict__ scales,
    const half* __restrict__ x,
    half* __restrict__ y,
    int N, int K
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane    = threadIdx.x % 32;
    if (warp_id >= N) return;
    int num_groups = K / 128;
    const uint8_t* row_bytes = W_q + static_cast<size_t>(warp_id) * (K / 2);
    const half* row_scales = scales + static_cast<size_t>(warp_id) * num_groups;
    float acc = 0.0f;
    for (int k = lane * 2; k < K; k += 64) {
        uint8_t b = row_bytes[k / 2];
        float scale = __half2float(row_scales[k / 128]);
        uint16_t lo16 = 0x6400u | static_cast<uint16_t>((b & 0x0Fu) ^ 0x08u);
        uint16_t hi16 = 0x6400u | static_cast<uint16_t>(((b >> 4) & 0x0Fu) ^ 0x08u);
        float dq0 = (__half2float(*reinterpret_cast<half*>(&lo16)) - 1032.0f) * scale;
        float dq1 = (__half2float(*reinterpret_cast<half*>(&hi16)) - 1032.0f) * scale;
        acc += dq0 * __half2float(x[k]) + dq1 * __half2float(x[k + 1]);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) y[warp_id] = __float2half(acc);
}

// 128-thread block variant (4 warps/block -> 4 rows/block in legacy)
__global__ void gemv_legacy_128(
    const uint8_t* __restrict__ W_q,
    const half* __restrict__ scales,
    const half* __restrict__ x,
    half* __restrict__ y,
    int N, int K
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane    = threadIdx.x % 32;
    if (warp_id >= N) return;
    int num_groups = K / 128;
    const uint8_t* row_bytes = W_q + static_cast<size_t>(warp_id) * (K / 2);
    const half* row_scales = scales + static_cast<size_t>(warp_id) * num_groups;
    float acc = 0.0f;
    for (int k = lane * 2; k < K; k += 64) {
        uint8_t b = row_bytes[k / 2];
        float scale = __half2float(row_scales[k / 128]);
        uint16_t lo16 = 0x6400u | static_cast<uint16_t>((b & 0x0Fu) ^ 0x08u);
        uint16_t hi16 = 0x6400u | static_cast<uint16_t>(((b >> 4) & 0x0Fu) ^ 0x08u);
        float dq0 = (__half2float(*reinterpret_cast<half*>(&lo16)) - 1032.0f) * scale;
        float dq1 = (__half2float(*reinterpret_cast<half*>(&hi16)) - 1032.0f) * scale;
        acc += dq0 * __half2float(x[k]) + dq1 * __half2float(x[k + 1]);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) y[warp_id] = __float2half(acc);
}

int main() {
    int K = 896;
    int max_m = 14;
    int max_N = 24 * 32 * max_m;

    uint8_t* d_W; half* d_s; half* d_x; half* d_y; int* d_flush;
    CUDA_CHECK(cudaMalloc(&d_W, static_cast<size_t>(max_N) * (K / 2)));
    CUDA_CHECK(cudaMalloc(&d_s, static_cast<size_t>(max_N) * (K / 128) * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_x, K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_y, max_N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_flush, 128 * 1024 * 1024));

    CUDA_CHECK(cudaMemset(d_W, 0x55, static_cast<size_t>(max_N) * (K / 2)));
    CUDA_CHECK(cudaMemset(d_s, 0x3c, static_cast<size_t>(max_N) * (K / 128) * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_x, 0x3c, K * sizeof(half)));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    std::cout << "========================================================================================================\n";
    std::cout << " WAVE TEST (K = 896): Sweeping m = 1 to 14 (N = 24 * 32 * m = 768 * m rows)\n";
    std::cout << "========================================================================================================\n";
    std::cout << std::left << std::setw(6) << "m"
              << std::setw(8) << "N"
              << std::setw(8) << "Bytes"
              << std::setw(14) << "Waves(Leg1024)"
              << std::setw(14) << "Time(Leg1024)"
              << std::setw(14) << "Waves(SpK1024)"
              << std::setw(14) << "Time(SpK1024)"
              << std::setw(14) << "Time(Leg256)"
              << std::setw(14) << "Time(Leg128)" << "\n";
    std::cout << std::string(105, '-') << "\n";

    auto time_kernel = [&](auto fn) -> float {
        for (int w = 0; w < 5; ++w) fn();
        CUDA_CHECK(cudaDeviceSynchronize());
        float total_ms = 0.0f;
        for (int i = 0; i < 20; ++i) {
            CUDA_CHECK(cudaMemset(d_flush, 0, 128 * 1024 * 1024));
            CUDA_CHECK(cudaEventRecord(ev_start));
            fn();
            CUDA_CHECK(cudaEventRecord(ev_stop));
            CUDA_CHECK(cudaEventSynchronize(ev_stop));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));
            total_ms += ms;
        }
        return (total_ms / 20.0f) * 1000.0f; // us
    };

    for (int m = 1; m <= max_m; ++m) {
        int N = 24 * 32 * m;
        size_t bytes = static_cast<size_t>(N) * (K / 2) + static_cast<size_t>(N) * (K / 128) * sizeof(half);

        int grid_leg1024 = (N + 31) / 32;
        float waves_leg1024 = grid_leg1024 / 24.0f;
        float t_leg1024 = time_kernel([&](){ gemv_legacy_1024<<<grid_leg1024, 1024>>>(d_W, d_s, d_x, d_y, N, K); });

        int grid_spk1024 = (N + 15) / 16;
        float waves_spk1024 = grid_spk1024 / 24.0f;
        float t_spk1024 = time_kernel([&](){ gemv_splitk_1024<<<grid_spk1024, 1024>>>(d_W, d_s, d_x, d_y, N, K); });

        int grid_leg256 = (N + 7) / 8;
        float t_leg256 = time_kernel([&](){ gemv_legacy_256<<<grid_leg256, 256>>>(d_W, d_s, d_x, d_y, N, K); });

        int grid_leg128 = (N + 3) / 4;
        float t_leg128 = time_kernel([&](){ gemv_legacy_128<<<grid_leg128, 128>>>(d_W, d_s, d_x, d_y, N, K); });

        std::cout << std::left << std::setw(6) << m
                  << std::setw(8) << N
                  << std::setw(8) << bytes / 1024 << " KB"
                  << std::setw(14) << std::fixed << std::setprecision(1) << waves_leg1024
                  << std::setw(14) << std::setprecision(2) << t_leg1024
                  << std::setw(14) << std::setprecision(1) << waves_spk1024
                  << std::setw(14) << std::setprecision(2) << t_spk1024
                  << std::setw(14) << std::setprecision(2) << t_leg256
                  << std::setw(14) << std::setprecision(2) << t_leg128 << "\n";
    }

    return 0;
}
