#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <string>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

#define WARPS_PER_ROW 2

__global__ void gemv_int4_baseline(
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

__global__ void gemv_int4_stripped_with_sink(
    const uint8_t* __restrict__ W_q,
    const half* __restrict__ scales,
    const half* __restrict__ x,
    half* __restrict__ y,
    int N, int K,
    float* __restrict__ checksum_sink
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane    = threadIdx.x % 32;
    if (warp_id >= N) return;
    int num_groups = K / 128;
    const uint8_t* row_bytes = W_q + static_cast<size_t>(warp_id) * (K / 2);
    const half* row_scales = scales + static_cast<size_t>(warp_id) * num_groups;
    float acc = 0.0f;
    uint32_t load_checksum = 0;
    for (int k = lane * 2; k < K; k += 64) {
        uint8_t b = row_bytes[k / 2];
        load_checksum ^= b;
        float scale = __half2float(row_scales[k / 128]);
        float dq = static_cast<float>(b) * scale;
        acc += dq * __half2float(x[k]);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
        load_checksum ^= __shfl_down_sync(0xffffffff, load_checksum, offset);
    }
    if (lane == 0) {
        y[warp_id] = __float2half(acc);
        if (checksum_sink && warp_id == 0) {
            atomicAdd(checksum_sink, acc + static_cast<float>(load_checksum));
        }
    }
}

__global__ void gemv_int4_splitk(
    const uint8_t* __restrict__ W_q,
    const half* __restrict__ scales,
    const half* __restrict__ x,
    half* __restrict__ y,
    int N, int K
) {
    constexpr int ROWS_PER_BLOCK = 32 / WARPS_PER_ROW;
    int block_warp_id = threadIdx.x / 32;
    int lane          = threadIdx.x % 32;
    int row_in_block = block_warp_id / WARPS_PER_ROW;
    int slice_idx    = block_warp_id % WARPS_PER_ROW;
    int row_id = blockIdx.x * ROWS_PER_BLOCK + row_in_block;
    if (row_id >= N) return;
    int num_groups = K / 128;
    const uint8_t* row_bytes = W_q + static_cast<size_t>(row_id) * (K / 2);
    const half* row_scales = scales + static_cast<size_t>(row_id) * num_groups;
    int slice_K = (K + WARPS_PER_ROW - 1) / WARPS_PER_ROW;
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
        float total_acc = 0.0f;
        #pragma unroll
        for (int w = 0; w < WARPS_PER_ROW; ++w) {
            total_acc += s_partial[row_in_block * WARPS_PER_ROW + w];
        }
        y[row_id] = __float2half(total_acc);
    }
}

int main(int argc, char** argv) {
    int target_shape = 0; // 0: qkv, 1: o_proj, 2: gate_up, 3: down_proj, 4: lm_head
    int target_kernel = 0; // 0: legacy, 1: stripped, 2: splitk
    if (argc > 1) target_shape = std::stoi(argv[1]);
    if (argc > 2) target_kernel = std::stoi(argv[2]);

    struct Shape { int N, K; };
    Shape shapes[] = {
        {1152, 896},
        {896, 896},
        {9728, 896},
        {896, 4864},
        {151936, 896}
    };
    int N = shapes[target_shape].N;
    int K = shapes[target_shape].K;

    uint8_t* d_W; half* d_s; half* d_x; half* d_y; float* d_sink;
    CUDA_CHECK(cudaMalloc(&d_W, static_cast<size_t>(N) * (K / 2)));
    CUDA_CHECK(cudaMalloc(&d_s, static_cast<size_t>(N) * (K / 128) * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_x, K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(float)));

    CUDA_CHECK(cudaMemset(d_W, 0x55, static_cast<size_t>(N) * (K / 2)));
    CUDA_CHECK(cudaMemset(d_s, 0x3c, static_cast<size_t>(N) * (K / 128) * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_x, 0x3c, K * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_sink, 0, sizeof(float)));

    int grid_legacy = (N + 31) / 32;
    int grid_splitk = (N + 15) / 16;

    if (target_kernel == 0) {
        gemv_int4_baseline<<<grid_legacy, 1024>>>(d_W, d_s, d_x, d_y, N, K);
    } else if (target_kernel == 1) {
        gemv_int4_stripped_with_sink<<<grid_legacy, 1024>>>(d_W, d_s, d_x, d_y, N, K, d_sink);
    } else {
        gemv_int4_splitk<<<grid_splitk, 1024>>>(d_W, d_s, d_x, d_y, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}
