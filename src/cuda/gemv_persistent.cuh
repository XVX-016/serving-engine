#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define WARPS_PER_ROW 2

// Persistent INT4 GEMV
// Launch: Grid = 24 * k blocks (e.g. 24 or 48 blocks), Block = 1024 threads (32 warps)
// Each warp processes rows via persistent grid-stride / block-stride loop.
// In each step, warp loads the next row's scales/data before completing dot product reduction.
__global__ void gemv_int4_persistent(
    const uint8_t* __restrict__ W_q,
    const half* __restrict__ scales,
    const half* __restrict__ x,
    half* __restrict__ y,
    int N, int K
) {
    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int total_warps    = (gridDim.x * blockDim.x) / 32;
    int lane           = threadIdx.x % 32;

    int num_groups = K / 128;

    // Persistent warp loop over rows
    for (int row_id = global_warp_id; row_id < N; row_id += total_warps) {
        const uint8_t* row_bytes = W_q + static_cast<size_t>(row_id) * (K / 2);
        const half* row_scales = scales + static_cast<size_t>(row_id) * num_groups;

        // Prefetch scale for next row if within bounds
        int next_row = row_id + total_warps;
        half prefetched_scale = 0;
        if (next_row < N) {
            prefetched_scale = scales[static_cast<size_t>(next_row) * num_groups];
        }

        float acc = 0.0f;
        // Software pipelined loop
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

        if (lane == 0) {
            y[row_id] = __float2half(acc);
        }
    }
}
