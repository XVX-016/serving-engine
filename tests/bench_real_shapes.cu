#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <iomanip>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

// FP16 Baseline GEMV
template <typename Epi>
__global__ void gemv_fp16_kernel(
    const half* __restrict__ W,
    const half* __restrict__ x,
    const half* __restrict__ extra,
    half* __restrict__ y,
    int N,
    int K
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane    = threadIdx.x % 32;
    if (warp_id >= N) return;

    const half* row = W + static_cast<size_t>(warp_id) * K;
    float acc = 0.0f;
    for (int k = lane; k < K; k += 32) {
        acc += __half2float(row[k]) * __half2float(x[k]);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) {
        y[warp_id] = __float2half(acc);
    }
}

// INT8 Baseline GEMV
__global__ void gemv_int8_kernel(
    const int8_t* __restrict__ W_q,
    const half* __restrict__ scales,
    const half* __restrict__ x,
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
    for (int r = k_tile + lane; r < K; r += 32) {
        acc += (static_cast<float>(row[r]) * scale) * __half2float(x[r]);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) {
        y[warp_id] = __float2half(acc);
    }
}

// INT4 Baseline GEMV (Exact code from src/cuda/gemv_quant.cu)
__global__ void gemv_int4_baseline(
    const uint8_t* __restrict__ W_q,
    const half* __restrict__ scales,
    const half* __restrict__ x,
    half* __restrict__ y,
    int N,
    int K,
    float* __restrict__ checksum_sink
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

    // Remainder loop for shapes where K % 1024 != 0 (e.g. K=896)
    for (int r = k_tile + lane * 2; r + 1 < K; r += 32 * 2) {
        uint8_t b = row_bytes[r / 2];
        int g_idx0 = r / 128;
        float scale0 = __half2float(row_scales[g_idx0]);
        int v0 = b & 0x0F;
        int q0 = (v0 >= 8) ? (v0 - 16) : v0;
        acc += (static_cast<float>(q0) * scale0) * __half2float(x[r]);

        int g_idx1 = (r + 1) / 128;
        float scale1 = __half2float(row_scales[g_idx1]);
        int v1 = (b >> 4) & 0x0F;
        int q1 = (v1 >= 8) ? (v1 - 16) : v1;
        acc += (static_cast<float>(q1) * scale1) * __half2float(x[r + 1]);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) {
        y[warp_id] = __float2half(acc);
        if (checksum_sink && warp_id == 0) {
            atomicAdd(checksum_sink, acc);
        }
    }
}

// INT4 Stripped-Math Variant WITH Checksum Sink (forces every byte loaded to contribute)
__global__ void gemv_int4_stripped_with_sink(
    const uint8_t* __restrict__ W_q,
    const half* __restrict__ scales,
    const half* __restrict__ x,
    half* __restrict__ y,
    int N,
    int K,
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

    int k_tile = 0;
    for (; k_tile + 32 * 32 <= K; k_tile += 32 * 32) {
        int k = k_tile + lane * 32;
        uint4 raw = *reinterpret_cast<const uint4*>(row_bytes + k / 2);
        load_checksum ^= raw.x ^ raw.y ^ raw.z ^ raw.w;
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&raw);

        float4 x_vec0 = *reinterpret_cast<const float4*>(x + k);
        const half* x_h0 = reinterpret_cast<const half*>(&x_vec0);

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            uint8_t b = bytes[i];
            int elem_idx = k + i * 2;
            int g_idx = elem_idx / 128;
            float scale = __half2float(row_scales[g_idx]);

            float dq = static_cast<float>(b) * scale;
            acc += dq * __half2float(x_h0[i % 8]);
        }
    }

    for (int r = k_tile + lane * 2; r + 1 < K; r += 32 * 2) {
        uint8_t b = row_bytes[r / 2];
        load_checksum ^= b;
        int g_idx0 = r / 128;
        float scale0 = __half2float(row_scales[g_idx0]);
        acc += (static_cast<float>(b) * scale0) * __half2float(x[r]);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
        load_checksum ^= __shfl_down_sync(0xffffffff, load_checksum, offset);
    }
    if (lane == 0) {
        y[warp_id] = __float2half(acc + (load_checksum & 0x1 ? 0.0f : 0.0f));
        if (checksum_sink && warp_id == 0) {
            atomicAdd(checksum_sink, acc + static_cast<float>(load_checksum));
        }
    }
}

// Split-K INT4 GEMV (Optimized Fix Attempt: 2 warps per row with shared memory reduction)
template <int WARPS_PER_ROW = 2>
__global__ void gemv_int4_splitk(
    const uint8_t* __restrict__ W_q,
    const half* __restrict__ scales,
    const half* __restrict__ x,
    half* __restrict__ y,
    int N,
    int K
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
    // Each thread processes 8 elements (4 bytes) per step across the 32 threads in the warp
    // 32 threads * 8 elements = 256 elements per tile
    int k_tile = slice_start;
    for (; k_tile + 32 * 8 <= slice_end; k_tile += 32 * 8) {
        int k = k_tile + lane * 8;
        uint32_t raw_u32 = *reinterpret_cast<const uint32_t*>(row_bytes + k / 2);
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&raw_u32);

        float4 x_vec = *reinterpret_cast<const float4*>(x + k);
        const half* x_h = reinterpret_cast<const half*>(&x_vec);

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint8_t b = bytes[i];
            int elem_idx = k + i * 2;
            int g_idx = elem_idx / 128;
            float scale = __half2float(row_scales[g_idx]);

            uint16_t lo16 = 0x6400u | static_cast<uint16_t>((b & 0x0Fu) ^ 0x08u);
            uint16_t hi16 = 0x6400u | static_cast<uint16_t>(((b >> 4) & 0x0Fu) ^ 0x08u);
            float dq0 = (__half2float(*reinterpret_cast<half*>(&lo16)) - 1032.0f) * scale;
            float dq1 = (__half2float(*reinterpret_cast<half*>(&hi16)) - 1032.0f) * scale;

            acc += dq0 * __half2float(x_h[i * 2]);
            acc += dq1 * __half2float(x_h[i * 2 + 1]);
        }
    }

    // Remainder for slice
    for (int r = k_tile + lane * 2; r + 1 < slice_end; r += 32 * 2) {
        uint8_t b = row_bytes[r / 2];
        int g_idx0 = r / 128;
        float scale0 = __half2float(row_scales[g_idx0]);
        int v0 = b & 0x0F;
        int q0 = (v0 >= 8) ? (v0 - 16) : v0;
        acc += (static_cast<float>(q0) * scale0) * __half2float(x[r]);

        int g_idx1 = (r + 1) / 128;
        float scale1 = __half2float(row_scales[g_idx1]);
        int v1 = (b >> 4) & 0x0F;
        int q1 = (v1 >= 8) ? (v1 - 16) : v1;
        acc += (static_cast<float>(q1) * scale1) * __half2float(x[r + 1]);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    __shared__ float s_partial[32];
    if (lane == 0) {
        s_partial[block_warp_id] = acc;
    }
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

struct ShapeInfo {
    std::string name;
    int N;
    int K;
};

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << " (SMs: " << prop.multiProcessorCount << ", L2: " << prop.l2CacheSize / 1e6 << " MB)\n";

    std::vector<ShapeInfo> shapes = {
        {"qkv", 1152, 896},
        {"o_proj", 896, 896},
        {"gate_up", 9728, 896},
        {"down_proj", 896, 4864},
        {"lm_head", 151936, 896}
    };

    // 128 MB L2 flush buffer
    const size_t flush_bytes = 128 * 1024 * 1024;
    int* d_flush = nullptr;
    CUDA_CHECK(cudaMalloc(&d_flush, flush_bytes));

    float* d_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sink, sizeof(float)));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    std::cout << "\n========================================================================================================================\n";
    std::cout << "REAL LAYER SHAPES BENCHMARK (With 128 MB L2 Flush per timed launch)\n";
    std::cout << "========================================================================================================================\n";
    std::cout << std::left << std::setw(12) << "Shape"
              << std::setw(12) << "Dim [NxK]"
              << std::setw(8)  << "Prec"
              << std::setw(12) << "Grid"
              << std::setw(12) << "Warps/SM"
              << std::setw(16) << "Bytes/SM (flight)"
              << std::setw(14) << "Time (us)"
              << std::setw(14) << "GB/s"
              << std::setw(12) << "% Roofline" << "\n";
    std::cout << std::string(120, '-') << "\n";

    for (const auto& sh : shapes) {
        int N = sh.N;
        int K = sh.K;

        // Allocations
        size_t bytes_fp16 = static_cast<size_t>(N) * K * sizeof(half);
        size_t bytes_int8 = static_cast<size_t>(N) * K + N * sizeof(half);
        size_t bytes_int4 = static_cast<size_t>(N) * (K / 2) + N * (K / 128) * sizeof(half);

        half* d_W_fp16; half* d_x_fp16; half* d_y_fp16;
        CUDA_CHECK(cudaMalloc(&d_W_fp16, bytes_fp16));
        CUDA_CHECK(cudaMalloc(&d_x_fp16, K * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_y_fp16, N * sizeof(half)));

        int8_t* d_W_int8; half* d_scales_int8;
        CUDA_CHECK(cudaMalloc(&d_W_int8, N * K));
        CUDA_CHECK(cudaMalloc(&d_scales_int8, N * sizeof(half)));

        uint8_t* d_W_int4; half* d_scales_int4;
        CUDA_CHECK(cudaMalloc(&d_W_int4, N * (K / 2)));
        CUDA_CHECK(cudaMalloc(&d_scales_int4, N * (K / 128) * sizeof(half)));

        CUDA_CHECK(cudaMemset(d_W_fp16, 0x3c, bytes_fp16));
        CUDA_CHECK(cudaMemset(d_x_fp16, 0x3c, K * sizeof(half)));
        CUDA_CHECK(cudaMemset(d_W_int8, 0x55, N * K));
        CUDA_CHECK(cudaMemset(d_scales_int8, 0x3c, N * sizeof(half)));
        CUDA_CHECK(cudaMemset(d_W_int4, 0x55, N * (K / 2)));
        CUDA_CHECK(cudaMemset(d_scales_int4, 0x3c, N * (K / 128) * sizeof(half)));

        // Benchmark helper
        auto run_bench = [&](const std::string& prec, size_t total_traffic, int grid_dim, int block_dim, auto kernel_call) {
            // 5 warmup
            for (int w = 0; w < 5; ++w) kernel_call();
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<float> times(20);
            for (int i = 0; i < 20; ++i) {
                CUDA_CHECK(cudaMemset(d_flush, 0, flush_bytes));
                CUDA_CHECK(cudaEventRecord(ev_start));
                kernel_call();
                CUDA_CHECK(cudaEventRecord(ev_stop));
                CUDA_CHECK(cudaEventSynchronize(ev_stop));
                CUDA_CHECK(cudaEventElapsedTime(&times[i], ev_start, ev_stop));
            }
            std::sort(times.begin(), times.end());
            float med_us = times[10] * 1000.0f;
            double gbps = (static_cast<double>(total_traffic) / 1e9) / (med_us / 1e6);
            double roof_pct = (gbps / 228.15) * 100.0;

            int total_warps = grid_dim * (block_dim / 32);
            double warps_per_sm = static_cast<double>(total_warps) / prop.multiProcessorCount;
            // Bytes in flight per SM = (loads per thread * bytes per load * active threads per SM)
            // For 16-byte load per thread * resident threads
            int bytes_per_thread = (prec == "INT4" ? 16 : (prec == "INT8" ? 16 : 8));
            double resident_threads = std::min<double>(total_warps * 32 / static_cast<double>(prop.multiProcessorCount), 1536.0);
            double bytes_in_flight_sm = bytes_per_thread * resident_threads;

            std::string dim_str = std::to_string(N) + "x" + std::to_string(K);
            std::string grid_str = "(" + std::to_string(grid_dim) + ",1,1)";
            std::cout << std::left << std::setw(12) << sh.name
                      << std::setw(12) << dim_str
                      << std::setw(8)  << prec
                      << std::setw(12) << grid_str
                      << std::setw(12) << std::fixed << std::setprecision(1) << warps_per_sm
                      << std::setw(16) << std::fixed << std::setprecision(0) << bytes_in_flight_sm
                      << std::setw(14) << std::fixed << std::setprecision(2) << med_us
                      << std::setw(14) << std::fixed << std::setprecision(1) << gbps
                      << std::setw(12) << std::fixed << std::setprecision(1) << roof_pct << "\n";
        };

        // 1. FP16
        int grid_fp16 = (N + 31) / 32;
        run_bench("FP16", bytes_fp16, grid_fp16, 1024, [&]() {
            gemv_fp16_kernel<void><<<grid_fp16, 1024>>>(d_W_fp16, d_x_fp16, nullptr, d_y_fp16, N, K);
        });

        // 2. INT8
        int grid_int8 = (N + 31) / 32;
        run_bench("INT8", bytes_int8, grid_int8, 1024, [&]() {
            gemv_int8_kernel<<<grid_int8, 1024>>>(d_W_int8, d_scales_int8, d_x_fp16, d_y_fp16, N, K);
        });

        // 3. INT4 Baseline
        int grid_int4 = (N + 31) / 32;
        run_bench("INT4-base", bytes_int4, grid_int4, 1024, [&]() {
            gemv_int4_baseline<<<grid_int4, 1024>>>(d_W_int4, d_scales_int4, d_x_fp16, d_y_fp16, N, K, d_sink);
        });

        // 4. INT4 Stripped with sink
        run_bench("INT4-sink", bytes_int4, grid_int4, 1024, [&]() {
            gemv_int4_stripped_with_sink<<<grid_int4, 1024>>>(d_W_int4, d_scales_int4, d_x_fp16, d_y_fp16, N, K, d_sink);
        });

        // 5. INT4 Split-K (2 warps per row)
        int grid_splitk = (N + 15) / 16; // 16 rows per block
        run_bench("INT4-sK2", bytes_int4, grid_splitk, 1024, [&]() {
            gemv_int4_splitk<2><<<grid_splitk, 1024>>>(d_W_int4, d_scales_int4, d_x_fp16, d_y_fp16, N, K);
        });

        CUDA_CHECK(cudaFree(d_W_fp16)); CUDA_CHECK(cudaFree(d_x_fp16)); CUDA_CHECK(cudaFree(d_y_fp16));
        CUDA_CHECK(cudaFree(d_W_int8)); CUDA_CHECK(cudaFree(d_scales_int8));
        CUDA_CHECK(cudaFree(d_W_int4)); CUDA_CHECK(cudaFree(d_scales_int4));
    }

    CUDA_CHECK(cudaFree(d_flush));
    CUDA_CHECK(cudaFree(d_sink));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    return 0;
}
