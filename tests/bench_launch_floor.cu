#include "src/common/cuda_utils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iomanip>

__global__ void empty_kernel() {}

__global__ void read_kernel(const float4* __restrict__ data, size_t num_float4, float* __restrict__ out) {
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
    float sum = acc0 + acc1 + acc2 + acc3;
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (threadIdx.x == 0) {
        atomicAdd(out, sum);
    }
}

struct BenchResult {
    double eager_us_per_launch;
    double eager_floor_315_us;
    double graph_us_per_launch;
    double graph_floor_315_us;
};

BenchResult run_microbenchmark(size_t bytes, int num_launches = 1000) {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    float4* d_data = nullptr;
    float* d_out = nullptr;
    size_t num_float4 = bytes / sizeof(float4);

    if (bytes > 0) {
        CUDA_CHECK(cudaMalloc(&d_data, bytes));
        CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));
        CUDA_CHECK(cudaMemset(d_data, 1, bytes));
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
    }

    int block_size = 256;
    int grid_size = 32; // representative small grid like linear decode projections

    auto launch_one = [&]() {
        if (bytes == 0) {
            empty_kernel<<<1, 1, 0, stream>>>();
        } else {
            read_kernel<<<grid_size, block_size, 0, stream>>>(d_data, num_float4, d_out);
        }
    };

    // Warmup eager
    for (int i = 0; i < 50; ++i) launch_one();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Timed eager (1000 launches back-to-back)
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    CUDA_CHECK(cudaEventRecord(ev_start, stream));
    for (int i = 0; i < num_launches; ++i) {
        launch_one();
    }
    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float eager_total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&eager_total_ms, ev_start, ev_stop));
    double eager_us_per_launch = (eager_total_ms * 1000.0) / num_launches;

    // Graph capture of 1000 back-to-back launches
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < num_launches; ++i) {
        launch_one();
    }
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    // Warmup graph
    for (int i = 0; i < 5; ++i) {
        CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Timed graph launch
    CUDA_CHECK(cudaEventRecord(ev_start, stream));
    CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    CUDA_CHECK(cudaEventRecord(ev_stop, stream));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float graph_total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&graph_total_ms, ev_start, ev_stop));
    double graph_us_per_launch = (graph_total_ms * 1000.0) / num_launches;

    // Clean up
    CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    if (bytes > 0) {
        CUDA_CHECK(cudaFree(d_data));
        CUDA_CHECK(cudaFree(d_out));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));

    return {
        eager_us_per_launch,
        eager_us_per_launch * 315.0,
        graph_us_per_launch,
        graph_us_per_launch * 315.0
    };
}

int main() {
    std::cout << "========================================================================\n";
    std::cout << " Task P11a Deliverable 1: Launch-Floor Microbenchmark (1000 Launches)\n";
    std::cout << "========================================================================\n";

    std::vector<std::pair<std::string, size_t>> cases = {
        {"Empty Kernel (0 MB)", 0},
        {"Read Kernel 0.5 MB", 512 * 1024},
        {"Read Kernel 1.0 MB", 1024 * 1024},
        {"Read Kernel 2.0 MB", 2048 * 1024}
    };

    std::cout << std::fixed << std::setprecision(3);
    std::cout << std::left << std::setw(24) << "Workload"
              << std::right << std::setw(15) << "Eager (us/lnch)"
              << std::setw(18) << "Eager Floor (315)"
              << std::setw(15) << "Graph (us/lnch)"
              << std::setw(18) << "Graph Floor (315)"
              << "\n";
    std::cout << "------------------------------------------------------------------------------------\n";

    for (const auto& [name, bytes] : cases) {
        BenchResult res = run_microbenchmark(bytes, 1000);
        std::cout << std::left << std::setw(24) << name
                  << std::right << std::setw(15) << res.eager_us_per_launch
                  << std::setw(15) << (res.eager_floor_315_us / 1000.0) << " ms"
                  << std::setw(15) << res.graph_us_per_launch
                  << std::setw(15) << (res.graph_floor_315_us / 1000.0) << " ms"
                  << "\n";
    }

    std::cout << "========================================================================\n";
    return 0;
}

