#include "src/cuda/kernels.h"
#include "src/cuda/kv_view.h"
#include "src/common/cuda_utils.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>

using namespace engine;
using namespace engine::cuda;

// Preallocate 128 MB buffer for L2 cache flushing
static DeviceBuffer<int32_t> g_flush_buffer;

void flush_l2_cache() {
    g_flush_buffer.zero();
}

int main(int argc, char** argv) {
    std::cout << "==========================================================================================" << std::endl;
    std::cout << "  C++ DECODE ATTENTION (GQA-GROUPED + SPLIT-KV) BENCHMARK (L2 FLUSHED)" << std::endl;
    std::cout << "==========================================================================================" << std::endl;

    g_flush_buffer.allocate(32 * 1024 * 1024); // 32M int32 = 128 MB

    int num_q_heads = 14;
    int num_kv_heads = 2;
    int head_dim = 64;

    std::vector<int> batches = {1, 4, 16, 32};
    std::vector<int> contexts = {128, 512, 2048, 4096};

    int num_warmup = 5;
    int num_timed = 25;

    for (int B : batches) {
        for (int S : contexts) {
            size_t q_size = static_cast<size_t>(B) * num_q_heads * head_dim;
            size_t kv_size = static_cast<size_t>(B) * num_kv_heads * S * head_dim;

            DeviceBuffer<half> d_q(q_size);
            DeviceBuffer<half> d_k(kv_size);
            DeviceBuffer<half> d_v(kv_size);
            DeviceBuffer<half> d_out(q_size);
            DeviceBuffer<int> d_seq_lens(B);

            std::vector<int> h_seq_lens(B, S);
            d_seq_lens.copy_from_host(h_seq_lens.data(), B);

            // Up to 16 splits scratch buffer
            size_t split_buffer_elems = 16 * B * num_q_heads * (head_dim + 2);
            DeviceBuffer<float> d_split_buf(split_buffer_elems);

            d_q.zero();
            d_k.zero();
            d_v.zero();

            KVView k_view{
                d_k.data(),
                static_cast<int64_t>(num_kv_heads * S * head_dim),
                static_cast<int64_t>(S * head_dim),
                static_cast<int64_t>(head_dim),
                1
            };

            KVView v_view{
                d_v.data(),
                static_cast<int64_t>(num_kv_heads * S * head_dim),
                static_cast<int64_t>(S * head_dim),
                static_cast<int64_t>(head_dim),
                1
            };

            // Warmup
            for (int w = 0; w < num_warmup; ++w) {
                decode_gqa_split(
                    d_q.data(),
                    k_view,
                    v_view,
                    d_seq_lens.data(),
                    d_out.data(),
                    B,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    S,
                    d_split_buf.data()
                );
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            // Timed runs with L2 cache flush
            std::vector<cudaEvent_t> start_events(num_timed);
            std::vector<cudaEvent_t> stop_events(num_timed);
            for (int i = 0; i < num_timed; ++i) {
                CUDA_CHECK(cudaEventCreate(&start_events[i]));
                CUDA_CHECK(cudaEventCreate(&stop_events[i]));
            }

            for (int iter = 0; iter < num_timed; ++iter) {
                flush_l2_cache();
                CUDA_CHECK(cudaEventRecord(start_events[iter]));
                decode_gqa_split(
                    d_q.data(),
                    k_view,
                    v_view,
                    d_seq_lens.data(),
                    d_out.data(),
                    B,
                    num_q_heads,
                    num_kv_heads,
                    head_dim,
                    S,
                    d_split_buf.data()
                );
                CUDA_CHECK(cudaEventRecord(stop_events[iter]));
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<float> times(num_timed);
            for (int iter = 0; iter < num_timed; ++iter) {
                float ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&ms, start_events[iter], stop_events[iter]));
                times[iter] = ms;
                CUDA_CHECK(cudaEventDestroy(start_events[iter]));
                CUDA_CHECK(cudaEventDestroy(stop_events[iter]));
            }

            std::sort(times.begin(), times.end());
            float median_ms = times[num_timed / 2];

            // Effective Bandwidth calculation:
            // total_bytes = B * S * num_kv_heads * head_dim * 2 * 2 (K + V read in FP16)
            //             + B * num_q_heads * head_dim * 2 * 2 (Q read + Out write in FP16)
            double total_kv_bytes = static_cast<double>(B) * S * num_kv_heads * head_dim * 4.0;
            double total_qo_bytes = static_cast<double>(B) * num_q_heads * head_dim * 4.0;
            double total_bytes = total_kv_bytes + total_qo_bytes;

            double bw_gb_s = (total_bytes / (median_ms * 1e-3)) / 1e9;
            double pct_copy = (bw_gb_s / 228.15) * 100.0; // 228.15 GB/s is copy/read bandwidth on this machine

            std::cout << "B=" << B << "\tS=" << S
                      << "\tMedian: " << median_ms << " ms"
                      << "\tSpread: [" << times.front() << ", " << times.back() << "] ms"
                      << "\tBW: " << bw_gb_s << " GB/s"
                      << "\t(" << pct_copy << "% copy BW)" << std::endl;
        }
    }

    return 0;
}
