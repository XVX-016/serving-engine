#pragma once

#include "src/common/cuda_utils.h"
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>

namespace engine {

struct SamplerConfig {
    float temperature = 0.0f;       // 0.0 = greedy argmax
    int top_k = 0;                  // <= 0: disabled
    float top_p = 1.0f;             // >= 1.0: disabled
    float repetition_penalty = 1.0f;// 1.0 = no penalty
    uint64_t seed = 42;
};

class SamplerGPU {
public:
    SamplerGPU() = default;
    explicit SamplerGPU(int vocab_size, int max_top_k = 1024);

    void init(int vocab_size, int max_top_k = 1024);

    // Sample next token ID from raw logits on device.
    // logits: device pointer [vocab_size]
    // history: host vector of previously generated token IDs (for repetition penalty)
    int sample(
        const half* d_logits,
        const std::vector<int64_t>& history,
        const SamplerConfig& config,
        cudaStream_t stream = 0
    );

    // Device-only version for benchmarking without host sync if needed
    // Returns sampled token to host
    int sample_from_float_host(
        const float* h_logits,
        const std::vector<int64_t>& history,
        const SamplerConfig& config
    );

private:
    int vocab_size_ = 0;
    int max_top_k_ = 1024;
    uint64_t rng_state_ = 0;

    // GPU workspace buffers
    DeviceBuffer<half> d_penalized_logits_; // [vocab_size]
    DeviceBuffer<int64_t> d_history_;       // [history_len]
    DeviceBuffer<float> d_block_topk_val_;  // [num_blocks * max_top_k]
    DeviceBuffer<int> d_block_topk_idx_;    // [num_blocks * max_top_k]
    DeviceBuffer<float> d_final_topk_val_;  // [max_top_k]
    DeviceBuffer<int> d_final_topk_idx_;    // [max_top_k]
    DeviceBuffer<float> d_topk_probs_;      // [max_top_k]
    DeviceBuffer<int> d_sampled_token_;     // [1]
};

} // namespace engine

