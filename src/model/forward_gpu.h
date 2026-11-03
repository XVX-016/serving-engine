#pragma once

#include "src/model/config.h"
#include "src/common/cuda_utils.h"
#include "src/runtime/kv_cache.h"
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <vector>
#include <string>
#include <cstdint>

namespace engine {

// GPU model weights: all FP16 on device
struct ModelWeightsGPU {
    DeviceBuffer<half> embed_tokens; // [vocab_size * hidden_size]
    DeviceBuffer<half> final_norm;   // [hidden_size]
    // lm_head is tied to embed_tokens (shares same buffer)

    struct LayerGPU {
        DeviceBuffer<half> input_layernorm;          // [hidden_size]
        DeviceBuffer<half> post_attention_layernorm; // [hidden_size]

        DeviceBuffer<half> q_proj_weight; // [hidden_size * hidden_size]
        DeviceBuffer<half> q_proj_bias;   // [hidden_size]
        DeviceBuffer<half> k_proj_weight; // [kv_dim * hidden_size]
        DeviceBuffer<half> k_proj_bias;   // [kv_dim]
        DeviceBuffer<half> v_proj_weight; // [kv_dim * hidden_size]
        DeviceBuffer<half> v_proj_bias;   // [kv_dim]
        DeviceBuffer<half> o_proj_weight; // [hidden_size * hidden_size]

        DeviceBuffer<half> gate_proj_weight; // [intermediate_size * hidden_size]
        DeviceBuffer<half> up_proj_weight;   // [intermediate_size * hidden_size]
        DeviceBuffer<half> down_proj_weight; // [hidden_size * intermediate_size]
    };

    std::vector<LayerGPU> layers;

    // Convert BF16 weights from safetensors mmap to FP16 on GPU
    bool init_from_weights(const ModelConfig& config, const ModelWeights& weights);
};

// Pre-allocated activation buffers on GPU (allocated once for max_seq_len)
struct ActivationBuffersGPU {
    DeviceBuffer<half> hidden_states; // [max_seq * hidden_size]
    DeviceBuffer<half> norm_out;      // [max_seq * hidden_size]
    DeviceBuffer<half> q;             // [max_seq * hidden_size]
    DeviceBuffer<half> k;             // [max_seq * kv_dim]
    DeviceBuffer<half> v;             // [max_seq * kv_dim]
    DeviceBuffer<half> attn_out;      // [max_seq * hidden_size]
    DeviceBuffer<half> proj_out;      // [max_seq * hidden_size]
    DeviceBuffer<half> gate;          // [max_seq * intermediate_size]
    DeviceBuffer<half> up;            // [max_seq * intermediate_size]
    DeviceBuffer<half> act;           // [max_seq * intermediate_size]
    DeviceBuffer<half> mlp_out;       // [max_seq * hidden_size]
    DeviceBuffer<float> logits;       // [max_seq * vocab_size] in FP32
    DeviceBuffer<int64_t> token_ids_d;// [max_seq]
    DeviceBuffer<int> decode_seq_len; // [1] sequence length for decode
    DeviceBuffer<float> split_kv_buf; // [splits * heads * head_dim]

    int64_t max_seq = 0;

    void allocate(const ModelConfig& config, int64_t max_seq_len);
};

// Per-kernel timing record for profiling
struct KernelRecord {
    std::string name;
    float ms;
};

// GPU forward pass returning logits as host float32 [seq_len * vocab_size]
// If dump_dir is non-empty, dumps same intermediate tensors as CPU path.
// If profile_records is non-null, records per-kernel timing.
std::vector<float> forward_gpu(
    const std::vector<int64_t>& token_ids,
    const ModelConfig& config,
    const ModelWeightsGPU& weights,
    ActivationBuffersGPU& buffers,
    cublasHandle_t cublas_handle,
    const std::string& dump_dir = "",
    std::vector<KernelRecord>* profile_records = nullptr
);

// Prefill: processes token_ids, appends K/V to kv_cache starting at pos 0,
// returns last-position logits as host float32 [vocab_size].
// d_last_logits: optional device buffer pointer to receive logits on device [vocab_size] in FP32.
std::vector<float> prefill_gpu(
    const std::vector<int64_t>& token_ids,
    const ModelConfig& config,
    const ModelWeightsGPU& weights,
    ActivationBuffersGPU& buffers,
    KVCache& kv_cache,
    cublasHandle_t cublas_handle,
    float* d_last_logits = nullptr,
    std::vector<KernelRecord>* profile_records = nullptr
);

// Single-token decode step:
// Processes single token_id at current_pos, appends K/V to kv_cache,
// returns last-position logits as host float32 [vocab_size].
// d_last_logits: optional device buffer pointer to receive logits on device [vocab_size] in FP32.
std::vector<float> decode_step_gpu(
    int64_t token_id,
    int current_pos,
    const ModelConfig& config,
    const ModelWeightsGPU& weights,
    ActivationBuffersGPU& buffers,
    KVCache& kv_cache,
    cublasHandle_t cublas_handle,
    float* d_last_logits = nullptr,
    std::vector<KernelRecord>* profile_records = nullptr
);

} // namespace engine

