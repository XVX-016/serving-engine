#pragma once

#include "src/model/config.h"
#include <vector>
#include <string>
#include <cstdint>

namespace engine {

// Storage for converted FP32 model weights on host CPU
struct ModelWeightsFP32 {
    std::vector<float> embed_tokens; // [vocab_size * hidden_size]
    std::vector<float> final_norm;   // [hidden_size]

    struct LayerFP32 {
        std::vector<float> input_layernorm;          // [hidden_size]
        std::vector<float> post_attention_layernorm; // [hidden_size]

        // Attention weights and biases
        std::vector<float> q_proj_weight; // [hidden_size, hidden_size]
        std::vector<float> q_proj_bias;   // [hidden_size]
        std::vector<float> k_proj_weight; // [kv_dim, hidden_size]
        std::vector<float> k_proj_bias;   // [kv_dim]
        std::vector<float> v_proj_weight; // [kv_dim, hidden_size]
        std::vector<float> v_proj_bias;   // [kv_dim]
        std::vector<float> o_proj_weight; // [hidden_size, hidden_size]

        // MLP
        std::vector<float> gate_proj_weight; // [intermediate_size, hidden_size]
        std::vector<float> up_proj_weight;   // [intermediate_size, hidden_size]
        std::vector<float> down_proj_weight; // [hidden_size, intermediate_size]
    };

    std::vector<LayerFP32> layers;

    // Convert BF16 ModelWeights from Safetensors to FP32 in CPU memory
    bool init_from_weights(const ModelConfig& config, const ModelWeights& weights);
};

// Full CPU Forward Pass over a sequence of token IDs returning logits [seq_len, vocab_size]
// If dump_dir is non-empty, exports:
//   - embedding.npy [1, seq_len, hidden_size]
//   - layer_{0,1,23}_post_attn.npy [1, seq_len, hidden_size]
//   - layer_{0,1,23}_post_mlp.npy [1, seq_len, hidden_size]
//   - final_norm.npy [1, seq_len, hidden_size]
//   - logits.npy [1, seq_len, vocab_size]
std::vector<float> forward_cpu(
    const std::vector<int64_t>& token_ids,
    const ModelConfig& config,
    const ModelWeightsFP32& weights,
    const std::string& dump_dir = ""
);

} // namespace engine

