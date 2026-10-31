#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace engine {

struct ModelConfig {
    std::string model_type = "qwen2";
    int vocab_size = 151936;
    int hidden_size = 896;
    int intermediate_size = 4864;
    int num_hidden_layers = 24;
    int num_attention_heads = 14;
    int num_key_value_heads = 2;
    int head_dim = 64; // hidden_size / num_attention_heads = 896 / 14 = 64
    int max_position_embeddings = 32768;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
    bool tie_word_embeddings = true;
    std::string torch_dtype = "bfloat16";

    static ModelConfig from_file(const std::string& path);
};

struct LayerWeights {
    // Attention
    const void* input_layernorm = nullptr; // [hidden_size]
    const void* q_proj_weight = nullptr;   // [hidden_size, hidden_size] = [896, 896]
    const void* q_proj_bias = nullptr;     // [hidden_size] = [896]
    const void* k_proj_weight = nullptr;   // [num_kv_heads * head_dim, hidden_size] = [128, 896]
    const void* k_proj_bias = nullptr;     // [128]
    const void* v_proj_weight = nullptr;   // [num_kv_heads * head_dim, hidden_size] = [128, 896]
    const void* v_proj_bias = nullptr;     // [128]
    const void* o_proj_weight = nullptr;   // [hidden_size, hidden_size] = [896, 896]

    // MLP
    const void* post_attention_layernorm = nullptr; // [hidden_size]
    const void* gate_proj_weight = nullptr; // [intermediate_size, hidden_size] = [4864, 896]
    const void* up_proj_weight = nullptr;   // [intermediate_size, hidden_size] = [4864, 896]
    const void* down_proj_weight = nullptr; // [hidden_size, intermediate_size] = [896, 4864]
};

class SafetensorsLoader;

struct ModelWeights {
    const void* embed_tokens = nullptr; // [vocab_size, hidden_size] = [151936, 896]
    std::vector<LayerWeights> layers;
    const void* final_norm = nullptr;   // [hidden_size] = [896]
    const void* lm_head = nullptr;      // tied to embed_tokens

    // Validate that all expected tensors exist, have correct shapes, and bind to pointers
    bool load_and_validate(const ModelConfig& config, const SafetensorsLoader& loader);
};

} // namespace engine
