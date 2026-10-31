#include "src/model/config.h"
#include "src/loader/safetensors.h"
#include "third_party/nlohmann/json.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace engine {

ModelConfig ModelConfig::from_file(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Failed to open config file: " + path);
    }
    nlohmann::json j;
    f >> j;

    ModelConfig cfg;
    cfg.model_type = j.value("model_type", "qwen2");
    cfg.vocab_size = j.value("vocab_size", 151936);
    cfg.hidden_size = j.value("hidden_size", 896);
    cfg.intermediate_size = j.value("intermediate_size", 4864);
    cfg.num_hidden_layers = j.value("num_hidden_layers", 24);
    cfg.num_attention_heads = j.value("num_attention_heads", 14);
    cfg.num_key_value_heads = j.value("num_key_value_heads", 2);
    cfg.head_dim = cfg.hidden_size / cfg.num_attention_heads;
    cfg.max_position_embeddings = j.value("max_position_embeddings", 32768);
    cfg.rms_norm_eps = j.value("rms_norm_eps", 1e-6f);
    cfg.rope_theta = j.value("rope_theta", 1000000.0f);
    cfg.tie_word_embeddings = j.value("tie_word_embeddings", true);
    cfg.torch_dtype = j.value("torch_dtype", "bfloat16");

    return cfg;
}

static bool check_tensor(const SafetensorsLoader& loader,
                         const std::string& name,
                         const std::vector<int64_t>& expected_shape,
                         const void*& out_ptr) {
    const TensorInfo* t = loader.get_tensor(name);
    if (!t) {
        std::cerr << "Missing required tensor: " << name << std::endl;
        return false;
    }
    if (t->shape != expected_shape) {
        std::cerr << "Shape mismatch for tensor " << name << ": expected [";
        for (size_t i = 0; i < expected_shape.size(); ++i) {
            std::cerr << (i > 0 ? ", " : "") << expected_shape[i];
        }
        std::cerr << "] but got [";
        for (size_t i = 0; i < t->shape.size(); ++i) {
            std::cerr << (i > 0 ? ", " : "") << t->shape[i];
        }
        std::cerr << "]" << std::endl;
        return false;
    }
    out_ptr = t->data;
    return true;
}

bool ModelWeights::load_and_validate(const ModelConfig& config, const SafetensorsLoader& loader) {
    // 1. Embedding
    if (!check_tensor(loader, "model.embed_tokens.weight", {config.vocab_size, config.hidden_size}, embed_tokens)) {
        return false;
    }

    // 2. Final norm
    if (!check_tensor(loader, "model.norm.weight", {config.hidden_size}, final_norm)) {
        return false;
    }

    // 3. Tied lm_head
    if (config.tie_word_embeddings) {
        lm_head = embed_tokens;
    } else {
        if (!check_tensor(loader, "lm_head.weight", {config.vocab_size, config.hidden_size}, lm_head)) {
            return false;
        }
    }

    // 4. Layers
    layers.resize(config.num_hidden_layers);
    int kv_dim = config.num_key_value_heads * config.head_dim;

    for (int i = 0; i < config.num_hidden_layers; ++i) {
        std::string prefix = "model.layers." + std::to_string(i) + ".";
        LayerWeights& lw = layers[i];

        // Norms
        if (!check_tensor(loader, prefix + "input_layernorm.weight", {config.hidden_size}, lw.input_layernorm)) return false;
        if (!check_tensor(loader, prefix + "post_attention_layernorm.weight", {config.hidden_size}, lw.post_attention_layernorm)) return false;

        // Attention weights and biases
        if (!check_tensor(loader, prefix + "self_attn.q_proj.weight", {config.hidden_size, config.hidden_size}, lw.q_proj_weight)) return false;
        if (!check_tensor(loader, prefix + "self_attn.q_proj.bias", {config.hidden_size}, lw.q_proj_bias)) return false;

        if (!check_tensor(loader, prefix + "self_attn.k_proj.weight", {kv_dim, config.hidden_size}, lw.k_proj_weight)) return false;
        if (!check_tensor(loader, prefix + "self_attn.k_proj.bias", {kv_dim}, lw.k_proj_bias)) return false;

        if (!check_tensor(loader, prefix + "self_attn.v_proj.weight", {kv_dim, config.hidden_size}, lw.v_proj_weight)) return false;
        if (!check_tensor(loader, prefix + "self_attn.v_proj.bias", {kv_dim}, lw.v_proj_bias)) return false;

        if (!check_tensor(loader, prefix + "self_attn.o_proj.weight", {config.hidden_size, config.hidden_size}, lw.o_proj_weight)) return false;

        // MLP weights
        if (!check_tensor(loader, prefix + "mlp.gate_proj.weight", {config.intermediate_size, config.hidden_size}, lw.gate_proj_weight)) return false;
        if (!check_tensor(loader, prefix + "mlp.up_proj.weight", {config.intermediate_size, config.hidden_size}, lw.up_proj_weight)) return false;
        if (!check_tensor(loader, prefix + "mlp.down_proj.weight", {config.hidden_size, config.intermediate_size}, lw.down_proj_weight)) return false;
    }

    return true;
}

} // namespace engine
