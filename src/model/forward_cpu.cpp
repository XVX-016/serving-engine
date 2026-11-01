#include "src/model/forward_cpu.h"
#include "src/cpu/ops.h"
#include "src/cpu/npy.h"
#include "src/loader/safetensors.h"

#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <cstring>

namespace fs = std::filesystem;

namespace engine {

static void copy_bf16_to_fp32(const void* src, std::vector<float>& dst, size_t count) {
    dst.resize(count);
    SafetensorsLoader::bf16_to_fp32(static_cast<const uint16_t*>(src), dst.data(), count);
}

bool ModelWeightsFP32::init_from_weights(const ModelConfig& config, const ModelWeights& weights) {
    try {
        // Embeddings and final norm
        copy_bf16_to_fp32(weights.embed_tokens, embed_tokens, static_cast<size_t>(config.vocab_size) * config.hidden_size);
        copy_bf16_to_fp32(weights.final_norm, final_norm, config.hidden_size);

        int kv_dim = config.num_key_value_heads * config.head_dim;
        layers.resize(config.num_hidden_layers);

        for (int i = 0; i < config.num_hidden_layers; ++i) {
            const auto& lw = weights.layers[i];
            auto& lfp = layers[i];

            copy_bf16_to_fp32(lw.input_layernorm, lfp.input_layernorm, config.hidden_size);
            copy_bf16_to_fp32(lw.post_attention_layernorm, lfp.post_attention_layernorm, config.hidden_size);

            copy_bf16_to_fp32(lw.q_proj_weight, lfp.q_proj_weight, static_cast<size_t>(config.hidden_size) * config.hidden_size);
            copy_bf16_to_fp32(lw.q_proj_bias, lfp.q_proj_bias, config.hidden_size);

            copy_bf16_to_fp32(lw.k_proj_weight, lfp.k_proj_weight, static_cast<size_t>(kv_dim) * config.hidden_size);
            copy_bf16_to_fp32(lw.k_proj_bias, lfp.k_proj_bias, kv_dim);

            copy_bf16_to_fp32(lw.v_proj_weight, lfp.v_proj_weight, static_cast<size_t>(kv_dim) * config.hidden_size);
            copy_bf16_to_fp32(lw.v_proj_bias, lfp.v_proj_bias, kv_dim);

            copy_bf16_to_fp32(lw.o_proj_weight, lfp.o_proj_weight, static_cast<size_t>(config.hidden_size) * config.hidden_size);

            copy_bf16_to_fp32(lw.gate_proj_weight, lfp.gate_proj_weight, static_cast<size_t>(config.intermediate_size) * config.hidden_size);
            copy_bf16_to_fp32(lw.up_proj_weight, lfp.up_proj_weight, static_cast<size_t>(config.intermediate_size) * config.hidden_size);
            copy_bf16_to_fp32(lw.down_proj_weight, lfp.down_proj_weight, static_cast<size_t>(config.hidden_size) * config.intermediate_size);
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error in init_from_weights: " << e.what() << std::endl;
        return false;
    }
}

std::vector<float> forward_cpu(
    const std::vector<int64_t>& token_ids,
    const ModelConfig& config,
    const ModelWeightsFP32& weights,
    const std::string& dump_dir
) {
    int64_t S = static_cast<int64_t>(token_ids.size());
    int64_t H = config.hidden_size;
    int64_t num_q_heads = config.num_attention_heads;
    int64_t num_kv_heads = config.num_key_value_heads;
    int64_t head_dim = config.head_dim;
    int64_t kv_dim = num_kv_heads * head_dim;
    int64_t inter_dim = config.intermediate_size;
    int64_t V = config.vocab_size;

    bool do_dump = !dump_dir.empty();
    if (do_dump) {
        fs::create_directories(dump_dir);
    }

    // 1. Embedding lookup: [S, H]
    std::vector<float> hidden_states(S * H);
    cpu::embedding_lookup(token_ids.data(), weights.embed_tokens.data(), hidden_states.data(), S, H);

    if (do_dump) {
        NpyWriter::write_f32(dump_dir + "/embedding.npy", hidden_states.data(), {1, S, H});
    }

    // Allocate layer buffers once
    std::vector<float> norm_out(S * H);
    std::vector<float> q(S * H);
    std::vector<float> k(S * kv_dim);
    std::vector<float> v(S * kv_dim);
    std::vector<float> attn_out(S * H);
    std::vector<float> proj_out(S * H);

    std::vector<float> gate(S * inter_dim);
    std::vector<float> up(S * inter_dim);
    std::vector<float> act(S * inter_dim);
    std::vector<float> mlp_out(S * H);

    for (int layer_idx = 0; layer_idx < config.num_hidden_layers; ++layer_idx) {
        const auto& lw = weights.layers[layer_idx];

        // 2a. Input RMSNorm: x_norm = rmsnorm(hidden_states)
        cpu::rmsnorm(hidden_states.data(), lw.input_layernorm.data(), norm_out.data(), S, H, config.rms_norm_eps);

        // 2b. Q, K, V projections:
        // q = x_norm * W_q^T + b_q: [S, H] * [H, H]^T -> [S, H]
        cpu::matmul_transB_blocked(norm_out.data(), lw.q_proj_weight.data(), q.data(), S, H, H);
        cpu::bias_add(q.data(), lw.q_proj_bias.data(), q.data(), S, H);

        // k = x_norm * W_k^T + b_k: [S, H] * [kv_dim, H]^T -> [S, kv_dim]
        cpu::matmul_transB_blocked(norm_out.data(), lw.k_proj_weight.data(), k.data(), S, H, kv_dim);
        cpu::bias_add(k.data(), lw.k_proj_bias.data(), k.data(), S, kv_dim);

        // v = x_norm * W_v^T + b_v: [S, H] * [kv_dim, H]^T -> [S, kv_dim]
        cpu::matmul_transB_blocked(norm_out.data(), lw.v_proj_weight.data(), v.data(), S, H, kv_dim);
        cpu::bias_add(v.data(), lw.v_proj_bias.data(), v.data(), S, kv_dim);

        // 2c. RoPE: apply in-place to q and k
        cpu::apply_rope(q.data(), k.data(), S, num_q_heads, num_kv_heads, head_dim, 0, config.rope_theta);

        // 2d. Causal Grouped-Query Attention:
        cpu::causal_gqa(q.data(), k.data(), v.data(), attn_out.data(), S, num_q_heads, num_kv_heads, head_dim);

        // 2e. Output projection: proj_out = attn_out * W_o^T (bias is false)
        cpu::matmul_transB_blocked(attn_out.data(), lw.o_proj_weight.data(), proj_out.data(), S, H, H);

        // 2f. Residual add: hidden_states = hidden_states + proj_out
        cpu::residual_add(hidden_states.data(), proj_out.data(), hidden_states.data(), S * H);

        if (do_dump && (layer_idx == 0 || layer_idx == 1 || layer_idx == 23)) {
            NpyWriter::write_f32(dump_dir + "/layer_" + std::to_string(layer_idx) + "_post_attn.npy", hidden_states.data(), {1, S, H});
        }

        // 3a. Post-attention layernorm
        cpu::rmsnorm(hidden_states.data(), lw.post_attention_layernorm.data(), norm_out.data(), S, H, config.rms_norm_eps);

        // 3b. MLP gate and up projections
        // gate = norm_out * W_gate^T: [S, H] * [inter_dim, H]^T -> [S, inter_dim]
        cpu::matmul_transB_blocked(norm_out.data(), lw.gate_proj_weight.data(), gate.data(), S, H, inter_dim);

        // up = norm_out * W_up^T: [S, H] * [inter_dim, H]^T -> [S, inter_dim]
        cpu::matmul_transB_blocked(norm_out.data(), lw.up_proj_weight.data(), up.data(), S, H, inter_dim);

        // 3c. SwiGLU: act = silu(gate) * up
        cpu::swiglu(gate.data(), up.data(), act.data(), S * inter_dim);

        // 3d. Down projection: mlp_out = act * W_down^T: [S, inter_dim] * [H, inter_dim]^T -> [S, H]
        cpu::matmul_transB_blocked(act.data(), lw.down_proj_weight.data(), mlp_out.data(), S, inter_dim, H);

        // 3e. Residual add: hidden_states = hidden_states + mlp_out
        cpu::residual_add(hidden_states.data(), mlp_out.data(), hidden_states.data(), S * H);

        if (do_dump && (layer_idx == 0 || layer_idx == 1 || layer_idx == 23)) {
            NpyWriter::write_f32(dump_dir + "/layer_" + std::to_string(layer_idx) + "_post_mlp.npy", hidden_states.data(), {1, S, H});
        }
    }

    // 4. Final RMSNorm
    cpu::rmsnorm(hidden_states.data(), weights.final_norm.data(), norm_out.data(), S, H, config.rms_norm_eps);

    if (do_dump) {
        NpyWriter::write_f32(dump_dir + "/final_norm.npy", norm_out.data(), {1, S, H});
    }

    // 5. LM Head: logits = norm_out * W_embed^T: [S, H] * [V, H]^T -> [S, V]
    std::vector<float> logits(S * V);
    cpu::matmul_transB_blocked(norm_out.data(), weights.embed_tokens.data(), logits.data(), S, H, V);

    if (do_dump) {
        NpyWriter::write_f32(dump_dir + "/logits.npy", logits.data(), {1, S, V});
    }

    return logits;
}

} // namespace engine

