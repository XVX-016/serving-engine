#include "src/model/forward_gpu.h"
#include "src/cuda/kernels.h"
#include "src/cpu/npy.h"
#include "src/loader/safetensors.h"

#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <cstring>
#include <algorithm>

namespace fs = std::filesystem;

namespace engine {

// ---- Weight upload: BF16 host -> FP16 device ----

static void upload_bf16_to_fp16(const void* bf16_host, DeviceBuffer<half>& dst, size_t count) {
    // Convert BF16 -> FP16 on host, then upload to device
    std::vector<uint16_t> fp16_host(count);
    SafetensorsLoader::bf16_to_f16(
        static_cast<const uint16_t*>(bf16_host),
        fp16_host.data(),
        count
    );
    dst.allocate(count);
    dst.copy_from_host(reinterpret_cast<const half*>(fp16_host.data()), count);
}

bool ModelWeightsGPU::init_from_weights(const ModelConfig& config, const ModelWeights& weights) {
    try {
        int kv_dim = config.num_key_value_heads * config.head_dim;

        upload_bf16_to_fp16(weights.embed_tokens, embed_tokens,
            static_cast<size_t>(config.vocab_size) * config.hidden_size);
        upload_bf16_to_fp16(weights.final_norm, final_norm, config.hidden_size);

        layers.resize(config.num_hidden_layers);
        for (int i = 0; i < config.num_hidden_layers; ++i) {
            const auto& lw = weights.layers[i];
            auto& lg = layers[i];

            upload_bf16_to_fp16(lw.input_layernorm, lg.input_layernorm, config.hidden_size);
            upload_bf16_to_fp16(lw.post_attention_layernorm, lg.post_attention_layernorm, config.hidden_size);

            upload_bf16_to_fp16(lw.q_proj_weight, lg.q_proj_weight,
                static_cast<size_t>(config.hidden_size) * config.hidden_size);
            upload_bf16_to_fp16(lw.q_proj_bias, lg.q_proj_bias, config.hidden_size);

            upload_bf16_to_fp16(lw.k_proj_weight, lg.k_proj_weight,
                static_cast<size_t>(kv_dim) * config.hidden_size);
            upload_bf16_to_fp16(lw.k_proj_bias, lg.k_proj_bias, kv_dim);

            upload_bf16_to_fp16(lw.v_proj_weight, lg.v_proj_weight,
                static_cast<size_t>(kv_dim) * config.hidden_size);
            upload_bf16_to_fp16(lw.v_proj_bias, lg.v_proj_bias, kv_dim);

            upload_bf16_to_fp16(lw.o_proj_weight, lg.o_proj_weight,
                static_cast<size_t>(config.hidden_size) * config.hidden_size);

            upload_bf16_to_fp16(lw.gate_proj_weight, lg.gate_proj_weight,
                static_cast<size_t>(config.intermediate_size) * config.hidden_size);
            upload_bf16_to_fp16(lw.up_proj_weight, lg.up_proj_weight,
                static_cast<size_t>(config.intermediate_size) * config.hidden_size);
            upload_bf16_to_fp16(lw.down_proj_weight, lg.down_proj_weight,
                static_cast<size_t>(config.hidden_size) * config.intermediate_size);
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error in ModelWeightsGPU::init_from_weights: " << e.what() << std::endl;
        return false;
    }
}

// ---- Activation buffer allocation ----

void ActivationBuffersGPU::allocate(const ModelConfig& config, int64_t max_seq_len) {
    max_seq = max_seq_len;
    int64_t H = config.hidden_size;
    int64_t kv_dim = config.num_key_value_heads * config.head_dim;
    int64_t inter = config.intermediate_size;
    int64_t V = config.vocab_size;

    hidden_states.allocate(max_seq * H);
    norm_out.allocate(max_seq * H);
    q.allocate(max_seq * H);
    k.allocate(max_seq * kv_dim);
    v.allocate(max_seq * kv_dim);
    attn_out.allocate(max_seq * H);
    proj_out.allocate(max_seq * H);
    gate.allocate(max_seq * inter);
    up.allocate(max_seq * inter);
    act.allocate(max_seq * inter);
    mlp_out.allocate(max_seq * H);
    logits.allocate(max_seq * V);
    token_ids_d.allocate(max_seq);
}

// ---- Helper: download FP16 device to host FP32 for npy dump ----

static void dump_half_to_npy(const half* d_ptr, int64_t count,
                              const std::string& path,
                              const std::vector<int64_t>& shape) {
    std::vector<half> host_h(count);
    CUDA_CHECK(cudaMemcpy(host_h.data(), d_ptr, count * sizeof(half), cudaMemcpyDeviceToHost));
    std::vector<float> host_f(count);
    for (int64_t i = 0; i < count; ++i) {
        host_f[i] = __half2float(host_h[i]);
    }
    NpyWriter::write_f32(path, host_f.data(), shape);
}

// ---- GPU Forward Pass ----

std::vector<float> forward_gpu(
    const std::vector<int64_t>& token_ids,
    const ModelConfig& config,
    const ModelWeightsGPU& weights,
    ActivationBuffersGPU& buffers,
    cublasHandle_t cublas_handle,
    const std::string& dump_dir,
    std::vector<KernelRecord>* profile_records
) {
    int64_t S = static_cast<int64_t>(token_ids.size());
    int64_t H = config.hidden_size;
    int64_t num_q_heads = config.num_attention_heads;
    int64_t num_kv_heads = config.num_key_value_heads;
    int64_t head_dim = config.head_dim;
    int64_t kv_dim = num_kv_heads * head_dim;
    int64_t inter_dim = config.intermediate_size;
    int64_t V = config.vocab_size;

    if (S > buffers.max_seq) {
        throw std::runtime_error("forward_gpu: seq_len " + std::to_string(S) +
                                 " exceeds max_seq " + std::to_string(buffers.max_seq));
    }

    bool do_dump = !dump_dir.empty();
    if (do_dump) {
        fs::create_directories(dump_dir);
    }

    // Macro for optional per-kernel timing
    CudaTimer timer;
    auto timed_begin = [&](const std::string& name) {
        if (profile_records) {
            timer.start();
        }
    };
    auto timed_end = [&](const std::string& name) {
        if (profile_records) {
            timer.stop();
            float ms = timer.elapsed_ms();
            profile_records->push_back({name, ms});
        }
    };

    // Upload token IDs
    buffers.token_ids_d.copy_from_host(token_ids.data(), S);

    // 1. Embedding lookup
    timed_begin("embedding");
    cuda::embedding_lookup(buffers.token_ids_d.data(), weights.embed_tokens.data(),
                           buffers.hidden_states.data(), S, H);
    timed_end("embedding");

    if (do_dump) {
        dump_half_to_npy(buffers.hidden_states.data(), S * H,
                        dump_dir + "/embedding.npy", {1, S, H});
    }

    for (int layer_idx = 0; layer_idx < config.num_hidden_layers; ++layer_idx) {
        const auto& lw = weights.layers[layer_idx];
        std::string lp = "layer" + std::to_string(layer_idx) + ".";

        // 2a. Input RMSNorm
        timed_begin(lp + "input_rmsnorm");
        cuda::rmsnorm(buffers.hidden_states.data(), lw.input_layernorm.data(),
                      buffers.norm_out.data(), S, H, config.rms_norm_eps);
        timed_end(lp + "input_rmsnorm");

        // 2b. Q projection: [S, H] * [H, H]^T -> [S, H]
        timed_begin(lp + "q_proj");
        cuda::matmul_transB(cublas_handle, buffers.norm_out.data(),
                           lw.q_proj_weight.data(), buffers.q.data(),
                           static_cast<int>(S), static_cast<int>(H), static_cast<int>(H));
        timed_end(lp + "q_proj");

        timed_begin(lp + "q_bias");
        cuda::bias_add(buffers.q.data(), lw.q_proj_bias.data(),
                      buffers.q.data(), S, H);
        timed_end(lp + "q_bias");

        // K projection: [S, H] * [kv_dim, H]^T -> [S, kv_dim]
        timed_begin(lp + "k_proj");
        cuda::matmul_transB(cublas_handle, buffers.norm_out.data(),
                           lw.k_proj_weight.data(), buffers.k.data(),
                           static_cast<int>(S), static_cast<int>(H), static_cast<int>(kv_dim));
        timed_end(lp + "k_proj");

        timed_begin(lp + "k_bias");
        cuda::bias_add(buffers.k.data(), lw.k_proj_bias.data(),
                      buffers.k.data(), S, kv_dim);
        timed_end(lp + "k_bias");

        // V projection: [S, H] * [kv_dim, H]^T -> [S, kv_dim]
        timed_begin(lp + "v_proj");
        cuda::matmul_transB(cublas_handle, buffers.norm_out.data(),
                           lw.v_proj_weight.data(), buffers.v.data(),
                           static_cast<int>(S), static_cast<int>(H), static_cast<int>(kv_dim));
        timed_end(lp + "v_proj");

        timed_begin(lp + "v_bias");
        cuda::bias_add(buffers.v.data(), lw.v_proj_bias.data(),
                      buffers.v.data(), S, kv_dim);
        timed_end(lp + "v_bias");

        // 2c. RoPE
        timed_begin(lp + "rope");
        cuda::apply_rope(buffers.q.data(), buffers.k.data(),
                        S, num_q_heads, num_kv_heads, head_dim, 0, config.rope_theta);
        timed_end(lp + "rope");

        // 2d. Causal GQA
        timed_begin(lp + "attention");
        cuda::causal_gqa(buffers.q.data(), buffers.k.data(), buffers.v.data(),
                        buffers.attn_out.data(), S, num_q_heads, num_kv_heads, head_dim);
        timed_end(lp + "attention");

        // 2e. Output projection: [S, H] * [H, H]^T -> [S, H]
        timed_begin(lp + "o_proj");
        cuda::matmul_transB(cublas_handle, buffers.attn_out.data(),
                           lw.o_proj_weight.data(), buffers.proj_out.data(),
                           static_cast<int>(S), static_cast<int>(H), static_cast<int>(H));
        timed_end(lp + "o_proj");

        // 2f. Residual add
        timed_begin(lp + "attn_residual");
        cuda::residual_add(buffers.hidden_states.data(), buffers.proj_out.data(),
                          buffers.hidden_states.data(), S * H);
        timed_end(lp + "attn_residual");

        if (do_dump && (layer_idx == 0 || layer_idx == 1 || layer_idx == 23)) {
            dump_half_to_npy(buffers.hidden_states.data(), S * H,
                dump_dir + "/layer_" + std::to_string(layer_idx) + "_post_attn.npy", {1, S, H});
        }

        // 3a. Post-attention layernorm
        timed_begin(lp + "post_rmsnorm");
        cuda::rmsnorm(buffers.hidden_states.data(), lw.post_attention_layernorm.data(),
                      buffers.norm_out.data(), S, H, config.rms_norm_eps);
        timed_end(lp + "post_rmsnorm");

        // 3b. Gate projection: [S, H] * [inter_dim, H]^T -> [S, inter_dim]
        timed_begin(lp + "gate_proj");
        cuda::matmul_transB(cublas_handle, buffers.norm_out.data(),
                           lw.gate_proj_weight.data(), buffers.gate.data(),
                           static_cast<int>(S), static_cast<int>(H), static_cast<int>(inter_dim));
        timed_end(lp + "gate_proj");

        // Up projection: [S, H] * [inter_dim, H]^T -> [S, inter_dim]
        timed_begin(lp + "up_proj");
        cuda::matmul_transB(cublas_handle, buffers.norm_out.data(),
                           lw.up_proj_weight.data(), buffers.up.data(),
                           static_cast<int>(S), static_cast<int>(H), static_cast<int>(inter_dim));
        timed_end(lp + "up_proj");

        // 3c. SwiGLU
        timed_begin(lp + "swiglu");
        cuda::swiglu(buffers.gate.data(), buffers.up.data(),
                    buffers.act.data(), S * inter_dim);
        timed_end(lp + "swiglu");

        // 3d. Down projection: [S, inter_dim] * [H, inter_dim]^T -> [S, H]
        timed_begin(lp + "down_proj");
        cuda::matmul_transB(cublas_handle, buffers.act.data(),
                           lw.down_proj_weight.data(), buffers.mlp_out.data(),
                           static_cast<int>(S), static_cast<int>(inter_dim), static_cast<int>(H));
        timed_end(lp + "down_proj");

        // 3e. Residual add
        timed_begin(lp + "mlp_residual");
        cuda::residual_add(buffers.hidden_states.data(), buffers.mlp_out.data(),
                          buffers.hidden_states.data(), S * H);
        timed_end(lp + "mlp_residual");

        if (do_dump && (layer_idx == 0 || layer_idx == 1 || layer_idx == 23)) {
            dump_half_to_npy(buffers.hidden_states.data(), S * H,
                dump_dir + "/layer_" + std::to_string(layer_idx) + "_post_mlp.npy", {1, S, H});
        }
    }

    // 4. Final RMSNorm
    timed_begin("final_rmsnorm");
    cuda::rmsnorm(buffers.hidden_states.data(), weights.final_norm.data(),
                  buffers.norm_out.data(), S, H, config.rms_norm_eps);
    timed_end("final_rmsnorm");

    if (do_dump) {
        dump_half_to_npy(buffers.norm_out.data(), S * H,
                        dump_dir + "/final_norm.npy", {1, S, H});
    }

    // 5. LM Head: logits = norm_out * embed_tokens^T (tied weights)
    // [S, H] * [V, H]^T -> [S, V]
    timed_begin("lm_head");
    cuda::matmul_transB(cublas_handle, buffers.norm_out.data(),
                       weights.embed_tokens.data(), buffers.logits.data(),
                       static_cast<int>(S), static_cast<int>(H), static_cast<int>(V));
    timed_end("lm_head");

    // Download logits to host as FP32
    std::vector<half> logits_h(S * V);
    CUDA_CHECK(cudaMemcpy(logits_h.data(), buffers.logits.data(),
               S * V * sizeof(half), cudaMemcpyDeviceToHost));

    std::vector<float> logits_f(S * V);
    for (int64_t i = 0; i < S * V; ++i) {
        logits_f[i] = __half2float(logits_h[i]);
    }

    if (do_dump) {
        NpyWriter::write_f32(dump_dir + "/logits.npy", logits_f.data(), {1, S, V});
    }

    return logits_f;
}

// ---- Prefill Implementation ----
std::vector<float> prefill_gpu(
    const std::vector<int64_t>& token_ids,
    const ModelConfig& config,
    const ModelWeightsGPU& weights,
    ActivationBuffersGPU& buffers,
    KVCache& kv_cache,
    cublasHandle_t cublas_handle,
    half* d_last_logits,
    std::vector<KernelRecord>* profile_records
) {
    int64_t S = static_cast<int64_t>(token_ids.size());
    int64_t H = config.hidden_size;
    int64_t num_q_heads = config.num_attention_heads;
    int64_t num_kv_heads = config.num_key_value_heads;
    int64_t head_dim = config.head_dim;
    int64_t kv_dim = num_kv_heads * head_dim;
    int64_t inter_dim = config.intermediate_size;
    int64_t V = config.vocab_size;

    if (S > buffers.max_seq) {
        throw std::runtime_error("prefill_gpu: seq_len " + std::to_string(S) +
                                 " exceeds max_seq " + std::to_string(buffers.max_seq));
    }

    CudaTimer timer;
    auto timed_begin = [&](const std::string& name) {
        if (profile_records) timer.start();
    };
    auto timed_end = [&](const std::string& name) {
        if (profile_records) {
            timer.stop();
            profile_records->push_back({name, timer.elapsed_ms()});
        }
    };

    // Upload token IDs
    buffers.token_ids_d.copy_from_host(token_ids.data(), S);

    // 1. Embedding lookup
    timed_begin("embedding");
    cuda::embedding_lookup(buffers.token_ids_d.data(), weights.embed_tokens.data(),
                           buffers.hidden_states.data(), S, H);
    timed_end("embedding");

    for (int layer_idx = 0; layer_idx < config.num_hidden_layers; ++layer_idx) {
        const auto& lw = weights.layers[layer_idx];
        std::string lp = "layer" + std::to_string(layer_idx) + ".";

        // 2a. Input RMSNorm
        timed_begin(lp + "input_rmsnorm");
        cuda::rmsnorm(buffers.hidden_states.data(), lw.input_layernorm.data(),
                      buffers.norm_out.data(), S, H, config.rms_norm_eps);
        timed_end(lp + "input_rmsnorm");

        // 2b. Q/K/V projections
        timed_begin(lp + "q_proj");
        cuda::matmul_transB(cublas_handle, buffers.norm_out.data(),
                           lw.q_proj_weight.data(), buffers.q.data(),
                           static_cast<int>(S), static_cast<int>(H), static_cast<int>(H));
        timed_end(lp + "q_proj");

        timed_begin(lp + "q_bias");
        cuda::bias_add(buffers.q.data(), lw.q_proj_bias.data(),
                      buffers.q.data(), S, H);
        timed_end(lp + "q_bias");

        timed_begin(lp + "k_proj");
        cuda::matmul_transB(cublas_handle, buffers.norm_out.data(),
                           lw.k_proj_weight.data(), buffers.k.data(),
                           static_cast<int>(S), static_cast<int>(H), static_cast<int>(kv_dim));
        timed_end(lp + "k_proj");

        timed_begin(lp + "k_bias");
        cuda::bias_add(buffers.k.data(), lw.k_proj_bias.data(),
                      buffers.k.data(), S, kv_dim);
        timed_end(lp + "k_bias");

        timed_begin(lp + "v_proj");
        cuda::matmul_transB(cublas_handle, buffers.norm_out.data(),
                           lw.v_proj_weight.data(), buffers.v.data(),
                           static_cast<int>(S), static_cast<int>(H), static_cast<int>(kv_dim));
        timed_end(lp + "v_proj");

        timed_begin(lp + "v_bias");
        cuda::bias_add(buffers.v.data(), lw.v_proj_bias.data(),
                      buffers.v.data(), S, kv_dim);
        timed_end(lp + "v_bias");

        // 2c. RoPE
        timed_begin(lp + "rope");
        cuda::apply_rope(buffers.q.data(), buffers.k.data(),
                        S, num_q_heads, num_kv_heads, head_dim, 0, config.rope_theta);
        timed_end(lp + "rope");

        // Append K and V to KV Cache for this layer
        timed_begin(lp + "cache_append");
        cuda::append_kv_cache(
            buffers.k.data(), buffers.v.data(),
            kv_cache.get_head_ptr(layer_idx, 0, 0),
            kv_cache.get_head_ptr(layer_idx, 1, 0),
            static_cast<int>(S), 0,
            static_cast<int>(num_kv_heads),
            kv_cache.max_seq(),
            static_cast<int>(head_dim)
        );
        timed_end(lp + "cache_append");

        // 2d. Causal GQA
        timed_begin(lp + "attention");
        cuda::causal_gqa(buffers.q.data(), buffers.k.data(), buffers.v.data(),
                        buffers.attn_out.data(), S, num_q_heads, num_kv_heads, head_dim);
        timed_end(lp + "attention");

        // 2e. Output projection
        timed_begin(lp + "o_proj");
        cuda::matmul_transB(cublas_handle, buffers.attn_out.data(),
                           lw.o_proj_weight.data(), buffers.proj_out.data(),
                           static_cast<int>(S), static_cast<int>(H), static_cast<int>(H));
        timed_end(lp + "o_proj");

        // 2f. Residual add
        timed_begin(lp + "attn_residual");
        cuda::residual_add(buffers.hidden_states.data(), buffers.proj_out.data(),
                          buffers.hidden_states.data(), S * H);
        timed_end(lp + "attn_residual");

        // 3a. Post-attention layernorm
        timed_begin(lp + "post_rmsnorm");
        cuda::rmsnorm(buffers.hidden_states.data(), lw.post_attention_layernorm.data(),
                      buffers.norm_out.data(), S, H, config.rms_norm_eps);
        timed_end(lp + "post_rmsnorm");

        // 3b. Gate & Up proj
        timed_begin(lp + "gate_proj");
        cuda::matmul_transB(cublas_handle, buffers.norm_out.data(),
                           lw.gate_proj_weight.data(), buffers.gate.data(),
                           static_cast<int>(S), static_cast<int>(H), static_cast<int>(inter_dim));
        timed_end(lp + "gate_proj");

        timed_begin(lp + "up_proj");
        cuda::matmul_transB(cublas_handle, buffers.norm_out.data(),
                           lw.up_proj_weight.data(), buffers.up.data(),
                           static_cast<int>(S), static_cast<int>(H), static_cast<int>(inter_dim));
        timed_end(lp + "up_proj");

        // 3c. SwiGLU
        timed_begin(lp + "swiglu");
        cuda::swiglu(buffers.gate.data(), buffers.up.data(),
                    buffers.act.data(), S * inter_dim);
        timed_end(lp + "swiglu");

        // 3d. Down projection
        timed_begin(lp + "down_proj");
        cuda::matmul_transB(cublas_handle, buffers.act.data(),
                           lw.down_proj_weight.data(), buffers.mlp_out.data(),
                           static_cast<int>(S), static_cast<int>(inter_dim), static_cast<int>(H));
        timed_end(lp + "down_proj");

        // 3e. Residual add
        timed_begin(lp + "mlp_residual");
        cuda::residual_add(buffers.hidden_states.data(), buffers.mlp_out.data(),
                          buffers.hidden_states.data(), S * H);
        timed_end(lp + "mlp_residual");
    }

    kv_cache.set_current_seq_len(static_cast<int>(S));

    // 4. Final RMSNorm on LAST position only
    const half* last_hidden = buffers.hidden_states.data() + (S - 1) * H;
    half* last_norm = buffers.norm_out.data(); // reuse norm_out[0..H-1]
    timed_begin("final_rmsnorm");
    cuda::rmsnorm(last_hidden, weights.final_norm.data(),
                  last_norm, 1, H, config.rms_norm_eps);
    timed_end("final_rmsnorm");

    // 5. LM Head on last position only: [1, H] * [V, H]^T -> [1, V]
    half* target_logits = d_last_logits ? d_last_logits : buffers.logits.data();
    timed_begin("lm_head");
    cuda::matmul_transB(cublas_handle, last_norm,
                       weights.embed_tokens.data(), target_logits,
                       1, static_cast<int>(H), static_cast<int>(V));
    timed_end("lm_head");

    // Download only last-position logits (1 * V)
    std::vector<half> logits_h(V);
    CUDA_CHECK(cudaMemcpy(logits_h.data(), target_logits,
               V * sizeof(half), cudaMemcpyDeviceToHost));

    std::vector<float> logits_f(V);
    for (int64_t i = 0; i < V; ++i) {
        logits_f[i] = __half2float(logits_h[i]);
    }
    return logits_f;
}

// ---- Decode Step Implementation ----
std::vector<float> decode_step_gpu(
    int64_t token_id,
    int current_pos,
    const ModelConfig& config,
    const ModelWeightsGPU& weights,
    ActivationBuffersGPU& buffers,
    KVCache& kv_cache,
    cublasHandle_t cublas_handle,
    half* d_last_logits,
    std::vector<KernelRecord>* profile_records
) {
    int64_t H = config.hidden_size;
    int64_t num_q_heads = config.num_attention_heads;
    int64_t num_kv_heads = config.num_key_value_heads;
    int64_t head_dim = config.head_dim;
    int64_t kv_dim = num_kv_heads * head_dim;
    int64_t inter_dim = config.intermediate_size;
    int64_t V = config.vocab_size;

    CudaTimer timer;
    auto timed_begin = [&](const std::string& name) {
        if (profile_records) timer.start();
    };
    auto timed_end = [&](const std::string& name) {
        if (profile_records) {
            timer.stop();
            profile_records->push_back({name, timer.elapsed_ms()});
        }
    };

    // Upload single token ID
    buffers.token_ids_d.copy_from_host(&token_id, 1);

    // 1. Embedding lookup (M=1)
    timed_begin("embedding");
    cuda::embedding_lookup(buffers.token_ids_d.data(), weights.embed_tokens.data(),
                           buffers.hidden_states.data(), 1, H);
    timed_end("embedding");

    for (int layer_idx = 0; layer_idx < config.num_hidden_layers; ++layer_idx) {
        const auto& lw = weights.layers[layer_idx];
        std::string lp = "layer" + std::to_string(layer_idx) + ".";

        // 2a. Input RMSNorm (M=1)
        timed_begin(lp + "input_rmsnorm");
        cuda::rmsnorm(buffers.hidden_states.data(), lw.input_layernorm.data(),
                      buffers.norm_out.data(), 1, H, config.rms_norm_eps);
        timed_end(lp + "input_rmsnorm");

        // 2b. Q/K/V projections (M=1)
        timed_begin(lp + "q_proj");
        cuda::matmul_transB(cublas_handle, buffers.norm_out.data(),
                           lw.q_proj_weight.data(), buffers.q.data(),
                           1, static_cast<int>(H), static_cast<int>(H));
        timed_end(lp + "q_proj");

        timed_begin(lp + "q_bias");
        cuda::bias_add(buffers.q.data(), lw.q_proj_bias.data(),
                      buffers.q.data(), 1, H);
        timed_end(lp + "q_bias");

        timed_begin(lp + "k_proj");
        cuda::matmul_transB(cublas_handle, buffers.norm_out.data(),
                           lw.k_proj_weight.data(), buffers.k.data(),
                           1, static_cast<int>(H), static_cast<int>(kv_dim));
        timed_end(lp + "k_proj");

        timed_begin(lp + "k_bias");
        cuda::bias_add(buffers.k.data(), lw.k_proj_bias.data(),
                      buffers.k.data(), 1, kv_dim);
        timed_end(lp + "k_bias");

        timed_begin(lp + "v_proj");
        cuda::matmul_transB(cublas_handle, buffers.norm_out.data(),
                           lw.v_proj_weight.data(), buffers.v.data(),
                           1, static_cast<int>(H), static_cast<int>(kv_dim));
        timed_end(lp + "v_proj");

        timed_begin(lp + "v_bias");
        cuda::bias_add(buffers.v.data(), lw.v_proj_bias.data(),
                      buffers.v.data(), 1, kv_dim);
        timed_end(lp + "v_bias");

        // 2c. RoPE with absolute position current_pos
        timed_begin(lp + "rope");
        cuda::apply_rope(buffers.q.data(), buffers.k.data(),
                        1, num_q_heads, num_kv_heads, head_dim, current_pos, config.rope_theta);
        timed_end(lp + "rope");

        // Append single key and value to cache at current_pos
        timed_begin(lp + "cache_append");
        cuda::append_kv_cache(
            buffers.k.data(), buffers.v.data(),
            kv_cache.get_head_ptr(layer_idx, 0, 0),
            kv_cache.get_head_ptr(layer_idx, 1, 0),
            1, current_pos,
            static_cast<int>(num_kv_heads),
            kv_cache.max_seq(),
            static_cast<int>(head_dim)
        );
        timed_end(lp + "cache_append");

        // 2d. Decode GQA attending over cache [0 .. current_pos]
        timed_begin(lp + "decode_attention");
        cuda::decode_gqa(
            buffers.q.data(),
            kv_cache.get_head_ptr(layer_idx, 0, 0),
            kv_cache.get_head_ptr(layer_idx, 1, 0),
            buffers.attn_out.data(),
            current_pos + 1,
            static_cast<int>(num_q_heads),
            static_cast<int>(num_kv_heads),
            kv_cache.max_seq(),
            static_cast<int>(head_dim)
        );
        timed_end(lp + "decode_attention");

        // 2e. Output projection (M=1)
        timed_begin(lp + "o_proj");
        cuda::matmul_transB(cublas_handle, buffers.attn_out.data(),
                           lw.o_proj_weight.data(), buffers.proj_out.data(),
                           1, static_cast<int>(H), static_cast<int>(H));
        timed_end(lp + "o_proj");

        // 2f. Residual add
        timed_begin(lp + "attn_residual");
        cuda::residual_add(buffers.hidden_states.data(), buffers.proj_out.data(),
                          buffers.hidden_states.data(), H);
        timed_end(lp + "attn_residual");

        // 3a. Post-attention RMSNorm
        timed_begin(lp + "post_rmsnorm");
        cuda::rmsnorm(buffers.hidden_states.data(), lw.post_attention_layernorm.data(),
                      buffers.norm_out.data(), 1, H, config.rms_norm_eps);
        timed_end(lp + "post_rmsnorm");

        // 3b. Gate & Up projections (M=1)
        timed_begin(lp + "gate_proj");
        cuda::matmul_transB(cublas_handle, buffers.norm_out.data(),
                           lw.gate_proj_weight.data(), buffers.gate.data(),
                           1, static_cast<int>(H), static_cast<int>(inter_dim));
        timed_end(lp + "gate_proj");

        timed_begin(lp + "up_proj");
        cuda::matmul_transB(cublas_handle, buffers.norm_out.data(),
                           lw.up_proj_weight.data(), buffers.up.data(),
                           1, static_cast<int>(H), static_cast<int>(inter_dim));
        timed_end(lp + "up_proj");

        // 3c. SwiGLU
        timed_begin(lp + "swiglu");
        cuda::swiglu(buffers.gate.data(), buffers.up.data(),
                    buffers.act.data(), inter_dim);
        timed_end(lp + "swiglu");

        // 3d. Down projection
        timed_begin(lp + "down_proj");
        cuda::matmul_transB(cublas_handle, buffers.act.data(),
                           lw.down_proj_weight.data(), buffers.mlp_out.data(),
                           1, static_cast<int>(inter_dim), static_cast<int>(H));
        timed_end(lp + "down_proj");

        // 3e. Residual add
        timed_begin(lp + "mlp_residual");
        cuda::residual_add(buffers.hidden_states.data(), buffers.mlp_out.data(),
                          buffers.hidden_states.data(), H);
        timed_end(lp + "mlp_residual");
    }

    kv_cache.set_current_seq_len(current_pos + 1);

    // 4. Final RMSNorm
    timed_begin("final_rmsnorm");
    cuda::rmsnorm(buffers.hidden_states.data(), weights.final_norm.data(),
                  buffers.norm_out.data(), 1, H, config.rms_norm_eps);
    timed_end("final_rmsnorm");

    // 5. LM Head: [1, H] * [V, H]^T -> [1, V]
    half* target_logits = d_last_logits ? d_last_logits : buffers.logits.data();
    timed_begin("lm_head");
    cuda::matmul_transB(cublas_handle, buffers.norm_out.data(),
                       weights.embed_tokens.data(), target_logits,
                       1, static_cast<int>(H), static_cast<int>(V));
    timed_end("lm_head");

    // Download only last-position logits (1 * V)
    std::vector<half> logits_h(V);
    CUDA_CHECK(cudaMemcpy(logits_h.data(), target_logits,
               V * sizeof(half), cudaMemcpyDeviceToHost));

    std::vector<float> logits_f(V);
    for (int64_t i = 0; i < V; ++i) {
        logits_f[i] = __half2float(logits_h[i]);
    }
    return logits_f;
}

} // namespace engine

