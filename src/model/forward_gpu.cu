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
    quant_mode = QuantMode::FP16;
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

static void upload_f16(const void* host_f16, DeviceBuffer<half>& dst, size_t count) {
    dst.allocate(count);
    dst.copy_from_host(reinterpret_cast<const half*>(host_f16), count);
}

static void upload_bytes(const void* host_bytes, DeviceBuffer<uint8_t>& dst, size_t count) {
    dst.allocate(count);
    dst.copy_from_host(reinterpret_cast<const uint8_t*>(host_bytes), count);
}

static void load_linear_weight(
    const SafetensorsLoader& loader,
    const std::string& prefix,
    int N,
    int K,
    QuantMode mode,
    DeviceBuffer<half>& fp16_buf,
    QuantWeightGPU& quant_buf
) {
    if (mode == QuantMode::FP16) {
        const auto* t = loader.get_tensor(prefix);
        if (!t) throw std::runtime_error("Missing tensor: " + prefix);
        if (t->dtype == DataType::BF16) {
            upload_bf16_to_fp16(t->data, fp16_buf, N * K);
        } else {
            upload_f16(t->data, fp16_buf, N * K);
        }
    } else if (mode == QuantMode::INT8) {
        const auto* t_qw = loader.get_tensor(prefix + ".qweight");
        const auto* t_sc = loader.get_tensor(prefix + ".scales");
        if (!t_qw || !t_sc) throw std::runtime_error("Missing INT8 tensor: " + prefix);
        upload_bytes(t_qw->data, quant_buf.qweight, N * K);
        upload_f16(t_sc->data, quant_buf.scales, N);
        // Pre-allocate dequantized buffer for prefill path
        quant_buf.fp16.allocate(N * K);
        cuda::dequantize_int8(reinterpret_cast<const int8_t*>(quant_buf.qweight.data()),
                             quant_buf.scales.data(), quant_buf.fp16.data(), N, K);
    } else if (mode == QuantMode::INT4_G128) {
        const auto* t_qw = loader.get_tensor(prefix + ".qweight");
        const auto* t_sc = loader.get_tensor(prefix + ".scales");
        if (!t_qw || !t_sc) throw std::runtime_error("Missing INT4 tensor: " + prefix);
        upload_bytes(t_qw->data, quant_buf.qweight, N * (K / 2));
        upload_f16(t_sc->data, quant_buf.scales, N * (K / 128));
        quant_buf.fp16.allocate(N * K);
        cuda::dequantize_int4_g128(quant_buf.qweight.data(), quant_buf.scales.data(),
                                   quant_buf.fp16.data(), N, K);
    }
}

bool ModelWeightsGPU::init_from_safetensors(const ModelConfig& config, const SafetensorsLoader& loader) {
    try {
        // Detect quant mode from tensor names (check both unfused and fused)
        if (loader.has_tensor("model.layers.0.self_attn.q_proj.weight.qweight")) {
            const auto* t_qw = loader.get_tensor("model.layers.0.self_attn.q_proj.weight.qweight");
            if (t_qw->dtype == DataType::I8) {
                quant_mode = QuantMode::INT8;
            } else {
                quant_mode = QuantMode::INT4_G128;
            }
        } else if (loader.has_tensor("model.layers.0.self_attn.qkv_proj.weight.qweight")) {
            const auto* t_qw = loader.get_tensor("model.layers.0.self_attn.qkv_proj.weight.qweight");
            if (t_qw->dtype == DataType::I8) {
                quant_mode = QuantMode::INT8;
            } else {
                quant_mode = QuantMode::INT4_G128;
            }
        } else {
            quant_mode = QuantMode::FP16;
        }

        int H = config.hidden_size;
        int kv_dim = config.num_key_value_heads * config.head_dim;
        int inter = config.intermediate_size;
        int V = config.vocab_size;

        // Embed tokens & final norm
        const auto* t_embed = loader.get_tensor("model.embed_tokens.weight");
        if (t_embed) {
            // FP16/BF16 embed — always loaded for embedding lookup
            if (t_embed->dtype == DataType::BF16) {
                upload_bf16_to_fp16(t_embed->data, embed_tokens, static_cast<size_t>(V) * H);
            } else {
                upload_f16(t_embed->data, embed_tokens, static_cast<size_t>(V) * H);
            }
            embed_is_quantized = false;
        } else {
            // Check for INT8-quantized embed_tokens (head-int8 models)
            const auto* t_eq = loader.get_tensor("model.embed_tokens.weight.qweight");
            const auto* t_es = loader.get_tensor("model.embed_tokens.weight.scales");
            if (t_eq && t_es) {
                // Load INT8 quantized embed for LM head GEMV
                embed_quant.qweight.allocate(static_cast<size_t>(V) * H);
                CUDA_CHECK(cudaMemcpy(embed_quant.qweight.data(), t_eq->data,
                           static_cast<size_t>(V) * H, cudaMemcpyHostToDevice));
                embed_quant.scales.allocate(V);
                CUDA_CHECK(cudaMemcpy(embed_quant.scales.data(), t_es->data,
                           static_cast<size_t>(V) * sizeof(half), cudaMemcpyHostToDevice));
                // Also dequantize to FP16 for embedding lookup
                embed_tokens.allocate(static_cast<size_t>(V) * H);
                cuda::dequantize_int8(
                    reinterpret_cast<const int8_t*>(embed_quant.qweight.data()),
                    embed_quant.scales.data(),
                    embed_tokens.data(), V, H);
                CUDA_CHECK(cudaDeviceSynchronize());
                embed_is_quantized = true;
            } else {
                throw std::runtime_error("Missing model.embed_tokens.weight tensor");
            }
        }

        const auto* t_norm = loader.get_tensor("model.norm.weight");
        if (t_norm) {
            if (t_norm->dtype == DataType::BF16) {
                upload_bf16_to_fp16(t_norm->data, final_norm, H);
            } else {
                upload_f16(t_norm->data, final_norm, H);
            }
        }

        layers.resize(config.num_hidden_layers);
        for (int i = 0; i < config.num_hidden_layers; ++i) {
            auto& lg = layers[i];
            std::string lp = "model.layers." + std::to_string(i) + ".";

            // Layer norms
            const auto* t_in_norm = loader.get_tensor(lp + "input_layernorm.weight");
            if (t_in_norm->dtype == DataType::BF16) upload_bf16_to_fp16(t_in_norm->data, lg.input_layernorm, H);
            else upload_f16(t_in_norm->data, lg.input_layernorm, H);

            const auto* t_post_norm = loader.get_tensor(lp + "post_attention_layernorm.weight");
            if (t_post_norm->dtype == DataType::BF16) upload_bf16_to_fp16(t_post_norm->data, lg.post_attention_layernorm, H);
            else upload_f16(t_post_norm->data, lg.post_attention_layernorm, H);

            // Check for fused QKV
            int qkv_dim = H + 2 * kv_dim;
            if (loader.has_tensor(lp + "self_attn.qkv_proj.weight") ||
                loader.has_tensor(lp + "self_attn.qkv_proj.weight.qweight")) {
                lg.has_fused_qkv = true;
                const auto* t_qkvb = loader.get_tensor(lp + "self_attn.qkv_proj.bias");
                if (t_qkvb) {
                    if (t_qkvb->dtype == DataType::BF16) upload_bf16_to_fp16(t_qkvb->data, lg.qkv_proj_bias, qkv_dim);
                    else upload_f16(t_qkvb->data, lg.qkv_proj_bias, qkv_dim);
                }
                load_linear_weight(loader, lp + "self_attn.qkv_proj.weight", qkv_dim, H, quant_mode, lg.qkv_proj_weight, lg.qkv_proj_quant);
            } else {
                lg.has_fused_qkv = false;
                // Biases (FP16)
                const auto* t_qb = loader.get_tensor(lp + "self_attn.q_proj.bias");
                if (t_qb->dtype == DataType::BF16) upload_bf16_to_fp16(t_qb->data, lg.q_proj_bias, H);
                else upload_f16(t_qb->data, lg.q_proj_bias, H);

                const auto* t_kb = loader.get_tensor(lp + "self_attn.k_proj.bias");
                if (t_kb->dtype == DataType::BF16) upload_bf16_to_fp16(t_kb->data, lg.k_proj_bias, kv_dim);
                else upload_f16(t_kb->data, lg.k_proj_bias, kv_dim);

                const auto* t_vb = loader.get_tensor(lp + "self_attn.v_proj.bias");
                if (t_vb->dtype == DataType::BF16) upload_bf16_to_fp16(t_vb->data, lg.v_proj_bias, kv_dim);
                else upload_f16(t_vb->data, lg.v_proj_bias, kv_dim);

                load_linear_weight(loader, lp + "self_attn.q_proj.weight", H, H, quant_mode, lg.q_proj_weight, lg.q_proj_quant);
                load_linear_weight(loader, lp + "self_attn.k_proj.weight", kv_dim, H, quant_mode, lg.k_proj_weight, lg.k_proj_quant);
                load_linear_weight(loader, lp + "self_attn.v_proj.weight", kv_dim, H, quant_mode, lg.v_proj_weight, lg.v_proj_quant);
            }

            // o_proj is always separate
            load_linear_weight(loader, lp + "self_attn.o_proj.weight", H, H, quant_mode, lg.o_proj_weight, lg.o_proj_quant);

            // Check for fused Gate+Up
            int gate_up_dim = 2 * inter;
            if (loader.has_tensor(lp + "mlp.gate_up_proj.weight") ||
                loader.has_tensor(lp + "mlp.gate_up_proj.weight.qweight")) {
                lg.has_fused_gate_up = true;
                load_linear_weight(loader, lp + "mlp.gate_up_proj.weight", gate_up_dim, H, quant_mode, lg.gate_up_proj_weight, lg.gate_up_proj_quant);
            } else {
                lg.has_fused_gate_up = false;
                load_linear_weight(loader, lp + "mlp.gate_proj.weight", inter, H, quant_mode, lg.gate_proj_weight, lg.gate_proj_quant);
                load_linear_weight(loader, lp + "mlp.up_proj.weight", inter, H, quant_mode, lg.up_proj_weight, lg.up_proj_quant);
            }

            // down_proj is always separate
            load_linear_weight(loader, lp + "mlp.down_proj.weight", H, inter, quant_mode, lg.down_proj_weight, lg.down_proj_quant);
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error in ModelWeightsGPU::init_from_safetensors: " << e.what() << std::endl;
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
    decode_seq_len.allocate(1);
    split_kv_buf.allocate(64 * 14 * (config.head_dim + 2));

    q8_act_H.allocate(H);
    q8_act_inter.allocate(inter);
    act_scale.allocate(1);

    // Fused activation buffers (Task P11a)
    qkv_out.allocate(max_seq * (H + 2 * kv_dim));
    gate_up_out.allocate(max_seq * (2 * inter));

    // CUDA Graph Device Parameters (Task P11a D3)
    d_current_pos.allocate(1);
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

        auto get_w = [&](const DeviceBuffer<half>& fp16_w, const QuantWeightGPU& qw) -> const half* {
            return (weights.quant_mode == QuantMode::FP16) ? fp16_w.data() : qw.fp16.data();
        };

        // 2b. Q projection: [S, H] * [H, H]^T -> [S, H]
        timed_begin(lp + "q_proj");
        cuda::matmul_transB(cublas_handle, buffers.norm_out.data(),
                           get_w(lw.q_proj_weight, lw.q_proj_quant), buffers.q.data(),
                           static_cast<int>(S), static_cast<int>(H), static_cast<int>(H));
        timed_end(lp + "q_proj");

        timed_begin(lp + "q_bias");
        cuda::bias_add(buffers.q.data(), lw.q_proj_bias.data(),
                      buffers.q.data(), S, H);
        timed_end(lp + "q_bias");

        // K projection: [S, H] * [kv_dim, H]^T -> [S, kv_dim]
        timed_begin(lp + "k_proj");
        cuda::matmul_transB(cublas_handle, buffers.norm_out.data(),
                           get_w(lw.k_proj_weight, lw.k_proj_quant), buffers.k.data(),
                           static_cast<int>(S), static_cast<int>(H), static_cast<int>(kv_dim));
        timed_end(lp + "k_proj");

        timed_begin(lp + "k_bias");
        cuda::bias_add(buffers.k.data(), lw.k_proj_bias.data(),
                      buffers.k.data(), S, kv_dim);
        timed_end(lp + "k_bias");

        // V projection: [S, H] * [kv_dim, H]^T -> [S, kv_dim]
        timed_begin(lp + "v_proj");
        cuda::matmul_transB(cublas_handle, buffers.norm_out.data(),
                           get_w(lw.v_proj_weight, lw.v_proj_quant), buffers.v.data(),
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

        // 2d. Flash-Style Causal GQA (tiled shared memory, online softmax)
        timed_begin(lp + "attention");
        cuda::flash_prefill_gqa(buffers.q.data(), buffers.k.data(), buffers.v.data(),
                                buffers.attn_out.data(), S, num_q_heads, num_kv_heads, head_dim);
        timed_end(lp + "attention");

        // 2e. Output projection: [S, H] * [H, H]^T -> [S, H]
        timed_begin(lp + "o_proj");
        cuda::matmul_transB(cublas_handle, buffers.attn_out.data(),
                           get_w(lw.o_proj_weight, lw.o_proj_quant), buffers.proj_out.data(),
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
                           get_w(lw.gate_proj_weight, lw.gate_proj_quant), buffers.gate.data(),
                           static_cast<int>(S), static_cast<int>(H), static_cast<int>(inter_dim));
        timed_end(lp + "gate_proj");

        // Up projection: [S, H] * [inter_dim, H]^T -> [S, inter_dim]
        timed_begin(lp + "up_proj");
        cuda::matmul_transB(cublas_handle, buffers.norm_out.data(),
                           get_w(lw.up_proj_weight, lw.up_proj_quant), buffers.up.data(),
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
                           get_w(lw.down_proj_weight, lw.down_proj_quant), buffers.mlp_out.data(),
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
    // [S, H] * [V, H]^T -> [S, V] in FP32
    timed_begin("lm_head");
    cuda::matmul_transB_f32(cublas_handle, buffers.norm_out.data(),
                           weights.embed_tokens.data(), buffers.logits.data(),
                           static_cast<int>(S), static_cast<int>(H), static_cast<int>(V));
    timed_end("lm_head");

    // Download logits to host as FP32
    std::vector<float> logits_f(S * V);
    CUDA_CHECK(cudaMemcpy(logits_f.data(), buffers.logits.data(),
               S * V * sizeof(float), cudaMemcpyDeviceToHost));

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
    float* d_last_logits,
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

        // Helper to get linear weight pointer (FP16 or dequantized)
        auto get_w = [&](const DeviceBuffer<half>& fp16_w, const QuantWeightGPU& qw) -> const half* {
            return (weights.quant_mode == QuantMode::FP16) ? fp16_w.data() : qw.fp16.data();
        };

        // 2b. Q/K/V projections
        if (lw.has_fused_qkv) {
            int qkv_dim = static_cast<int>(H + 2 * kv_dim);
            timed_begin(lp + "qkv_proj");
            cuda::matmul_transB(cublas_handle, buffers.norm_out.data(),
                               get_w(lw.qkv_proj_weight, lw.qkv_proj_quant), buffers.qkv_out.data(),
                               static_cast<int>(S), static_cast<int>(H), qkv_dim);
            timed_end(lp + "qkv_proj");

            timed_begin(lp + "qkv_bias");
            cuda::bias_add(buffers.qkv_out.data(), lw.qkv_proj_bias.data(),
                          buffers.qkv_out.data(), S, qkv_dim);
            timed_end(lp + "qkv_bias");

            for (int s = 0; s < S; ++s) {
                CUDA_CHECK(cudaMemcpyAsync(buffers.q.data() + s * H, buffers.qkv_out.data() + s * qkv_dim,
                                           H * sizeof(half), cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpyAsync(buffers.k.data() + s * kv_dim, buffers.qkv_out.data() + s * qkv_dim + H,
                                           kv_dim * sizeof(half), cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpyAsync(buffers.v.data() + s * kv_dim, buffers.qkv_out.data() + s * qkv_dim + H + kv_dim,
                                           kv_dim * sizeof(half), cudaMemcpyDeviceToDevice));
            }
        } else {
            timed_begin(lp + "q_proj");
            cuda::matmul_transB(cublas_handle, buffers.norm_out.data(),
                               get_w(lw.q_proj_weight, lw.q_proj_quant), buffers.q.data(),
                               static_cast<int>(S), static_cast<int>(H), static_cast<int>(H));
            timed_end(lp + "q_proj");

            timed_begin(lp + "q_bias");
            cuda::bias_add(buffers.q.data(), lw.q_proj_bias.data(),
                          buffers.q.data(), S, H);
            timed_end(lp + "q_bias");

            timed_begin(lp + "k_proj");
            cuda::matmul_transB(cublas_handle, buffers.norm_out.data(),
                               get_w(lw.k_proj_weight, lw.k_proj_quant), buffers.k.data(),
                               static_cast<int>(S), static_cast<int>(H), static_cast<int>(kv_dim));
            timed_end(lp + "k_proj");

            timed_begin(lp + "k_bias");
            cuda::bias_add(buffers.k.data(), lw.k_proj_bias.data(),
                          buffers.k.data(), S, kv_dim);
            timed_end(lp + "k_bias");

            timed_begin(lp + "v_proj");
            cuda::matmul_transB(cublas_handle, buffers.norm_out.data(),
                               get_w(lw.v_proj_weight, lw.v_proj_quant), buffers.v.data(),
                               static_cast<int>(S), static_cast<int>(H), static_cast<int>(kv_dim));
            timed_end(lp + "v_proj");

            timed_begin(lp + "v_bias");
            cuda::bias_add(buffers.v.data(), lw.v_proj_bias.data(),
                          buffers.v.data(), S, kv_dim);
            timed_end(lp + "v_bias");
        }

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

        // 2d. Flash-Style Causal GQA (tiled shared memory, online softmax)
        timed_begin(lp + "attention");
        cuda::flash_prefill_gqa(buffers.q.data(), buffers.k.data(), buffers.v.data(),
                                buffers.attn_out.data(), S, num_q_heads, num_kv_heads, head_dim);
        timed_end(lp + "attention");

        // 2e. Output projection
        timed_begin(lp + "o_proj");
        cuda::matmul_transB(cublas_handle, buffers.attn_out.data(),
                           get_w(lw.o_proj_weight, lw.o_proj_quant), buffers.proj_out.data(),
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
        if (lw.has_fused_gate_up) {
            int gate_up_dim = static_cast<int>(2 * inter_dim);
            timed_begin(lp + "gate_up_proj");
            cuda::matmul_transB(cublas_handle, buffers.norm_out.data(),
                               get_w(lw.gate_up_proj_weight, lw.gate_up_proj_quant), buffers.gate_up_out.data(),
                               static_cast<int>(S), static_cast<int>(H), gate_up_dim);
            timed_end(lp + "gate_up_proj");

            for (int s = 0; s < S; ++s) {
                CUDA_CHECK(cudaMemcpyAsync(buffers.gate.data() + s * inter_dim, buffers.gate_up_out.data() + s * gate_up_dim,
                                           inter_dim * sizeof(half), cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpyAsync(buffers.up.data() + s * inter_dim, buffers.gate_up_out.data() + s * gate_up_dim + inter_dim,
                                           inter_dim * sizeof(half), cudaMemcpyDeviceToDevice));
            }
        } else {
            timed_begin(lp + "gate_proj");
            cuda::matmul_transB(cublas_handle, buffers.norm_out.data(),
                               get_w(lw.gate_proj_weight, lw.gate_proj_quant), buffers.gate.data(),
                               static_cast<int>(S), static_cast<int>(H), static_cast<int>(inter_dim));
            timed_end(lp + "gate_proj");

            timed_begin(lp + "up_proj");
            cuda::matmul_transB(cublas_handle, buffers.norm_out.data(),
                               get_w(lw.up_proj_weight, lw.up_proj_quant), buffers.up.data(),
                               static_cast<int>(S), static_cast<int>(H), static_cast<int>(inter_dim));
            timed_end(lp + "up_proj");
        }

        // 3c. SwiGLU
        timed_begin(lp + "swiglu");
        cuda::swiglu(buffers.gate.data(), buffers.up.data(),
                    buffers.act.data(), S * inter_dim);
        timed_end(lp + "swiglu");

        // 3d. Down projection
        timed_begin(lp + "down_proj");
        cuda::matmul_transB(cublas_handle, buffers.act.data(),
                           get_w(lw.down_proj_weight, lw.down_proj_quant), buffers.mlp_out.data(),
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

    // 5. LM Head on last position only: [1, H] * [V, H]^T -> [1, V] in FP32
    float* target_logits = d_last_logits ? d_last_logits : buffers.logits.data();
    timed_begin("lm_head");
    if (weights.embed_is_quantized) {
        cuda::gemv_int8_f32(
            reinterpret_cast<const int8_t*>(weights.embed_quant.qweight.data()),
            weights.embed_quant.scales.data(),
            last_norm,
            target_logits,
            static_cast<int>(V), static_cast<int>(H));
    } else {
        cuda::matmul_transB_f32(cublas_handle, last_norm,
                               weights.embed_tokens.data(), target_logits,
                               1, static_cast<int>(H), static_cast<int>(V));
    }
    timed_end("lm_head");

    // Download only last-position logits (1 * V) if requested (skip when d_last_logits provided for async pipeline)
    std::vector<float> logits_f;
    if (!d_last_logits) {
        logits_f.resize(V);
        CUDA_CHECK(cudaMemcpy(logits_f.data(), target_logits,
                   V * sizeof(float), cudaMemcpyDeviceToHost));
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
    float* d_last_logits,
    std::vector<KernelRecord>* profile_records,
    bool use_legacy_gemv
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

    // Upload single token ID and sequence length
    buffers.token_ids_d.copy_from_host(&token_id, 1);
    int total_tokens = current_pos + 1;
    buffers.decode_seq_len.copy_from_host(&total_tokens, 1);

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

        // If using dp4a, quantize norm_out activation vector to INT8
        bool do_dp4a = !use_legacy_gemv && (weights.quant_mode == QuantMode::INT8 || weights.quant_mode == QuantMode::INT4_G128);
        if (do_dp4a) {
            timed_begin(lp + "quant_act_norm");
            cuda::quantize_activation_int8(buffers.norm_out.data(), buffers.q8_act_H.data(),
                                           buffers.act_scale.data(), static_cast<int>(H));
            timed_end(lp + "quant_act_norm");
        }

        // 2b. Q/K/V projections (M=1) — custom GEMV with fused bias
        if (lw.has_fused_qkv) {
            int qkv_dim = static_cast<int>(H + 2 * kv_dim);
            timed_begin(lp + "qkv_proj");
            if (weights.quant_mode == QuantMode::INT8) {
                if (do_dp4a) {
                    cuda::gemv_int8_dp4a_bias(reinterpret_cast<const int8_t*>(lw.qkv_proj_quant.qweight.data()),
                                             lw.qkv_proj_quant.scales.data(), buffers.q8_act_H.data(),
                                             buffers.act_scale.data(), lw.qkv_proj_bias.data(), buffers.qkv_out.data(),
                                             qkv_dim, static_cast<int>(H));
                } else {
                    cuda::gemv_int8_bias(reinterpret_cast<const int8_t*>(lw.qkv_proj_quant.qweight.data()),
                                        lw.qkv_proj_quant.scales.data(), buffers.norm_out.data(),
                                        lw.qkv_proj_bias.data(), buffers.qkv_out.data(),
                                        qkv_dim, static_cast<int>(H));
                }
            } else if (weights.quant_mode == QuantMode::INT4_G128) {
                if (do_dp4a) {
                    cuda::gemv_int4_g128_dp4a_bias(lw.qkv_proj_quant.qweight.data(),
                                                  lw.qkv_proj_quant.scales.data(), buffers.q8_act_H.data(),
                                                  buffers.act_scale.data(), lw.qkv_proj_bias.data(), buffers.qkv_out.data(),
                                                  qkv_dim, static_cast<int>(H));
                } else {
                    cuda::gemv_int4_g128_bias(lw.qkv_proj_quant.qweight.data(),
                                             lw.qkv_proj_quant.scales.data(), buffers.norm_out.data(),
                                             lw.qkv_proj_bias.data(), buffers.qkv_out.data(),
                                             qkv_dim, static_cast<int>(H));
                }
            } else {
                cuda::gemv_bias(lw.qkv_proj_weight.data(), buffers.norm_out.data(),
                                lw.qkv_proj_bias.data(), buffers.qkv_out.data(),
                                qkv_dim, static_cast<int>(H));
            }
            timed_end(lp + "qkv_proj");
        } else {
            timed_begin(lp + "q_proj");
            if (weights.quant_mode == QuantMode::INT8) {
                if (do_dp4a) {
                    cuda::gemv_int8_dp4a_bias(reinterpret_cast<const int8_t*>(lw.q_proj_quant.qweight.data()),
                                             lw.q_proj_quant.scales.data(), buffers.q8_act_H.data(),
                                             buffers.act_scale.data(), lw.q_proj_bias.data(), buffers.q.data(),
                                             static_cast<int>(H), static_cast<int>(H));
                } else {
                    cuda::gemv_int8_bias(reinterpret_cast<const int8_t*>(lw.q_proj_quant.qweight.data()),
                                        lw.q_proj_quant.scales.data(), buffers.norm_out.data(),
                                        lw.q_proj_bias.data(), buffers.q.data(),
                                        static_cast<int>(H), static_cast<int>(H));
                }
            } else if (weights.quant_mode == QuantMode::INT4_G128) {
                if (do_dp4a) {
                    cuda::gemv_int4_g128_dp4a_bias(lw.q_proj_quant.qweight.data(),
                                                  lw.q_proj_quant.scales.data(), buffers.q8_act_H.data(),
                                                  buffers.act_scale.data(), lw.q_proj_bias.data(), buffers.q.data(),
                                                  static_cast<int>(H), static_cast<int>(H));
                } else {
                    cuda::gemv_int4_g128_bias(lw.q_proj_quant.qweight.data(),
                                             lw.q_proj_quant.scales.data(), buffers.norm_out.data(),
                                             lw.q_proj_bias.data(), buffers.q.data(),
                                             static_cast<int>(H), static_cast<int>(H));
                }
            } else {
                cuda::gemv_bias(lw.q_proj_weight.data(), buffers.norm_out.data(),
                                lw.q_proj_bias.data(), buffers.q.data(),
                                static_cast<int>(H), static_cast<int>(H));
            }
            timed_end(lp + "q_proj");

            timed_begin(lp + "k_proj");
            if (weights.quant_mode == QuantMode::INT8) {
                if (do_dp4a) {
                    cuda::gemv_int8_dp4a_bias(reinterpret_cast<const int8_t*>(lw.k_proj_quant.qweight.data()),
                                             lw.k_proj_quant.scales.data(), buffers.q8_act_H.data(),
                                             buffers.act_scale.data(), lw.k_proj_bias.data(), buffers.k.data(),
                                             static_cast<int>(kv_dim), static_cast<int>(H));
                } else {
                    cuda::gemv_int8_bias(reinterpret_cast<const int8_t*>(lw.k_proj_quant.qweight.data()),
                                        lw.k_proj_quant.scales.data(), buffers.norm_out.data(),
                                        lw.k_proj_bias.data(), buffers.k.data(),
                                        static_cast<int>(kv_dim), static_cast<int>(H));
                }
            } else if (weights.quant_mode == QuantMode::INT4_G128) {
                if (do_dp4a) {
                    cuda::gemv_int4_g128_dp4a_bias(lw.k_proj_quant.qweight.data(),
                                                  lw.k_proj_quant.scales.data(), buffers.q8_act_H.data(),
                                                  buffers.act_scale.data(), lw.k_proj_bias.data(), buffers.k.data(),
                                                  static_cast<int>(kv_dim), static_cast<int>(H));
                } else {
                    cuda::gemv_int4_g128_bias(lw.k_proj_quant.qweight.data(),
                                             lw.k_proj_quant.scales.data(), buffers.norm_out.data(),
                                             lw.k_proj_bias.data(), buffers.k.data(),
                                             static_cast<int>(kv_dim), static_cast<int>(H));
                }
            } else {
                cuda::gemv_bias(lw.k_proj_weight.data(), buffers.norm_out.data(),
                                lw.k_proj_bias.data(), buffers.k.data(),
                                static_cast<int>(kv_dim), static_cast<int>(H));
            }
            timed_end(lp + "k_proj");

            timed_begin(lp + "v_proj");
            if (weights.quant_mode == QuantMode::INT8) {
                if (do_dp4a) {
                    cuda::gemv_int8_dp4a_bias(reinterpret_cast<const int8_t*>(lw.v_proj_quant.qweight.data()),
                                             lw.v_proj_quant.scales.data(), buffers.q8_act_H.data(),
                                             buffers.act_scale.data(), lw.v_proj_bias.data(), buffers.v.data(),
                                             static_cast<int>(kv_dim), static_cast<int>(H));
                } else {
                    cuda::gemv_int8_bias(reinterpret_cast<const int8_t*>(lw.v_proj_quant.qweight.data()),
                                        lw.v_proj_quant.scales.data(), buffers.norm_out.data(),
                                        lw.v_proj_bias.data(), buffers.v.data(),
                                        static_cast<int>(kv_dim), static_cast<int>(H));
                }
            } else if (weights.quant_mode == QuantMode::INT4_G128) {
                if (do_dp4a) {
                    cuda::gemv_int4_g128_dp4a_bias(lw.v_proj_quant.qweight.data(),
                                                  lw.v_proj_quant.scales.data(), buffers.q8_act_H.data(),
                                                  buffers.act_scale.data(), lw.v_proj_bias.data(), buffers.v.data(),
                                                  static_cast<int>(kv_dim), static_cast<int>(H));
                } else {
                    cuda::gemv_int4_g128_bias(lw.v_proj_quant.qweight.data(),
                                             lw.v_proj_quant.scales.data(), buffers.norm_out.data(),
                                             lw.v_proj_bias.data(), buffers.v.data(),
                                             static_cast<int>(kv_dim), static_cast<int>(H));
                }
            } else {
                cuda::gemv_bias(lw.v_proj_weight.data(), buffers.norm_out.data(),
                                lw.v_proj_bias.data(), buffers.v.data(),
                                static_cast<int>(kv_dim), static_cast<int>(H));
            }
            timed_end(lp + "v_proj");
        }

        half* q_step_ptr = lw.has_fused_qkv ? buffers.qkv_out.data() : buffers.q.data();
        half* k_step_ptr = lw.has_fused_qkv ? (buffers.qkv_out.data() + H) : buffers.k.data();
        half* v_step_ptr = lw.has_fused_qkv ? (buffers.qkv_out.data() + H + kv_dim) : buffers.v.data();

        // 2c. RoPE with absolute position current_pos
        timed_begin(lp + "rope");
        cuda::apply_rope(q_step_ptr, k_step_ptr,
                        1, num_q_heads, num_kv_heads, head_dim, current_pos, config.rope_theta);
        timed_end(lp + "rope");

        // Append single key and value to cache at current_pos
        timed_begin(lp + "cache_append");
        cuda::append_kv_cache(
            k_step_ptr, v_step_ptr,
            kv_cache.get_head_ptr(layer_idx, 0, 0),
            kv_cache.get_head_ptr(layer_idx, 1, 0),
            1, current_pos,
            static_cast<int>(num_kv_heads),
            kv_cache.max_seq(),
            static_cast<int>(head_dim)
        );
        timed_end(lp + "cache_append");

        // 2d. Decode Attention (GQA-grouped, Split-KV with KVView)
        timed_begin(lp + "decode_attention");
        cuda::KVView k_view{
            kv_cache.get_head_ptr(layer_idx, 0, 0),
            0,
            static_cast<int64_t>(kv_cache.max_seq() * head_dim),
            static_cast<int64_t>(head_dim),
            1
        };
        cuda::KVView v_view{
            kv_cache.get_head_ptr(layer_idx, 1, 0),
            0,
            static_cast<int64_t>(kv_cache.max_seq() * head_dim),
            static_cast<int64_t>(head_dim),
            1
        };
        cuda::decode_gqa_split(
            q_step_ptr,
            k_view,
            v_view,
            buffers.decode_seq_len.data(),
            buffers.attn_out.data(),
            1,
            static_cast<int>(num_q_heads),
            static_cast<int>(num_kv_heads),
            static_cast<int>(head_dim),
            total_tokens,
            buffers.split_kv_buf.data()
        );
        timed_end(lp + "decode_attention");

        // Quantize attn_out to INT8 if do_dp4a
        if (do_dp4a) {
            timed_begin(lp + "quant_act_attn");
            cuda::quantize_activation_int8(buffers.attn_out.data(), buffers.q8_act_H.data(),
                                           buffers.act_scale.data(), static_cast<int>(H));
            timed_end(lp + "quant_act_attn");
        }

        // 2e. Output projection (M=1) fused with residual add into hidden_states
        timed_begin(lp + "o_proj_residual");
        if (weights.quant_mode == QuantMode::INT8) {
            if (do_dp4a) {
                cuda::gemv_int8_dp4a_residual(reinterpret_cast<const int8_t*>(lw.o_proj_quant.qweight.data()),
                                             lw.o_proj_quant.scales.data(), buffers.q8_act_H.data(),
                                             buffers.act_scale.data(), buffers.hidden_states.data(),
                                             buffers.hidden_states.data(),
                                             static_cast<int>(H), static_cast<int>(H));
            } else {
                cuda::gemv_int8_residual(reinterpret_cast<const int8_t*>(lw.o_proj_quant.qweight.data()),
                                        lw.o_proj_quant.scales.data(), buffers.attn_out.data(),
                                        buffers.hidden_states.data(), buffers.hidden_states.data(),
                                        static_cast<int>(H), static_cast<int>(H));
            }
        } else if (weights.quant_mode == QuantMode::INT4_G128) {
            if (do_dp4a) {
                cuda::gemv_int4_g128_dp4a_residual(lw.o_proj_quant.qweight.data(),
                                                  lw.o_proj_quant.scales.data(), buffers.q8_act_H.data(),
                                                  buffers.act_scale.data(), buffers.hidden_states.data(),
                                                  buffers.hidden_states.data(),
                                                  static_cast<int>(H), static_cast<int>(H));
            } else {
                cuda::gemv_int4_g128_residual(lw.o_proj_quant.qweight.data(),
                                             lw.o_proj_quant.scales.data(), buffers.attn_out.data(),
                                             buffers.hidden_states.data(), buffers.hidden_states.data(),
                                             static_cast<int>(H), static_cast<int>(H));
            }
        } else {
            cuda::gemv_residual(lw.o_proj_weight.data(), buffers.attn_out.data(),
                                buffers.hidden_states.data(), buffers.hidden_states.data(),
                                static_cast<int>(H), static_cast<int>(H));
        }
        timed_end(lp + "o_proj_residual");

        // 3a. Post-attention RMSNorm
        timed_begin(lp + "post_rmsnorm");
        cuda::rmsnorm(buffers.hidden_states.data(), lw.post_attention_layernorm.data(),
                      buffers.norm_out.data(), 1, H, config.rms_norm_eps);
        timed_end(lp + "post_rmsnorm");

        // Quantize post_norm_out to INT8 if do_dp4a
        if (do_dp4a) {
            timed_begin(lp + "quant_act_post_norm");
            cuda::quantize_activation_int8(buffers.norm_out.data(), buffers.q8_act_H.data(),
                                           buffers.act_scale.data(), static_cast<int>(H));
            timed_end(lp + "quant_act_post_norm");
        }

        // 3b. Gate & Up projections (M=1) — custom GEMV
        if (lw.has_fused_gate_up) {
            int gate_up_dim = static_cast<int>(2 * inter_dim);
            timed_begin(lp + "gate_up_proj");
            if (weights.quant_mode == QuantMode::INT8) {
                if (do_dp4a) {
                    cuda::gemv_int8_dp4a(reinterpret_cast<const int8_t*>(lw.gate_up_proj_quant.qweight.data()),
                                        lw.gate_up_proj_quant.scales.data(), buffers.q8_act_H.data(),
                                        buffers.act_scale.data(), buffers.gate_up_out.data(),
                                        gate_up_dim, static_cast<int>(H));
                } else {
                    cuda::gemv_int8(reinterpret_cast<const int8_t*>(lw.gate_up_proj_quant.qweight.data()),
                                   lw.gate_up_proj_quant.scales.data(), buffers.norm_out.data(),
                                   buffers.gate_up_out.data(),
                                   gate_up_dim, static_cast<int>(H));
                }
            } else if (weights.quant_mode == QuantMode::INT4_G128) {
                if (do_dp4a) {
                    cuda::gemv_int4_g128_dp4a(lw.gate_up_proj_quant.qweight.data(),
                                             lw.gate_up_proj_quant.scales.data(), buffers.q8_act_H.data(),
                                             buffers.act_scale.data(), buffers.gate_up_out.data(),
                                             gate_up_dim, static_cast<int>(H));
                } else {
                    cuda::gemv_int4_g128(lw.gate_up_proj_quant.qweight.data(),
                                        lw.gate_up_proj_quant.scales.data(), buffers.norm_out.data(),
                                        buffers.gate_up_out.data(),
                                        gate_up_dim, static_cast<int>(H));
                }
            } else {
                cuda::gemv(lw.gate_up_proj_weight.data(), buffers.norm_out.data(),
                           buffers.gate_up_out.data(),
                           gate_up_dim, static_cast<int>(H));
            }
            timed_end(lp + "gate_up_proj");
        } else {
            timed_begin(lp + "gate_proj");
            if (weights.quant_mode == QuantMode::INT8) {
                if (do_dp4a) {
                    cuda::gemv_int8_dp4a(reinterpret_cast<const int8_t*>(lw.gate_proj_quant.qweight.data()),
                                        lw.gate_proj_quant.scales.data(), buffers.q8_act_H.data(),
                                        buffers.act_scale.data(), buffers.gate.data(),
                                        static_cast<int>(inter_dim), static_cast<int>(H));
                } else {
                    cuda::gemv_int8(reinterpret_cast<const int8_t*>(lw.gate_proj_quant.qweight.data()),
                                   lw.gate_proj_quant.scales.data(), buffers.norm_out.data(),
                                   buffers.gate.data(),
                                   static_cast<int>(inter_dim), static_cast<int>(H));
                }
            } else if (weights.quant_mode == QuantMode::INT4_G128) {
                if (do_dp4a) {
                    cuda::gemv_int4_g128_dp4a(lw.gate_proj_quant.qweight.data(),
                                             lw.gate_proj_quant.scales.data(), buffers.q8_act_H.data(),
                                             buffers.act_scale.data(), buffers.gate.data(),
                                             static_cast<int>(inter_dim), static_cast<int>(H));
                } else {
                    cuda::gemv_int4_g128(lw.gate_proj_quant.qweight.data(),
                                        lw.gate_proj_quant.scales.data(), buffers.norm_out.data(),
                                        buffers.gate.data(),
                                        static_cast<int>(inter_dim), static_cast<int>(H));
                }
            } else {
                cuda::gemv(lw.gate_proj_weight.data(), buffers.norm_out.data(),
                           buffers.gate.data(),
                           static_cast<int>(inter_dim), static_cast<int>(H));
            }
            timed_end(lp + "gate_proj");

            timed_begin(lp + "up_proj");
            if (weights.quant_mode == QuantMode::INT8) {
                if (do_dp4a) {
                    cuda::gemv_int8_dp4a(reinterpret_cast<const int8_t*>(lw.up_proj_quant.qweight.data()),
                                        lw.up_proj_quant.scales.data(), buffers.q8_act_H.data(),
                                        buffers.act_scale.data(), buffers.up.data(),
                                        static_cast<int>(inter_dim), static_cast<int>(H));
                } else {
                    cuda::gemv_int8(reinterpret_cast<const int8_t*>(lw.up_proj_quant.qweight.data()),
                                   lw.up_proj_quant.scales.data(), buffers.norm_out.data(),
                                   buffers.up.data(),
                                   static_cast<int>(inter_dim), static_cast<int>(H));
                }
            } else if (weights.quant_mode == QuantMode::INT4_G128) {
                if (do_dp4a) {
                    cuda::gemv_int4_g128_dp4a(lw.up_proj_quant.qweight.data(),
                                             lw.up_proj_quant.scales.data(), buffers.q8_act_H.data(),
                                             buffers.act_scale.data(), buffers.up.data(),
                                             static_cast<int>(inter_dim), static_cast<int>(H));
                } else {
                    cuda::gemv_int4_g128(lw.up_proj_quant.qweight.data(),
                                        lw.up_proj_quant.scales.data(), buffers.norm_out.data(),
                                        buffers.up.data(),
                                        static_cast<int>(inter_dim), static_cast<int>(H));
                }
            } else {
                cuda::gemv(lw.up_proj_weight.data(), buffers.norm_out.data(),
                           buffers.up.data(),
                           static_cast<int>(inter_dim), static_cast<int>(H));
            }
            timed_end(lp + "up_proj");
        }

        const half* gate_step_ptr = lw.has_fused_gate_up ? buffers.gate_up_out.data() : buffers.gate.data();
        const half* up_step_ptr = lw.has_fused_gate_up ? (buffers.gate_up_out.data() + inter_dim) : buffers.up.data();

        // 3c. SwiGLU
        timed_begin(lp + "swiglu");
        cuda::swiglu(gate_step_ptr, up_step_ptr,
                    buffers.act.data(), inter_dim);
        timed_end(lp + "swiglu");

        // Quantize act (SwiGLU output [inter_dim]) to INT8 if do_dp4a
        if (do_dp4a) {
            timed_begin(lp + "quant_act_mlp");
            cuda::quantize_activation_int8(buffers.act.data(), buffers.q8_act_inter.data(),
                                           buffers.act_scale.data(), static_cast<int>(inter_dim));
            timed_end(lp + "quant_act_mlp");
        }

        // 3d. Down projection fused with residual add into hidden_states
        timed_begin(lp + "down_proj_residual");
        if (weights.quant_mode == QuantMode::INT8) {
            if (do_dp4a) {
                cuda::gemv_int8_dp4a_residual(reinterpret_cast<const int8_t*>(lw.down_proj_quant.qweight.data()),
                                             lw.down_proj_quant.scales.data(), buffers.q8_act_inter.data(),
                                             buffers.act_scale.data(), buffers.hidden_states.data(),
                                             buffers.hidden_states.data(),
                                             static_cast<int>(H), static_cast<int>(inter_dim));
            } else {
                cuda::gemv_int8_residual(reinterpret_cast<const int8_t*>(lw.down_proj_quant.qweight.data()),
                                        lw.down_proj_quant.scales.data(), buffers.act.data(),
                                        buffers.hidden_states.data(), buffers.hidden_states.data(),
                                        static_cast<int>(H), static_cast<int>(inter_dim));
            }
        } else if (weights.quant_mode == QuantMode::INT4_G128) {
            if (do_dp4a) {
                cuda::gemv_int4_g128_dp4a_residual(lw.down_proj_quant.qweight.data(),
                                                  lw.down_proj_quant.scales.data(), buffers.q8_act_inter.data(),
                                                  buffers.act_scale.data(), buffers.hidden_states.data(),
                                                  buffers.hidden_states.data(),
                                                  static_cast<int>(H), static_cast<int>(inter_dim));
            } else {
                cuda::gemv_int4_g128_residual(lw.down_proj_quant.qweight.data(),
                                             lw.down_proj_quant.scales.data(), buffers.act.data(),
                                             buffers.hidden_states.data(), buffers.hidden_states.data(),
                                             static_cast<int>(H), static_cast<int>(inter_dim));
            }
        } else {
            cuda::gemv_residual(lw.down_proj_weight.data(), buffers.act.data(),
                                buffers.hidden_states.data(), buffers.hidden_states.data(),
                                static_cast<int>(H), static_cast<int>(inter_dim));
        }
        timed_end(lp + "down_proj_residual");
    }

    kv_cache.set_current_seq_len(current_pos + 1);

    // 4. Final RMSNorm
    timed_begin("final_rmsnorm");
    cuda::rmsnorm(buffers.hidden_states.data(), weights.final_norm.data(),
                  buffers.norm_out.data(), 1, H, config.rms_norm_eps);
    timed_end("final_rmsnorm");

    // 5. LM Head: [1, H] * [V, H]^T -> [1, V] in FP32
    float* target_logits = d_last_logits ? d_last_logits : buffers.logits.data();
    timed_begin("lm_head");
    if (weights.embed_is_quantized) {
        if (!use_legacy_gemv) {
            // Quantize final norm_out for dp4a LM head
            cuda::quantize_activation_int8(buffers.norm_out.data(), buffers.q8_act_H.data(),
                                           buffers.act_scale.data(), static_cast<int>(H));
            cuda::gemv_int8_dp4a_f32(
                reinterpret_cast<const int8_t*>(weights.embed_quant.qweight.data()),
                weights.embed_quant.scales.data(),
                buffers.q8_act_H.data(),
                buffers.act_scale.data(),
                target_logits,
                static_cast<int>(V), static_cast<int>(H));
        } else {
            // Legacy GEMV directly to FP32 logits
            cuda::gemv_int8_f32(
                reinterpret_cast<const int8_t*>(weights.embed_quant.qweight.data()),
                weights.embed_quant.scales.data(),
                buffers.norm_out.data(),
                target_logits,
                static_cast<int>(V), static_cast<int>(H));
        }
    } else {
        cuda::matmul_transB_f32(cublas_handle, buffers.norm_out.data(),
                               weights.embed_tokens.data(), target_logits,
                               1, static_cast<int>(H), static_cast<int>(V));
    }
    timed_end("lm_head");

    // Download only last-position logits (1 * V) if requested (skip when d_last_logits provided for async pipeline)
    std::vector<float> logits_f;
    if (!d_last_logits) {
        logits_f.resize(V);
        CUDA_CHECK(cudaMemcpy(logits_f.data(), target_logits,
                   V * sizeof(float), cudaMemcpyDeviceToHost));
    }
    return logits_f;
}

// ----------------------------------------------------------------------------
// Graphable Decode Step Implementation (Task P11a D3)
// ----------------------------------------------------------------------------
void decode_step_gpu_graphable(
    const ModelConfig& config,
    const ModelWeightsGPU& weights,
    ActivationBuffersGPU& buffers,
    KVCache& kv_cache,
    cublasHandle_t cublas_handle,
    float* d_last_logits,
    bool use_legacy_gemv,
    cudaStream_t stream
) {
    int64_t H = config.hidden_size;
    int64_t num_q_heads = config.num_attention_heads;
    int64_t num_kv_heads = config.num_key_value_heads;
    int64_t head_dim = config.head_dim;
    int64_t kv_dim = num_kv_heads * head_dim;
    int64_t inter_dim = config.intermediate_size;
    int64_t V = config.vocab_size;

    // 1. Embedding lookup (reads token_ids_d from device memory)
    cuda::embedding_lookup(buffers.token_ids_d.data(), weights.embed_tokens.data(),
                           buffers.hidden_states.data(), 1, H, stream);

    for (int layer_idx = 0; layer_idx < config.num_hidden_layers; ++layer_idx) {
        const auto& lw = weights.layers[layer_idx];

        // 2a. Input RMSNorm
        cuda::rmsnorm(buffers.hidden_states.data(), lw.input_layernorm.data(),
                      buffers.norm_out.data(), 1, H, config.rms_norm_eps, stream);

        bool do_dp4a = !use_legacy_gemv && (weights.quant_mode == QuantMode::INT8 || weights.quant_mode == QuantMode::INT4_G128);
        if (do_dp4a) {
            cuda::quantize_activation_int8(buffers.norm_out.data(), buffers.q8_act_H.data(),
                                           buffers.act_scale.data(), static_cast<int>(H), stream);
        }

        // 2b. Q/K/V projections
        if (lw.has_fused_qkv) {
            int qkv_dim = static_cast<int>(H + 2 * kv_dim);
            if (weights.quant_mode == QuantMode::INT8) {
                if (do_dp4a) {
                    cuda::gemv_int8_dp4a_bias(reinterpret_cast<const int8_t*>(lw.qkv_proj_quant.qweight.data()),
                                             lw.qkv_proj_quant.scales.data(), buffers.q8_act_H.data(),
                                             buffers.act_scale.data(), lw.qkv_proj_bias.data(), buffers.qkv_out.data(),
                                             qkv_dim, static_cast<int>(H), stream);
                } else {
                    cuda::gemv_int8_bias(reinterpret_cast<const int8_t*>(lw.qkv_proj_quant.qweight.data()),
                                        lw.qkv_proj_quant.scales.data(), buffers.norm_out.data(),
                                        lw.qkv_proj_bias.data(), buffers.qkv_out.data(),
                                        qkv_dim, static_cast<int>(H), stream);
                }
            } else if (weights.quant_mode == QuantMode::INT4_G128) {
                if (do_dp4a) {
                    cuda::gemv_int4_g128_dp4a_bias(lw.qkv_proj_quant.qweight.data(),
                                                  lw.qkv_proj_quant.scales.data(), buffers.q8_act_H.data(),
                                                  buffers.act_scale.data(), lw.qkv_proj_bias.data(), buffers.qkv_out.data(),
                                                  qkv_dim, static_cast<int>(H), stream);
                } else {
                    cuda::gemv_int4_g128_bias(lw.qkv_proj_quant.qweight.data(),
                                             lw.qkv_proj_quant.scales.data(), buffers.norm_out.data(),
                                             lw.qkv_proj_bias.data(), buffers.qkv_out.data(),
                                             qkv_dim, static_cast<int>(H), stream);
                }
            } else {
                cuda::gemv_bias(lw.qkv_proj_weight.data(), buffers.norm_out.data(),
                                lw.qkv_proj_bias.data(), buffers.qkv_out.data(),
                                qkv_dim, static_cast<int>(H), stream);
            }
        } else {
            if (weights.quant_mode == QuantMode::INT8) {
                if (do_dp4a) {
                    cuda::gemv_int8_dp4a_bias(reinterpret_cast<const int8_t*>(lw.q_proj_quant.qweight.data()),
                                             lw.q_proj_quant.scales.data(), buffers.q8_act_H.data(),
                                             buffers.act_scale.data(), lw.q_proj_bias.data(), buffers.q.data(),
                                             static_cast<int>(H), static_cast<int>(H), stream);
                    cuda::gemv_int8_dp4a_bias(reinterpret_cast<const int8_t*>(lw.k_proj_quant.qweight.data()),
                                             lw.k_proj_quant.scales.data(), buffers.q8_act_H.data(),
                                             buffers.act_scale.data(), lw.k_proj_bias.data(), buffers.k.data(),
                                             static_cast<int>(kv_dim), static_cast<int>(H), stream);
                    cuda::gemv_int8_dp4a_bias(reinterpret_cast<const int8_t*>(lw.v_proj_quant.qweight.data()),
                                             lw.v_proj_quant.scales.data(), buffers.q8_act_H.data(),
                                             buffers.act_scale.data(), lw.v_proj_bias.data(), buffers.v.data(),
                                             static_cast<int>(kv_dim), static_cast<int>(H), stream);
                } else {
                    cuda::gemv_int8_bias(reinterpret_cast<const int8_t*>(lw.q_proj_quant.qweight.data()),
                                        lw.q_proj_quant.scales.data(), buffers.norm_out.data(),
                                        lw.q_proj_bias.data(), buffers.q.data(),
                                        static_cast<int>(H), static_cast<int>(H), stream);
                    cuda::gemv_int8_bias(reinterpret_cast<const int8_t*>(lw.k_proj_quant.qweight.data()),
                                        lw.k_proj_quant.scales.data(), buffers.norm_out.data(),
                                        lw.k_proj_bias.data(), buffers.k.data(),
                                        static_cast<int>(kv_dim), static_cast<int>(H), stream);
                    cuda::gemv_int8_bias(reinterpret_cast<const int8_t*>(lw.v_proj_quant.qweight.data()),
                                        lw.v_proj_quant.scales.data(), buffers.norm_out.data(),
                                        lw.v_proj_bias.data(), buffers.v.data(),
                                        static_cast<int>(kv_dim), static_cast<int>(H), stream);
                }
            } else if (weights.quant_mode == QuantMode::INT4_G128) {
                if (do_dp4a) {
                    cuda::gemv_int4_g128_dp4a_bias(lw.q_proj_quant.qweight.data(),
                                                  lw.q_proj_quant.scales.data(), buffers.q8_act_H.data(),
                                                  buffers.act_scale.data(), lw.q_proj_bias.data(), buffers.q.data(),
                                                  static_cast<int>(H), static_cast<int>(H), stream);
                    cuda::gemv_int4_g128_dp4a_bias(lw.k_proj_quant.qweight.data(),
                                                  lw.k_proj_quant.scales.data(), buffers.q8_act_H.data(),
                                                  buffers.act_scale.data(), lw.k_proj_bias.data(), buffers.k.data(),
                                                  static_cast<int>(kv_dim), static_cast<int>(H), stream);
                    cuda::gemv_int4_g128_dp4a_bias(lw.v_proj_quant.qweight.data(),
                                                  lw.v_proj_quant.scales.data(), buffers.q8_act_H.data(),
                                                  buffers.act_scale.data(), lw.v_proj_bias.data(), buffers.v.data(),
                                                  static_cast<int>(kv_dim), static_cast<int>(H), stream);
                } else {
                    cuda::gemv_int4_g128_bias(lw.q_proj_quant.qweight.data(),
                                             lw.q_proj_quant.scales.data(), buffers.norm_out.data(),
                                             lw.q_proj_bias.data(), buffers.q.data(),
                                             static_cast<int>(H), static_cast<int>(H), stream);
                    cuda::gemv_int4_g128_bias(lw.k_proj_quant.qweight.data(),
                                             lw.k_proj_quant.scales.data(), buffers.norm_out.data(),
                                             lw.k_proj_bias.data(), buffers.k.data(),
                                             static_cast<int>(kv_dim), static_cast<int>(H), stream);
                    cuda::gemv_int4_g128_bias(lw.v_proj_quant.qweight.data(),
                                             lw.v_proj_quant.scales.data(), buffers.norm_out.data(),
                                             lw.v_proj_bias.data(), buffers.v.data(),
                                             static_cast<int>(kv_dim), static_cast<int>(H), stream);
                }
            } else {
                cuda::gemv_bias(lw.q_proj_weight.data(), buffers.norm_out.data(),
                                lw.q_proj_bias.data(), buffers.q.data(),
                                static_cast<int>(H), static_cast<int>(H), stream);
                cuda::gemv_bias(lw.k_proj_weight.data(), buffers.norm_out.data(),
                                lw.k_proj_bias.data(), buffers.k.data(),
                                static_cast<int>(kv_dim), static_cast<int>(H), stream);
                cuda::gemv_bias(lw.v_proj_weight.data(), buffers.norm_out.data(),
                                lw.v_proj_bias.data(), buffers.v.data(),
                                static_cast<int>(kv_dim), static_cast<int>(H), stream);
            }
        }

        half* q_step_ptr = lw.has_fused_qkv ? buffers.qkv_out.data() : buffers.q.data();
        half* k_step_ptr = lw.has_fused_qkv ? (buffers.qkv_out.data() + H) : buffers.k.data();
        half* v_step_ptr = lw.has_fused_qkv ? (buffers.qkv_out.data() + H + kv_dim) : buffers.v.data();

        // 2c. RoPE with device pointer position
        cuda::apply_rope_device_pos(q_step_ptr, k_step_ptr,
                                    1, num_q_heads, num_kv_heads, head_dim,
                                    buffers.d_current_pos.data(), config.rope_theta, stream);

        // Append key and value to cache using device pointer position
        cuda::append_kv_cache_device_pos(
            k_step_ptr, v_step_ptr,
            kv_cache.get_head_ptr(layer_idx, 0, 0),
            kv_cache.get_head_ptr(layer_idx, 1, 0),
            1, buffers.d_current_pos.data(),
            static_cast<int>(num_kv_heads),
            kv_cache.max_seq(),
            static_cast<int>(head_dim),
            stream
        );

        // 2d. Decode Attention (reads sequence length from buffers.decode_seq_len device buffer)
        cuda::KVView k_view{
            kv_cache.get_head_ptr(layer_idx, 0, 0),
            0,
            static_cast<int64_t>(kv_cache.max_seq() * head_dim),
            static_cast<int64_t>(head_dim),
            1
        };
        cuda::KVView v_view{
            kv_cache.get_head_ptr(layer_idx, 1, 0),
            0,
            static_cast<int64_t>(kv_cache.max_seq() * head_dim),
            static_cast<int64_t>(head_dim),
            1
        };
        cuda::decode_gqa_split(
            q_step_ptr,
            k_view,
            v_view,
            buffers.decode_seq_len.data(),
            buffers.attn_out.data(),
            1,
            static_cast<int>(num_q_heads),
            static_cast<int>(num_kv_heads),
            static_cast<int>(head_dim),
            kv_cache.max_seq(),
            buffers.split_kv_buf.data(),
            stream
        );

        if (do_dp4a) {
            cuda::quantize_activation_int8(buffers.attn_out.data(), buffers.q8_act_H.data(),
                                           buffers.act_scale.data(), static_cast<int>(H), stream);
        }

        // 2e. Output projection fused with residual add
        if (weights.quant_mode == QuantMode::INT8) {
            if (do_dp4a) {
                cuda::gemv_int8_dp4a_residual(reinterpret_cast<const int8_t*>(lw.o_proj_quant.qweight.data()),
                                             lw.o_proj_quant.scales.data(), buffers.q8_act_H.data(),
                                             buffers.act_scale.data(), buffers.hidden_states.data(),
                                             buffers.hidden_states.data(),
                                             static_cast<int>(H), static_cast<int>(H), stream);
            } else {
                cuda::gemv_int8_residual(reinterpret_cast<const int8_t*>(lw.o_proj_quant.qweight.data()),
                                        lw.o_proj_quant.scales.data(), buffers.attn_out.data(),
                                        buffers.hidden_states.data(), buffers.hidden_states.data(),
                                        static_cast<int>(H), static_cast<int>(H), stream);
            }
        } else if (weights.quant_mode == QuantMode::INT4_G128) {
            if (do_dp4a) {
                cuda::gemv_int4_g128_dp4a_residual(lw.o_proj_quant.qweight.data(),
                                                  lw.o_proj_quant.scales.data(), buffers.q8_act_H.data(),
                                                  buffers.act_scale.data(), buffers.hidden_states.data(),
                                                  buffers.hidden_states.data(),
                                                  static_cast<int>(H), static_cast<int>(H), stream);
            } else {
                cuda::gemv_int4_g128_residual(lw.o_proj_quant.qweight.data(),
                                             lw.o_proj_quant.scales.data(), buffers.attn_out.data(),
                                             buffers.hidden_states.data(), buffers.hidden_states.data(),
                                             static_cast<int>(H), static_cast<int>(H), stream);
            }
        } else {
            cuda::gemv_residual(lw.o_proj_weight.data(), buffers.attn_out.data(),
                                buffers.hidden_states.data(), buffers.hidden_states.data(),
                                static_cast<int>(H), static_cast<int>(H), stream);
        }

        // 3a. Post-attention RMSNorm
        cuda::rmsnorm(buffers.hidden_states.data(), lw.post_attention_layernorm.data(),
                      buffers.norm_out.data(), 1, H, config.rms_norm_eps, stream);

        if (do_dp4a) {
            cuda::quantize_activation_int8(buffers.norm_out.data(), buffers.q8_act_H.data(),
                                           buffers.act_scale.data(), static_cast<int>(H), stream);
        }

        // 3b. Gate & Up projections
        if (lw.has_fused_gate_up) {
            int gate_up_dim = static_cast<int>(2 * inter_dim);
            if (weights.quant_mode == QuantMode::INT8) {
                if (do_dp4a) {
                    cuda::gemv_int8_dp4a(reinterpret_cast<const int8_t*>(lw.gate_up_proj_quant.qweight.data()),
                                        lw.gate_up_proj_quant.scales.data(), buffers.q8_act_H.data(),
                                        buffers.act_scale.data(), buffers.gate_up_out.data(),
                                        gate_up_dim, static_cast<int>(H), stream);
                } else {
                    cuda::gemv_int8(reinterpret_cast<const int8_t*>(lw.gate_up_proj_quant.qweight.data()),
                                   lw.gate_up_proj_quant.scales.data(), buffers.norm_out.data(),
                                   buffers.gate_up_out.data(),
                                   gate_up_dim, static_cast<int>(H), stream);
                }
            } else if (weights.quant_mode == QuantMode::INT4_G128) {
                if (do_dp4a) {
                    cuda::gemv_int4_g128_dp4a(lw.gate_up_proj_quant.qweight.data(),
                                             lw.gate_up_proj_quant.scales.data(), buffers.q8_act_H.data(),
                                             buffers.act_scale.data(), buffers.gate_up_out.data(),
                                             gate_up_dim, static_cast<int>(H), stream);
                } else {
                    cuda::gemv_int4_g128(lw.gate_up_proj_quant.qweight.data(),
                                        lw.gate_up_proj_quant.scales.data(), buffers.norm_out.data(),
                                        buffers.gate_up_out.data(),
                                        gate_up_dim, static_cast<int>(H), stream);
                }
            } else {
                cuda::gemv(lw.gate_up_proj_weight.data(), buffers.norm_out.data(),
                           buffers.gate_up_out.data(),
                           gate_up_dim, static_cast<int>(H), stream);
            }
        } else {
            if (weights.quant_mode == QuantMode::INT8) {
                if (do_dp4a) {
                    cuda::gemv_int8_dp4a(reinterpret_cast<const int8_t*>(lw.gate_proj_quant.qweight.data()),
                                        lw.gate_proj_quant.scales.data(), buffers.q8_act_H.data(),
                                        buffers.act_scale.data(), buffers.gate.data(),
                                        static_cast<int>(inter_dim), static_cast<int>(H), stream);
                    cuda::gemv_int8_dp4a(reinterpret_cast<const int8_t*>(lw.up_proj_quant.qweight.data()),
                                        lw.up_proj_quant.scales.data(), buffers.q8_act_H.data(),
                                        buffers.act_scale.data(), buffers.up.data(),
                                        static_cast<int>(inter_dim), static_cast<int>(H), stream);
                } else {
                    cuda::gemv_int8(reinterpret_cast<const int8_t*>(lw.gate_proj_quant.qweight.data()),
                                   lw.gate_proj_quant.scales.data(), buffers.norm_out.data(),
                                   buffers.gate.data(),
                                   static_cast<int>(inter_dim), static_cast<int>(H), stream);
                    cuda::gemv_int8(reinterpret_cast<const int8_t*>(lw.up_proj_quant.qweight.data()),
                                   lw.up_proj_quant.scales.data(), buffers.norm_out.data(),
                                   buffers.up.data(),
                                   static_cast<int>(inter_dim), static_cast<int>(H), stream);
                }
            } else if (weights.quant_mode == QuantMode::INT4_G128) {
                if (do_dp4a) {
                    cuda::gemv_int4_g128_dp4a(lw.gate_proj_quant.qweight.data(),
                                             lw.gate_proj_quant.scales.data(), buffers.q8_act_H.data(),
                                             buffers.act_scale.data(), buffers.gate.data(),
                                             static_cast<int>(inter_dim), static_cast<int>(H), stream);
                    cuda::gemv_int4_g128_dp4a(lw.up_proj_quant.qweight.data(),
                                             lw.up_proj_quant.scales.data(), buffers.q8_act_H.data(),
                                             buffers.act_scale.data(), buffers.up.data(),
                                             static_cast<int>(inter_dim), static_cast<int>(H), stream);
                } else {
                    cuda::gemv_int4_g128(lw.gate_proj_quant.qweight.data(),
                                        lw.gate_proj_quant.scales.data(), buffers.norm_out.data(),
                                        buffers.gate.data(),
                                        static_cast<int>(inter_dim), static_cast<int>(H), stream);
                    cuda::gemv_int4_g128(lw.up_proj_quant.qweight.data(),
                                        lw.up_proj_quant.scales.data(), buffers.norm_out.data(),
                                        buffers.up.data(),
                                        static_cast<int>(inter_dim), static_cast<int>(H), stream);
                }
            } else {
                cuda::gemv(lw.gate_proj_weight.data(), buffers.norm_out.data(),
                           buffers.gate.data(),
                           static_cast<int>(inter_dim), static_cast<int>(H), stream);
                cuda::gemv(lw.up_proj_weight.data(), buffers.norm_out.data(),
                           buffers.up.data(),
                           static_cast<int>(inter_dim), static_cast<int>(H), stream);
            }
        }

        const half* gate_step_ptr = lw.has_fused_gate_up ? buffers.gate_up_out.data() : buffers.gate.data();
        const half* up_step_ptr = lw.has_fused_gate_up ? (buffers.gate_up_out.data() + inter_dim) : buffers.up.data();

        // 3c. SwiGLU
        cuda::swiglu(gate_step_ptr, up_step_ptr,
                    buffers.act.data(), inter_dim, stream);

        if (do_dp4a) {
            cuda::quantize_activation_int8(buffers.act.data(), buffers.q8_act_inter.data(),
                                           buffers.act_scale.data(), static_cast<int>(inter_dim), stream);
        }

        // 3d. Down projection fused with residual add
        if (weights.quant_mode == QuantMode::INT8) {
            if (do_dp4a) {
                cuda::gemv_int8_dp4a_residual(reinterpret_cast<const int8_t*>(lw.down_proj_quant.qweight.data()),
                                             lw.down_proj_quant.scales.data(), buffers.q8_act_inter.data(),
                                             buffers.act_scale.data(), buffers.hidden_states.data(),
                                             buffers.hidden_states.data(),
                                             static_cast<int>(H), static_cast<int>(inter_dim), stream);
            } else {
                cuda::gemv_int8_residual(reinterpret_cast<const int8_t*>(lw.down_proj_quant.qweight.data()),
                                        lw.down_proj_quant.scales.data(), buffers.act.data(),
                                        buffers.hidden_states.data(), buffers.hidden_states.data(),
                                        static_cast<int>(H), static_cast<int>(inter_dim), stream);
            }
        } else if (weights.quant_mode == QuantMode::INT4_G128) {
            if (do_dp4a) {
                cuda::gemv_int4_g128_dp4a_residual(lw.down_proj_quant.qweight.data(),
                                                  lw.down_proj_quant.scales.data(), buffers.q8_act_inter.data(),
                                                  buffers.act_scale.data(), buffers.hidden_states.data(),
                                                  buffers.hidden_states.data(),
                                                  static_cast<int>(H), static_cast<int>(inter_dim), stream);
            } else {
                cuda::gemv_int4_g128_residual(lw.down_proj_quant.qweight.data(),
                                             lw.down_proj_quant.scales.data(), buffers.act.data(),
                                             buffers.hidden_states.data(), buffers.hidden_states.data(),
                                             static_cast<int>(H), static_cast<int>(inter_dim), stream);
            }
        } else {
            cuda::gemv_residual(lw.down_proj_weight.data(), buffers.act.data(),
                                buffers.hidden_states.data(), buffers.hidden_states.data(),
                                static_cast<int>(H), static_cast<int>(inter_dim), stream);
        }
    }

    // 4. Final RMSNorm
    cuda::rmsnorm(buffers.hidden_states.data(), weights.final_norm.data(),
                  buffers.norm_out.data(), 1, H, config.rms_norm_eps, stream);

    // 5. LM Head
    if (weights.embed_is_quantized) {
        if (!use_legacy_gemv) {
            cuda::quantize_activation_int8(buffers.norm_out.data(), buffers.q8_act_H.data(),
                                           buffers.act_scale.data(), static_cast<int>(H), stream);
            cuda::gemv_int8_dp4a_f32(
                reinterpret_cast<const int8_t*>(weights.embed_quant.qweight.data()),
                weights.embed_quant.scales.data(),
                buffers.q8_act_H.data(),
                buffers.act_scale.data(),
                d_last_logits,
                static_cast<int>(V), static_cast<int>(H), stream);
        } else {
            cuda::gemv_int8_f32(
                reinterpret_cast<const int8_t*>(weights.embed_quant.qweight.data()),
                weights.embed_quant.scales.data(),
                buffers.norm_out.data(),
                d_last_logits,
                static_cast<int>(V), static_cast<int>(H), stream);
        }
    } else {
        cublasSetStream(cublas_handle, stream);
        cuda::matmul_transB_f32(cublas_handle, buffers.norm_out.data(),
                               weights.embed_tokens.data(), d_last_logits,
                               1, static_cast<int>(H), static_cast<int>(V));
    }
}

void DecodeCudaGraph::capture(
    const ModelConfig& config,
    const ModelWeightsGPU& weights,
    ActivationBuffersGPU& buffers,
    KVCache& kv_cache,
    cublasHandle_t cublas_handle,
    float* d_last_logits,
    bool use_legacy_gemv,
    cudaStream_t stream
) {
    destroy();
    if (!stream_) {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }
    cudaStream_t cap_stream = stream ? stream : stream_;

    CUDA_CHECK(cudaStreamBeginCapture(cap_stream, cudaStreamCaptureModeGlobal));
    decode_step_gpu_graphable(config, weights, buffers, kv_cache, cublas_handle, d_last_logits, use_legacy_gemv, cap_stream);
    CUDA_CHECK(cudaStreamEndCapture(cap_stream, &graph_));
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0));
    captured_ = true;
}

void DecodeCudaGraph::launch(
    int64_t token_id,
    int current_pos,
    ActivationBuffersGPU& buffers,
    cudaStream_t stream
) {
    if (!captured_) return;
    cudaStream_t run_stream = stream ? stream : (stream_ ? stream_ : 0);
    int total_seq = current_pos + 1;
    CUDA_CHECK(cudaMemcpyAsync(buffers.token_ids_d.data(), &token_id, sizeof(int64_t), cudaMemcpyHostToDevice, run_stream));
    CUDA_CHECK(cudaMemcpyAsync(buffers.d_current_pos.data(), &current_pos, sizeof(int), cudaMemcpyHostToDevice, run_stream));
    CUDA_CHECK(cudaMemcpyAsync(buffers.decode_seq_len.data(), &total_seq, sizeof(int), cudaMemcpyHostToDevice, run_stream));
    CUDA_CHECK(cudaGraphLaunch(graph_exec_, run_stream));
}

void DecodeCudaGraph::destroy() {
    if (graph_exec_) {
        cudaGraphExecDestroy(graph_exec_);
        graph_exec_ = nullptr;
    }
    if (graph_) {
        cudaGraphDestroy(graph_);
        graph_ = nullptr;
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    captured_ = false;
}

} // namespace engine

