#include "src/model/config.h"
#include "src/model/forward_gpu.h"
#include "src/runtime/kv_cache.h"
#include "src/loader/safetensors.h"
#include "src/common/cuda_utils.h"

#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <iomanip>

using namespace engine;

struct LogitDiffResult {
    float max_diff;
    int max_diff_idx;
};

LogitDiffResult compare_decode_logits(
    const std::string& model_unfused,
    const std::string& model_fused,
    const std::vector<int64_t>& prompt_tokens
) {
    ModelConfig cfg = ModelConfig::from_file(model_unfused + "/config.json");
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    // Load unfused
    SafetensorsLoader loader_u;
    if (!loader_u.open(model_unfused + "/model.safetensors")) {
        std::cerr << "Failed to open unfused safetensors: " << model_unfused << "\n";
        return {-1.0f, -1};
    }
    ModelWeightsGPU weights_gpu_u;
    weights_gpu_u.init_from_safetensors(cfg, loader_u);

    ActivationBuffersGPU buffers_u;
    buffers_u.allocate(cfg, 4096);
    KVCache kv_u(cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, 4096);

    // Prefill prompt
    kv_u.set_current_seq_len(0);
    prefill_gpu(prompt_tokens, cfg, weights_gpu_u, buffers_u, kv_u, cublas_handle);

    // Decode 1 step
    int next_tok = 100;
    int cur_pos = static_cast<int>(prompt_tokens.size());
    std::vector<float> logits_u = decode_step_gpu(
        next_tok, cur_pos, cfg, weights_gpu_u, buffers_u, kv_u, cublas_handle,
        nullptr, nullptr, true
    );

    // Load fused
    SafetensorsLoader loader_f;
    if (!loader_f.open(model_fused + "/model.safetensors")) {
        std::cerr << "Failed to open fused safetensors: " << model_fused << "\n";
        return {-1.0f, -1};
    }
    ModelWeightsGPU weights_gpu_f;
    weights_gpu_f.init_from_safetensors(cfg, loader_f);

    ActivationBuffersGPU buffers_f;
    buffers_f.allocate(cfg, 4096);
    KVCache kv_f(cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, 4096);

    // Prefill prompt
    kv_f.set_current_seq_len(0);
    prefill_gpu(prompt_tokens, cfg, weights_gpu_f, buffers_f, kv_f, cublas_handle);

    // Decode 1 step
    std::vector<float> logits_f = decode_step_gpu(
        next_tok, cur_pos, cfg, weights_gpu_f, buffers_f, kv_f, cublas_handle,
        nullptr, nullptr, true
    );

    cublasDestroy(cublas_handle);

    float max_d = 0.0f;
    int max_idx = 0;
    for (size_t i = 0; i < logits_u.size(); ++i) {
        float d = std::abs(logits_u[i] - logits_f[i]);
        if (d > max_d) {
            max_d = d;
            max_idx = static_cast<int>(i);
        }
    }
    return {max_d, max_idx};
}

std::vector<int64_t> read_token_ids(const std::string& pname) {
    std::string token_file = "data/reference/" + pname + "/token_ids.npy";
    std::ifstream tf(token_file, std::ios::binary);
    if (!tf.is_open()) return {};

    char magic[6];
    tf.read(magic, 6);
    char ver[2];
    tf.read(ver, 2);
    uint16_t hlen = 0;
    tf.read(reinterpret_cast<char*>(&hlen), 2);
    std::string header(hlen, ' ');
    tf.read(&header[0], hlen);

    tf.seekg(0, std::ios::end);
    size_t file_len = tf.tellg();
    size_t data_len = file_len - (10 + hlen);
    size_t S = data_len / sizeof(int64_t);

    tf.seekg(10 + hlen, std::ios::beg);
    std::vector<int64_t> tokens(S);
    tf.read(reinterpret_cast<char*>(tokens.data()), data_len);
    return tokens;
}

int main() {
    std::vector<std::string> prompt_names = {
        "01_short", "02_medium", "04_code", "06_cjk_emoji", "08_whitespace_indent"
    };

    std::vector<std::pair<std::string, std::pair<std::string, std::string>>> models = {
        {"FP16", {"data/models/hf-cache/snapshots/7ae557604adf67be50417f59c2c2f167def9a775", "data/models/qwen2.5-0.5b-fp16-fused"}},
        {"INT8", {"data/models/qwen2.5-0.5b-int8", "data/models/qwen2.5-0.5b-int8-fused"}},
        {"INT4", {"data/models/qwen2.5-0.5b-int4-g128-head-int8", "data/models/qwen2.5-0.5b-int4-g128-head-int8-fused"}}
    };

    std::cout << "========================================================================\n";
    std::cout << " Fused vs Unfused Decode Logits Verification (Max Abs Diff)\n";
    std::cout << "========================================================================\n";

    for (const auto& [prec, paths] : models) {
        std::cout << "\nTesting Precision: " << prec << "\n";
        float prec_max_diff = 0.0f;
        for (const auto& pname : prompt_names) {
            std::vector<int64_t> tokens = read_token_ids(pname);
            LogitDiffResult res = compare_decode_logits(paths.first, paths.second, tokens);
            std::cout << "  Prompt [" << std::setw(20) << std::left << pname << "]: Max Abs Diff = "
                      << std::scientific << std::setprecision(6) << res.max_diff
                      << " (at vocab index " << res.max_diff_idx << ")\n";
            if (res.max_diff > prec_max_diff) prec_max_diff = res.max_diff;
        }
        std::cout << "--> " << prec << " Global Max Diff across all 5 prompts: "
                  << std::scientific << std::setprecision(6) << prec_max_diff << "\n";
    }

    return 0;
}
