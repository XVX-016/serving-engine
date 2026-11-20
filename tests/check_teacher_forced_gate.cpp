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
#include <algorithm>

using namespace engine;

static std::vector<int64_t> read_token_ids(const std::string& pname) {
    std::string token_file = "data/reference/" + pname + "/token_ids.npy";
    std::ifstream tf(token_file, std::ios::binary);
    if (!tf.is_open()) return {};
    char magic[6]; tf.read(magic, 6);
    char ver[2];   tf.read(ver, 2);
    uint16_t hlen = 0;
    tf.read(reinterpret_cast<char*>(&hlen), 2);
    std::string header(hlen, ' ');
    tf.read(&header[0], hlen);
    tf.seekg(0, std::ios::end);
    size_t file_len = static_cast<size_t>(tf.tellg());
    size_t data_len = file_len - (10 + hlen);
    size_t S = data_len / sizeof(int64_t);
    tf.seekg(10 + hlen, std::ios::beg);
    std::vector<int64_t> tokens(S);
    tf.read(reinterpret_cast<char*>(tokens.data()), data_len);
    return tokens;
}

static std::vector<float> read_ref_logits(const std::string& pname) {
    std::string npy_path = "data/reference/" + pname + "/logits.npy";
    std::ifstream tf(npy_path, std::ios::binary);
    if (!tf.is_open()) return {};
    char magic[6]; tf.read(magic, 6);
    char ver[2];   tf.read(ver, 2);
    uint16_t hlen = 0;
    tf.read(reinterpret_cast<char*>(&hlen), 2);
    std::string header(hlen, ' ');
    tf.read(&header[0], hlen);
    tf.seekg(0, std::ios::end);
    size_t file_len = static_cast<size_t>(tf.tellg());
    size_t data_len = file_len - (10 + hlen);
    size_t N = data_len / sizeof(float);
    tf.seekg(10 + hlen, std::ios::beg);
    std::vector<float> data(N);
    tf.read(reinterpret_cast<char*>(data.data()), data_len);
    return data;
}

int main() {
    const std::string model_dir = "data/models/qwen2.5-0.5b-int4-g128-head-int8-fused";
    ModelConfig cfg = ModelConfig::from_file(model_dir + "/config.json");
    SafetensorsLoader loader;
    if (!loader.open(model_dir + "/model.safetensors")) {
        std::cerr << "Failed to open safetensors\n";
        return 1;
    }
    ModelWeightsGPU weights;
    weights.init_from_safetensors(cfg, loader);
    ActivationBuffersGPU buffers;
    buffers.allocate(cfg, 4096);
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    std::vector<std::string> prompt_names = {
        "01_short", "02_medium", "03_long", "04_code", "05_unicode", "06_cjk_emoji", "07_chat_template", "08_whitespace_indent"
    };

    std::cout << "========================================================================================\n";
    std::cout << " Gate 1.7 Verification: INT4 Split-K vs Legacy Teacher-Forced Error vs CPU FP32 Ref\n";
    std::cout << "========================================================================================\n";
    std::cout << std::left << std::setw(24) << "Prompt"
              << std::setw(16) << "MeanErr(Leg)"
              << std::setw(16) << "MeanErr(SpK)"
              << std::setw(12) << "MeanRatio"
              << std::setw(16) << "MaxErr(Leg)"
              << std::setw(16) << "MaxErr(SpK)"
              << std::setw(12) << "MaxRatio"
              << "Gate (<=1.5x)\n";
    std::cout << std::string(105, '-') << "\n";

    bool all_pass = true;

    for (const auto& pname : prompt_names) {
        std::vector<int64_t> tokens = read_token_ids(pname);
        std::vector<float> ref_logits = read_ref_logits(pname);
        if (tokens.empty() || ref_logits.empty()) {
            std::cerr << "Skipping " << pname << "\n";
            continue;
        }

        size_t S = tokens.size();

        // 1. Run Legacy teacher-forced
        KVCache kv_leg(cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, 4096);
        kv_leg.set_current_seq_len(0);
        std::vector<int64_t> prefill_tok = {tokens[0]};
        std::vector<float> pre_leg = prefill_gpu(prefill_tok, cfg, weights, buffers, kv_leg, cublas_handle);
        std::vector<float> leg_logits(S * cfg.vocab_size);
        std::copy(pre_leg.begin(), pre_leg.end(), leg_logits.begin());

        for (size_t pos = 1; pos < S; ++pos) {
            std::vector<float> dec = decode_step_gpu(
                tokens[pos], static_cast<int>(pos), cfg, weights, buffers, kv_leg, cublas_handle,
                nullptr, nullptr, /*use_legacy_gemv=*/true
            );
            std::copy(dec.begin(), dec.end(), leg_logits.begin() + pos * cfg.vocab_size);
        }

        // 2. Run Split-K teacher-forced
        KVCache kv_spk(cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, 4096);
        kv_spk.set_current_seq_len(0);
        std::vector<float> pre_spk = prefill_gpu(prefill_tok, cfg, weights, buffers, kv_spk, cublas_handle);
        std::vector<float> spk_logits(S * cfg.vocab_size);
        std::copy(pre_spk.begin(), pre_spk.end(), spk_logits.begin());

        for (size_t pos = 1; pos < S; ++pos) {
            std::vector<float> dec = decode_step_gpu(
                tokens[pos], static_cast<int>(pos), cfg, weights, buffers, kv_spk, cublas_handle,
                nullptr, nullptr, /*use_legacy_gemv=*/false
            );
            std::copy(dec.begin(), dec.end(), spk_logits.begin() + pos * cfg.vocab_size);
        }

        // Compute error vs CPU FP32 reference
        double sum_err_leg = 0.0;
        double sum_err_spk = 0.0;
        float max_err_leg = 0.0f;
        float max_err_spk = 0.0f;

        for (size_t i = 0; i < S * cfg.vocab_size; ++i) {
            float err_leg = std::abs(leg_logits[i] - ref_logits[i]);
            float err_spk = std::abs(spk_logits[i] - ref_logits[i]);

            sum_err_leg += err_leg;
            sum_err_spk += err_spk;
            if (err_leg > max_err_leg) max_err_leg = err_leg;
            if (err_spk > max_err_spk) max_err_spk = err_spk;
        }

        double mean_err_leg = sum_err_leg / (S * cfg.vocab_size);
        double mean_err_spk = sum_err_spk / (S * cfg.vocab_size);

        double mean_ratio = mean_err_spk / mean_err_leg;
        double max_ratio = static_cast<double>(max_err_spk) / max_err_leg;

        bool pass = (mean_ratio <= 1.5) && (max_ratio <= 1.5);
        if (!pass) all_pass = false;

        std::cout << std::left << std::setw(24) << pname
                  << std::setw(16) << std::fixed << std::setprecision(5) << mean_err_leg
                  << std::setw(16) << mean_err_spk
                  << std::setw(12) << std::setprecision(3) << mean_ratio
                  << std::setw(16) << std::setprecision(5) << max_err_leg
                  << std::setw(16) << max_err_spk
                  << std::setw(12) << std::setprecision(3) << max_ratio
                  << (pass ? "PASS" : "FAIL") << "\n";
    }
    std::cout << std::string(105, '-') << "\n";
    std::cout << "Overall Gate 1.7 Ratio Rule: " << (all_pass ? "PASS" : "FAIL") << "\n";

    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    return 0;
}
