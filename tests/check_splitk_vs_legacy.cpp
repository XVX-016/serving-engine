// check_splitk_vs_legacy.cpp
// Compares Split-K vs Legacy INT4 GEMV kernels:
//   1. Max abs logit difference (one decode step per prompt)
//   2. 256-token greedy generation match (all 5 prompts)
//   3. Perplexity is handled via generate.exe --ppl-tokens (see P11d report)
//
// Both runs use the same INT4 fused model; use_legacy_gemv controls which kernel.

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
#include <cstdint>
#include <string>
#include <algorithm>

using namespace engine;

// Read token_ids.npy (same helper as check_fused_logits.cpp)
static std::vector<int64_t> read_token_ids(const std::string& pname) {
    std::string token_file = "data/reference/" + pname + "/token_ids.npy";
    std::ifstream tf(token_file, std::ios::binary);
    if (!tf.is_open()) {
        std::cerr << "Cannot open " << token_file << "\n";
        return {};
    }
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
    tf.read(reinterpret_cast<char*>(tokens.data()), static_cast<std::streamsize>(data_len));
    return tokens;
}

int main() {
    const std::string model_dir = "data/models/qwen2.5-0.5b-int4-g128-head-int8-fused";

    ModelConfig cfg = ModelConfig::from_file(model_dir + "/config.json");

    SafetensorsLoader loader;
    if (!loader.open(model_dir + "/model.safetensors")) {
        std::cerr << "Failed to open safetensors: " << model_dir << "\n";
        return 1;
    }

    ModelWeightsGPU weights;
    if (!weights.init_from_safetensors(cfg, loader)) {
        std::cerr << "Failed to init weights\n";
        return 1;
    }

    ActivationBuffersGPU buffers;
    buffers.allocate(cfg, 4096);

    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    std::vector<std::string> prompt_names = {
        "01_short", "02_medium", "04_code", "06_cjk_emoji", "08_whitespace_indent"
    };

    std::cout << "========================================================================\n";
    std::cout << " INT4 Split-K vs Legacy GEMV: Logit Diff & 256-Token Greedy Match\n";
    std::cout << "========================================================================\n";
    std::cout << std::left << std::setw(26) << "Prompt"
              << std::setw(28) << "MaxAbsLogitDiff"
              << std::setw(16) << "GreedyMatch/256"
              << "\n";
    std::cout << std::string(70, '-') << "\n";

    int total_match = 0;
    float global_max_diff = 0.0f;

    for (const auto& pname : prompt_names) {
        std::vector<int64_t> prompt_toks = read_token_ids(pname);
        if (prompt_toks.empty()) {
            std::cerr << "Empty tokens for " << pname << "\n";
            return 1;
        }

        // --- Split-K run (use_legacy_gemv = false) ---
        KVCache kv_splitk(cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, 4096);
        kv_splitk.set_current_seq_len(0);
        prefill_gpu(prompt_toks, cfg, weights, buffers, kv_splitk, cublas_handle);
        CUDA_CHECK(cudaDeviceSynchronize());

        int64_t first_tok = prompt_toks.back();
        int first_pos = static_cast<int>(prompt_toks.size());

        std::vector<float> logits_splitk = decode_step_gpu(
            first_tok, first_pos, cfg, weights, buffers, kv_splitk, cublas_handle,
            nullptr, nullptr, /*use_legacy_gemv=*/false
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<int64_t> gen_splitk;
        {
            int64_t next_tok = first_tok;
            for (int step = 0; step < 256; ++step) {
                int pos = first_pos + step;
                std::vector<float> l = decode_step_gpu(
                    next_tok, pos, cfg, weights, buffers, kv_splitk, cublas_handle,
                    nullptr, nullptr, /*use_legacy_gemv=*/false
                );
                int best = static_cast<int>(
                    std::max_element(l.begin(), l.end()) - l.begin());
                gen_splitk.push_back(best);
                next_tok = best;
            }
        }

        // --- Legacy run (use_legacy_gemv = true) ---
        KVCache kv_legacy(cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, 4096);
        kv_legacy.set_current_seq_len(0);
        prefill_gpu(prompt_toks, cfg, weights, buffers, kv_legacy, cublas_handle);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> logits_legacy = decode_step_gpu(
            first_tok, first_pos, cfg, weights, buffers, kv_legacy, cublas_handle,
            nullptr, nullptr, /*use_legacy_gemv=*/true
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<int64_t> gen_legacy;
        {
            int64_t next_tok = first_tok;
            for (int step = 0; step < 256; ++step) {
                int pos = first_pos + step;
                std::vector<float> l = decode_step_gpu(
                    next_tok, pos, cfg, weights, buffers, kv_legacy, cublas_handle,
                    nullptr, nullptr, /*use_legacy_gemv=*/true
                );
                int best = static_cast<int>(
                    std::max_element(l.begin(), l.end()) - l.begin());
                gen_legacy.push_back(best);
                next_tok = best;
            }
        }

        // --- Compare logits ---
        float max_diff = 0.0f;
        int max_idx = 0;
        for (size_t i = 0; i < logits_splitk.size(); ++i) {
            float d = std::abs(logits_splitk[i] - logits_legacy[i]);
            if (d > max_diff) { max_diff = d; max_idx = static_cast<int>(i); }
        }
        if (max_diff > global_max_diff) global_max_diff = max_diff;

        // --- Compare 256-token greedy prefix ---
        int match = 0;
        for (size_t i = 0; i < gen_splitk.size(); ++i) {
            if (gen_splitk[i] == gen_legacy[i]) ++match;
            else {
                std::cout << "    [" << pname << " mismatch at step " << i << "]: splitk="
                          << gen_splitk[i] << " vs legacy=" << gen_legacy[i] << "\n";
                break;
            }
        }
        total_match += match;

        std::cout << std::left << std::setw(26) << pname
                  << std::scientific << std::setprecision(6)
                  << std::setw(28) << max_diff
                  << match << "/256  (max_idx=" << max_idx << ")\n";
    }

    std::cout << std::string(70, '-') << "\n";
    std::cout << "Global max abs logit diff (5 prompts): "
              << std::scientific << std::setprecision(6) << global_max_diff << "\n";
    std::cout << "Total greedy prefix match: " << total_match << " / " << (5*256) << "\n\n";

    // Split-K reorders FP32 partial sums across two warps per row.
    // Floating point summation is non-associative, so Delta > 0.0 is mathematically expected.
    // Numerical gate passes if max diff is bounded (< 5e-2, typical logit scale is ~10-20).
    if (global_max_diff < 5e-2f && total_match >= 1000) {
        std::cout << "GATE: PASS (numerical parity preserved across FP32 summation reordering)\n";
    } else {
        std::cout << "GATE: FAIL (unexpected divergence)\n";
    }

    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    return 0;
}
