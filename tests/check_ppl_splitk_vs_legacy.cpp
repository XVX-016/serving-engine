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
#include <string>

using namespace engine;

int main(int argc, char** argv) {
    std::string tokens_file = "data/ppl_tokens.bin";
    int max_len = 1024;
    int stride = 512;
    int max_chunks = 10; // default 10 chunks for quick test, or 0 for all
    if (argc > 1) max_chunks = std::stoi(argv[1]);

    const std::string model_dir = "data/models/qwen2.5-0.5b-int4-g128-head-int8-fused";
    ModelConfig cfg = ModelConfig::from_file(model_dir + "/config.json");
    SafetensorsLoader loader;
    if (!loader.open(model_dir + "/model.safetensors")) {
        std::cerr << "Failed to open safetensors: " << model_dir << "\n";
        return 1;
    }
    ModelWeightsGPU weights;
    weights.init_from_safetensors(cfg, loader);
    ActivationBuffersGPU buffers;
    buffers.allocate(cfg, 4096);
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    std::ifstream tf(tokens_file, std::ios::binary);
    if (!tf.is_open()) {
        std::cerr << "Failed to open tokens file: " << tokens_file << "\n";
        return 1;
    }
    uint32_t num_tokens = 0;
    tf.read(reinterpret_cast<char*>(&num_tokens), sizeof(num_tokens));
    std::vector<int32_t> raw_tokens(num_tokens);
    tf.read(reinterpret_cast<char*>(raw_tokens.data()), num_tokens * sizeof(int32_t));
    std::vector<int64_t> all_tokens(raw_tokens.begin(), raw_tokens.end());

    std::cout << "Loaded " << all_tokens.size() << " tokens from " << tokens_file << "\n";

    auto eval_ppl = [&](bool use_legacy) -> double {
        KVCache kv(cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, 4096);
        double total_nll = 0.0;
        int64_t total_pred = 0;
        int64_t seq_len = static_cast<int64_t>(all_tokens.size());
        int64_t prev_end = 0;
        int chunk_idx = 0;

        for (int64_t begin_loc = 0; begin_loc < seq_len; begin_loc += stride) {
            if (max_chunks > 0 && chunk_idx >= max_chunks) break;
            int64_t end_loc = std::min(begin_loc + max_len, seq_len);
            int64_t trg_len = end_loc - prev_end;
            int64_t cur_len = end_loc - begin_loc;

            std::vector<int64_t> chunk(all_tokens.begin() + begin_loc, all_tokens.begin() + end_loc);
            
            // Evaluate chunk step by step with KV cache to exercise decode GEMV
            kv.set_current_seq_len(0);
            std::vector<int64_t> first_t = {chunk[0]};
            prefill_gpu(first_t, cfg, weights, buffers, kv, cublas_handle);

            std::vector<float> logits(cur_len * cfg.vocab_size);
            // decode rest
            for (int64_t pos = 1; pos < cur_len; ++pos) {
                std::vector<float> step_logits = decode_step_gpu(
                    chunk[pos], static_cast<int>(pos), cfg, weights, buffers, kv, cublas_handle,
                    nullptr, nullptr, use_legacy
                );
                std::copy(step_logits.begin(), step_logits.end(), logits.begin() + pos * cfg.vocab_size);
            }

            for (int64_t t = cur_len - trg_len; t < cur_len; ++t) {
                int64_t target_tok = chunk[t];
                int64_t pred_pos = t - 1;
                if (pred_pos < 0) continue;
                const float* row = logits.data() + pred_pos * cfg.vocab_size;
                float max_val = -1e30f;
                for (int v = 0; v < cfg.vocab_size; ++v) {
                    if (row[v] > max_val) max_val = row[v];
                }
                double sum_exp = 0.0;
                for (int v = 0; v < cfg.vocab_size; ++v) {
                    sum_exp += std::exp(static_cast<double>(row[v] - max_val));
                }
                double lse = max_val + std::log(sum_exp);
                double target_lp = static_cast<double>(row[target_tok]) - lse;
                total_nll += (-target_lp);
                total_pred++;
            }

            prev_end = end_loc;
            chunk_idx++;
            if (end_loc == seq_len) break;
        }

        return std::exp(total_nll / total_pred);
    };

    std::cout << "Evaluating Legacy GEMV PPL (" << max_chunks << " chunks)...\n";
    double ppl_leg = eval_ppl(true);
    std::cout << "Legacy PPL: " << ppl_leg << "\n";

    std::cout << "Evaluating Split-K GEMV PPL (" << max_chunks << " chunks)...\n";
    double ppl_spk = eval_ppl(false);
    std::cout << "Split-K PPL: " << ppl_spk << "\n";

    std::cout << "Delta PPL (Split-K - Legacy): " << (ppl_spk - ppl_leg) << " (" 
              << ((ppl_spk - ppl_leg) / ppl_leg * 100.0) << "%)\n";

    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    return 0;
}
