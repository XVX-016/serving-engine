#include "src/model/config.h"
#include "src/model/forward_gpu.h"
#include "src/runtime/kv_cache.h"
#include "src/loader/safetensors.h"
#include "src/common/cuda_utils.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <numeric>

using namespace engine;

int main() {
    const std::string model_dir = "data/models/qwen2.5-0.5b-int4-g128-head-int8-fused";
    ModelConfig cfg = ModelConfig::from_file(model_dir + "/config.json");
    SafetensorsLoader loader;
    loader.open(model_dir + "/model.safetensors");
    ModelWeightsGPU weights;
    weights.init_from_safetensors(cfg, loader);
    ActivationBuffersGPU buffers;
    buffers.allocate(cfg, 4096);
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    std::string token_file = "data/reference/06_cjk_emoji/token_ids.npy";
    std::ifstream tf(token_file, std::ios::binary);
    char magic[6]; tf.read(magic, 6);
    char ver[2]; tf.read(ver, 2);
    uint16_t hlen = 0; tf.read(reinterpret_cast<char*>(&hlen), 2);
    std::string header(hlen, ' '); tf.read(&header[0], hlen);
    tf.seekg(0, std::ios::end);
    size_t file_len = static_cast<size_t>(tf.tellg());
    size_t data_len = file_len - (10 + hlen);
    size_t S = data_len / sizeof(int64_t);
    tf.seekg(10 + hlen, std::ios::beg);
    std::vector<int64_t> prompt_toks(S);
    tf.read(reinterpret_cast<char*>(prompt_toks.data()), data_len);

    KVCache kv_splitk(cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, 4096);
    kv_splitk.set_current_seq_len(0);
    prefill_gpu(prompt_toks, cfg, weights, buffers, kv_splitk, cublas_handle);
    int64_t first_tok = prompt_toks.back();
    int first_pos = static_cast<int>(prompt_toks.size());

    std::vector<float> l0_splitk = decode_step_gpu(first_tok, first_pos, cfg, weights, buffers, kv_splitk, cublas_handle, nullptr, nullptr, false);
    int top0_splitk = std::max_element(l0_splitk.begin(), l0_splitk.end()) - l0_splitk.begin();

    std::vector<float> l1_splitk = decode_step_gpu(top0_splitk, first_pos + 1, cfg, weights, buffers, kv_splitk, cublas_handle, nullptr, nullptr, false);
    int top1_splitk = std::max_element(l1_splitk.begin(), l1_splitk.end()) - l1_splitk.begin();

    KVCache kv_legacy(cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, 4096);
    kv_legacy.set_current_seq_len(0);
    prefill_gpu(prompt_toks, cfg, weights, buffers, kv_legacy, cublas_handle);
    std::vector<float> l0_legacy = decode_step_gpu(first_tok, first_pos, cfg, weights, buffers, kv_legacy, cublas_handle, nullptr, nullptr, true);
    int top0_legacy = std::max_element(l0_legacy.begin(), l0_legacy.end()) - l0_legacy.begin();
    std::vector<float> l1_legacy = decode_step_gpu(top0_legacy, first_pos + 1, cfg, weights, buffers, kv_legacy, cublas_handle, nullptr, nullptr, true);
    int top1_legacy = std::max_element(l1_legacy.begin(), l1_legacy.end()) - l1_legacy.begin();

    std::cout << "step 0: splitk=" << top0_splitk << " (logit=" << l0_splitk[top0_splitk] << "), legacy=" << top0_legacy << " (logit=" << l0_legacy[top0_legacy] << ")\n";
    std::cout << "step 1: splitk=" << top1_splitk << " (logit=" << l1_splitk[top1_splitk] << "), legacy=" << top1_legacy << " (logit=" << l1_legacy[top1_legacy] << ")\n";
    std::cout << "At step 1: splitk l1[105953]=" << l1_splitk[105953] << " l1[271]=" << l1_splitk[271] << "\n";
    std::cout << "At step 1: legacy l1[105953]=" << l1_legacy[105953] << " l1[271]=" << l1_legacy[271] << "\n";

    // Let's also check top 5 tokens at step 1 for both:
    std::vector<size_t> idx_sp(l1_splitk.size());
    std::iota(idx_sp.begin(), idx_sp.end(), 0);
    std::sort(idx_sp.begin(), idx_sp.end(), [&](size_t a, size_t b){ return l1_splitk[a] > l1_splitk[b]; });
    std::cout << "Split-K step 1 top 5:\n";
    for(int i=0; i<5; ++i) {
        std::cout << "  #" << i << ": token " << idx_sp[i] << " logit=" << l1_splitk[idx_sp[i]] << "\n";
    }

    std::vector<size_t> idx_leg(l1_legacy.size());
    std::iota(idx_leg.begin(), idx_leg.end(), 0);
    std::sort(idx_leg.begin(), idx_leg.end(), [&](size_t a, size_t b){ return l1_legacy[a] > l1_legacy[b]; });
    std::cout << "Legacy step 1 top 5:\n";
    for(int i=0; i<5; ++i) {
        std::cout << "  #" << i << ": token " << idx_leg[i] << " logit=" << l1_legacy[idx_leg[i]] << "\n";
    }

    return 0;
}
