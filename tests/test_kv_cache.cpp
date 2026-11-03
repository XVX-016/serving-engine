#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "third_party/doctest.h"
#include "src/runtime/kv_cache.h"
#include "src/cuda/kernels.h"
#include "src/model/config.h"
#include "src/loader/safetensors.h"
#include "src/model/forward_gpu.h"
#include "src/common/cuda_utils.h"
#include "src/cpu/ops.h"

#include <vector>
#include <random>
#include <cmath>
#include <iostream>

using namespace engine;

static std::vector<half> to_half(const std::vector<float>& src) {
    std::vector<half> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = __float2half(src[i]);
    }
    return dst;
}

static std::vector<float> to_float(const std::vector<half>& src) {
    std::vector<float> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = __half2float(src[i]);
    }
    return dst;
}

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float max_d = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float d = std::abs(a[i] - b[i]);
        if (d > max_d) max_d = d;
    }
    return max_d;
}

TEST_CASE("CUDA Kernel Test: append_kv_cache (20+ random shapes)") {
    std::mt19937 rng(101);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    const float tolerance = 0.0f; // Exact copy

    for (int test_idx = 0; test_idx < 25; ++test_idx) {
        int num_tokens = (test_idx == 0) ? 1 : (1 + rng() % 37);
        int start_pos = (test_idx == 0) ? 0 : (rng() % 50);
        int num_kv_heads = 1 + rng() % 4;
        int max_seq = start_pos + num_tokens + (rng() % 30);
        int head_dim = 16 * (1 + rng() % 4);

        size_t src_size = static_cast<size_t>(num_tokens) * num_kv_heads * head_dim;
        std::vector<float> k_src_f(src_size), v_src_f(src_size);
        for (auto& val : k_src_f) val = dist(rng);
        for (auto& val : v_src_f) val = dist(rng);

        std::vector<half> k_src_h = to_half(k_src_f);
        std::vector<half> v_src_h = to_half(v_src_f);

        size_t cache_size = static_cast<size_t>(num_kv_heads) * max_seq * head_dim;
        DeviceBuffer<half> d_k_src(src_size), d_v_src(src_size);
        DeviceBuffer<half> d_k_cache(cache_size), d_v_cache(cache_size);

        d_k_src.copy_from_host(k_src_h.data(), src_size);
        d_v_src.copy_from_host(v_src_h.data(), src_size);
        d_k_cache.zero();
        d_v_cache.zero();

        cuda::append_kv_cache(
            d_k_src.data(), d_v_src.data(),
            d_k_cache.data(), d_v_cache.data(),
            num_tokens, start_pos, num_kv_heads, max_seq, head_dim
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<half> k_cache_h(cache_size), v_cache_h(cache_size);
        d_k_cache.copy_to_host(k_cache_h.data(), cache_size);
        d_v_cache.copy_to_host(v_cache_h.data(), cache_size);

        // Verify elements at target slices
        for (int t = 0; t < num_tokens; ++t) {
            for (int h = 0; h < num_kv_heads; ++h) {
                for (int d = 0; d < head_dim; ++d) {
                    size_t s_off = (static_cast<size_t>(t) * num_kv_heads + h) * head_dim + d;
                    size_t d_off = (static_cast<size_t>(h) * max_seq + (start_pos + t)) * head_dim + d;
                    CHECK(__half2float(k_cache_h[d_off]) == __half2float(k_src_h[s_off]));
                    CHECK(__half2float(v_cache_h[d_off]) == __half2float(v_src_h[s_off]));
                }
            }
        }
    }
}

TEST_CASE("CUDA Kernel Test: decode_gqa vs CPU reference (20+ random shapes)") {
    std::mt19937 rng(102);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const float tolerance = 3e-3f;

    for (int test_idx = 0; test_idx < 25; ++test_idx) {
        int total_seq = (test_idx == 0) ? 1 : (1 + rng() % 64);
        int max_seq = total_seq + (rng() % 32);
        int num_kv_heads = 1 + rng() % 3;
        int group_size = 1 + rng() % 5;
        int num_q_heads = num_kv_heads * group_size;
        int head_dim = 16 * (1 + rng() % 4); // 16, 32, 48, 64

        // Q: [num_q_heads, head_dim]
        size_t q_size = static_cast<size_t>(num_q_heads) * head_dim;
        std::vector<float> q_f(q_size);
        for (auto& val : q_f) val = dist(rng);
        std::vector<half> q_h = to_half(q_f);

        // K and V cache: [num_kv_heads, max_seq, head_dim]
        size_t cache_size = static_cast<size_t>(num_kv_heads) * max_seq * head_dim;
        std::vector<float> k_cache_f(cache_size), v_cache_f(cache_size);
        for (auto& val : k_cache_f) val = dist(rng);
        for (auto& val : v_cache_f) val = dist(rng);
        std::vector<half> k_cache_h = to_half(k_cache_f);
        std::vector<half> v_cache_h = to_half(v_cache_f);

        // CPU reference: decode attention for 1 query token attending over total_seq tokens
        std::vector<float> ref_out(q_size, 0.0f);
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        std::vector<float> q_h_f = to_float(q_h);
        std::vector<float> k_cache_h_f = to_float(k_cache_h);
        std::vector<float> v_cache_h_f = to_float(v_cache_h);

        for (int qh = 0; qh < num_q_heads; ++qh) {
            int kvh = qh / group_size;
            const float* cur_q = q_h_f.data() + qh * head_dim;
            const float* cur_k_head = k_cache_h_f.data() + static_cast<size_t>(kvh) * max_seq * head_dim;
            const float* cur_v_head = v_cache_h_f.data() + static_cast<size_t>(kvh) * max_seq * head_dim;

            // Compute attention scores
            std::vector<float> scores(total_seq);
            float max_s = -1e20f;
            for (int t = 0; t < total_seq; ++t) {
                const float* cur_k = cur_k_head + static_cast<size_t>(t) * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += cur_q[d] * cur_k[d];
                }
                scores[t] = dot * scale;
                if (scores[t] > max_s) max_s = scores[t];
            }

            // Softmax
            float sum_exp = 0.0f;
            for (int t = 0; t < total_seq; ++t) {
                scores[t] = std::exp(scores[t] - max_s);
                sum_exp += scores[t];
            }
            for (int t = 0; t < total_seq; ++t) {
                scores[t] /= sum_exp;
            }

            // Weighted sum of V
            float* cur_out = ref_out.data() + qh * head_dim;
            for (int t = 0; t < total_seq; ++t) {
                const float* cur_v = cur_v_head + static_cast<size_t>(t) * head_dim;
                float w = scores[t];
                for (int d = 0; d < head_dim; ++d) {
                    cur_out[d] += w * cur_v[d];
                }
            }
        }

        // Run GPU kernel
        DeviceBuffer<half> d_q(q_size);
        DeviceBuffer<half> d_k_cache(cache_size), d_v_cache(cache_size);
        DeviceBuffer<half> d_out(q_size);

        d_q.copy_from_host(q_h.data(), q_size);
        d_k_cache.copy_from_host(k_cache_h.data(), cache_size);
        d_v_cache.copy_from_host(v_cache_h.data(), cache_size);

        cuda::decode_gqa(
            d_q.data(), d_k_cache.data(), d_v_cache.data(), d_out.data(),
            total_seq, num_q_heads, num_kv_heads, max_seq, head_dim
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<half> gpu_out_h(q_size);
        d_out.copy_to_host(gpu_out_h.data(), q_size);
        std::vector<float> gpu_out_f = to_float(gpu_out_h);

        float diff = max_abs_diff(gpu_out_f, ref_out);
        CHECK(diff <= tolerance);
    }
}


TEST_CASE("Gate 1.7 Equivalence: Cached decode logits equal full-recompute logits at every position") {
    // Model configuration and weights
    std::string model_path = "data/models/hf-cache/snapshots/7ae557604adf67be50417f59c2c2f167def9a775";
    ModelConfig cfg = ModelConfig::from_file(model_path + "/config.json");
    SafetensorsLoader loader;
    REQUIRE(loader.open(model_path + "/model.safetensors"));

    ModelWeights weights;
    REQUIRE(weights.load_and_validate(cfg, loader));

    ModelWeightsGPU weights_gpu;
    REQUIRE(weights_gpu.init_from_weights(cfg, weights));

    ActivationBuffersGPU buffers_gpu;
    buffers_gpu.allocate(cfg, 4096);

    KVCache kv_cache(cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, 4096);

    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    // Test across ALL 8 reference prompts
    std::vector<std::string> prompt_names = {
        "01_short", "02_medium", "03_long", "04_code",
        "05_unicode", "06_cjk_emoji", "07_chat_template", "08_whitespace_indent"
    };

    float global_max_diff = 0.0f;
    bool all_gate_passed = true;

    for (const auto& pname : prompt_names) {
        std::string token_file = "data/reference/" + pname + "/token_ids.npy";
        std::ifstream tf(token_file, std::ios::binary);
        REQUIRE(tf.is_open());

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

        // Load CPU FP32 reference logits
        std::string ref_logits_file = "data/reference/" + pname + "/logits.npy";
        std::ifstream rlf(ref_logits_file, std::ios::binary);
        REQUIRE(rlf.is_open());
        char r_magic[6], r_ver[2];
        rlf.read(r_magic, 6);
        rlf.read(r_ver, 2);
        uint16_t r_hlen = 0;
        rlf.read(reinterpret_cast<char*>(&r_hlen), 2);
        std::string r_header(r_hlen, ' ');
        rlf.read(&r_header[0], r_hlen);
        rlf.seekg(0, std::ios::end);
        size_t r_data_len = static_cast<size_t>(rlf.tellg()) - (10 + r_hlen);
        rlf.seekg(10 + r_hlen, std::ios::beg);
        std::vector<float> ref_all_logits(r_data_len / sizeof(float));
        rlf.read(reinterpret_cast<char*>(ref_all_logits.data()), r_data_len);

        // Run full recompute on all tokens to get recompute logits for all positions
        std::vector<float> recompute_all = forward_gpu(
            tokens, cfg, weights_gpu, buffers_gpu, cublas_handle
        );

        // Prefill with first token to initialize cache, then decode step by step
        kv_cache.set_current_seq_len(0);

        std::vector<float> cached_decode_logits(S * cfg.vocab_size);

        // Run prefill on token 0
        std::vector<int64_t> prefill_tok = {tokens[0]};
        std::vector<float> prefill_logits = prefill_gpu(
            prefill_tok, cfg, weights_gpu, buffers_gpu, kv_cache, cublas_handle
        );
        std::copy(prefill_logits.begin(), prefill_logits.end(), cached_decode_logits.begin());

        // Decode tokens 1 .. S-1
        for (size_t pos = 1; pos < S; ++pos) {
            std::vector<float> decode_logits = decode_step_gpu(
                tokens[pos], static_cast<int>(pos), cfg, weights_gpu, buffers_gpu, kv_cache, cublas_handle
            );
            std::copy(decode_logits.begin(), decode_logits.end(), cached_decode_logits.begin() + pos * cfg.vocab_size);
        }

        // Now evaluate Gate 1.7 criteria at every position:
        // 1. Top-1 identical or near-tie (ref margin < 0.05)
        // 2. Cached path error vs CPU FP32 reference <= 1.5x recompute path error vs CPU reference (mean & max)
        double sum_cached_err = 0.0;
        double sum_recompute_err = 0.0;
        float prompt_max_cached_err = 0.0f;
        float prompt_max_recompute_err = 0.0f;

        for (size_t pos = 0; pos < S; ++pos) {
            const float* c_logits = cached_decode_logits.data() + pos * cfg.vocab_size;
            const float* r_logits = recompute_all.data() + pos * cfg.vocab_size;
            const float* ref_logits = ref_all_logits.data() + pos * cfg.vocab_size;

            int c_top1 = 0, r_top1 = 0, ref_top1 = 0;
            float c_max = c_logits[0], r_max = r_logits[0], ref_max = -1e30f;
            for (int64_t v = 0; v < cfg.vocab_size; ++v) {
                if (c_logits[v] > c_max) { c_max = c_logits[v]; c_top1 = static_cast<int>(v); }
                if (r_logits[v] > r_max) { r_max = r_logits[v]; r_top1 = static_cast<int>(v); }
                if (ref_logits[v] > ref_max) { ref_max = ref_logits[v]; ref_top1 = static_cast<int>(v); }
            }

            // Check top-1 match
            if (c_top1 != r_top1) {
                // Check near-tie against reference
                std::vector<float> sorted_ref(ref_logits, ref_logits + cfg.vocab_size);
                std::sort(sorted_ref.rbegin(), sorted_ref.rend());
                float margin = sorted_ref[0] - sorted_ref[1];
                bool passes_near_tie = (margin < 0.05f);
                CHECK(passes_near_tie);
                if (!passes_near_tie) all_gate_passed = false;
            }

            for (int64_t v = 0; v < cfg.vocab_size; ++v) {
                float c_err = std::abs(c_logits[v] - ref_logits[v]);
                float r_err = std::abs(r_logits[v] - ref_logits[v]);
                float c_r_diff = std::abs(c_logits[v] - r_logits[v]);
                if (c_r_diff > global_max_diff) global_max_diff = c_r_diff;

                sum_cached_err += c_err;
                sum_recompute_err += r_err;
                if (c_err > prompt_max_cached_err) prompt_max_cached_err = c_err;
                if (r_err > prompt_max_recompute_err) prompt_max_recompute_err = r_err;
            }
        }

        double mean_cached_err = sum_cached_err / (S * cfg.vocab_size);
        double mean_recompute_err = sum_recompute_err / (S * cfg.vocab_size);
        double mean_ratio = mean_cached_err / mean_recompute_err;
        double max_ratio = static_cast<double>(prompt_max_cached_err) / prompt_max_recompute_err;

        std::cout << "Prompt [" << pname << "] (" << S << " tokens): "
                  << "Mean Err: cached=" << mean_cached_err << ", recompute=" << mean_recompute_err << " (ratio " << mean_ratio << "x) | "
                  << "Max Err: cached=" << prompt_max_cached_err << ", recompute=" << prompt_max_recompute_err << " (ratio " << max_ratio << "x)\n";

        // Gate: cached path error vs CPU FP32 reference is at most 1.5x the recompute path error (mean and max)
        CHECK(mean_ratio <= 1.5);
        CHECK(max_ratio <= 1.5);
        if (mean_ratio > 1.5 || max_ratio > 1.5) all_gate_passed = false;
    }

    std::cout << "\n=== Gate 1.7 Summary: All 8 Prompts Verified. Global Max Logit Diff (cached vs recompute): "
              << global_max_diff << " | Gate 1.7 Status: " << (all_gate_passed ? "PASS" : "FAIL") << " ===\n";
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
}
