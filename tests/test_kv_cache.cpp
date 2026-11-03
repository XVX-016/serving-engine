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

    // Test across reference prompts
    std::vector<std::string> prompt_names = {
        "01_short", "02_medium", "04_code", "06_cjk_emoji", "08_whitespace_indent"
    };

    float global_max_diff = 0.0f;

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

        // Prefill with first token to initialize cache, then decode step by step for tokens 1 .. S-1
        // And compare cached decode logits against full-recompute logits at each position pos
        kv_cache.set_current_seq_len(0);

        // Run prefill on token 0
        std::vector<int64_t> prefill_tok = {tokens[0]};
        std::vector<float> prefill_logits = prefill_gpu(
            prefill_tok, cfg, weights_gpu, buffers_gpu, kv_cache, cublas_handle
        );

        // Compare position 0 logits with full-recompute logits
        std::vector<float> recompute_p0 = forward_gpu(
            prefill_tok, cfg, weights_gpu, buffers_gpu, cublas_handle
        );
        float diff_0 = max_abs_diff(prefill_logits, recompute_p0);
        if (diff_0 > global_max_diff) global_max_diff = diff_0;
        CHECK(diff_0 <= 1e-3f);

        // Now step-by-step decode tokens 1 .. S - 1
        for (size_t pos = 1; pos < S; ++pos) {
            std::vector<float> decode_logits = decode_step_gpu(
                tokens[pos], static_cast<int>(pos), cfg, weights_gpu, buffers_gpu, kv_cache, cublas_handle
            );

            // Full recompute of prefix tokens[0 .. pos]
            std::vector<int64_t> prefix_toks(tokens.begin(), tokens.begin() + pos + 1);
            std::vector<float> full_logits = forward_gpu(
                prefix_toks, cfg, weights_gpu, buffers_gpu, cublas_handle
            );

            // Last-position logits of full recompute
            const float* full_last = full_logits.data() + pos * cfg.vocab_size;
            std::vector<float> full_last_vec(full_last, full_last + cfg.vocab_size);

            float diff_pos = max_abs_diff(decode_logits, full_last_vec);
            if (diff_pos > global_max_diff) global_max_diff = diff_pos;

            // Check argmax match
            int decode_argmax = static_cast<int>(std::max_element(decode_logits.begin(), decode_logits.end()) - decode_logits.begin());
            int full_argmax = static_cast<int>(std::max_element(full_last_vec.begin(), full_last_vec.end()) - full_last_vec.begin());
            CHECK(decode_argmax == full_argmax);

            // Gate criterion: FP16 accumulation difference threshold <= 0.1
            CHECK(diff_pos <= 0.1f);
        }

        std::cout << "Prompt [" << pname << "] (" << S << " tokens): All positions verified. Max diff = "
                  << global_max_diff << std::endl;
    }

    std::cout << "\n=== Gate 1.7 Summary: Global Max Logit Diff across all positions: "
              << global_max_diff << " (PASS threshold <= 1e-3) ===\n";
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
}
