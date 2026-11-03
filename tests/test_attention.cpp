#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "third_party/doctest.h"
#include "src/cuda/kernels.h"
#include "src/cuda/kv_view.h"
#include "src/common/cuda_utils.h"

#include <vector>
#include <random>
#include <cmath>
#include <iostream>

using namespace engine;
using namespace engine::cuda;

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

// ============================================================================
// Unit Tests: decode_gqa_split vs CPU reference (25 random shapes)
// ============================================================================
TEST_CASE("CUDA Kernel Test: decode_gqa_split vs CPU Reference (random shapes, length-1, non-tile multiples)") {
    std::mt19937 rng(701);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const float tolerance = 3.5e-3f; // FP16 tolerance

    for (int test_idx = 0; test_idx < 25; ++test_idx) {
        int batch_size = (test_idx % 4 == 0) ? 1 : (1 + rng() % 3);
        int num_kv_heads = 2;
        int group_size = (test_idx % 3 == 0) ? 7 : (1 + rng() % 7); // include exact Qwen group_size=7
        int num_q_heads = num_kv_heads * group_size;
        int head_dim = 64; // Qwen head dim is 64

        std::vector<int> seq_lens(batch_size);
        int max_seq = 0;
        for (int b = 0; b < batch_size; ++b) {
            if (test_idx == 0) seq_lens[b] = 1; // test length-1 sequence!
            else if (test_idx == 1) seq_lens[b] = 17; // non-tile multiple
            else if (test_idx == 2) seq_lens[b] = 513; // split boundary
            else seq_lens[b] = 1 + rng() % 600;
            if (seq_lens[b] > max_seq) max_seq = seq_lens[b];
        }

        // Q: [batch_size, num_q_heads, head_dim]
        size_t q_size = static_cast<size_t>(batch_size) * num_q_heads * head_dim;
        std::vector<float> q_f(q_size);
        for (auto& val : q_f) val = dist(rng);
        std::vector<half> q_h = to_half(q_f);

        // K and V: [batch_size, num_kv_heads, max_seq, head_dim]
        size_t kv_size = static_cast<size_t>(batch_size) * num_kv_heads * max_seq * head_dim;
        std::vector<float> k_f(kv_size), v_f(kv_size);
        for (auto& val : k_f) val = dist(rng);
        for (auto& val : v_f) val = dist(rng);
        std::vector<half> k_h = to_half(k_f);
        std::vector<half> v_h = to_half(v_f);

        // CPU Reference computation
        std::vector<float> ref_out(q_size, 0.0f);
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        std::vector<float> q_h_f = to_float(q_h);
        std::vector<float> k_h_f = to_float(k_h);
        std::vector<float> v_h_f = to_float(v_h);

        for (int b = 0; b < batch_size; ++b) {
            int cur_len = seq_lens[b];
            for (int qh = 0; qh < num_q_heads; ++qh) {
                int kvh = qh / group_size;
                const float* cur_q = q_h_f.data() + (static_cast<size_t>(b) * num_q_heads + qh) * head_dim;
                const float* cur_k_head = k_h_f.data() + (static_cast<size_t>(b) * num_kv_heads + kvh) * max_seq * head_dim;
                const float* cur_v_head = v_h_f.data() + (static_cast<size_t>(b) * num_kv_heads + kvh) * max_seq * head_dim;

                std::vector<float> scores(cur_len);
                float max_s = -1e20f;
                for (int t = 0; t < cur_len; ++t) {
                    const float* cur_k = cur_k_head + static_cast<size_t>(t) * head_dim;
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        dot += cur_q[d] * cur_k[d];
                    }
                    scores[t] = dot * scale;
                    if (scores[t] > max_s) max_s = scores[t];
                }

                float sum_exp = 0.0f;
                for (int t = 0; t < cur_len; ++t) {
                    scores[t] = std::exp(scores[t] - max_s);
                    sum_exp += scores[t];
                }
                for (int t = 0; t < cur_len; ++t) {
                    scores[t] /= sum_exp;
                }

                float* cur_out = ref_out.data() + (static_cast<size_t>(b) * num_q_heads + qh) * head_dim;
                for (int d = 0; d < head_dim; ++d) {
                    float val = 0.0f;
                    for (int t = 0; t < cur_len; ++t) {
                        const float* cur_v = cur_v_head + static_cast<size_t>(t) * head_dim;
                        val += scores[t] * cur_v[d];
                    }
                    cur_out[d] = val;
                }
            }
        }

        // GPU execution
        DeviceBuffer<half> d_q(q_size), d_k(kv_size), d_v(kv_size), d_out(q_size);
        DeviceBuffer<int> d_seq_lens(batch_size);
        // Scratch buffer for splits: up to 8 splits
        size_t split_buffer_elems = 8 * batch_size * num_q_heads * (head_dim + 2);
        DeviceBuffer<float> d_split_buf(split_buffer_elems);

        d_q.copy_from_host(q_h.data(), q_size);
        d_k.copy_from_host(k_h.data(), kv_size);
        d_v.copy_from_host(v_h.data(), kv_size);
        d_seq_lens.copy_from_host(seq_lens.data(), batch_size);

        KVView k_view{
            d_k.data(),
            static_cast<int64_t>(num_kv_heads * max_seq * head_dim),
            static_cast<int64_t>(max_seq * head_dim),
            static_cast<int64_t>(head_dim),
            1
        };

        KVView v_view{
            d_v.data(),
            static_cast<int64_t>(num_kv_heads * max_seq * head_dim),
            static_cast<int64_t>(max_seq * head_dim),
            static_cast<int64_t>(head_dim),
            1
        };

        decode_gqa_split(
            d_q.data(),
            k_view,
            v_view,
            d_seq_lens.data(),
            d_out.data(),
            batch_size,
            num_q_heads,
            num_kv_heads,
            head_dim,
            max_seq,
            d_split_buf.data()
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<half> out_h(q_size);
        d_out.copy_to_host(out_h.data(), q_size);
        std::vector<float> gpu_out = to_float(out_h);

        float max_diff = 0.0f;
        for (size_t i = 0; i < q_size; ++i) {
            float d = std::abs(gpu_out[i] - ref_out[i]);
            if (d > max_diff) max_diff = d;
        }

        CHECK(max_diff < tolerance);
    }
}

// ============================================================================
// Unit Tests: flash_prefill_gqa vs CPU causal GQA reference (25 random shapes)
// ============================================================================
TEST_CASE("CUDA Kernel Test: flash_prefill_gqa vs CPU Reference (random shapes, causal mask, length-1)") {
    std::mt19937 rng(702);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const float tolerance = 3.5e-3f; // FP16 tolerance

    for (int test_idx = 0; test_idx < 25; ++test_idx) {
        int batch_size = 1;
        int num_kv_heads = 2;
        int group_size = (test_idx % 3 == 0) ? 7 : (1 + rng() % 7);
        int num_q_heads = num_kv_heads * group_size;
        int head_dim = 64;

        int seq_len = 0;
        if (test_idx == 0) seq_len = 1;
        else if (test_idx == 1) seq_len = 15; // less than tile size 16
        else if (test_idx == 2) seq_len = 17; // non-tile multiple
        else if (test_idx == 3) seq_len = 65; // spans multiple tiles
        else seq_len = 1 + rng() % 120;

        size_t q_size = static_cast<size_t>(batch_size) * seq_len * num_q_heads * head_dim;
        size_t kv_size = static_cast<size_t>(batch_size) * seq_len * num_kv_heads * head_dim;

        std::vector<float> q_f(q_size), k_f(kv_size), v_f(kv_size);
        for (auto& val : q_f) val = dist(rng);
        for (auto& val : k_f) val = dist(rng);
        for (auto& val : v_f) val = dist(rng);

        std::vector<half> q_h = to_half(q_f);
        std::vector<half> k_h = to_half(k_f);
        std::vector<half> v_h = to_half(v_f);

        // CPU Causal Attention Reference
        std::vector<float> ref_out(q_size, 0.0f);
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        std::vector<float> q_h_f = to_float(q_h);
        std::vector<float> k_h_f = to_float(k_h);
        std::vector<float> v_h_f = to_float(v_h);

        for (int q_pos = 0; q_pos < seq_len; ++q_pos) {
            for (int qh = 0; qh < num_q_heads; ++qh) {
                int kvh = qh / group_size;
                const float* cur_q = q_h_f.data() + (static_cast<size_t>(q_pos) * num_q_heads + qh) * head_dim;

                int valid_k_len = q_pos + 1; // causal mask: keys 0 .. q_pos
                std::vector<float> scores(valid_k_len);
                float max_s = -1e20f;

                for (int k_pos = 0; k_pos < valid_k_len; ++k_pos) {
                    const float* cur_k = k_h_f.data() + (static_cast<size_t>(k_pos) * num_kv_heads + kvh) * head_dim;
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        dot += cur_q[d] * cur_k[d];
                    }
                    scores[k_pos] = dot * scale;
                    if (scores[k_pos] > max_s) max_s = scores[k_pos];
                }

                float sum_exp = 0.0f;
                for (int k_pos = 0; k_pos < valid_k_len; ++k_pos) {
                    scores[k_pos] = std::exp(scores[k_pos] - max_s);
                    sum_exp += scores[k_pos];
                }
                for (int k_pos = 0; k_pos < valid_k_len; ++k_pos) {
                    scores[k_pos] /= sum_exp;
                }

                float* cur_out = ref_out.data() + (static_cast<size_t>(q_pos) * num_q_heads + qh) * head_dim;
                for (int d = 0; d < head_dim; ++d) {
                    float val = 0.0f;
                    for (int k_pos = 0; k_pos < valid_k_len; ++k_pos) {
                        const float* cur_v = v_h_f.data() + (static_cast<size_t>(k_pos) * num_kv_heads + kvh) * head_dim;
                        val += scores[k_pos] * cur_v[d];
                    }
                    cur_out[d] = val;
                }
            }
        }

        // GPU execution
        DeviceBuffer<half> d_q(q_size), d_k(kv_size), d_v(kv_size), d_out(q_size);
        d_q.copy_from_host(q_h.data(), q_size);
        d_k.copy_from_host(k_h.data(), kv_size);
        d_v.copy_from_host(v_h.data(), kv_size);

        flash_prefill_gqa(
            d_q.data(),
            d_k.data(),
            d_v.data(),
            d_out.data(),
            seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            batch_size
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<half> out_h(q_size);
        d_out.copy_to_host(out_h.data(), q_size);
        std::vector<float> gpu_out = to_float(out_h);

        float max_diff = 0.0f;
        for (size_t i = 0; i < q_size; ++i) {
            float d = std::abs(gpu_out[i] - ref_out[i]);
            if (d > max_diff) max_diff = d;
        }

        CHECK(max_diff < tolerance);
    }
}
