#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "third_party/doctest.h"
#include "src/cpu/ops.h"

#include <vector>
#include <cmath>
#include <random>
#include <iostream>

using namespace engine::cpu;

TEST_CASE("CPU Ops: RMSNorm") {
    int64_t rows = 5;
    int64_t dim = 17; // non-power-of-2 / prime shape
    float eps = 1e-6f;

    std::vector<float> x(rows * dim);
    std::vector<float> weight(dim);
    std::vector<float> out(rows * dim);

    for (int64_t i = 0; i < rows * dim; ++i) x[i] = static_cast<float>(i % 7) - 3.0f;
    for (int64_t i = 0; i < dim; ++i) weight[i] = 1.0f + 0.1f * static_cast<float>(i);

    rmsnorm(x.data(), weight.data(), out.data(), rows, dim, eps);

    // Verify row by row
    for (int64_t r = 0; r < rows; ++r) {
        double sq = 0.0;
        for (int64_t c = 0; c < dim; ++c) {
            sq += x[r * dim + c] * x[r * dim + c];
        }
        double mean = sq / dim;
        float rsqrt_val = 1.0f / std::sqrt(static_cast<float>(mean) + eps);
        for (int64_t c = 0; c < dim; ++c) {
            float expected = x[r * dim + c] * rsqrt_val * weight[c];
            CHECK(std::abs(out[r * dim + c] - expected) < 1e-5f);
        }
    }
}

TEST_CASE("CPU Ops: Matmul Naive vs Blocked transB on unaligned shapes") {
    int64_t M = 37;
    int64_t K = 79;
    int64_t N = 53;

    std::vector<float> A(M * K);
    std::vector<float> B(N * K);
    std::vector<float> C_naive(M * N);
    std::vector<float> C_blocked(M * N);

    for (size_t i = 0; i < A.size(); ++i) A[i] = (static_cast<float>(static_cast<int64_t>(i % 13)) - 6.0f) * 0.1f;
    for (size_t i = 0; i < B.size(); ++i) B[i] = (static_cast<float>(static_cast<int64_t>(i % 17)) - 8.0f) * 0.1f;

    matmul_transB_naive(A.data(), B.data(), C_naive.data(), M, K, N);
    matmul_transB_blocked(A.data(), B.data(), C_blocked.data(), M, K, N);

    for (size_t i = 0; i < C_naive.size(); ++i) {
        CHECK(std::abs(C_naive[i] - C_blocked[i]) < 1e-4f);
    }



}

TEST_CASE("CPU Ops: RoPE Half-Split") {
    int64_t seq_len = 3;
    int64_t num_q_heads = 2;
    int64_t num_kv_heads = 1;
    int64_t head_dim = 8;
    float theta = 10000.0f;

    std::vector<float> q(seq_len * num_q_heads * head_dim, 1.0f);
    std::vector<float> k(seq_len * num_kv_heads * head_dim, 1.0f);

    apply_rope(q.data(), k.data(), seq_len, num_q_heads, num_kv_heads, head_dim, 0, theta);

    // At pos 0: cos=1, sin=0, so q and k should be unmodified (1.0f)
    for (int64_t i = 0; i < num_q_heads * head_dim; ++i) {
        CHECK(std::abs(q[i] - 1.0f) < 1e-5f);
    }
    for (int64_t i = 0; i < num_kv_heads * head_dim; ++i) {
        CHECK(std::abs(k[i] - 1.0f) < 1e-5f);
    }

    // At pos 1: check element 0 and element 4 (half_dim = 4)
    // inv_freq[0] = 1.0 / theta^0 = 1.0
    // theta0 = 1.0 * 1.0 = 1.0 rad. cos(1.0), sin(1.0)
    // q[0] was 1, q[4] was 1
    // q_new[0] = 1 * cos(1) - 1 * sin(1)
    // q_new[4] = 1 * cos(1) + 1 * sin(1)
    float expected_0 = std::cos(1.0f) - std::sin(1.0f);
    float expected_4 = std::cos(1.0f) + std::sin(1.0f);
    int64_t pos1_q_offset = 1 * num_q_heads * head_dim;
    CHECK(std::abs(q[pos1_q_offset + 0] - expected_0) < 1e-5f);
    CHECK(std::abs(q[pos1_q_offset + 4] - expected_4) < 1e-5f);
}

TEST_CASE("CPU Ops: SwiGLU") {
    std::vector<float> gate = {-2.0f, 0.0f, 2.0f};
    std::vector<float> up = {1.5f, 3.0f, -0.5f};
    std::vector<float> out(3);

    swiglu(gate.data(), up.data(), out.data(), 3);

    // silu(0) = 0 * 0.5 = 0
    CHECK(std::abs(out[1] - 0.0f) < 1e-6f);
    // silu(2.0) = 2.0 / (1.0 + exp(-2.0))
    float silu2 = 2.0f / (1.0f + std::exp(-2.0f));
    CHECK(std::abs(out[2] - (silu2 * -0.5f)) < 1e-5f);
}

TEST_CASE("CPU Ops: Causal GQA") {
    // 2 tokens, 2 Q heads, 1 KV head, head_dim 4
    int64_t seq_len = 2;
    int64_t num_q_heads = 2;
    int64_t num_kv_heads = 1;
    int64_t head_dim = 4;

    std::vector<float> q(seq_len * num_q_heads * head_dim, 0.5f);
    std::vector<float> k(seq_len * num_kv_heads * head_dim, 0.5f);
    std::vector<float> v(seq_len * num_kv_heads * head_dim, 0.0f);
    // Let v[0] = 1.0, v[1] = 2.0
    for (int64_t d = 0; d < head_dim; ++d) {
        v[0 * head_dim + d] = 1.0f;
        v[1 * head_dim + d] = 2.0f;
    }

    std::vector<float> out(seq_len * num_q_heads * head_dim, 0.0f);
    causal_gqa(q.data(), k.data(), v.data(), out.data(), seq_len, num_q_heads, num_kv_heads, head_dim);

    // At position 0, causal mask only allows looking at key 0. So score weight is 1.0 on v[0].
    for (int64_t h = 0; h < num_q_heads; ++h) {
        for (int64_t d = 0; d < head_dim; ++d) {
            CHECK(std::abs(out[(0 * num_q_heads + h) * head_dim + d] - 1.0f) < 1e-5f);
        }
    }
}
