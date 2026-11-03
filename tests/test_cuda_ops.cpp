#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "src/cuda/kernels.h"
#include "src/cpu/ops.h"
#include "src/common/cuda_utils.h"
#include "third_party/doctest.h"

#include <vector>
#include <random>
#include <cmath>
#include <iostream>

using namespace engine;

// Helper to convert float vector to half vector
static std::vector<half> to_half(const std::vector<float>& src) {
    std::vector<half> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        dst[i] = __float2half(src[i]);
    }
    return dst;
}

// Helper to convert half vector to float vector
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

TEST_CASE("CUDA Kernel Test: embedding_lookup (20+ random shapes)") {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Stated tolerance for embedding lookup: exact FP16 copy, tolerance 0.0
    const float tolerance = 0.0f;

    for (int test_idx = 0; test_idx < 25; ++test_idx) {
        int64_t seq_len = (test_idx == 0) ? 1 : (1 + rng() % 73); // Includes length 1
        int64_t vocab_size = 50 + rng() % 200;
        int64_t hidden_size = 7 + rng() % 129; // Non-multiple of tile size

        std::vector<int64_t> token_ids(seq_len);
        for (int64_t i = 0; i < seq_len; ++i) {
            token_ids[i] = rng() % vocab_size;
        }

        std::vector<float> table_f(vocab_size * hidden_size);
        for (auto& val : table_f) val = dist(rng);
        std::vector<half> table_h = to_half(table_f);

        // CPU reference
        std::vector<float> ref_out_f(seq_len * hidden_size);
        std::vector<float> table_h_as_f = to_float(table_h);
        cpu::embedding_lookup(token_ids.data(), table_h_as_f.data(), ref_out_f.data(), seq_len, hidden_size);

        // CUDA execution
        DeviceBuffer<int64_t> d_token_ids(seq_len);
        DeviceBuffer<half> d_table(table_h.size());
        DeviceBuffer<half> d_out(seq_len * hidden_size);

        d_token_ids.copy_from_host(token_ids.data(), seq_len);
        d_table.copy_from_host(table_h.data(), table_h.size());

        cuda::embedding_lookup(d_token_ids.data(), d_table.data(), d_out.data(), seq_len, hidden_size);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<half> gpu_out_h(seq_len * hidden_size);
        d_out.copy_to_host(gpu_out_h.data(), gpu_out_h.size());
        std::vector<float> gpu_out_f = to_float(gpu_out_h);

        float diff = max_abs_diff(gpu_out_f, ref_out_f);
        CHECK(diff <= tolerance);
    }
}

TEST_CASE("CUDA Kernel Test: rmsnorm (20+ random shapes)") {
    std::mt19937 rng(43);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Stated tolerance: FP16 precision difference vs CPU double-acc rmsnorm <= 2e-3
    const float tolerance = 2e-3f;

    for (int test_idx = 0; test_idx < 25; ++test_idx) {
        int64_t rows = (test_idx == 0) ? 1 : (1 + rng() % 50);
        int64_t dim = 17 + rng() % 300; // unaligned / prime

        std::vector<float> x_f(rows * dim);
        for (auto& val : x_f) val = dist(rng);
        std::vector<half> x_h = to_half(x_f);

        std::vector<float> weight_f(dim);
        for (auto& val : weight_f) val = dist(rng);
        std::vector<half> weight_h = to_half(weight_f);

        // CPU reference using half inputs converted to float
        std::vector<float> x_h_f = to_float(x_h);
        std::vector<float> weight_h_f = to_float(weight_h);
        std::vector<float> ref_out(rows * dim);
        cpu::rmsnorm(x_h_f.data(), weight_h_f.data(), ref_out.data(), rows, dim, 1e-6f);

        // GPU execution
        DeviceBuffer<half> d_x(x_h.size());
        DeviceBuffer<half> d_weight(weight_h.size());
        DeviceBuffer<half> d_out(rows * dim);

        d_x.copy_from_host(x_h.data(), x_h.size());
        d_weight.copy_from_host(weight_h.data(), weight_h.size());

        cuda::rmsnorm(d_x.data(), d_weight.data(), d_out.data(), rows, dim, 1e-6f);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<half> gpu_out_h(rows * dim);
        d_out.copy_to_host(gpu_out_h.data(), gpu_out_h.size());
        std::vector<float> gpu_out_f = to_float(gpu_out_h);

        float diff = max_abs_diff(gpu_out_f, ref_out);
        CHECK(diff <= tolerance);
    }
}

TEST_CASE("CUDA Kernel Test: rope (20+ random shapes)") {
    std::mt19937 rng(44);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Stated tolerance: FP16 precision difference <= 2e-3
    const float tolerance = 2e-3f;

    for (int test_idx = 0; test_idx < 25; ++test_idx) {
        int64_t seq_len = (test_idx == 0) ? 1 : (1 + rng() % 35);
        int64_t num_q_heads = 1 + rng() % 16;
        int64_t num_kv_heads = 1 + rng() % 4;
        int64_t head_dim = ((1 + rng() % 32) * 2); // even dimension, e.g. 2..64
        int64_t start_pos = rng() % 100;

        std::vector<float> q_f(seq_len * num_q_heads * head_dim);
        std::vector<float> k_f(seq_len * num_kv_heads * head_dim);
        for (auto& val : q_f) val = dist(rng);
        for (auto& val : k_f) val = dist(rng);

        std::vector<half> q_h = to_half(q_f);
        std::vector<half> k_h = to_half(k_f);

        // CPU reference
        std::vector<float> ref_q = to_float(q_h);
        std::vector<float> ref_k = to_float(k_h);
        cpu::apply_rope(ref_q.data(), ref_k.data(), seq_len, num_q_heads, num_kv_heads, head_dim, start_pos, 1000000.0f);

        // GPU execution
        DeviceBuffer<half> d_q(q_h.size());
        DeviceBuffer<half> d_k(k_h.size());
        d_q.copy_from_host(q_h.data(), q_h.size());
        d_k.copy_from_host(k_h.data(), k_h.size());

        cuda::apply_rope(d_q.data(), d_k.data(), seq_len, num_q_heads, num_kv_heads, head_dim, start_pos, 1000000.0f);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<half> gpu_q_h(q_h.size());
        std::vector<half> gpu_k_h(k_h.size());
        d_q.copy_to_host(gpu_q_h.data(), gpu_q_h.size());
        d_k.copy_to_host(gpu_k_h.data(), gpu_k_h.size());

        std::vector<float> gpu_q_f = to_float(gpu_q_h);
        std::vector<float> gpu_k_f = to_float(gpu_k_h);

        float diff_q = max_abs_diff(gpu_q_f, ref_q);
        float diff_k = max_abs_diff(gpu_k_f, ref_k);
        CHECK(diff_q <= tolerance);
        CHECK(diff_k <= tolerance);
    }
}

TEST_CASE("CUDA Kernel Test: causal_gqa (20+ random shapes)") {
    std::mt19937 rng(45);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Stated tolerance: online FP32 softmax vs CPU FP32 softmax, converted to FP16 <= 3e-3
    const float tolerance = 3e-3f;

    for (int test_idx = 0; test_idx < 25; ++test_idx) {
        int64_t seq_len = (test_idx == 0) ? 1 : (1 + rng() % 25);
        int64_t num_kv_heads = 1 + rng() % 3;
        int64_t group_size = 1 + rng() % 5;
        int64_t num_q_heads = num_kv_heads * group_size;
        int64_t head_dim = 16 * (1 + rng() % 4); // 16, 32, 48, 64

        std::vector<float> q_f(seq_len * num_q_heads * head_dim);
        std::vector<float> k_f(seq_len * num_kv_heads * head_dim);
        std::vector<float> v_f(seq_len * num_kv_heads * head_dim);
        for (auto& val : q_f) val = dist(rng);
        for (auto& val : k_f) val = dist(rng);
        for (auto& val : v_f) val = dist(rng);

        std::vector<half> q_h = to_half(q_f);
        std::vector<half> k_h = to_half(k_f);
        std::vector<half> v_h = to_half(v_f);

        // CPU reference
        std::vector<float> q_h_f = to_float(q_h);
        std::vector<float> k_h_f = to_float(k_h);
        std::vector<float> v_h_f = to_float(v_h);
        std::vector<float> ref_out(seq_len * num_q_heads * head_dim);
        cpu::causal_gqa(q_h_f.data(), k_h_f.data(), v_h_f.data(), ref_out.data(), seq_len, num_q_heads, num_kv_heads, head_dim);

        // GPU execution
        DeviceBuffer<half> d_q(q_h.size());
        DeviceBuffer<half> d_k(k_h.size());
        DeviceBuffer<half> d_v(v_h.size());
        DeviceBuffer<half> d_out(ref_out.size());

        d_q.copy_from_host(q_h.data(), q_h.size());
        d_k.copy_from_host(k_h.data(), k_h.size());
        d_v.copy_from_host(v_h.data(), v_h.size());

        cuda::causal_gqa(d_q.data(), d_k.data(), d_v.data(), d_out.data(), seq_len, num_q_heads, num_kv_heads, head_dim);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<half> gpu_out_h(ref_out.size());
        d_out.copy_to_host(gpu_out_h.data(), gpu_out_h.size());
        std::vector<float> gpu_out_f = to_float(gpu_out_h);

        float diff = max_abs_diff(gpu_out_f, ref_out);
        CHECK(diff <= tolerance);
    }
}

TEST_CASE("CUDA Kernel Test: softmax (20+ random shapes)") {
    std::mt19937 rng(46);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

    // Stated tolerance: <= 2e-3
    const float tolerance = 2e-3f;

    for (int test_idx = 0; test_idx < 25; ++test_idx) {
        int64_t rows = (test_idx == 0) ? 1 : (1 + rng() % 30);
        int64_t cols = 1 + rng() % 500;

        std::vector<float> x_f(rows * cols);
        for (auto& val : x_f) val = dist(rng);
        std::vector<half> x_h = to_half(x_f);

        // CPU reference
        std::vector<float> ref_out(rows * cols);
        for (int64_t r = 0; r < rows; ++r) {
            float max_v = -1e20f;
            for (int64_t c = 0; c < cols; ++c) {
                float v = __half2float(x_h[r * cols + c]);
                if (v > max_v) max_v = v;
            }
            float sum_e = 0.0f;
            for (int64_t c = 0; c < cols; ++c) {
                float e = std::exp(__half2float(x_h[r * cols + c]) - max_v);
                ref_out[r * cols + c] = e;
                sum_e += e;
            }
            for (int64_t c = 0; c < cols; ++c) {
                ref_out[r * cols + c] /= sum_e;
            }
        }

        DeviceBuffer<half> d_x(x_h.size());
        DeviceBuffer<half> d_out(rows * cols);
        d_x.copy_from_host(x_h.data(), x_h.size());

        cuda::softmax(d_x.data(), d_out.data(), rows, cols);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<half> gpu_out_h(rows * cols);
        d_out.copy_to_host(gpu_out_h.data(), gpu_out_h.size());
        std::vector<float> gpu_out_f = to_float(gpu_out_h);

        float diff = max_abs_diff(gpu_out_f, ref_out);
        CHECK(diff <= tolerance);
    }
}

TEST_CASE("CUDA Kernel Test: swiglu (20+ random shapes)") {
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

    // Stated tolerance: <= 2e-3
    const float tolerance = 2e-3f;

    for (int test_idx = 0; test_idx < 25; ++test_idx) {
        int64_t count = (test_idx == 0) ? 1 : (1 + rng() % 10000);

        std::vector<float> gate_f(count);
        std::vector<float> up_f(count);
        for (auto& val : gate_f) val = dist(rng);
        for (auto& val : up_f) val = dist(rng);

        std::vector<half> gate_h = to_half(gate_f);
        std::vector<half> up_h = to_half(up_f);

        // CPU reference
        std::vector<float> gate_h_f = to_float(gate_h);
        std::vector<float> up_h_f = to_float(up_h);
        std::vector<float> ref_out(count);
        cpu::swiglu(gate_h_f.data(), up_h_f.data(), ref_out.data(), count);

        // GPU execution
        DeviceBuffer<half> d_gate(count);
        DeviceBuffer<half> d_up(count);
        DeviceBuffer<half> d_out(count);

        d_gate.copy_from_host(gate_h.data(), count);
        d_up.copy_from_host(up_h.data(), count);

        cuda::swiglu(d_gate.data(), d_up.data(), d_out.data(), count);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<half> gpu_out_h(count);
        d_out.copy_to_host(gpu_out_h.data(), count);
        std::vector<float> gpu_out_f = to_float(gpu_out_h);

        float diff = max_abs_diff(gpu_out_f, ref_out);
        CHECK(diff <= tolerance);
    }
}

TEST_CASE("CUDA Kernel Test: residual_add (20+ random shapes)") {
    std::mt19937 rng(48);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

    // Stated tolerance: exact FP16 addition, tolerance 0.0
    const float tolerance = 0.0f;

    for (int test_idx = 0; test_idx < 25; ++test_idx) {
        int64_t count = (test_idx == 0) ? 1 : (1 + rng() % 10000);

        std::vector<float> a_f(count);
        std::vector<float> b_f(count);
        for (auto& val : a_f) val = dist(rng);
        for (auto& val : b_f) val = dist(rng);

        std::vector<half> a_h = to_half(a_f);
        std::vector<half> b_h = to_half(b_f);

        // CPU reference
        std::vector<half> ref_out(count);
        for (int64_t i = 0; i < count; ++i) {
            float va = __half2float(a_h[i]);
            float vb = __half2float(b_h[i]);
            ref_out[i] = __float2half(va + vb);
        }

        DeviceBuffer<half> d_a(count);
        DeviceBuffer<half> d_b(count);
        DeviceBuffer<half> d_out(count);

        d_a.copy_from_host(a_h.data(), count);
        d_b.copy_from_host(b_h.data(), count);

        cuda::residual_add(d_a.data(), d_b.data(), d_out.data(), count);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<half> gpu_out_h(count);
        d_out.copy_to_host(gpu_out_h.data(), count);

        std::vector<float> gpu_out_f = to_float(gpu_out_h);
        std::vector<float> ref_out_f = to_float(ref_out);

        float diff = max_abs_diff(gpu_out_f, ref_out_f);
        CHECK(diff <= tolerance);
    }
}

TEST_CASE("CUDA Kernel Test: bias_add (20+ random shapes)") {
    std::mt19937 rng(49);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

    // Stated tolerance: exact FP16 addition, tolerance 0.0
    const float tolerance = 0.0f;

    for (int test_idx = 0; test_idx < 25; ++test_idx) {
        int64_t M = (test_idx == 0) ? 1 : (1 + rng() % 50);
        int64_t N = (test_idx == 0) ? 1 : (1 + rng() % 200);

        std::vector<float> x_f(M * N);
        std::vector<float> bias_f(N);
        for (auto& val : x_f) val = dist(rng);
        for (auto& val : bias_f) val = dist(rng);

        std::vector<half> x_h = to_half(x_f);
        std::vector<half> bias_h = to_half(bias_f);

        // CPU reference
        std::vector<half> ref_out(M * N);
        for (int64_t r = 0; r < M; ++r) {
            for (int64_t c = 0; c < N; ++c) {
                float vx = __half2float(x_h[r * N + c]);
                float vb = __half2float(bias_h[c]);
                ref_out[r * N + c] = __float2half(vx + vb);
            }
        }

        DeviceBuffer<half> d_x(M * N);
        DeviceBuffer<half> d_bias(N);
        DeviceBuffer<half> d_out(M * N);

        d_x.copy_from_host(x_h.data(), M * N);
        d_bias.copy_from_host(bias_h.data(), N);

        cuda::bias_add(d_x.data(), d_bias.data(), d_out.data(), M, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<half> gpu_out_h(M * N);
        d_out.copy_to_host(gpu_out_h.data(), M * N);

        std::vector<float> gpu_out_f = to_float(gpu_out_h);
        std::vector<float> ref_out_f = to_float(ref_out);

        float diff = max_abs_diff(gpu_out_f, ref_out_f);
        CHECK(diff <= tolerance);
    }
}

TEST_CASE("CUDA Kernel Test: cuBLAS matmul_transB (20+ random shapes)") {
    std::mt19937 rng(50);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Stated tolerance: cuBLAS FP16 in / FP32 accumulate vs CPU FP32 reference <= 2e-3
    const float tolerance = 2e-3f;

    for (int test_idx = 0; test_idx < 25; ++test_idx) {
        int M = (test_idx == 0) ? 1 : (1 + rng() % 33);
        int K = 1 + rng() % 65;
        int N = 1 + rng() % 49;

        std::vector<float> A_f(M * K);
        std::vector<float> B_f(N * K);
        for (auto& val : A_f) val = dist(rng);
        for (auto& val : B_f) val = dist(rng);

        std::vector<half> A_h = to_half(A_f);
        std::vector<half> B_h = to_half(B_f);

        // CPU reference: C = A * B^T with FP16 output rounding
        std::vector<float> A_h_f = to_float(A_h);
        std::vector<float> B_h_f = to_float(B_h);
        std::vector<float> ref_C_f(M * N);
        cpu::matmul_transB_naive(A_h_f.data(), B_h_f.data(), ref_C_f.data(), M, K, N);
        // Round reference to FP16 since output buffer C is FP16
        std::vector<float> ref_C = to_float(to_half(ref_C_f));

        DeviceBuffer<half> d_A(M * K);
        DeviceBuffer<half> d_B(N * K);
        DeviceBuffer<half> d_C(M * N);

        d_A.copy_from_host(A_h.data(), M * K);
        d_B.copy_from_host(B_h.data(), N * K);

        cuda::matmul_transB(handle, d_A.data(), d_B.data(), d_C.data(), M, K, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<half> gpu_C_h(M * N);
        d_C.copy_to_host(gpu_C_h.data(), M * N);
        std::vector<float> gpu_C_f = to_float(gpu_C_h);

        float diff = max_abs_diff(gpu_C_f, ref_C);
        CHECK(diff <= tolerance);
    }

    cublasDestroy(handle);
}

TEST_CASE("CUDA Kernel Test: cuBLAS matmul_transB_f32 (25 random shapes)") {
    std::mt19937 rng(51);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Stated tolerance: cuBLAS FP16 in / FP32 accumulate / FP32 out vs CPU FP32 reference <= 2e-4
    const float tolerance = 2e-4f;

    for (int test_idx = 0; test_idx < 25; ++test_idx) {
        int M = (test_idx == 0) ? 1 : (1 + rng() % 33);
        int K = 1 + rng() % 65;
        int N = 1 + rng() % 49;

        std::vector<float> A_f(M * K);
        std::vector<float> B_f(N * K);
        for (auto& val : A_f) val = dist(rng);
        for (auto& val : B_f) val = dist(rng);

        std::vector<half> A_h = to_half(A_f);
        std::vector<half> B_h = to_half(B_f);

        // CPU reference: C = A * B^T with full FP32 output precision
        std::vector<float> A_h_f = to_float(A_h);
        std::vector<float> B_h_f = to_float(B_h);
        std::vector<float> ref_C_f(M * N);
        cpu::matmul_transB_naive(A_h_f.data(), B_h_f.data(), ref_C_f.data(), M, K, N);

        DeviceBuffer<half> d_A(M * K);
        DeviceBuffer<half> d_B(N * K);
        DeviceBuffer<float> d_C(M * N);

        d_A.copy_from_host(A_h.data(), M * K);
        d_B.copy_from_host(B_h.data(), N * K);

        cuda::matmul_transB_f32(handle, d_A.data(), d_B.data(), d_C.data(), M, K, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> gpu_C_f(M * N);
        d_C.copy_to_host(gpu_C_f.data(), M * N);

        float diff = max_abs_diff(gpu_C_f, ref_C_f);
        CHECK(diff <= tolerance);
    }

    cublasDestroy(handle);
}
