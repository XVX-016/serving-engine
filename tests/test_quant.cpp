#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "third_party/doctest.h"
#include "src/common/cuda_utils.h"
#include "src/cuda/kernels.h"
#include <cuda_fp16.h>
#include <vector>
#include <random>
#include <cmath>
#include <iostream>

static float half_to_float(half h) {
    return __half2float(h);
}

static half float_to_half(float f) {
    return __float2half(f);
}

TEST_CASE("INT8 GEMV CPU vs GPU Reference Across Diverse Shapes") {
    // Shapes: multiple of 32, multiple of 16, non-multiples, small and large
    std::vector<std::pair<int, int>> shapes = {
        {1, 128}, {7, 128}, {16, 128}, {32, 128},
        {128, 896}, {896, 896}, {4864, 896}, {896, 4864},
        {15, 64}, {33, 160}, {67, 352}, {100, 512},
        {24, 768}, {55, 1024}, {3, 2048}, {896, 128}
    };

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist_w(-1.5f, 1.5f);
    std::uniform_real_distribution<float> dist_x(-1.0f, 1.0f);

    for (const auto& [N, K] : shapes) {
        // Allocate host data
        std::vector<int8_t> h_W_q(N * K);
        std::vector<half> h_scales(N);
        std::vector<half> h_x(K);
        std::vector<half> h_bias(N);
        std::vector<half> h_y_ref(N);
        std::vector<half> h_y_ref_bias(N);

        for (int k = 0; k < K; ++k) {
            h_x[k] = float_to_half(dist_x(rng));
        }

        for (int n = 0; n < N; ++n) {
            float max_v = 0.0f;
            std::vector<float> row_f(K);
            for (int k = 0; k < K; ++k) {
                float v = dist_w(rng);
                row_f[k] = v;
                if (std::abs(v) > max_v) max_v = std::abs(v);
            }
            float s = std::max(max_v / 127.0f, 1e-6f);
            h_scales[n] = float_to_half(s);
            h_bias[n] = float_to_half(dist_w(rng));

            float s_f = half_to_float(h_scales[n]);
            float dot = 0.0f;
            for (int k = 0; k < K; ++k) {
                int q = std::clamp(static_cast<int>(std::round(row_f[k] / s_f)), -127, 127);
                h_W_q[n * K + k] = static_cast<int8_t>(q);
                float w_deq = static_cast<float>(q) * s_f;
                dot += w_deq * half_to_float(h_x[k]);
            }
            h_y_ref[n] = float_to_half(dot);
            h_y_ref_bias[n] = float_to_half(dot + half_to_float(h_bias[n]));
        }

        // Allocate device buffers
        engine::DeviceBuffer<int8_t> d_W_q(N * K);
        engine::DeviceBuffer<half> d_scales(N);
        engine::DeviceBuffer<half> d_x(K);
        engine::DeviceBuffer<half> d_bias(N);
        engine::DeviceBuffer<half> d_y(N);

        d_W_q.copy_from_host(h_W_q.data(), N * K);
        d_scales.copy_from_host(h_scales.data(), N);
        d_x.copy_from_host(h_x.data(), K);
        d_bias.copy_from_host(h_bias.data(), N);

        // Run GPU kernel: none
        engine::cuda::gemv_int8(d_W_q.data(), d_scales.data(), d_x.data(), d_y.data(), N, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<half> h_y_gpu(N);
        d_y.copy_to_host(h_y_gpu.data(), N);

        for (int n = 0; n < N; ++n) {
            float ref = half_to_float(h_y_ref[n]);
            float gpu = half_to_float(h_y_gpu[n]);
            float diff = std::abs(ref - gpu);
            float tol = 1e-2f * std::max(1.0f, std::abs(ref));
            CHECK(diff <= tol);
        }

        // Run GPU kernel: bias
        engine::cuda::gemv_int8_bias(d_W_q.data(), d_scales.data(), d_x.data(), d_bias.data(), d_y.data(), N, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        d_y.copy_to_host(h_y_gpu.data(), N);
        for (int n = 0; n < N; ++n) {
            float ref = half_to_float(h_y_ref_bias[n]);
            float gpu = half_to_float(h_y_gpu[n]);
            float diff = std::abs(ref - gpu);
            float tol = 1e-2f * std::max(1.0f, std::abs(ref));
            CHECK(diff <= tol);
        }
    }
}

TEST_CASE("INT4 Group-128 GEMV CPU vs GPU Reference Across Diverse Shapes") {
    // K must be a multiple of group_size=128 for packed grouped GEMV
    std::vector<std::pair<int, int>> shapes = {
        {1, 128}, {7, 128}, {16, 256}, {32, 512},
        {128, 896}, {896, 896}, {4864, 896}, {896, 4864},
        {15, 128}, {33, 256}, {67, 384}, {100, 512},
        {24, 768}, {55, 1024}, {3, 2048}, {896, 128}
    };

    const int GROUP_SIZE = 128;
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist_w(-1.5f, 1.5f);
    std::uniform_real_distribution<float> dist_x(-1.0f, 1.0f);

    for (const auto& [N, K] : shapes) {
        int num_groups = K / GROUP_SIZE;
        int K_bytes = K / 2;

        std::vector<uint8_t> h_W_q(N * K_bytes);
        std::vector<half> h_scales(N * num_groups);
        std::vector<half> h_x(K);
        std::vector<half> h_bias(N);
        std::vector<half> h_y_ref(N);
        std::vector<half> h_y_ref_bias(N);

        for (int k = 0; k < K; ++k) {
            h_x[k] = float_to_half(dist_x(rng));
        }

        for (int n = 0; n < N; ++n) {
            h_bias[n] = float_to_half(dist_w(rng));
            float dot = 0.0f;

            for (int g = 0; g < num_groups; ++g) {
                float max_v = 0.0f;
                std::vector<float> g_vals(GROUP_SIZE);
                for (int i = 0; i < GROUP_SIZE; ++i) {
                    float v = dist_w(rng);
                    g_vals[i] = v;
                    if (std::abs(v) > max_v) max_v = std::abs(v);
                }
                float s = std::max(max_v / 7.0f, 1e-6f);
                h_scales[n * num_groups + g] = float_to_half(s);
                float s_f = half_to_float(h_scales[n * num_groups + g]);

                for (int i = 0; i < GROUP_SIZE; i += 2) {
                    int k0 = g * GROUP_SIZE + i;
                    int k1 = k0 + 1;

                    int q0 = std::clamp(static_cast<int>(std::round(g_vals[i] / s_f)), -8, 7);
                    int q1 = std::clamp(static_cast<int>(std::round(g_vals[i + 1] / s_f)), -8, 7);

                    uint8_t u0 = static_cast<uint8_t>(q0 & 0x0F);
                    uint8_t u1 = static_cast<uint8_t>(q1 & 0x0F);
                    uint8_t packed_b = (u0 & 0x0F) | ((u1 & 0x0F) << 4);

                    h_W_q[n * K_bytes + k0 / 2] = packed_b;

                    float w0 = static_cast<float>(q0) * s_f;
                    float w1 = static_cast<float>(q1) * s_f;

                    dot += w0 * half_to_float(h_x[k0]);
                    dot += w1 * half_to_float(h_x[k1]);
                }
            }
            h_y_ref[n] = float_to_half(dot);
            h_y_ref_bias[n] = float_to_half(dot + half_to_float(h_bias[n]));
        }

        engine::DeviceBuffer<uint8_t> d_W_q(N * K_bytes);
        engine::DeviceBuffer<half> d_scales(N * num_groups);
        engine::DeviceBuffer<half> d_x(K);
        engine::DeviceBuffer<half> d_bias(N);
        engine::DeviceBuffer<half> d_y(N);

        d_W_q.copy_from_host(h_W_q.data(), N * K_bytes);
        d_scales.copy_from_host(h_scales.data(), N * num_groups);
        d_x.copy_from_host(h_x.data(), K);
        d_bias.copy_from_host(h_bias.data(), N);

        // Run GPU kernel: none
        engine::cuda::gemv_int4_g128(d_W_q.data(), d_scales.data(), d_x.data(), d_y.data(), N, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<half> h_y_gpu(N);
        d_y.copy_to_host(h_y_gpu.data(), N);

        for (int n = 0; n < N; ++n) {
            float ref = half_to_float(h_y_ref[n]);
            float gpu = half_to_float(h_y_gpu[n]);
            float diff = std::abs(ref - gpu);
            float tol = 1e-2f * std::max(1.0f, std::abs(ref));
            CHECK(diff <= tol);
        }

        // Run GPU kernel: bias
        engine::cuda::gemv_int4_g128_bias(d_W_q.data(), d_scales.data(), d_x.data(), d_bias.data(), d_y.data(), N, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        d_y.copy_to_host(h_y_gpu.data(), N);
        for (int n = 0; n < N; ++n) {
            float ref = half_to_float(h_y_ref_bias[n]);
            float gpu = half_to_float(h_y_gpu[n]);
            float diff = std::abs(ref - gpu);
            float tol = 1e-2f * std::max(1.0f, std::abs(ref));
            CHECK(diff <= tol);
        }
    }
}
