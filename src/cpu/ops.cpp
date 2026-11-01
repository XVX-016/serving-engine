#include "src/cpu/ops.h"

#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

namespace engine {
namespace cpu {

void embedding_lookup(
    const int64_t* token_ids,
    const float* table,
    float* out,
    int64_t seq_len,
    int64_t hidden_size
) {
    for (int64_t i = 0; i < seq_len; ++i) {
        int64_t id = token_ids[i];
        const float* src = table + id * hidden_size;
        float* dst = out + i * hidden_size;
        std::memcpy(dst, src, hidden_size * sizeof(float));
    }
}

void rmsnorm(
    const float* x,
    const float* weight,
    float* out,
    int64_t rows,
    int64_t dim,
    float eps
) {
    for (int64_t r = 0; r < rows; ++r) {
        const float* row_x = x + r * dim;
        float* row_out = out + r * dim;

        double sum_sq = 0.0;
        for (int64_t c = 0; c < dim; ++c) {
            double v = static_cast<double>(row_x[c]);
            sum_sq += v * v;
        }

        double mean_sq = sum_sq / static_cast<double>(dim);
        float rsqrt_val = static_cast<float>(1.0 / std::sqrt(mean_sq + static_cast<double>(eps)));

        for (int64_t c = 0; c < dim; ++c) {
            row_out[c] = row_x[c] * rsqrt_val * weight[c];
        }
    }
}

void matmul_transB_naive(
    const float* A,
    const float* B,
    float* C,
    int64_t M,
    int64_t K,
    int64_t N
) {
    // C[m, n] = sum_k A[m, k] * B[n, k]
    for (int64_t m = 0; m < M; ++m) {
        const float* row_a = A + m * K;
        float* row_c = C + m * N;
        for (int64_t n = 0; n < N; ++n) {
            const float* row_b = B + n * K;
            double sum = 0.0;
            for (int64_t k = 0; k < K; ++k) {
                sum += static_cast<double>(row_a[k]) * static_cast<double>(row_b[k]);
            }
            row_c[n] = static_cast<float>(sum);
        }
    }
}

void matmul_transB_blocked(
    const float* A,
    const float* B,
    float* C,
    int64_t M,
    int64_t K,
    int64_t N
) {
    // Zero out C
    std::memset(C, 0, M * N * sizeof(float));

    constexpr int64_t BM = 64;
    constexpr int64_t BN = 64;
    constexpr int64_t BK = 64;

    for (int64_t m0 = 0; m0 < M; m0 += BM) {
        int64_t m_end = std::min(m0 + BM, M);
        for (int64_t n0 = 0; n0 < N; n0 += BN) {
            int64_t n_end = std::min(n0 + BN, N);
            for (int64_t k0 = 0; k0 < K; k0 += BK) {
                int64_t k_end = std::min(k0 + BK, K);

                for (int64_t m = m0; m < m_end; ++m) {
                    const float* a_ptr = A + m * K;
                    float* c_ptr = C + m * N;
                    for (int64_t n = n0; n < n_end; ++n) {
                        const float* b_ptr = B + n * K;
                        double sum = 0.0;
                        for (int64_t k = k0; k < k_end; ++k) {
                            sum += static_cast<double>(a_ptr[k]) * static_cast<double>(b_ptr[k]);
                        }
                        c_ptr[n] += static_cast<float>(sum);
                    }
                }

            }
        }
    }
}

void matmul_naive(
    const float* A,
    const float* B,
    float* C,
    int64_t M,
    int64_t K,
    int64_t N
) {
    // C[m, n] = sum_k A[m, k] * B[k, n]
    std::memset(C, 0, M * N * sizeof(float));
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            float a = A[m * K + k];
            const float* b_row = B + k * N;
            float* c_row = C + m * N;
            for (int64_t n = 0; n < N; ++n) {
                c_row[n] += a * b_row[n];
            }
        }
    }
}

void bias_add(
    const float* x,
    const float* bias,
    float* out,
    int64_t M,
    int64_t N
) {
    for (int64_t m = 0; m < M; ++m) {
        const float* x_row = x + m * N;
        float* out_row = out + m * N;
        for (int64_t n = 0; n < N; ++n) {
            out_row[n] = x_row[n] + bias[n];
        }
    }
}

void residual_add(
    const float* a,
    const float* b,
    float* out,
    int64_t count
) {
    for (int64_t i = 0; i < count; ++i) {
        out[i] = a[i] + b[i];
    }
}

void apply_rope(
    float* q,
    float* k,
    int64_t seq_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t start_pos,
    float rope_theta
) {
    int64_t half = head_dim / 2;
    std::vector<float> inv_freq(half);
    for (int64_t i = 0; i < half; ++i) {
        inv_freq[i] = 1.0f / std::pow(rope_theta, static_cast<float>(2 * i) / static_cast<float>(head_dim));
    }

    std::vector<float> cos_buf(half);
    std::vector<float> sin_buf(half);

    for (int64_t s = 0; s < seq_len; ++s) {
        int64_t pos = start_pos + s;
        for (int64_t i = 0; i < half; ++i) {
            float theta = static_cast<float>(pos) * inv_freq[i];
            cos_buf[i] = std::cos(theta);
            sin_buf[i] = std::sin(theta);
        }

        // Apply to Q: shape [seq_len, num_q_heads, head_dim]
        for (int64_t h = 0; h < num_q_heads; ++h) {
            float* vec = q + (s * num_q_heads + h) * head_dim;
            for (int64_t i = 0; i < half; ++i) {
                float x1 = vec[i];
                float x2 = vec[half + i];
                // rotate_half: [-x2, x1]
                // out1 = x1 * cos - x2 * sin
                // out2 = x2 * cos + x1 * sin
                float c = cos_buf[i];
                float sn = sin_buf[i];
                vec[i] = x1 * c - x2 * sn;
                vec[half + i] = x2 * c + x1 * sn;
            }
        }

        // Apply to K: shape [seq_len, num_kv_heads, head_dim]
        for (int64_t h = 0; h < num_kv_heads; ++h) {
            float* vec = k + (s * num_kv_heads + h) * head_dim;
            for (int64_t i = 0; i < half; ++i) {
                float x1 = vec[i];
                float x2 = vec[half + i];
                float c = cos_buf[i];
                float sn = sin_buf[i];
                vec[i] = x1 * c - x2 * sn;
                vec[half + i] = x2 * c + x1 * sn;
            }
        }
    }
}

void causal_gqa(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int64_t seq_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim
) {
    int64_t group_size = num_q_heads / num_kv_heads;
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Temporary buffer for attention scores: [seq_len]
    std::vector<float> scores(seq_len);

    for (int64_t q_h = 0; q_h < num_q_heads; ++q_h) {
        int64_t kv_h = q_h / group_size;

        for (int64_t i = 0; i < seq_len; ++i) {
            const float* q_vec = q + (i * num_q_heads + q_h) * head_dim;

            // 1. Compute dot products with all causal keys j <= i
            float max_score = -1e30f;
            for (int64_t j = 0; j <= i; ++j) {
                const float* k_vec = k + (j * num_kv_heads + kv_h) * head_dim;
                double dot = 0.0;
                for (int64_t d = 0; d < head_dim; ++d) {
                    dot += static_cast<double>(q_vec[d]) * static_cast<double>(k_vec[d]);
                }
                float score = static_cast<float>(dot) * scale;
                scores[j] = score;
                if (score > max_score) {
                    max_score = score;
                }
            }

            // 2. Softmax over j in [0, i]
            double sum_exp = 0.0;
            for (int64_t j = 0; j <= i; ++j) {
                float exp_val = std::exp(scores[j] - max_score);
                scores[j] = exp_val;
                sum_exp += static_cast<double>(exp_val);
            }
            float inv_sum = static_cast<float>(1.0 / sum_exp);
            for (int64_t j = 0; j <= i; ++j) {
                scores[j] *= inv_sum;
            }

            // 3. Accumulate weighted values into out[i, q_h, :]
            float* out_vec = out + (i * num_q_heads + q_h) * head_dim;
            for (int64_t d = 0; d < head_dim; ++d) {
                double val_sum = 0.0;
                for (int64_t j = 0; j <= i; ++j) {
                    const float* v_vec = v + (j * num_kv_heads + kv_h) * head_dim;
                    val_sum += static_cast<double>(scores[j]) * static_cast<double>(v_vec[d]);
                }
                out_vec[d] = static_cast<float>(val_sum);
            }
        }
    }
}

void swiglu(
    const float* gate,
    const float* up,
    float* out,
    int64_t count
) {
    for (int64_t i = 0; i < count; ++i) {
        float g = gate[i];
        float silu_g = g / (1.0f + std::exp(-g));
        out[i] = silu_g * up[i];
    }
}

} // namespace cpu
} // namespace engine

