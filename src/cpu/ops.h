#pragma once

#include <cstdint>
#include <cstddef>

namespace engine {
namespace cpu {

// 1. Embedding lookup: out[i, :] = table[token_ids[i], :]
// token_ids: [seq_len], table: [vocab_size, hidden_size], out: [seq_len, hidden_size]
void embedding_lookup(
    const int64_t* token_ids,
    const float* table,
    float* out,
    int64_t seq_len,
    int64_t hidden_size
);

// 2. RMSNorm: out = (x / sqrt(mean(x^2) + eps)) * weight
// x: [N, D], weight: [D], out: [N, D]
void rmsnorm(
    const float* x,
    const float* weight,
    float* out,
    int64_t rows,
    int64_t dim,
    float eps = 1e-6f
);

// 3. Matrix Multiplication: C = A * B^T or C = A * B
// In transformer weights, linear layer is y = x * W^T + b, where W is [out_features, in_features] and x is [seq_len, in_features].
// matmul_transB: A: [M, K], B: [N, K], C: [M, N] where C = A * B^T
// Both naive and cache-blocked versions provided.
void matmul_transB_naive(
    const float* A,
    const float* B,
    float* C,
    int64_t M,
    int64_t K,
    int64_t N
);

void matmul_transB_blocked(
    const float* A,
    const float* B,
    float* C,
    int64_t M,
    int64_t K,
    int64_t N
);

// Standard matmul: A: [M, K], B: [K, N], C: [M, N]
void matmul_naive(
    const float* A,
    const float* B,
    float* C,
    int64_t M,
    int64_t K,
    int64_t N
);

// 4. Bias add: out[i, j] = x[i, j] + bias[j]
// x: [M, N], bias: [N], out: [M, N] (can be in-place: out == x)
void bias_add(
    const float* x,
    const float* bias,
    float* out,
    int64_t M,
    int64_t N
);

// 5. Residual add: out[i] = a[i] + b[i]
void residual_add(
    const float* a,
    const float* b,
    float* out,
    int64_t count
);

// 6. RoPE: Half-split rotary position embedding matching HF rotate_half:
// rotate_half(x) = [-x[..., D/2:], x[..., :D/2]]
// x_rot = x * cos + rotate_half(x) * sin
// freqs = 1.0 / (rope_theta ^ (2*i / head_dim)) for i in [0, head_dim/2)
// For pos: cos[pos, i] = cos(pos * freqs[i]), sin[pos, i] = sin(pos * freqs[i])
// cos and sin are duplicated across halves: cos = [cos, cos], sin = [sin, sin] of length head_dim.
void apply_rope(
    float* q, // [seq_len, num_q_heads, head_dim]
    float* k, // [seq_len, num_kv_heads, head_dim]
    int64_t seq_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t start_pos = 0,
    float rope_theta = 1000000.0f
);

// 7. Grouped-Query Attention with Causal Mask:
// q: [seq_len, num_q_heads, head_dim]
// k: [seq_len, num_kv_heads, head_dim]
// v: [seq_len, num_kv_heads, head_dim]
// out: [seq_len, num_q_heads * head_dim]
// Each query head h_q maps to KV head h_kv = h_q / (num_q_heads / num_kv_heads).
// For Qwen2.5-0.5B: num_q_heads = 14, num_kv_heads = 2 => group_size = 7.
// Attention score scaling = 1.0f / sqrt(head_dim) = 1.0f / 8.0f = 0.125f.
void causal_gqa(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int64_t seq_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim
);

// 8. SwiGLU: out = (gate * sigmoid(gate)) * up
// gate: [M, N], up: [M, N], out: [M, N]
void swiglu(
    const float* gate,
    const float* up,
    float* out,
    int64_t count
);

} // namespace cpu
} // namespace engine

