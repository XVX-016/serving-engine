#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdint>

namespace engine {
namespace cuda {

// 1. Embedding lookup: out[i, :] = table[token_ids[i], :]
// token_ids: [seq_len], table: [vocab_size, hidden_size], out: [seq_len, hidden_size]
void embedding_lookup(
    const int64_t* token_ids,
    const half* table,
    half* out,
    int64_t seq_len,
    int64_t hidden_size,
    cudaStream_t stream = 0
);

// 2. RMSNorm: out = (x / sqrt(mean(x^2) + eps)) * weight
// x: [rows, dim], weight: [dim], out: [rows, dim]
// Uses FP32 accumulation and warp-level reduction
void rmsnorm(
    const half* x,
    const half* weight,
    half* out,
    int64_t rows,
    int64_t dim,
    float eps = 1e-6f,
    cudaStream_t stream = 0
);

// 3. Bias add: out[i, j] = x[i, j] + bias[j]
// x: [M, N], bias: [N], out: [M, N]
void bias_add(
    const half* x,
    const half* bias,
    half* out,
    int64_t M,
    int64_t N,
    cudaStream_t stream = 0
);

// 4. Residual add: out[i] = a[i] + b[i]
void residual_add(
    const half* a,
    const half* b,
    half* out,
    int64_t count,
    cudaStream_t stream = 0
);

// 5. RoPE: Half-split rotary position embedding matching HF rotate_half:
// rotate_half(x) = [-x[..., D/2:], x[..., :D/2]]
// x_rot = x * cos + rotate_half(x) * sin
void apply_rope(
    half* q, // [seq_len, num_q_heads, head_dim]
    half* k, // [seq_len, num_kv_heads, head_dim]
    int64_t seq_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t start_pos = 0,
    float rope_theta = 1000000.0f,
    cudaStream_t stream = 0
);

// 6. Causal Grouped-Query Attention:
// q: [seq_len, num_q_heads, head_dim]
// k: [seq_len, num_kv_heads, head_dim]
// v: [seq_len, num_kv_heads, head_dim]
// out: [seq_len, num_q_heads, head_dim]
// Scale factor: 1.0f / sqrt(head_dim)
void causal_gqa(
    const half* q,
    const half* k,
    const half* v,
    half* out,
    int64_t seq_len,
    int64_t num_q_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    cudaStream_t stream = 0
);

// 7. Standalone Softmax over last dimension (FP32 accumulation)
// x: [rows, cols], out: [rows, cols]
void softmax(
    const half* x,
    half* out,
    int64_t rows,
    int64_t cols,
    cudaStream_t stream = 0
);

// 7b. Single-token Decode Attention attending over head-major KV cache:
// q: [num_q_heads, head_dim]
// k_cache: [num_kv_heads, max_seq, head_dim]
// v_cache: [num_kv_heads, max_seq, head_dim]
// out: [num_q_heads, head_dim]
void decode_gqa(
    const half* q,
    const half* k_cache,
    const half* v_cache,
    half* out,
    int total_seq_len,
    int num_q_heads,
    int num_kv_heads,
    int max_seq,
    int head_dim,
    cudaStream_t stream = 0
);

// 8. SwiGLU: out = (gate * sigmoid(gate)) * up
// gate: [count], up: [count], out: [count]
void swiglu(
    const half* gate,
    const half* up,
    half* out,
    int64_t count,
    cudaStream_t stream = 0
);

// 9. cuBLAS Linear Projection: C = A * B^T
// A is [M, K] in row-major
// B is [N, K] in row-major (PyTorch linear weight layout)
// C is [M, N] in row-major
// Uses cublasGemmEx with FP16 input, FP32 accumulate, FP16 output
void matmul_transB(
    cublasHandle_t handle,
    const half* A,
    const half* B,
    half* C,
    int M,
    int K,
    int N
);

// 9b. cuBLAS Linear Projection to FP32 output: C = A * B^T
// A is [M, K] in FP16, B is [N, K] in FP16, C is [M, N] in FP32
// Uses cublasGemmEx with FP16 input, FP32 accumulate, FP32 output (CUDA_R_32F)
void matmul_transB_f32(
    cublasHandle_t handle,
    const half* A,
    const half* B,
    float* C,
    int M,
    int K,
    int N
);

// 10. Custom FP16 GEMV for decode (M=1), one warp per output neuron.
//     y[n] = sum_k(W[n,k] * x[k])   W:[N,K] FP16, x:[K] FP16, y:[N] FP16
void gemv(
    const half* W,     // [N, K] row-major
    const half* x,     // [K]
    half* y,           // [N]
    int N,
    int K,
    cudaStream_t stream = 0
);

// 10b. GEMV + bias:  y[n] = dot(W[n], x) + b[n]
void gemv_bias(
    const half* W,     // [N, K]
    const half* x,     // [K]
    const half* b,     // [N]
    half* y,           // [N]
    int N,
    int K,
    cudaStream_t stream = 0
);

// 10c. GEMV + residual (fused residual add):  y[n] = dot(W[n], x) + r[n]
// Used for o_proj and down_proj in decode.
void gemv_residual(
    const half* W,     // [N, K]
    const half* x,     // [K]
    const half* r,     // [N]  residual to add
    half* y,           // [N]  output
    int N,
    int K,
    cudaStream_t stream = 0
);

} // namespace cuda
} // namespace engine

