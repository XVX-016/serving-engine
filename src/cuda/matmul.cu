#include "src/cuda/kernels.h"
#include "src/common/cuda_utils.h"

namespace engine {
namespace cuda {

// C = A * B^T
// In row-major:
// A is [M, K], lda = K
// B is [N, K], ldb = K (so B^T is [K, N])
// C is [M, N], ldc = N
//
// In cuBLAS column-major interpretation:
// An [M, N] row-major matrix with leading dimension N is equivalent to an [N, M] column-major matrix with leading dimension N.
// Specifically: C_col = C^T = (A * B^T)^T = (B^T)^T * A^T = B * A^T.
// B is [N, K] row-major -> B_col is [K, N] column-major with ld = K.
// A is [M, K] row-major -> A_col is [K, M] column-major with ld = K.
// We want C_col = B * A^T:
// So in column-major:
// Matrix 1 (opA = N): we want [N, K]. B_col is [K, N]. So we transpose B_col! (opA = CUBLAS_OP_T), leading dimension = K.
// Matrix 2 (opB = N): we want [K, M]. A_col is [K, M]. So no transpose on A_col! (opB = CUBLAS_OP_N), leading dimension = K.
// Then the result is [N, M] with leading dimension N, exactly C_col!
void matmul_transB(
    cublasHandle_t handle,
    const half* A,
    const half* B,
    half* C,
    int M,
    int K,
    int N
) {
    float alpha = 1.0f;
    float beta = 0.0f;

    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,       // transpose B
        CUBLAS_OP_N,       // normal A
        N,                 // m for column-major result
        M,                 // n for column-major result
        K,                 // k for column-major inner dim
        &alpha,
        B, CUDA_R_16F, K,  // B_col is [K, N], transposed gives [N, K], ldb = K
        A, CUDA_R_16F, K,  // A_col is [K, M], not transposed gives [K, M], lda = K
        &beta,
        C, CUDA_R_16F, N,  // C_col is [N, M], ldc = N
        CUBLAS_COMPUTE_32F,// FP32 accumulation
        CUBLAS_GEMM_DEFAULT
    ));
}

void matmul_transB_f32(
    cublasHandle_t handle,
    const half* A,
    const half* B,
    float* C,
    int M,
    int K,
    int N
) {
    float alpha = 1.0f;
    float beta = 0.0f;

    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,       // transpose B
        CUBLAS_OP_N,       // normal A
        N,                 // m for column-major result
        M,                 // n for column-major result
        K,                 // k for column-major inner dim
        &alpha,
        B, CUDA_R_16F, K,  // B_col is [K, N], transposed gives [N, K], ldb = K
        A, CUDA_R_16F, K,  // A_col is [K, M], not transposed gives [K, M], lda = K
        &beta,
        C, CUDA_R_32F, N,  // C_col is [N, M] in FP32, ldc = N
        CUBLAS_COMPUTE_32F,// FP32 accumulation
        CUBLAS_GEMM_DEFAULT
    ));
}

} // namespace cuda
} // namespace engine

