with open('src/cuda/gemv_quant.cu', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''void gemv_int4_g128(
    const uint8_t* W_q,
    const half* scales,
    const half* x,
    half* y,
    int N,
    int K,
    cudaStream_t stream
) {
    if (N <= 0 || K <= 0) return;
    int blocks = (N + 15) / 16;
    gemv_int4_splitk_kernel<EpiNone, 128, 2><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(W_q, scales, x, nullptr, y, N, K);
    CUDA_CHECK(cudaGetLastError());
}

void gemv_int4_g128_bias(
    const uint8_t* W_q,
    const half* scales,
    const half* x,
    const half* b,
    half* y,
    int N,
    int K,
    cudaStream_t stream
) {
    if (N <= 0 || K <= 0) return;
    int blocks = (N + 15) / 16;
    gemv_int4_splitk_kernel<EpiBias, 128, 2><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(W_q, scales, x, b, y, N, K);
    CUDA_CHECK(cudaGetLastError());
}

void gemv_int4_g128_residual(
    const uint8_t* W_q,
    const half* scales,
    const half* x,
    const half* r,
    half* y,
    int N,
    int K,
    cudaStream_t stream
) {
    if (N <= 0 || K <= 0) return;
    int blocks = (N + 15) / 16;
    gemv_int4_splitk_kernel<EpiResidual, 128, 2><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(W_q, scales, x, r, y, N, K);
    CUDA_CHECK(cudaGetLastError());
}'''

repl = '''void gemv_int4_g128(
    const uint8_t* W_q,
    const half* scales,
    const half* x,
    half* y,
    int N,
    int K,
    cudaStream_t stream
) {
    if (N <= 0 || K <= 0) return;
    int blocks = (N + 15) / 16;
    gemv_int4_splitk_kernel<EpiNone, 128, 2><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(W_q, scales, x, nullptr, y, N, K);
    CUDA_CHECK(cudaGetLastError());
}

void gemv_int4_g128_bias(
    const uint8_t* W_q,
    const half* scales,
    const half* x,
    const half* b,
    half* y,
    int N,
    int K,
    cudaStream_t stream
) {
    if (N <= 0 || K <= 0) return;
    int blocks = (N + 15) / 16;
    gemv_int4_splitk_kernel<EpiBias, 128, 2><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(W_q, scales, x, b, y, N, K);
    CUDA_CHECK(cudaGetLastError());
}

void gemv_int4_g128_residual(
    const uint8_t* W_q,
    const half* scales,
    const half* x,
    const half* r,
    half* y,
    int N,
    int K,
    cudaStream_t stream
) {
    if (N <= 0 || K <= 0) return;
    int blocks = (N + 15) / 16;
    gemv_int4_splitk_kernel<EpiResidual, 128, 2><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(W_q, scales, x, r, y, N, K);
    CUDA_CHECK(cudaGetLastError());
}

void gemv_int4_g128_legacy(
    const uint8_t* W_q,
    const half* scales,
    const half* x,
    half* y,
    int N,
    int K,
    cudaStream_t stream
) {
    if (N <= 0 || K <= 0) return;
    int blocks = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    gemv_int4_kernel<EpiNone, 128><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(W_q, scales, x, nullptr, y, N, K);
    CUDA_CHECK(cudaGetLastError());
}

void gemv_int4_g128_bias_legacy(
    const uint8_t* W_q,
    const half* scales,
    const half* x,
    const half* b,
    half* y,
    int N,
    int K,
    cudaStream_t stream
) {
    if (N <= 0 || K <= 0) return;
    int blocks = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    gemv_int4_kernel<EpiBias, 128><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(W_q, scales, x, b, y, N, K);
    CUDA_CHECK(cudaGetLastError());
}

void gemv_int4_g128_residual_legacy(
    const uint8_t* W_q,
    const half* scales,
    const half* x,
    const half* r,
    half* y,
    int N,
    int K,
    cudaStream_t stream
) {
    if (N <= 0 || K <= 0) return;
    int blocks = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    gemv_int4_kernel<EpiResidual, 128><<<blocks, THREADS_PER_BLOCK, 0, stream>>>(W_q, scales, x, r, y, N, K);
    CUDA_CHECK(cudaGetLastError());
}'''

text = text.replace(target, repl, 1)

with open('src/cuda/gemv_quant.cu', 'w', encoding='utf-8') as f:
    f.write(text)

print('Added legacy functions back into gemv_quant.cu.')
