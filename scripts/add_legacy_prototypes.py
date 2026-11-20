with open('src/cuda/kernels.h', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''void gemv_int4_g128(
    const uint8_t* W_q,
    const half* scales,
    const half* x,
    half* y,
    int N,
    int K,
    cudaStream_t stream = 0
);'''

repl = '''void gemv_int4_g128(
    const uint8_t* W_q,
    const half* scales,
    const half* x,
    half* y,
    int N,
    int K,
    cudaStream_t stream = 0
);

void gemv_int4_g128_legacy(
    const uint8_t* W_q,
    const half* scales,
    const half* x,
    half* y,
    int N,
    int K,
    cudaStream_t stream = 0
);

void gemv_int4_g128_bias_legacy(
    const uint8_t* W_q,
    const half* scales,
    const half* x,
    const half* b,
    half* y,
    int N,
    int K,
    cudaStream_t stream = 0
);

void gemv_int4_g128_residual_legacy(
    const uint8_t* W_q,
    const half* scales,
    const half* x,
    const half* r,
    half* y,
    int N,
    int K,
    cudaStream_t stream = 0
);'''

text = text.replace(target, repl, 1)

with open('src/cuda/kernels.h', 'w', encoding='utf-8') as f:
    f.write(text)

print('Updated kernels.h with legacy prototypes.')
