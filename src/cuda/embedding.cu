#include "src/cuda/kernels.h"
#include "src/common/cuda_utils.h"

namespace engine {
namespace cuda {

__global__ void embedding_lookup_kernel(
    const int64_t* __restrict__ token_ids,
    const half* __restrict__ table,
    half* __restrict__ out,
    int64_t seq_len,
    int64_t hidden_size
) {
    int64_t s = blockIdx.x;
    if (s >= seq_len) return;

    int64_t tok_id = token_ids[s];
    const half* src = table + tok_id * hidden_size;
    half* dst = out + s * hidden_size;

    for (int64_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        dst[i] = src[i];
    }
}

void embedding_lookup(
    const int64_t* token_ids,
    const half* table,
    half* out,
    int64_t seq_len,
    int64_t hidden_size,
    cudaStream_t stream
) {
    if (seq_len <= 0) return;
    int threads = 256;
    if (hidden_size < 256) threads = static_cast<int>(hidden_size);
    embedding_lookup_kernel<<<static_cast<unsigned int>(seq_len), threads, 0, stream>>>(
        token_ids, table, out, seq_len, hidden_size
    );
    CUDA_CHECK(cudaGetLastError());
}

} // namespace cuda
} // namespace engine

