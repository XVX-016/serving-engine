#include "src/cuda/dummy_kernel.h"
#include "src/common/cuda_utils.h"

__global__ void vector_add_kernel(const float* a, const float* b, float* c, size_t n) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void launch_vector_add(const float* a, const float* b, float* c, size_t n) {
    size_t block_size = 256;
    size_t grid_size = (n + block_size - 1) / block_size;
    vector_add_kernel<<<grid_size, block_size>>>(a, b, c, n);
    CUDA_CHECK(cudaGetLastError());
}

