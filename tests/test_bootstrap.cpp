#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include "src/common/cuda_utils.h"
#include "src/cuda/dummy_kernel.h"

int main() {
    const size_t n = 1024;
    std::vector<float> h_a(n), h_b(n), h_c(n);
    for (size_t i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    launch_vector_add(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < n; ++i) {
        float expected = h_a[i] + h_b[i];
        if (std::fabs(h_c[i] - expected) > 1e-5f) {
            std::cerr << "Mismatch at index " << i << ": expected " << expected << ", got " << h_c[i] << std::endl;
            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_c);
            return 1;
        }
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    std::cout << "test_bootstrap PASSED: vector add verified across " << n << " elements." << std::endl;
    return 0;
}

