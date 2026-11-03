#pragma once

#include <cuda_fp16.h>
#include <cstdint>

namespace engine {
namespace cuda {

// KVView provides an abstracted strided view over contiguous or structured KV caches.
// In P10, this struct can be augmented with block tables for paged layout.
// For contiguous cache:
//   base pointer: ptr to start of K or V buffer for layer / batch
//   stride_b: stride across sequences (if batched contiguous)
//   stride_h: stride across KV heads
//   stride_s: stride across sequence positions
//   stride_d: stride across head dimension elements (typically 1)
struct KVView {
    const half* __restrict__ data;
    int64_t stride_b;
    int64_t stride_h;
    int64_t stride_s;
    int64_t stride_d;

    __device__ __forceinline__ const half* get_ptr(int b, int h, int s) const {
        return data + b * stride_b + h * stride_h + s * stride_s;
    }

    __device__ __forceinline__ const half* get_ptr(int b, int h, int s, int d) const {
        return data + b * stride_b + h * stride_h + s * stride_s + d * stride_d;
    }
};

} // namespace cuda
} // namespace engine

