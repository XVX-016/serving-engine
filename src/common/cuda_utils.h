#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            throw std::runtime_error(std::string("CUDA error: ") +            \
                                     cudaGetErrorString(err));                \
        }                                                                     \
    } while (0)

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t status = (call);                                       \
        if (status != CUBLAS_STATUS_SUCCESS) {                                \
            fprintf(stderr, "cuBLAS error at %s:%d: status %d\n",             \
                    __FILE__, __LINE__, static_cast<int>(status));            \
            throw std::runtime_error("cuBLAS error code: " +                  \
                                     std::to_string(static_cast<int>(status))); \
        }                                                                     \
    } while (0)

namespace engine {

// RAII wrapper for GPU device memory
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() : ptr_(nullptr), size_(0) {}
    explicit DeviceBuffer(size_t count) : ptr_(nullptr), size_(0) {
        allocate(count);
    }
    ~DeviceBuffer() {
        reset();
    }

    // Non-copyable
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // Move-constructible & move-assignable
    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    void allocate(size_t count) {
        reset();
        if (count > 0) {
            CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
            size_ = count;
        }
    }

    void reset() {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            size_ = 0;
        }
    }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    size_t size() const { return size_; }
    size_t size_bytes() const { return size_ * sizeof(T); }

    void copy_from_host(const T* src, size_t count) {
        if (count > size_) {
            throw std::runtime_error("DeviceBuffer::copy_from_host buffer overflow");
        }
        CUDA_CHECK(cudaMemcpy(ptr_, src, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    void copy_to_host(T* dst, size_t count) const {
        if (count > size_) {
            throw std::runtime_error("DeviceBuffer::copy_to_host buffer overflow");
        }
        CUDA_CHECK(cudaMemcpy(dst, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void zero() {
        if (ptr_ && size_ > 0) {
            CUDA_CHECK(cudaMemset(ptr_, 0, size_ * sizeof(T)));
        }
    }

private:
    T* ptr_;
    size_t size_;
};

// High-precision RAII CUDA Event Timer
class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(start_, stream));
    }

    void stop(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stop_, stream));
    }

    float elapsed_ms() {
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

    cudaEvent_t get_start_event() const { return start_; }
    cudaEvent_t get_stop_event() const { return stop_; }

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};

} // namespace engine
