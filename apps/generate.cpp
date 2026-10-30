#include <iostream>
#include <string>
#include <vector>
#include "src/common/cuda_utils.h"

int main(int argc, char** argv) {
    std::cout << "Engine v2 (C++/CUDA) CLI stub" << std::endl;
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err == cudaSuccess) {
        std::cout << "CUDA devices found: " << device_count << std::endl;
    } else {
        std::cerr << "Failed to query CUDA devices: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    return 0;
}

