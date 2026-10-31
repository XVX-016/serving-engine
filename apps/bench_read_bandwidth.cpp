#include "src/cuda/read_bandwidth.h"
#include "src/common/cuda_utils.h"
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "=== Running Read-Only Bandwidth Microbenchmark (256 MB, L2 flushed) ===" << std::endl;
    ReadBandwidthResult res = run_read_bandwidth_benchmark(5, 25);
    std::cout << "  Read Time: median=" << std::fixed << std::setprecision(3) << res.median_ms << " ms" << std::endl;
    std::cout << "  Bandwidth: median=" << std::fixed << std::setprecision(2) << res.median_gb_s 
              << " GB/s, spread=[" << res.min_gb_s << ", " << res.max_gb_s << "] GB/s" << std::endl;
    return 0;
}

