#pragma once

#include <cstddef>

struct ReadBandwidthResult {
    float median_gb_s;
    float min_gb_s;
    float max_gb_s;
    float median_ms;
};

ReadBandwidthResult run_read_bandwidth_benchmark(int num_warmup = 5, int num_timed = 25);

