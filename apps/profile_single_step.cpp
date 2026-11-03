#include "src/cuda/kernels.h"
#include "src/cuda/kv_view.h"
#include "src/common/cuda_utils.h"
#include <iostream>
#include <vector>

using namespace engine;
using namespace engine::cuda;

int main() {
    int B = 32;
    int S = 2048;
    int num_q_heads = 14;
    int num_kv_heads = 2;
    int head_dim = 64;

    size_t q_size = static_cast<size_t>(B) * num_q_heads * head_dim;
    size_t kv_size = static_cast<size_t>(B) * num_kv_heads * S * head_dim;

    DeviceBuffer<half> d_q(q_size);
    DeviceBuffer<half> d_k(kv_size);
    DeviceBuffer<half> d_v(kv_size);
    DeviceBuffer<half> d_out(q_size);
    DeviceBuffer<int> d_seq_lens(B);

    std::vector<int> h_seq_lens(B, S);
    d_seq_lens.copy_from_host(h_seq_lens.data(), B);

    d_q.zero();
    d_k.zero();
    d_v.zero();

    KVView k_view{
        d_k.data(),
        static_cast<int64_t>(num_kv_heads * S * head_dim),
        static_cast<int64_t>(S * head_dim),
        static_cast<int64_t>(head_dim),
        1
    };

    KVView v_view{
        d_v.data(),
        static_cast<int64_t>(num_kv_heads * S * head_dim),
        static_cast<int64_t>(S * head_dim),
        static_cast<int64_t>(head_dim),
        1
    };

    decode_gqa_split(
        d_q.data(),
        k_view,
        v_view,
        d_seq_lens.data(),
        d_out.data(),
        B,
        num_q_heads,
        num_kv_heads,
        head_dim,
        S
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}

