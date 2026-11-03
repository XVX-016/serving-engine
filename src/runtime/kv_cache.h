#pragma once

#include "src/common/cuda_utils.h"
#include <cuda_fp16.h>
#include <cstdint>

namespace engine {

// Contiguous KV Cache for a single sequence:
// Layout: [num_layers][2 (0=K, 1=V)][num_kv_heads][max_seq][head_dim]
// Along the inner dimension: head_dim (64) contiguous halfs = 128 bytes (1 cache line)
class KVCache {
public:
    KVCache() = default;
    KVCache(int num_layers, int num_kv_heads, int head_dim, int max_seq) {
        allocate(num_layers, num_kv_heads, head_dim, max_seq);
    }

    void allocate(int num_layers, int num_kv_heads, int head_dim, int max_seq) {
        num_layers_ = num_layers;
        num_kv_heads_ = num_kv_heads;
        head_dim_ = head_dim;
        max_seq_ = max_seq;

        // Total elements = num_layers * 2 * num_kv_heads * max_seq * head_dim
        size_t total_elements = static_cast<size_t>(num_layers_) * 2 * num_kv_heads_ * max_seq_ * head_dim_;
        data_.allocate(total_elements);
        data_.zero();
        current_seq_len_ = 0;
    }

    // Pointer to specific layer and K/V buffer:
    // is_v: 0 for K, 1 for V
    half* get_layer_ptr(int layer_idx, int is_v) {
        size_t offset = (static_cast<size_t>(layer_idx) * 2 + is_v) * num_kv_heads_ * max_seq_ * head_dim_;
        return data_.data() + offset;
    }

    const half* get_layer_ptr(int layer_idx, int is_v) const {
        size_t offset = (static_cast<size_t>(layer_idx) * 2 + is_v) * num_kv_heads_ * max_seq_ * head_dim_;
        return data_.data() + offset;
    }

    // Pointer to specific head buffer for a layer:
    half* get_head_ptr(int layer_idx, int is_v, int kv_head) {
        size_t offset = ((static_cast<size_t>(layer_idx) * 2 + is_v) * num_kv_heads_ + kv_head) * max_seq_ * head_dim_;
        return data_.data() + offset;
    }

    const half* get_head_ptr(int layer_idx, int is_v, int kv_head) const {
        size_t offset = ((static_cast<size_t>(layer_idx) * 2 + is_v) * num_kv_heads_ + kv_head) * max_seq_ * head_dim_;
        return data_.data() + offset;
    }

    int num_layers() const { return num_layers_; }
    int num_kv_heads() const { return num_kv_heads_; }
    int head_dim() const { return head_dim_; }
    int max_seq() const { return max_seq_; }
    int current_seq_len() const { return current_seq_len_; }
    void set_current_seq_len(int len) { current_seq_len_ = len; }

    half* raw_data() { return data_.data(); }
    const half* raw_data() const { return data_.data(); }

private:
    int num_layers_ = 0;
    int num_kv_heads_ = 0;
    int head_dim_ = 0;
    int max_seq_ = 0;
    int current_seq_len_ = 0;

    DeviceBuffer<half> data_;
};

namespace cuda {

// Append a range of tokens [start_pos, start_pos + num_tokens) for a layer into the KV cache.
// src_k / src_v layout: [num_tokens, num_kv_heads, head_dim]
// dst_k / dst_v in cache: [num_kv_heads, max_seq, head_dim]
void append_kv_cache(
    const half* src_k,
    const half* src_v,
    half* dst_k_head_base,
    half* dst_v_head_base,
    int num_tokens,
    int start_pos,
    int num_kv_heads,
    int max_seq,
    int head_dim,
    cudaStream_t stream = 0
);

} // namespace cuda
} // namespace engine
