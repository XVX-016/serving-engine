with open('src/cuda/gemv_quant.cu', 'r', encoding='utf-8') as f:
    text = f.read()

splitk_kernel_template = '''
// ============================================================================
// 2b. Split-K INT4 GEMV (Optimized Fix Attempt: 2 warps per row with shared reduction)
// ============================================================================
template <typename Epi, int GROUP_SIZE, int WARPS_PER_ROW = 2>
__global__ void gemv_int4_splitk_kernel(
    const uint8_t* __restrict__ W_q,
    const half* __restrict__ scales,
    const half* __restrict__ x,
    const half* __restrict__ extra,
    half* __restrict__ y,
    int N,
    int K
) {
    constexpr int ROWS_PER_BLOCK = 32 / WARPS_PER_ROW;
    int block_warp_id = threadIdx.x / 32;
    int lane          = threadIdx.x % 32;

    int row_in_block = block_warp_id / WARPS_PER_ROW;
    int slice_idx    = block_warp_id % WARPS_PER_ROW;

    int row_id = blockIdx.x * ROWS_PER_BLOCK + row_in_block;
    if (row_id >= N) return;

    int num_groups = K / GROUP_SIZE;
    const uint8_t* row_bytes = W_q + static_cast<size_t>(row_id) * (K / 2);
    const half* row_scales = scales + static_cast<size_t>(row_id) * num_groups;

    int slice_K = (K + WARPS_PER_ROW - 1) / WARPS_PER_ROW;
    int slice_start = slice_idx * slice_K;
    int slice_end   = min(slice_start + slice_K, K);

    float acc = 0.0f;
    int k_tile = slice_start;
    for (; k_tile + 32 * 8 <= slice_end; k_tile += 32 * 8) {
        int k = k_tile + lane * 8;
        uint32_t raw_u32 = *reinterpret_cast<const uint32_t*>(row_bytes + k / 2);
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&raw_u32);

        float4 x_vec = *reinterpret_cast<const float4*>(x + k);
        const half* x_h = reinterpret_cast<const half*>(&x_vec);

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint8_t b = bytes[i];
            int elem_idx = k + i * 2;
            int g_idx = elem_idx / GROUP_SIZE;
            float scale = __half2float(row_scales[g_idx]);

            uint16_t lo16 = 0x6400u | static_cast<uint16_t>((b & 0x0Fu) ^ 0x08u);
            uint16_t hi16 = 0x6400u | static_cast<uint16_t>(((b >> 4) & 0x0Fu) ^ 0x08u);
            float dq0 = (__half2float(*reinterpret_cast<half*>(&lo16)) - 1032.0f) * scale;
            float dq1 = (__half2float(*reinterpret_cast<half*>(&hi16)) - 1032.0f) * scale;

            acc += dq0 * __half2float(x_h[i * 2]);
            acc += dq1 * __half2float(x_h[i * 2 + 1]);
        }
    }

    for (int r = k_tile + lane * 2; r + 1 < slice_end; r += 32 * 2) {
        uint8_t b = row_bytes[r / 2];
        int g_idx0 = r / GROUP_SIZE;
        float scale0 = __half2float(row_scales[g_idx0]);
        int v0 = b & 0x0F;
        int q0 = (v0 >= 8) ? (v0 - 16) : v0;
        acc += (static_cast<float>(q0) * scale0) * __half2float(x[r]);

        int g_idx1 = (r + 1) / GROUP_SIZE;
        float scale1 = __half2float(row_scales[g_idx1]);
        int v1 = (b >> 4) & 0x0F;
        int q1 = (v1 >= 8) ? (v1 - 16) : v1;
        acc += (static_cast<float>(q1) * scale1) * __half2float(x[r + 1]);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    __shared__ float s_partial[32];
    if (lane == 0) {
        s_partial[block_warp_id] = acc;
    }
    __syncthreads();

    if (slice_idx == 0 && lane == 0) {
        float total_acc = 0.0f;
        #pragma unroll
        for (int w = 0; w < WARPS_PER_ROW; ++w) {
            total_acc += s_partial[row_in_block * WARPS_PER_ROW + w];
        }
        if constexpr (!std::is_same_v<Epi, EpiNone>) {
            total_acc += __half2float(extra[row_id]);
        }
        y[row_id] = __float2half(total_acc);
    }
}
'''

target_loc = '// ============================================================================\n// 3. Tile / Buffer Dequantization for Prefill Path'
text = text.replace(target_loc, splitk_kernel_template + '\n' + target_loc, 1)

with open('src/cuda/gemv_quant.cu', 'w', encoding='utf-8') as f:
    f.write(text)

print('Added gemv_int4_splitk_kernel to src/cuda/gemv_quant.cu')
