with open('tests/bench_real_shapes.cu', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace the INT4 baseline and stripped kernels with the exact ones from gemv_quant.cu including remainder loop
exact_int4_code = '''// INT4 Baseline GEMV (Exact code from src/cuda/gemv_quant.cu)
__global__ void gemv_int4_baseline(
    const uint8_t* __restrict__ W_q,
    const half* __restrict__ scales,
    const half* __restrict__ x,
    half* __restrict__ y,
    int N,
    int K,
    float* __restrict__ checksum_sink
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane    = threadIdx.x % 32;
    if (warp_id >= N) return;

    int num_groups = K / 128;
    const uint8_t* row_bytes = W_q + static_cast<size_t>(warp_id) * (K / 2);
    const half* row_scales = scales + static_cast<size_t>(warp_id) * num_groups;

    float acc = 0.0f;
    int k_tile = 0;
    for (; k_tile + 32 * 32 <= K; k_tile += 32 * 32) {
        int k = k_tile + lane * 32;
        uint4 raw = *reinterpret_cast<const uint4*>(row_bytes + k / 2);
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&raw);

        float4 x_vec0 = *reinterpret_cast<const float4*>(x + k);
        float4 x_vec1 = *reinterpret_cast<const float4*>(x + k + 8);
        float4 x_vec2 = *reinterpret_cast<const float4*>(x + k + 16);
        float4 x_vec3 = *reinterpret_cast<const float4*>(x + k + 24);

        const half* x_h[4] = {
            reinterpret_cast<const half*>(&x_vec0),
            reinterpret_cast<const half*>(&x_vec1),
            reinterpret_cast<const half*>(&x_vec2),
            reinterpret_cast<const half*>(&x_vec3)
        };

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            uint8_t b = bytes[i];
            int elem_idx = k + i * 2;
            int g_idx = elem_idx / 128;
            float scale = __half2float(row_scales[g_idx]);

            uint16_t lo16 = 0x6400u | static_cast<uint16_t>((b & 0x0Fu) ^ 0x08u);
            uint16_t hi16 = 0x6400u | static_cast<uint16_t>(((b >> 4) & 0x0Fu) ^ 0x08u);
            float dq0 = (__half2float(*reinterpret_cast<half*>(&lo16)) - 1032.0f) * scale;
            float dq1 = (__half2float(*reinterpret_cast<half*>(&hi16)) - 1032.0f) * scale;

            int h_chunk0 = (i * 2) / 8;
            int h_sub0   = (i * 2) % 8;
            acc += dq0 * __half2float(x_h[h_chunk0][h_sub0]);

            int h_chunk1 = (i * 2 + 1) / 8;
            int h_sub1   = (i * 2 + 1) % 8;
            acc += dq1 * __half2float(x_h[h_chunk1][h_sub1]);
        }
    }

    // Remainder loop for shapes where K % 1024 != 0 (e.g. K=896)
    for (int r = k_tile + lane * 2; r + 1 < K; r += 32 * 2) {
        uint8_t b = row_bytes[r / 2];
        int g_idx0 = r / 128;
        float scale0 = __half2float(row_scales[g_idx0]);
        int v0 = b & 0x0F;
        int q0 = (v0 >= 8) ? (v0 - 16) : v0;
        acc += (static_cast<float>(q0) * scale0) * __half2float(x[r]);

        int g_idx1 = (r + 1) / 128;
        float scale1 = __half2float(row_scales[g_idx1]);
        int v1 = (b >> 4) & 0x0F;
        int q1 = (v1 >= 8) ? (v1 - 16) : v1;
        acc += (static_cast<float>(q1) * scale1) * __half2float(x[r + 1]);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) {
        y[warp_id] = __float2half(acc);
        if (checksum_sink && warp_id == 0) {
            atomicAdd(checksum_sink, acc);
        }
    }
}

// INT4 Stripped-Math Variant WITH Checksum Sink (forces every byte loaded to contribute)
__global__ void gemv_int4_stripped_with_sink(
    const uint8_t* __restrict__ W_q,
    const half* __restrict__ scales,
    const half* __restrict__ x,
    half* __restrict__ y,
    int N,
    int K,
    float* __restrict__ checksum_sink
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane    = threadIdx.x % 32;
    if (warp_id >= N) return;

    int num_groups = K / 128;
    const uint8_t* row_bytes = W_q + static_cast<size_t>(warp_id) * (K / 2);
    const half* row_scales = scales + static_cast<size_t>(warp_id) * num_groups;

    float acc = 0.0f;
    uint32_t load_checksum = 0;

    int k_tile = 0;
    for (; k_tile + 32 * 32 <= K; k_tile += 32 * 32) {
        int k = k_tile + lane * 32;
        uint4 raw = *reinterpret_cast<const uint4*>(row_bytes + k / 2);
        load_checksum ^= raw.x ^ raw.y ^ raw.z ^ raw.w;
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&raw);

        float4 x_vec0 = *reinterpret_cast<const float4*>(x + k);
        const half* x_h0 = reinterpret_cast<const half*>(&x_vec0);

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            uint8_t b = bytes[i];
            int elem_idx = k + i * 2;
            int g_idx = elem_idx / 128;
            float scale = __half2float(row_scales[g_idx]);

            float dq = static_cast<float>(b) * scale;
            acc += dq * __half2float(x_h0[i % 8]);
        }
    }

    for (int r = k_tile + lane * 2; r + 1 < K; r += 32 * 2) {
        uint8_t b = row_bytes[r / 2];
        load_checksum ^= b;
        int g_idx0 = r / 128;
        float scale0 = __half2float(row_scales[g_idx0]);
        acc += (static_cast<float>(b) * scale0) * __half2float(x[r]);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
        load_checksum ^= __shfl_down_sync(0xffffffff, load_checksum, offset);
    }
    if (lane == 0) {
        y[warp_id] = __float2half(acc + (load_checksum & 0x1 ? 0.0f : 0.0f));
        if (checksum_sink && warp_id == 0) {
            atomicAdd(checksum_sink, acc + static_cast<float>(load_checksum));
        }
    }
}
'''

# Find where gemv_int4_baseline starts and gemv_int4_splitk starts
start_idx = text.find('// INT4 Baseline GEMV (Magic Float Mantissa Dequant)')
end_idx = text.find('// Split-K INT4 GEMV')
text = text[:start_idx] + exact_int4_code + '\n' + text[end_idx:]

with open('tests/bench_real_shapes.cu', 'w', encoding='utf-8') as f:
    f.write(text)

print('Updated bench_real_shapes.cu with full remainder loop.')
