# Quantization Format Specification (docs/quant_format.md)

## 1. Overview
This document specifies the binary storage layout and GPU memory organization for weight-only quantized models in the serving engine (Task P8).

Supported quantization formats:
1. **INT8 Per-Channel Symmetric (`Q8_0` style)**:
   - Quantized granularity: 1 scale per output row ($N$).
   - Weight elements: signed 8-bit integers (`int8_t`).
2. **INT4 Grouped Symmetric / Asymmetric (`Q4_0` / AWQ style)**:
   - Group size: $G = 128$ (or $G = 64$) along input dimension $K$.
   - Weight elements: 4-bit nibbles packed two per byte (`uint8_t`).
   - Scales: FP16 (`half`), 1 per group.
   - Optional zero points: FP16 (`half`), 1 per group (for asymmetric mode).

---

## 2. In-Register Dequantization & Bit Packing

### 2.1 INT8 Symmetric (Per-Channel)
For matrix $W \in \mathbb{R}^{N \times K}$ with output row index $n \in [0, N-1]$:
- Scale: $s[n] = \frac{\max_{k} |W[n, k]|}{127.0f}$
- Storage:
  - Weight matrix $Q \in \mathbb{Z}^{N \times K}$, stored as `int8_t` in row-major order.
  - Scale vector $S \in \mathbb{R}^N$, stored as `half` or `float` [N].
- Coalescing:
  - Rows are padded/aligned to 16 bytes so warps can execute vectorized 128-bit loads (`int4` / `uint4` = 16 bytes = 16 INT8 weights per thread).
- Arithmetic in register:
  $$W_{\text{dequant}}[n, k] = s[n] \times \text{float}(Q[n, k])$$

### 2.2 INT4 Grouped (Group Size 128)
For matrix $W \in \mathbb{R}^{N \times K}$:
Along each row $n$, weights are divided into blocks of $G = 128$ consecutive elements.
- Number of groups per row: $N_{\text{groups}} = K / G$.

#### Nibble Packing (Two Nibbles per Byte)
To achieve coalesced 16-byte memory loads and cheap in-register unpacking:
- Each byte stores two adjacent 4-bit elements along dimension $K$:
  - Low 4 bits (bits 0–3): element at even index $k$ ($q_0 = \text{byte} \& 0\text{x0F}$)
  - High 4 bits (bits 4–7): element at odd index $k+1$ ($q_1 = (\text{byte} \gg 4) \& 0\text{x0F}$)
- 16 bytes loaded by a single 128-bit instruction (`uint4`) yield 32 packed weights.
- Four threads in a warp can load an entire 128-element group in a single collective transaction.

#### In-Register Unpacking Sequence
For signed symmetric INT4 with range $[-8, 7]$:
Given an unpacked 4-bit integer $v \in [0, 15]$:
- Signed conversion:
  $$q = \begin{cases} v & \text{if } v < 8 \\ v - 16 & \text{if } v \ge 8 \end{cases}$$
- Or arithmetic bit-shift: $((\text{int32\_t})(v \ll 28)) \gg 28$.
- Dot product accumulation:
  $$acc += (\text{float}(q) \times s) \times x$$

#### Asymmetric Mode (Affine)
If asymmetric zero-points are enabled:
$$W_{\text{dequant}} = s \times (v - z)$$
where $s \in \text{FP16}$, $z \in \text{FP16}$ (or uint4 zero point).

---

## 3. File Container Format
The quantized model is exported by `scripts/quantize.py` into a single file or a directory containing:
1. `quant_config.json`:
   - `quant_type`: `"int8_per_channel"` | `"int4_g128_sym"` | `"int4_g128_asym"`
   - `group_size`: 128 (or 64)
   - `embed_type`: `"fp16"` | `"int8"`
   - `lm_head_type`: `"fp16"` | `"int8"`
2. Tensor files:
   - Stored with standard safetensors or contiguous flat binary format with header metadata:
     - `tensor_name.qweight`: Packed byte buffer
     - `tensor_name.scales`: FP16 scale factors
     - `tensor_name.zeros` (if asymmetric): FP16 zero points
     - `tensor_name.bias`: Unquantized FP16 biases

