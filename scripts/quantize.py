#!/usr/bin/env python3
"""
scripts/quantize.py

Quantizes model weights:
- INT8 per-output-channel symmetric
- INT4 group size 128 (and group size 64) with FP16 scales (symmetric and asymmetric)
- Options for embedding/LM head: FP16 or INT8

Exports quantized model weights and config into safetensors format.
"""

import os
import sys
import argparse
import json
import torch
import safetensors.torch

LINEAR_WEIGHT_SUFFIXES = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]

def quantize_int8_symmetric(weight: torch.Tensor):
    """
    weight: [N, K] in float/half
    Returns:
      qweight: [N, K] in int8
      scales: [N] in float16
    Formula:
      scale = max(|weight|, dim=1) / 127.0
      qweight = clamp(round(weight / scale), -127, 127)
    """
    orig_dtype = weight.dtype
    w_float = weight.to(torch.float32)
    max_val = torch.amax(torch.abs(w_float), dim=1, keepdim=True)
    scale = torch.clamp(max_val / 127.0, min=1e-8)
    q = torch.clamp(torch.round(w_float / scale), -127, 127).to(torch.int8)
    return q, scale.squeeze(1).to(torch.float16)

def quantize_int4_grouped(weight: torch.Tensor, group_size: int = 128, asymmetric: bool = False):
    """
    weight: [N, K]
    Returns:
      packed_qweight: [N, K // 2] in uint8 (two nibbles per byte: low=even k, high=odd k)
      scales: [N, K // group_size] in float16
      zeros: [N, K // group_size] in float16 (if asymmetric)
    """
    N, K = weight.shape
    assert K % group_size == 0, f"K={K} must be divisible by group_size={group_size}"
    num_groups = K // group_size

    w_reshaped = weight.to(torch.float32).reshape(N, num_groups, group_size)

    if not asymmetric:
        # Symmetric: range [-8, 7], scale = max(|w|) / 7.0 (or range [-7, 7])
        # Using [-7, 7] or [-8, 7]. For symmetric, scale = max(|w|) / 7.0
        max_val = torch.amax(torch.abs(w_reshaped), dim=2, keepdim=True)
        scale = torch.clamp(max_val / 7.0, min=1e-8)
        # Quantize to [-8, 7]
        q = torch.clamp(torch.round(w_reshaped / scale), -8, 7).to(torch.int32)
        # Convert signed [-8, 7] to 4-bit unsigned [0, 15] via & 0x0F
        u4 = (q & 0x0F).to(torch.uint8).reshape(N, K)
        zeros = None
    else:
        # Asymmetric: range [0, 15], w = scale * q + zero
        min_val = torch.amin(w_reshaped, dim=2, keepdim=True)
        max_val = torch.amax(w_reshaped, dim=2, keepdim=True)
        scale = torch.clamp((max_val - min_val) / 15.0, min=1e-8)
        q = torch.clamp(torch.round((w_reshaped - min_val) / scale), 0, 15).to(torch.int32)
        u4 = (q & 0x0F).to(torch.uint8).reshape(N, K)
        zeros = min_val.squeeze(2).to(torch.float16)

    scale_f16 = scale.squeeze(2).to(torch.float16)

    # Pack two 4-bit nibbles per byte:
    # low nibble = even k (k_even)
    # high nibble = odd k (k_odd)
    k_even = u4[:, 0::2]
    k_odd = u4[:, 1::2]
    packed = (k_even & 0x0F) | ((k_odd & 0x0F) << 4)

    return packed, scale_f16, zeros

def main():
    parser = argparse.ArgumentParser(description="Quantize Qwen2.5 weights to INT8 / INT4")
    parser.add_argument("--src-dir", type=str, default="data/models/hf-cache/snapshots/7ae557604adf67be50417f59c2c2f167def9a775")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--format", type=str, choices=["int8_sym", "int4_g128_sym", "int4_g128_asym", "int4_g64_sym", "int4_g64_asym"], required=True)
    parser.add_argument("--embed-mode", type=str, choices=["fp16", "int8"], default="fp16")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    src_safetensors = os.path.join(args.src_dir, "model.safetensors")
    print(f"Loading weights from {src_safetensors}...")
    weights = safetensors.torch.load_file(src_safetensors)

    out_tensors = {}
    total_orig_bytes = 0
    total_quant_bytes = 0

    for name, tensor in weights.items():
        is_linear = any(name.endswith(suffix) for suffix in LINEAR_WEIGHT_SUFFIXES)
        is_embed = (name == "model.embed_tokens.weight")

        if is_linear:
            if args.format == "int8_sym":
                qweight, scales = quantize_int8_symmetric(tensor)
                out_tensors[f"{name}.qweight"] = qweight
                out_tensors[f"{name}.scales"] = scales
                orig_b = tensor.numel() * 2
                quant_b = qweight.numel() * 1 + scales.numel() * 2
            elif args.format.startswith("int4"):
                g_size = 64 if "g64" in args.format else 128
                asym = ("asym" in args.format)
                packed, scales, zeros = quantize_int4_grouped(tensor, group_size=g_size, asymmetric=asym)
                out_tensors[f"{name}.qweight"] = packed
                out_tensors[f"{name}.scales"] = scales
                quant_b = packed.numel() * 1 + scales.numel() * 2
                if zeros is not None:
                    out_tensors[f"{name}.zeros"] = zeros
                    quant_b += zeros.numel() * 2
                orig_b = tensor.numel() * 2
        elif is_embed and args.embed_mode == "int8":
            qweight, scales = quantize_int8_symmetric(tensor)
            out_tensors[f"{name}.qweight"] = qweight
            out_tensors[f"{name}.scales"] = scales
            orig_b = tensor.numel() * 2
            quant_b = qweight.numel() * 1 + scales.numel() * 2
        else:
            # Keep as FP16
            f16_tensor = tensor.to(torch.float16)
            out_tensors[name] = f16_tensor
            orig_b = tensor.numel() * 2
            quant_b = orig_b

        total_orig_bytes += orig_b
        total_quant_bytes += quant_b

    out_file = os.path.join(args.out_dir, "model.safetensors")
    print(f"Saving {len(out_tensors)} quantized tensors to {out_file}...")
    safetensors.torch.save_file(out_tensors, out_file)

    meta = {
        "format": args.format,
        "embed_mode": args.embed_mode,
        "orig_bytes": total_orig_bytes,
        "quant_bytes": total_quant_bytes,
        "compression_ratio": total_orig_bytes / total_quant_bytes
    }
    with open(os.path.join(args.out_dir, "quant_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Copy config.json and tokenizer files if present
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json"]:
        src_path = os.path.join(args.src_dir, fname)
        if os.path.exists(src_path):
            with open(src_path, "rb") as rf:
                with open(os.path.join(args.out_dir, fname), "wb") as wf:
                    wf.write(rf.read())

    print(f"Quantization finished!")
    print(f"Original size: {total_orig_bytes / (1024*1024):.2f} MB")
    print(f"Quantized size: {total_quant_bytes / (1024*1024):.2f} MB")
    print(f"Compression ratio: {total_orig_bytes / total_quant_bytes:.2f}x")

if __name__ == "__main__":
    main()

