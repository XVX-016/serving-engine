import os
import sys
import json
import argparse
import torch
import safetensors.torch
import shutil

def fuse_model(src_dir, out_dir):
    print(f"Fusing model from {src_dir} to {out_dir}...")
    os.makedirs(out_dir, exist_ok=True)
    src_file = os.path.join(src_dir, "model.safetensors")
    if not os.path.exists(src_file):
        raise FileNotFoundError(f"Source file not found: {src_file}")
    weights = safetensors.torch.load_file(src_file)
    out_tensors = {}
    
    # Identify number of layers
    num_layers = 0
    while f"model.layers.{num_layers}.input_layernorm.weight" in weights:
        num_layers += 1
    print(f"Detected {num_layers} layers")
    
    # Check if quantized
    is_int8 = f"model.layers.0.self_attn.q_proj.weight.qweight" in weights and weights[f"model.layers.0.self_attn.q_proj.weight.qweight"].dtype == torch.int8
    is_int4 = f"model.layers.0.self_attn.q_proj.weight.qweight" in weights and weights[f"model.layers.0.self_attn.q_proj.weight.qweight"].dtype == torch.uint8
    is_fp16 = not is_int8 and not is_int4

    for name, tensor in weights.items():
        # Skip tensors that will be fused
        if any(proj in name for proj in [".q_proj.", ".k_proj.", ".v_proj.", ".gate_proj.", ".up_proj."]):
            continue
        out_tensors[name] = tensor

    for i in range(num_layers):
        lp = f"model.layers.{i}."
        attn = f"{lp}self_attn."
        mlp = f"{lp}mlp."
        
        # Fuse biases (always FP16/BF16)
        qb = weights[f"{attn}q_proj.bias"]
        kb = weights[f"{attn}k_proj.bias"]
        vb = weights[f"{attn}v_proj.bias"]
        qkv_b = torch.cat([qb, kb, vb], dim=0)
        out_tensors[f"{attn}qkv_proj.bias"] = qkv_b
        
        if is_fp16:
            qw = weights[f"{attn}q_proj.weight"]
            kw = weights[f"{attn}k_proj.weight"]
            vw = weights[f"{attn}v_proj.weight"]
            out_tensors[f"{attn}qkv_proj.weight"] = torch.cat([qw, kw, vw], dim=0)
            
            gw = weights[f"{mlp}gate_proj.weight"]
            uw = weights[f"{mlp}up_proj.weight"]
            out_tensors[f"{mlp}gate_up_proj.weight"] = torch.cat([gw, uw], dim=0)
        else:
            qw = weights[f"{attn}q_proj.weight.qweight"]
            kw = weights[f"{attn}k_proj.weight.qweight"]
            vw = weights[f"{attn}v_proj.weight.qweight"]
            out_tensors[f"{attn}qkv_proj.weight.qweight"] = torch.cat([qw, kw, vw], dim=0)
            
            qs = weights[f"{attn}q_proj.weight.scales"]
            ks = weights[f"{attn}k_proj.weight.scales"]
            vs = weights[f"{attn}v_proj.weight.scales"]
            out_tensors[f"{attn}qkv_proj.weight.scales"] = torch.cat([qs, ks, vs], dim=0)
            
            gw = weights[f"{mlp}gate_proj.weight.qweight"]
            uw = weights[f"{mlp}up_proj.weight.qweight"]
            out_tensors[f"{mlp}gate_up_proj.weight.qweight"] = torch.cat([gw, uw], dim=0)
            
            gs = weights[f"{mlp}gate_proj.weight.scales"]
            us = weights[f"{mlp}up_proj.weight.scales"]
            out_tensors[f"{mlp}gate_up_proj.weight.scales"] = torch.cat([gs, us], dim=0)
            
    out_file = os.path.join(out_dir, "model.safetensors")
    print(f"Saving {len(out_tensors)} tensors to {out_file}...")
    safetensors.torch.save_file(out_tensors, out_file)
    
    # Copy metadata files
    for fname in os.listdir(src_dir):
        if fname.endswith(".json") or fname.endswith(".txt") or fname.startswith("tokenizer"):
            src_p = os.path.join(src_dir, fname)
            dst_p = os.path.join(out_dir, fname)
            if os.path.isfile(src_p):
                shutil.copy2(src_p, dst_p)
    print("Done!")

def main():
    parser = argparse.ArgumentParser(description="Fuse QKV and Gate-Up weights for Qwen2.5")
    parser.add_argument("--src-dir", type=str, required=True, help="Source model directory")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for fused model")
    args = parser.parse_args()
    fuse_model(args.src_dir, args.out_dir)

if __name__ == "__main__":
    main()

