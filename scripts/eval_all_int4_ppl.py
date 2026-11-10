#!/usr/bin/env python3
"""
scripts/eval_all_int4_ppl.py

Evaluates full WikiText-2 test split perplexity (C=1024, S=512) for all INT4 configurations:
1. Symmetric g128
2. Asymmetric g128
3. Symmetric g64
4. Asymmetric g64
5. Symmetric g128 + INT8 head
6. Asymmetric g128 + INT8 head
7. Symmetric g64 + INT8 head
8. Asymmetric g64 + INT8 head
"""

import os
import sys
import torch
import safetensors.torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_BASE = "data/models/hf-cache/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"

def dequant_int4_tensor(packed, scales, zeros=None, group_size=128):
    N, K_half = packed.shape
    K = K_half * 2
    u0 = (packed & 0x0F).to(torch.int32)
    u1 = ((packed >> 4) & 0x0F).to(torch.int32)

    if zeros is None:
        q0 = torch.where(u0 >= 8, u0 - 16, u0)
        q1 = torch.where(u1 >= 8, u1 - 16, u1)
    else:
        q0 = u0
        q1 = u1

    q = torch.empty((N, K), dtype=torch.float32, device=packed.device)
    q[:, 0::2] = q0.float()
    q[:, 1::2] = q1.float()

    num_groups = K // group_size
    q_reshaped = q.reshape(N, num_groups, group_size)
    scales_reshaped = scales.unsqueeze(2).float()

    if zeros is None:
        w_deq = (q_reshaped * scales_reshaped).reshape(N, K)
    else:
        zeros_reshaped = zeros.unsqueeze(2).float()
        w_deq = (q_reshaped * scales_reshaped + zeros_reshaped).reshape(N, K)

    return w_deq

def dequant_int8_tensor(qweight, scales):
    # qweight: [N, K] int8, scales: [N] float16
    return (qweight.float() * scales.unsqueeze(1).float())

def eval_model_ppl(model_dir, group_size, is_asym, has_int8_head, encodings, max_length=1024, stride=512):
    print(f"\n=======================================================")
    print(f"Loading {model_dir} (G={group_size}, asym={is_asym}, int8_head={has_int8_head})...")
    
    # Load clean base architecture in FP16 on cuda
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_BASE,
        dtype=torch.float16,
        device_map="cuda"
    )
    model.eval()

    w_quant = safetensors.torch.load_file(os.path.join(model_dir, "model.safetensors"))

    # Dequantize layers into model
    for name, module in model.named_modules():
        if hasattr(module, "weight"):
            qw_key = f"{name}.weight.qweight"
            sc_key = f"{name}.weight.scales"
            zr_key = f"{name}.weight.zeros"

            if qw_key in w_quant:
                qw = w_quant[qw_key].cuda()
                sc = w_quant[sc_key].cuda()
                zr = w_quant[zr_key].cuda() if zr_key in w_quant else None

                if "embed_tokens" in name:
                    # INT8 head/embed
                    deq_w = dequant_int8_tensor(qw, sc).to(module.weight.dtype)
                else:
                    # INT4 linear
                    deq_w = dequant_int4_tensor(qw, sc, zr, group_size=group_size).to(module.weight.dtype)

                module.weight.data.copy_(deq_w)

    seq_len = encodings.size(1)
    nlls = []
    total_tokens = 0
    prev_end_loc = 0

    with torch.no_grad():
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings[:, begin_loc:end_loc].cuda()
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100
            outputs = model(input_ids, labels=target_ids)
            nlls.append(outputs.loss * trg_len)
            total_tokens += trg_len
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

    ppl = torch.exp(torch.stack(nlls).sum() / total_tokens).item()
    print(f"--> PPL: {ppl:.4f} (over {total_tokens} tokens)")
    return ppl

def main():
    print("Loading full WikiText-2 test split...")
    test_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    test_text = "\n\n".join(test_ds["text"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE)
    encodings = tokenizer(test_text, return_tensors="pt").input_ids

    configs = [
        ("Symmetric G128", "data/models/qwen2.5-0.5b-int4-g128", 128, False, False),
        ("Asymmetric G128", "data/models/qwen2.5-0.5b-int4-g128-asym", 128, True, False),
        ("Symmetric G64", "data/models/qwen2.5-0.5b-int4-g64", 64, False, False),
        ("Asymmetric G64", "data/models/qwen2.5-0.5b-int4-g64-asym", 64, True, False),
        ("Symmetric G128 + INT8 Head", "data/models/qwen2.5-0.5b-int4-g128-head-int8", 128, False, True),
        ("Asymmetric G128 + INT8 Head", "data/models/qwen2.5-0.5b-int4-g128-asym-head-int8", 128, True, True),
        ("Symmetric G64 + INT8 Head", "data/models/qwen2.5-0.5b-int4-g64-head-int8", 64, False, True),
        ("Asymmetric G64 + INT8 Head", "data/models/qwen2.5-0.5b-int4-g64-asym-head-int8", 64, True, True),
    ]

    results = {}
    for label, path, g, asym, head_i8 in configs:
        ppl = eval_model_ppl(path, g, asym, head_i8, encodings)
        results[label] = ppl

    print("\n================== FULL SUMMARY ==================")
    ref_fp16 = 13.6522
    for label, ppl in results.items():
        rel = ((ppl - ref_fp16) / ref_fp16) * 100.0
        print(f"{label:<30} | PPL: {ppl:<8.4f} | Rel: {rel:+.2f}%")

if __name__ == "__main__":
    main()

