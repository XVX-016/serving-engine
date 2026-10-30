#!/usr/bin/env python3
"""
scripts/export_reference.py

Generates numerical ground truth from Hugging Face transformers in FP32 on CPU.
Exports token IDs, embedding, post-attention residual, post-MLP residual (layers 0, 1, 23),
final norm output, and full logits as .npy files under data/reference/.
"""

import os
import json
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "reference")

PROMPTS = [
    {
        "id": "01_short",
        "name": "short",
        "text": "The capital of France is"
    },
    {
        "id": "02_medium",
        "name": "medium",
        "text": "Deep learning inference engines optimize memory bandwidth and cache locality to achieve high token generation speeds on modern GPUs."
    },
    {
        "id": "03_long",
        "name": "long",
        "text": (
            "Artificial intelligence and neural network systems have evolved rapidly over recent decades. "
            "From early perceptrons to modern large language models with billions of parameters, hardware "
            "architectures such as GPUs with specialized Tensor Cores have enabled unprecedented scale. "
            "Efficient serving requires optimizing KV cache memory layout, batch scheduling, and low-latency "
            "kernel launches to achieve peak throughput."
        )
    },
    {
        "id": "04_code",
        "name": "code",
        "text": (
            "def quicksort(arr):\n"
            "    if len(arr) <= 1:\n"
            "        return arr\n"
            "    pivot = arr[len(arr) // 2]\n"
            "    left = [x for x in arr if x < pivot]\n"
            "    middle = [x for x in arr if x == pivot]\n"
            "    right = [x for x in arr if x > pivot]\n"
            "    return quicksort(left) + middle + quicksort(right)\n"
        )
    },
    {
        "id": "05_unicode",
        "name": "unicode",
        "text": "L'élégance des mathématiques: ∀ x ∈ ℝ, e^(iπ) + 1 = 0. Σ_(n=1)^∞ 1/n² = π²/6. Café & naïve Schrödinger."
    },
    {
        "id": "06_cjk_emoji",
        "name": "cjk_emoji",
        "text": "你好，世界！🚀 こんにちは世界！🌟 안녕하세요 세계! 🇨🇳 🇯🇵 🇰🇷"
    },
    {
        "id": "07_chat_template",
        "name": "chat_template",
        "messages": [
            {"role": "user", "content": "Explain grouped-query attention in 2 sentences."}
        ]
    },
    {
        "id": "08_whitespace_indent",
        "name": "whitespace_indent",
        "text": (
            "class TransformerBlock:\n"
            "    def __init__(self, dim, heads):\n"
            "        self.dim = dim\n"
            "        self.heads = heads\n"
            "\n"
            "    def forward(self, x):\n"
            "        # 8 spaces indent\n"
            "        residual = x\n"
            "        x = norm(x)\n"
            "        return residual + x\n"
        )
    }
]

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Loading {MODEL_NAME} in FP32 on CPU...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        device_map="cpu"
    )
    model.eval()

    captured = {}

    def hook_embed(mod, inp, out):
        t = out.detach()
        if t.ndim == 2:
            t = t.unsqueeze(0)
        captured["embedding"] = t.cpu().numpy()

    def make_hook_post_attn(layer_idx):
        def hook(mod, inp):
            # Input to post_attention_layernorm is residual after attention addition
            t = inp[0].detach()
            if t.ndim == 2:
                t = t.unsqueeze(0)
            captured[f"layer_{layer_idx}_post_attn"] = t.cpu().numpy()
        return hook

    def make_hook_post_mlp(layer_idx):
        def hook(mod, inp, out):
            # Output of decoder layer is residual after MLP addition
            t = out[0].detach() if isinstance(out, (tuple, list)) else out.detach()
            if t.ndim == 2:
                t = t.unsqueeze(0)
            captured[f"layer_{layer_idx}_post_mlp"] = t.cpu().numpy()
        return hook

    def hook_norm(mod, inp, out):
        t = out.detach()
        if t.ndim == 2:
            t = t.unsqueeze(0)
        captured["final_norm"] = t.cpu().numpy()

    model.model.embed_tokens.register_forward_hook(hook_embed)
    model.model.norm.register_forward_hook(hook_norm)

    for idx in [0, 1, 23]:
        model.model.layers[idx].post_attention_layernorm.register_forward_pre_hook(make_hook_post_attn(idx))
        model.model.layers[idx].register_forward_hook(make_hook_post_mlp(idx))

    print(f"Exporting reference tensors for {len(PROMPTS)} prompts to {OUTPUT_DIR}...")

    manifest = []

    for p in PROMPTS:
        p_id = p["id"]
        p_name = p["name"]
        prompt_dir = os.path.join(OUTPUT_DIR, p_id)
        os.makedirs(prompt_dir, exist_ok=True)

        if "messages" in p:
            prompt_text = tokenizer.apply_chat_template(
                p["messages"],
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt_text = p["text"]

        inputs = tokenizer(prompt_text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        token_ids_np = input_ids.cpu().numpy()
        seq_len = int(token_ids_np.shape[1])

        captured.clear()
        with torch.no_grad():
            outputs = model(**inputs)

        logits_np = outputs.logits.detach().cpu().numpy()
        captured["logits"] = logits_np

        np.save(os.path.join(prompt_dir, "token_ids.npy"), token_ids_np)
        for tensor_name, arr in captured.items():
            np.save(os.path.join(prompt_dir, f"{tensor_name}.npy"), arr.astype(np.float32))

        meta = {
            "id": p_id,
            "name": p_name,
            "prompt_text": prompt_text,
            "seq_len": seq_len,
            "tensor_shapes": {k: list(v.shape) for k, v in captured.items()}
        }
        meta["tensor_shapes"]["token_ids"] = list(token_ids_np.shape)

        with open(os.path.join(prompt_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        manifest.append(meta)
        print(f"  [{p_id}] '{p_name}' (seq_len={seq_len}) -> exported {len(captured) + 1} tensors")

    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("Reference export complete.")

if __name__ == "__main__":
    main()

