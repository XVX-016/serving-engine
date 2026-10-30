#!/usr/bin/env python3
"""
scripts/download_gguf.py
Downloads official Qwen2.5-0.5B-Instruct GGUFs (fp16, q8_0, q4_k_m) from Hugging Face.
"""
import os
from huggingface_hub import hf_hub_download

REPO_ID = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
TARGET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "models")
os.makedirs(TARGET_DIR, exist_ok=True)

FILES = [
    "qwen2.5-0.5b-instruct-fp16.gguf",
    "qwen2.5-0.5b-instruct-q8_0.gguf",
    "qwen2.5-0.5b-instruct-q4_k_m.gguf",
]

def main():
    for f in FILES:
        print(f"Downloading {f} from {REPO_ID}...")
        path = hf_hub_download(repo_id=REPO_ID, filename=f, local_dir=TARGET_DIR)
        print(f"Downloaded {f} -> {path}")

if __name__ == "__main__":
    main()

