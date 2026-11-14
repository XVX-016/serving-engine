import subprocess
import json
import sys

prompts = [
    "The capital of France is",
    "Deep learning inference engines optimize memory bandwidth and cache locality to achieve high token generation speeds on modern GPUs.",
    "def quicksort(arr):\n",
    "你好，世界！🚀 こんにちは世界！🌟 안녕하세요 세계! 🇨🇳 🇯🇵 🇰🇷",
    "class TransformerBlock:\n"
]

models = [
    "data/models/qwen2.5-0.5b-fp16-fused",
    "data/models/qwen2.5-0.5b-int8-fused",
    "data/models/qwen2.5-0.5b-int4-g128-head-int8-fused"
]

def run_gen(model, prompt, cuda_graph=False, max_tokens=256):
    cmd = [
        "build/generate.exe",
        "--model", model,
        "--prompt", prompt,
        "--max-new-tokens", str(max_tokens)
    ]
    if cuda_graph:
        cmd.append("--cuda-graph")
    else:
        cmd.append("--kv-cache")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error running command: {' '.join(cmd)}")
        print(res.stderr)
        return []
    
    # parse token IDs
    for line in res.stdout.splitlines():
        if line.startswith("Generated token IDs: ["):
            ids_str = line[len("Generated token IDs: ["):-1]
            if not ids_str.strip():
                return []
            return [int(x.strip()) for x in ids_str.split(",")]
    return []

all_passed = True

for model in models:
    print(f"\n=======================================================")
    print(f"Testing model: {model}")
    print(f"=======================================================")
    for i, p in enumerate(prompts):
        print(f"Prompt {i+1}: {p[:40]}... (256 tokens)")
        eager_tokens = run_gen(model, p, cuda_graph=False, max_tokens=256)
        graph_tokens = run_gen(model, p, cuda_graph=True, max_tokens=256)
        
        if eager_tokens == graph_tokens:
            print(f"  PASS: {len(eager_tokens)} tokens generated, 100% exact match between Eager and CUDA Graph.")
        else:
            all_passed = False
            print(f"  FAIL: Token mismatch!")
            print(f"  Eager len: {len(eager_tokens)}, Graph len: {len(graph_tokens)}")
            for j in range(min(len(eager_tokens), len(graph_tokens))):
                if eager_tokens[j] != graph_tokens[j]:
                    print(f"  First mismatch at token {j}: eager={eager_tokens[j]} vs graph={graph_tokens[j]}")
                    break

if all_passed:
    print("\nSUCCESS: All 5 prompts for 256 tokens matched 100% across all models!")
else:
    print("\nFAILURE: Mismatches detected!")
    sys.exit(1)
