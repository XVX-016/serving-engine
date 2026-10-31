import os, sys, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT_DIR = r"c:\Computing\serving_engine"
PY_ENGINE_DIR = os.path.join(ROOT_DIR, "reference", "py-engine")
if PY_ENGINE_DIR not in sys.path:
    sys.path.insert(0, PY_ENGINE_DIR)

from engine.kernels.paged_attention_integration import KernelPagedManualDecode

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="cuda")
hf_model.eval()

runner = KernelPagedManualDecode(
    model=hf_model,
    tokenizer=tokenizer,
    block_size=16,
    num_blocks=256,
    device="cuda",
    dtype=torch.float16,
)

# Warmup
_ = runner.generate("Warmup text", max_new_tokens=2)

torch.cuda.synchronize()
print("Starting profiled generation of 1 token...")
# Generate exactly 1 token to trace one decode step
torch.cuda.cudart().cudaProfilerStart()
_ = runner.generate("The quick brown fox jumps over the lazy dog.", max_new_tokens=1)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print("Done profile step.")
