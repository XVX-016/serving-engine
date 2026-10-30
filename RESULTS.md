# RESULTS.md

| Task | Config | Batch | Prompt | Gen | Prefill tok/s | Decode ms/step | Decode tok/s | VRAM | Max logit diff | Notes |
|------|--------|-------|--------|-----|---------------|----------------|--------------|------|----------------|-------|
| P0   | HF eager FP16 | 1 | | | | | | | | |
| P0   | py-engine (Triton) | 1 | | | | | | | | |
| P0   | llama.cpp F16 | 1 | | | | | | | | |
| P0   | llama.cpp Q8_0 and Q4_K_M | 1 | | | | | | | | |
| P3   | Naive GPU, cuBLAS, no cache | 1 | | | | | | | | |
| P4   | KV cache + sampling | 1 | | | | | | | | |
| P5   | Custom GEMV/GEMM + fusion | 1 | | | | | | | | |
| P7   | Split-KV decode + flash prefill | 1, 32 | | | | | | | | |
| P8   | INT8 and INT4 weights | 1 | | | | | | | ppl: | |
| P9   | INT8, FP8, INT4 KV | 32 | | | | | | | ppl: | |
| P10  | Paged KV + batching | 1, 4, 8, 16 | | | | | | | | |
| P11  | CUDA graphs | 1, 8 | | | | | | | | |
| P12  | Megakernel | 1 | | | | | | | | |
