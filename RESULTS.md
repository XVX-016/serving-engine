# RESULTS.md

| Task | Config | Batch | Prompt | Gen | Prefill tok/s | Decode ms/step | Decode tok/s | VRAM | Max logit diff | Notes |
|------|--------|-------|--------|-----|---------------|----------------|--------------|------|----------------|-------|
| P0   | 256 MB D2D Copy (L2 flushed) | - | - | - | - | 2.37 ms | - | - | - | RTX 4060 Laptop (8GB, sm_89), Driver 616.64, CUDA 12.4, SM clock 2250 MHz, Mem clock 8001 MHz, 226.3 GB/s median (spread [225.4, 227.0] GB/s, 5 warmup + 25 timed), `python scripts/bench.py` |
| P0   | HF eager FP16 | 1 | 512 | 128 | 8175.3 | 56.71 | 17.6 | 1134.5 MB | - | PyTorch 2.6.0+cu124, transformers 5.15.0, median SM clock 2250 MHz, max temp 63C, `python scripts/bench.py` |
| P0   | py-engine (Triton) | 1 | 512 | 128 | 10984.0 | 46.61 | 21.5 | 1185.6 MB | - | Triton 3.8.0, KernelPagedManualDecode, block_size=16, 256 blocks, median SM clock 900 MHz (idle throttled between steps), max temp 62C, `python scripts/bench.py` |
| P0   | llama.cpp F16 | 1 | 512 | 128 | 24368.2 | 5.22 | 191.5 | ~1.2 GB | - | llama-bench build 3c809f4 CUDA, ngl 99, 5 reps: pp512=24368.16 ± 2701.70 t/s, tg128=191.49 ± 6.04 t/s (5.22 ms/tok), `llama-bench.exe -m qwen2.5-0.5b-instruct-fp16.gguf -p 512 -n 128 -ngl 99 -r 5` |
| P0   | llama.cpp Q8_0 | 1 | 512 | 128 | 26579.0 | 3.09 | 323.8 | ~0.7 GB | - | llama-bench build 3c809f4 CUDA, ngl 99, 5 reps: pp512=26578.98 ± 1569.80 t/s, tg128=323.75 ± 2.45 t/s (3.09 ms/tok), `llama-bench.exe -m qwen2.5-0.5b-instruct-q8_0.gguf -p 512 -n 128 -ngl 99 -r 5` |
| P0   | llama.cpp Q4_K_M | 1 | 512 | 128 | 23100.8 | 2.73 | 366.4 | ~0.6 GB | - | llama-bench build 3c809f4 CUDA, ngl 99, 5 reps: pp512=23100.82 ± 1307.20 t/s, tg128=366.42 ± 0.99 t/s (2.73 ms/tok), `llama-bench.exe -m qwen2.5-0.5b-instruct-q4_k_m.gguf -p 512 -n 128 -ngl 99 -r 5` |
| P3   | Naive GPU, cuBLAS, no cache | 1 | | | | | | | | |
| P4   | KV cache + sampling | 1 | | | | | | | | |
| P5   | Custom GEMV/GEMM + fusion | 1 | | | | | | | | |
| P7   | Split-KV decode + flash prefill | 1, 32 | | | | | | | | |
| P8   | INT8 and INT4 weights | 1 | | | | | | | ppl: | |
| P9   | INT8, FP8, INT4 KV | 32 | | | | | | | ppl: | |
| P10  | Paged KV + batching | 1, 4, 8, 16 | | | | | | | | |
| P11  | CUDA graphs | 1, 8 | | | | | | | | |
| P12  | Megakernel | 1 | | | | | | | | |
