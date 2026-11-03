# GATES.md: what PASS means (from the project plan, section 1.7)

Tokenizer: 100% identical ids vs HF on 300+ strings (done in P1, plus the Unicode sweep).
CPU FP32 forward: max abs diff vs HF <= 1e-3 on logits and on intermediates (layers 0, 1, 23);
  argmax equal at every position on all reference prompts.
GPU FP16 forward: teacher-forced, every top-1 mismatch vs HF must have an HF FP32 top-1/top-2
  logit margin < 0.05 (the "near-tie rule"). Free-running greedy, 64 tokens: matches HF, or the
  first divergence is at such a near-tie. Also check for inf/NaN in every intermediate (BF16
  weights converted to FP16 can overflow). Accumulate in FP32 in RMSNorm and attention scores.
KV-cache equivalence: at every position, top-1 is identical (or a near-tie: HF FP32 top-1/top-2
  margin < 0.05), AND the cached path's error vs the CPU FP32 reference is <= 1.5x the recompute
  path's error vs the same reference (check both mean and max over positions).
  (Changed after P4 from "cached vs recompute max abs diff <= 1e-3", which is infeasible for two
  FP16 paths whose GEMM and GEMV reduction orders differ. Approved by the human.)
Kernel unit tests: each kernel vs its CPU reference on >= 20 random shapes including sizes that
  are not tile multiples and length-1 sequences; tolerance stated in the test.
Performance claims: 5 warm-up and >= 20 timed iterations, CUDA events, L2 flushed for bandwidth
  tests, clocks/temperature/throttle logged, AC power, median and spread reported; headline
  claims repeated in an interleaved run.
Memory safety: compute-sanitizer memcheck clean on every new kernel's test.
Roofline reference: read bandwidth 228 GB/s; decode ceilings are bytes_read_per_step / 228 GB/s.
