# AGENTS.md: standing instructions

## Mission
Build a C++17/CUDA inference engine for Qwen2.5-0.5B-Instruct on an RTX 4060 Laptop
(8 GB, sm_89): KV cache, sampling, custom kernels, INT8/INT4 weights, low-bit KV cache,
paged KV with continuous batching, CUDA graphs, and a batch-1 megakernel experiment.
The goal is correct, measured, honestly reported results. Negative results are results.

## Ground truth
- Correctness oracle: Hugging Face transformers, FP32, CPU (scripts/export_reference.py).
- Previous engine: reference/py-engine (git submodule, tag v1.0 / main). READ-ONLY. Never edit it.
  You may run its benchmarks and read its code.
- Baselines: HF eager, py-engine, llama.cpp. Numbers live in RESULTS.md.
- config.json of the model is the source of truth for dimensions.
- Environment details and exact build/test commands: docs/ENV.md (created by task P0).

## Working rules
1. Plan first. At the start of every task, write the plan, file list, tests and acceptance
   commands into docs/reports/<TASK>.md before writing code.
2. Tests before kernels. Every kernel gets a CPU-reference test over random shapes, including
   shapes that are not multiples of tile sizes, before it is used by the model.
3. Never loosen a tolerance, threshold or test to make something pass. If a threshold looks
   wrong, stop and report evidence.
4. Never write a number into RESULTS.md or a report that did not come from a command you ran.
   Paste the command and its output into the report.
5. Performance methodology: 5 warm-up + at least 20 timed iterations, CUDA events, L2 flushed
   for bandwidth tests, NVML clocks/temperature logged, median and spread reported.
   Change one thing at a time and record before/after.
6. Check every CUDA and cuBLAS call. Run compute-sanitizer (memcheck) on each new kernel test.
7. Small commits, one logical change each, message = what + which test proves it.
8. Dependencies allowed: CUDA toolkit libs (cuBLAS), pybind11, nlohmann/json, doctest or
   GoogleTest. Anything else: ask first.
9. Never delete a result. A wrong number gets a new row and the old one is marked Retracted
   with the reason.
10. Stay in scope. Do not start the next task, refactor unrelated code, or change public
    interfaces (runner interface, weight format, file layout) without asking.
11. Windows note: a single kernel launch running longer than about 2 seconds can be killed by
    the display driver (TDR). Keep launches short.

## Report format (final section of docs/reports/<TASK>.md)
- Gate verdict: PASS | FAIL | NEGATIVE RESULT, with one sentence why.
- What was built (files, interfaces).
- Acceptance commands run, with pasted output.
- Numbers: table of before/after, with GPU clocks and the exact commands.
- Deviations from the plan and why.
- Open risks and suspected causes of anything unexplained.
- Suggested next steps (do not start them).

## Stop and ask the human when
- A gate fails after two serious fix attempts.
- A change would alter a public interface or the weight/cache format.
- A needed measurement is impossible here (profiler permissions, driver, missing tool).
- The task is on track to exceed about 12 hours of agent work.
- You are about to claim a performance win you could not reproduce twice.
