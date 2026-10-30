# Model Runner Interface Specification (docs/runner_interface.md)

This document formalizes the exact contract and interface between the `py-engine` scheduler (`engine/scheduler/continuous_batching.py`, `engine/serving/async_engine.py`) and the model runner (`engine/kernels/paged_attention_integration.py` / future `py/cpp_runner.py`).

## 1. Architectural Roles and Ownership

| Entity | Role | Ownership & Boundaries |
|---|---|---|
| **Scheduler** (`Scheduler`) | Drives iteration scheduling | **SOLE owner** of block allocation/deallocation. Only `Scheduler.step()` calls `allocator.allocate()`. `Scheduler.post_step()` is the **SOLE writer** of `generated_ids`, `last_token_id`, and `total_generated_tokens`. |
| **Model Runner** (`KernelPagedManualDecode` / C++ Runner) | Executes forward passes | Owns GPU physical memory tensors (`PhysicalBlockPool`) and kernel execution. **NEVER** calls `allocator.allocate()`. **NEVER** frees blocks. |
| **Physical Block Pool** (`PhysicalBlockPool`) | GPU memory storage | Holds K and V cache tensors for all layers. Shape per layer: `[num_blocks, block_size, num_kv_heads, head_dim]`. |
| **Block Allocator** (`BlockAllocator`) | Logical block management | Manages free lists, block reference counts, prefix caching LRU pool, and mapping from logical block index to physical block ID (`BlockTable`). |

---

## 2. Model Runner Contract

A model runner must support two execution methods: `prefill(seq, allocator=None)` and `decode_step(seqs, allocator=None)`.
Alternatively, or in addition, runners may expose `step(batch: ScheduledBatch) -> List[int]`.

### 2.1 `prefill(seq: Sequence, allocator: Optional[BlockAllocator] = None) -> int`

- **Purpose**: Runs the full prompt through the model, copies prompt KV states into the physical pool for blocks owned by `seq`, and returns the first generated token ID.
- **Preconditions**:
  - `seq.block_table.physical_block_ids` has already been allocated by `Scheduler.step()`.
  - `seq.block_table.num_tokens` is already set to `prompt_len`.
  - If prefix caching is active, `seq.num_cached_tokens` indicates the number of prompt tokens already cached in earlier blocks.
- **Contract & Invariants**:
  - The runner **MUST NOT** call `allocator.allocate()`.
  - The runner **MUST NOT** modify `seq.block_table.num_tokens` or `seq.cur_len`.
  - The runner writes KV tensors into `pool.k_cache` and `pool.v_cache` at indices specified in `seq.block_table.physical_block_ids[skip_blocks:]` where `skip_blocks = num_cached_tokens // block_size`.
  - In prefix mode `"full"`, the runner seeds generation with cached KV blocks and only forwards the uncached prompt suffix. In `"share"` mode, it recomputes the full prompt but skips overwriting shared blocks.
- **Returns**: `first_token_id: int` (argmax of the final prefill prompt position logits).
- **Diagnostics**:
  - The runner optionally sets `seq.last_logits` (tensor on CPU) and `seq.last_logit_margin` (float difference between top-1 and top-2 logits).

### 2.2 `decode_step(seqs: List[Sequence], allocator: Optional[BlockAllocator] = None) -> List[int]`

- **Purpose**: Executes a single autoregressive decoding step for a batch of $N$ sequences ($N \ge 1$).
- **Preconditions**:
  - For each `seq` in `seqs`:
    - `seq.block_table` is populated.
    - If `seq.block_table.num_tokens % block_size == 0`, `Scheduler.step()` has already allocated the next physical block.
    - `seq.last_token_id` contains the token generated at the previous step (or `seq.generated_ids[-1]`).
- **Contract & Invariants**:
  - The runner **bumps** `seq.block_table.num_tokens += 1` immediately before invoking the kernel.
    - *Rationale*: This communicates to the kernel that the new KV entry must be written at position `write_pos = old_num_tokens = new_num_tokens - 1`.
  - The runner **MUST NOT** call `allocator.allocate()`.
  - The runner verifies `write_block_id = block_table.physical_block_ids[num_tokens // block_size]`. If `allocator.ref_counts[write_block_id] > 1`, an assertion fails (a sequence must never mutate a shared block during decode).
  - The runner runs under `torch.inference_mode()`.
- **Returns**: `next_tokens: List[int]` of length $N$, where `next_tokens[i]` is the argmax token ID for `seqs[i]`.
- **Diagnostics**:
  - Updates `seq.last_logits` and `seq.last_logit_margin` for each sequence.

### 2.3 `step(batch: ScheduledBatch) -> List[int]`

- Unified dispatch method:
  - If `batch.batch_type == BatchType.PREFILL`: calls `prefill(batch.sequences[0])`, returns `[first_token]`.
  - If `batch.batch_type == BatchType.DECODE`: calls `decode_step(batch.sequences)`, returns `List[int]`.

---

## 3. Data Structures, Tensor Shapes, and Dtypes

### 3.1 Model Constants for Qwen2.5-0.5B-Instruct
- `num_layers`: 24
- `hidden_size`: 896
- `num_q_heads`: 14
- `num_kv_heads`: 2 (Grouped Query Attention ratio = 7)
- `head_dim`: 64
- `vocab_size`: 151,936
- `block_size`: 16 (default)
- `dtype`: `torch.float16` (`fp16`)

### 3.2 Physical Block Pool (`PhysicalBlockPool`)
- `k_cache`: List of 24 tensors, each `[num_blocks, block_size, num_kv_heads, head_dim]`, `torch.float16` on CUDA.
- `v_cache`: List of 24 tensors, each `[num_blocks, block_size, num_kv_heads, head_dim]`, `torch.float16` on CUDA.
- Memory size per block across 24 layers:
  $$24 \times 2 \times 16 \times 2 \times 64 \times 2\text{ bytes} = 98,304\text{ bytes} = 96\text{ KiB per block}.$$

### 3.3 Block Table Tensor & Padded Block Tables
- In decode steps:
  - `block_tables`: Tensor of shape `[batch_size, max_num_blocks_in_batch]`, dtype `torch.int32`. Unused slots padded with 0.
  - `seq_lens`: Tensor of shape `[batch_size]`, dtype `torch.int32`, containing the updated sequence length (including the token being decoded).

---

## 4. Open Questions & Uncertainties (Marked UNSURE)
- `UNSURE`: None. The contract is fully and consistently defined in `reference/py-engine`:
  - `Scheduler.step()` handles all block allocations and watermark preemption.
  - `Scheduler.post_step()` appends generated tokens and mirrors sequence length.
  - `runner.decode_step()` is responsible solely for bumping `block_table.num_tokens` by 1 and dispatching the forward pass.

