# ranxianglei/sglang — ours/main

Our deployment-tested patches on top of upstream SGLang **PR #36497 branch** (`Introduce Qwen 3.8 Flash Next`, commit `73a255206`). Use this branch if you want to serve **Qwen3.8-Flash-Next W4A16 (Intel AutoRound) pruned models** on a single 96GB GPU.

Base of this branch: `ed9cbb16c` (imported working tree of the PR branch) + two functional commits below.

## Patches in this branch

### 1. W4A16 (GPTQ int4) MoE marlin repack: OOM fix + scales dtype fix
`python/sglang/srt/layers/quantization/gptq_kernels.py` (+22 lines)

- **Problem**: loading Intel AutoRound W4A16 checkpoints on a 96GB GPU hits a deterministic 91.56GB OOM during `gptq_marlin_moe_repack`. Root cause: the loader materializes int4 weights as int32 (8× expansion, w13 37.5G + w2 18.75G), the repacked copies are added on top, and the stale int32 tensors are only released by the allocator after all 48 layers → guaranteed OOM.
- **Fix**: `gc.collect() + torch.cuda.empty_cache()` after each layer's `process_weights_after_loading` (peak drops 69.01→68.52GB per layer, verified).
- **Also**: AutoRound ships fp16 `scales` while hidden states are bf16 — marlin kernels assert on the dtype mismatch; we cast scales to bf16 after repack.

### 2. QSA × mixed-chunk: folded decode rows support
`qwen_sparse_attn_backend.py` / `qsa/metadata.py` / `qsa/qsa_indexer.py` (+104 lines)

- **Problem**: `--enable-mixed-chunk` crashes with a device-side assert as soon as a mixed batch folds decode rows (extend_len=1) into an extend pass. Root cause: `_qsa_build_write_plan` assumed page-aligned prefix lens (`prefix % ratio == 0`); decode rows folded by mixed-chunk carry arbitrary-alignment prefixes (their KV is written token-by-token, not in radix pages of 64).
- **Fix**: row-level hybrid — decode rows get `start_block = end - (len % ratio == 0)`, ring-slot membership -1 sentinel, forced-pending token, and a dual-path store in the indexer (direct vs ring sources selected only when mixed rows exist; zero overhead on the pure paths). Read path already worked (per-row granularity formula).
- Verified: previously-guaranteed crash under 8-way concurrency now stable (535 tok/s), greedy output identical to non-mixed, qbank quality within variance band.

## Serve command (reference)

```bash
python -m sglang.launch_server \
  --model-path <w4a16-pruned-checkpoint> \
  --ple-offload-embedding \
  --moe-a2a-backend none \
  --linear-attn-prefill-backend triton \
  --linear-attn-decode-backend triton \
  --mamba-ssm-dtype bfloat16 \
  --context-length 262144 \
  --mem-fraction-static 0.93 \
  --chunked-prefill-size 8192 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder
```

Hardware floor: 96GB VRAM + ≥64GB free host RAM (PLE is pinned to host). Model: see `ranxianglei/Qwen3.8-Flash-Next-W4A16-Pruned-294E` on Hugging Face (includes `toolkit/make_p294.sh` and the standalone diff `deploy/sglang_w4a16_marlin_gc.diff`).

## Relationship to upstream

- Both patches are clean candidates for upstream PRs once #36497 lands in `main` (they only touch files that #36497 introduces). Until then this branch is the usable artifact.
- License: Apache 2.0 (unchanged, inherited from sgl-project/sglang).
