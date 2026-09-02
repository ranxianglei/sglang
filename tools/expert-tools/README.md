# Expert Tools — inference-time MoE expert pruning pipeline

Produce a per-layer keep set from your own traffic, serve only those experts,
and re-slice checkpoints so future expert edits touch one small file.

## Pipeline

1. **Profile** — run your real traffic against a server started with
   `--expert-distribution-recorder-mode stat`, then dump:
   `POST /start_expert_distribution_record` → traffic → `POST /dump_expert_distribution_record`
   (dump lands in `/tmp/expert_distribution_recorder_<ts>.pt`).
   - `feed_prompts.py` / `profile_router.py` — helpers to replay sessions/prompts
2. **Keep set** — `make_keep.py --dump <capability.pt> [--daily-dump <daily.pt> --top 294 --heal-top 2]`
   - capability dump = sessions you must keep serving well (anomalous/degraded contexts)
   - daily dump = routine traffic; top-N per layer
   - output `keep.json`: `{"0": [gid...], ...}`
3. **Serve** — two modes (both in this fork):
   - keep-only offload: `SGLANG_EXPERT_KEEP_MASK=keep.json SGLANG_EXPERT_KEEP_OFFLOAD=1`
     (cold experts stay on host RAM; GPU pool shrinks to the keep set; KV pool grows)
   - mask-only: `SGLANG_EXPERT_KEEP_MASK=keep.json`
     (all experts loaded; router logits of non-kept experts biased to -inf)
4. **Re-slice** (optional, recommended):
   - `slice_modular.py` — per-layer expert files (`experts-L{N}.safetensors`), backbone untouched;
     edit one layer later = rewrite one ~180MB file
   - `slice_experts.py` — classic linear shards with num_experts rewritten

## Verified numbers (Qwen3.8-Flash-Next W4A16, 512→296 experts, 1×96GB GPU)

- GPU weights 58→44.9GB (-13GB), KV pool +31%, single-stream 100.6 tok/s (parity),
  quality gate green, anomalous-context self-heal 5/5 (vs 0/3 without heal-top union)
