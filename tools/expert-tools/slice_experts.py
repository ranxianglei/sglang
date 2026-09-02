#!/usr/bin/env python3
"""Slice AutoRound W4A16 into a 256-expert pruned variant using the d256v2
keep map (real pi-session router profile, reused as-is: router behavior is a
property of the base model, not of the quantization format).

Key-level surgery only: expert tensors keep their packed layouts untouched,
selected experts are renumbered 0..255, router gate rows are reindexed to
match, config num_experts 512 -> 256.
"""
import json, os, shutil, glob, re
import torch
from safetensors.torch import save_file

SRC = "/mnt/8t/models/qwen3.8-flash-next-w4a16-intel"
KEEP = json.load(open("/tmp/opencode/keep288.json"))
DST = "/mnt/8t/models/qwen3.8-flash-next-w4a16-p288"
SHARD_BYTES = 4_500_000_000

for small in ("chat_template.jinja", "configuration.json", "generation_config.json",
              "merges.txt", "vocab.json", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json", "added_tokens.json"):
    p = os.path.join(SRC, small)
    if os.path.exists(p):
        os.makedirs(DST, exist_ok=True)
        shutil.copy2(p, os.path.join(DST, small))

cfg = json.load(open(os.path.join(SRC, "config.json")))
tc = cfg.get("text_config", cfg)
tc["num_experts"] = 288
json.dump(cfg, open(os.path.join(DST, "config.json"), "w"), indent=1, ensure_ascii=False)

files = sorted(glob.glob(os.path.join(SRC, "model-*.safetensors")))
from safetensors.torch import safe_open

EXPERT_RE = re.compile(r"^(.*)\.mlp\.experts\.(\d+)\.(.+)$")

out_shards = []
cur = {}
cur_bytes = 0
shard_idx = 0

def flush():
    global cur, cur_bytes, shard_idx
    if not cur:
        return
    name = f"model-{shard_idx:05d}-of-XXXXX.safetensors"
    path = os.path.join(DST, name)
    save_file({k: v.contiguous() for k, v in cur.items()}, path, metadata={"format": "pt"})
    out_shards.append((name, {k: v.dtype for k, v in cur.items()}))
    print(f"wrote {name} {cur_bytes/1e9:.2f}GB {len(cur)} tensors", flush=True)
    cur = {}
    cur_bytes = 0
    shard_idx += 1

for fp in files:
    with safe_open(fp, framework="pt", device="cpu") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            m = EXPERT_RE.match(key)
            if m:
                layer_s, exp_s, rest = m.groups()
                keep = KEEP[layer_s.split(".layers.")[1].split(".")[0]]
                e = int(exp_s)
                if e not in keep:
                    continue
                new_id = keep.index(e)
                new_key = f"{layer_s}.mlp.experts.{new_id}.{rest}"
            elif key.endswith(".mlp.gate.weight"):
                layer_s = key[: -len(".mlp.gate.weight")]
                keep = KEEP[layer_s.split(".layers.")[1].split(".")[0]]
                t = t[keep, :]
                new_key = key
            else:
                new_key = key
            nbytes = t.numel() * t.element_size()
            if cur_bytes + nbytes > SHARD_BYTES:
                flush()
            cur[new_key] = t
            cur_bytes += nbytes
flush()

total = len(out_shards)
index = {"metadata": {"total_size": 0}, "weight_map": {}}
for i, (name, dtypes) in enumerate(out_shards):
    real = f"model-{i+1:05d}-of-{total:05d}.safetensors"
    os.rename(os.path.join(DST, name), os.path.join(DST, real))
    for k in dtypes:
        index["weight_map"][k] = real
for fp2 in glob.glob(os.path.join(DST, "model-*-of-*.safetensors")):
    index["metadata"]["total_size"] += os.path.getsize(fp2)
json.dump(index, open(os.path.join(DST, "model.safetensors.index.json"), "w"), indent=1)
print("done:", total, "shards", flush=True)
