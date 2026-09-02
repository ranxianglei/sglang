#!/usr/bin/env python3
"""Slice p296h into expert-modular format:
backbone-XXXXX-of-XXXXN.safetensors (dense/GDN/attn/embed/PLE, linear pack)
experts-L{0..47}.safetensors (per-layer all MoE expert tensors)
Rewrites index weight_map accordingly. Streaming: one output file assembled at
a time, source shards opened read-only via safetensors safe_open (mmap)."""
import json, os, re, sys, shutil
from safetensors.torch import safe_open, save_file

SRC = "/mnt/8t/models/qwen3.8-flash-next-w4a16-p296h"
DST = "/mnt/8t/models/qwen3.8-flash-next-w4a16-modular"
TARGET_GB = 4.4
expert_re = re.compile(r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.")

def main():
    idx = json.load(open(f"{SRC}/model.safetensors.index.json"))
    wm = idx["weight_map"]
    experts = {}
    backbone = []
    for k, shard in wm.items():
        m = expert_re.match(k)
        if m:
            experts.setdefault(int(m.group(1)), []).append((k, shard))
        else:
            backbone.append((k, shard))
    os.makedirs(DST, exist_ok=True)
    new_wm = {}
    handles = {}
    def get(shard):
        if shard not in handles:
            handles[shard] = safe_open(f"{SRC}/{shard}", framework="pt", device="cpu")
        return handles[shard]

    # BDD: experts dict maps layer_id -> [(tensor_key, source_shard)]
    n_layers = max(experts) + 1
    for L in range(n_layers):
        entries = experts.get(L, [])
        if not entries:
            continue
        fname = f"experts-L{L:02d}.safetensors"
        tensors, meta_set = {}, None
        for k, shard in entries:
            f = get(shard)
            t = f.get_tensor(k)
            tensors[k] = t
            if meta_set is None:
                meta_set = f.metadata()
        save_file(tensors, f"{DST}/{fname}", metadata=meta_set or {"format": "pt"})
        for k, _ in entries:
            new_wm[k] = fname
        del tensors
        print(f"{fname}: {len(entries)} keys", flush=True)

    order = sorted(backbone)
    i, cur, cur_bytes, n_files = 0, {}, 0, 0
    total = TARGET_GB * 1024**3
    def flush():
        nonlocal cur, cur_bytes, n_files
        if not cur:
            return
        n_files += 1
        fname = f"backbone-{n_files:05d}.safetensors"
        save_file(cur, f"{DST}/{fname}", metadata={"format": "pt"})
        for k in cur:
            new_wm[k] = fname
        print(f"{fname}: {len(cur)} keys {cur_bytes/1e9:.2f}GB", flush=True)
        cur, cur_bytes = {}, 0
    while i < len(order):
        k, shard = order[i]
        f = get(shard)
        t = f.get_tensor(k)
        b = t.numel() * t.element_size()
        if cur and cur_bytes + b > total:
            flush()
        cur[k] = t
        cur_bytes += b
        i += 1
    flush()

    idx["weight_map"] = new_wm
    json.dump(idx, open(f"{DST}/model.safetensors.index.json", "w"), indent=1)
    for aux in os.listdir(SRC):
        if aux.endswith(".safetensors"):
            continue
        if aux == "model.safetensors.index.json":
            continue
        s = f"{SRC}/{aux}"
        if os.path.isfile(s):
            shutil.copy(s, f"{DST}/{aux}")
    print("DONE", len(new_wm), "keys")

if __name__ == "__main__":
    main()
