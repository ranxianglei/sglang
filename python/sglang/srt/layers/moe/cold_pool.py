# ours: cold-expert pool — keep-set experts stay resident on GPU, cold experts
# are staged in host pinned memory after load and pulled into per-layer LRU
# slots on demand (one-step lookahead prefetch).
#
# Env:
#   SGLANG_EXPERT_COLD_POOL_SLOTS=32  per-layer LRU slot count (requires
#                                     SGLANG_EXPERT_KEEP_MASK to be set)
#
# Phases:
#   load:        full expert pool is loaded & repacked (keep-offload inactive)
#   post-process maybe_shrink_after_process(): cold rows -> pinned host
#                tensors; GPU tensors re-alloc'd to (keep_len + slots) rows,
#                cold slots zero-filled. activate() flips topk gating to
#                dynamic bias/remap tables (CUDA-graph safe, content-only
#                updates from the prefetcher).
#   runtime:     after_forward_hook() inspects last topk global ids per layer,
#                stages needed cold experts into LRU slots (H2D copy on the
#                current stream, outside graph replay), unmasking them for the
#                *next* forward step.

import logging
import os

import torch

logger = logging.getLogger(__name__)

_EXPERT_PARAM_NAMES = (
    "w13_qweight",
    "w2_qweight",
    "w13_scales",
    "w2_scales",
    "w13_qzeros",
    "w2_qzeros",
)

_STATE = {"active": False, "slots": 0, "layers": {}}


def cold_pool_slots() -> int:
    try:
        return int(os.environ.get("SGLANG_EXPERT_COLD_POOL_SLOTS", "0") or 0)
    except ValueError:
        return 0


def cold_pool_enabled() -> bool:
    return cold_pool_slots() > 0 and bool(
        os.environ.get("SGLANG_EXPERT_KEEP_MASK", "")
    )


def is_active() -> bool:
    return _STATE["active"]


class LayerColdPool:
    def __init__(self, layer_id: int, keep, cold, slots: int, module):
        self.layer_id = layer_id
        self.keep = list(keep)
        self.keep_set = set(keep)
        self.cold = list(cold)
        self.slots = slots
        self.module = module
        self.slot_gid = [-1] * slots
        self.slot_tick = [0] * slots
        self.gid_slot = {}
        self.tick = 0
        self.host = {}
        self.cold_gid_gpu = None
        self.cold_mask_gpu = None
        self.need_counts = None
        self.num_experts_total = 0
        self.bias_gpu = None
        self.remap_gpu = None

    def total_host_bytes(self) -> int:
        n = 0
        for d in self.host.values():
            for t in d.values():
                n += t.numel() * t.element_size()
        return n


def _iter_moe_modules(model):
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    for m in model.modules():
        if isinstance(m, FusedMoE):
            yield m


def maybe_shrink_after_process(model) -> None:
    """Loader hook: run after process_weights_after_loading for all modules."""
    if not cold_pool_enabled():
        return
    from sglang.srt.layers.moe.topk import (
        _load_expert_keep_mask,
        activate_cold_pool,
    )

    slots = cold_pool_slots()
    keep_mask = _load_expert_keep_mask() or {}
    from sglang.srt.layers.quantization.utils import replace_parameter

    total_host = 0
    shrunk = 0
    for module in _iter_moe_modules(model):
        layer_id = getattr(module, "layer_id", None)
        if layer_id is None:
            continue
        keep = keep_mask.get(int(layer_id))
        if not keep:
            continue
        ref = None
        for name in _EXPERT_PARAM_NAMES:
            t = getattr(module, name, None)
            if isinstance(t, torch.Tensor) and t.dim() >= 1:
                ref = t
                break
        if ref is None:
            continue
        E = ref.shape[0]
        keep_len = len(keep)
        cold_ids = [g for g in range(E) if g not in set(keep)]
        if not cold_ids:
            continue
        pool = LayerColdPool(layer_id, keep, cold_ids, slots, module)
        keep_idx = torch.tensor(keep, dtype=torch.long, device=ref.device)
        for name in _EXPERT_PARAM_NAMES:
            t = getattr(module, name, None)
            if not isinstance(t, torch.Tensor) or t.dim() < 1 or t.shape[0] != E:
                continue
            cold_rows = t[cold_ids].to("cpu", non_blocking=False).contiguous()
            cold_rows = cold_rows.pin_memory()
            for i, gid in enumerate(cold_ids):
                pool.host.setdefault(gid, {})[name] = cold_rows[i]
            new_shape = (keep_len + slots,) + tuple(t.shape[1:])
            new_t = torch.zeros(new_shape, dtype=t.dtype, device=t.device)
            new_t[:keep_len] = t.index_select(0, keep_idx)
            new_param = torch.nn.Parameter(new_t, requires_grad=False)
            replace_parameter(module, name, new_param)
            del t
        pool.cold_gid_gpu = torch.tensor(
            cold_ids, dtype=torch.long, device=ref.device
        )
        mask = torch.zeros(
            len(pool.keep) + len(pool.cold), dtype=torch.bool, device=ref.device
        )
        mask[pool.cold_gid_gpu] = True
        pool.cold_mask_gpu = mask
        pool.num_experts_total = len(pool.keep) + len(pool.cold)
        _STATE["layers"][int(layer_id)] = pool
        total_host += pool.total_host_bytes()
        shrunk += 1

    _STATE["slots"] = slots
    if shrunk:
        activate_cold_pool(num_experts=E)
        _STATE["active"] = True
        logger.info(
            f"[COLD-POOL] shrunk {shrunk} MoE layers: "
            f"host pinned {total_host / 1e9:.2f} GB, slots/layer={slots}"
        )


def _stage_expert(pool: LayerColdPool, gid: int, quiet: bool = False) -> None:
    """Copy one cold expert's rows from host into a (possibly evicted) LRU
    slot and unmask it in the dynamic gating tables."""
    from sglang.srt.layers.moe.topk import unmask_cold_expert

    slot = pool.gid_slot.get(gid)
    if slot is None:
        slot = min(range(pool.slots), key=lambda s: pool.slot_tick[s])
        old = pool.slot_gid[slot]
        if old != -1:
            mask_cold_expert(pool, old)
            pool.gid_slot.pop(old, None)
        pool.slot_gid[slot] = gid
        pool.gid_slot[gid] = slot
    pool.slot_tick[slot] = pool.tick
    pool.slot_gid[slot] = gid
    keep_len = len(pool.keep)
    rows = pool.host.get(gid)
    if rows is None:
        return
    if not quiet:
        logger.info(
            "[COLD-POOL] stage layer=%d expert=%d slot=%d",
            pool.layer_id,
            gid,
            slot,
        )
    for name, pinned in rows.items():
        gpu = getattr(pool.module, name, None)
        if gpu is None:
            continue
        gpu.data[keep_len + slot].copy_(pinned, non_blocking=False)
    unmask_cold_expert(pool, gid, slot)


def mask_cold_expert(pool: LayerColdPool, gid: int) -> None:
    if pool.bias_gpu is not None:
        pool.bias_gpu[gid] = float("-inf")


def _strong_alpha() -> float:
    try:
        return float(os.environ.get("SGLANG_COLD_STRONG_ALPHA", "0.8"))
    except ValueError:
        return 0.8


def _need_hits() -> int:
    try:
        return int(os.environ.get("SGLANG_COLD_NEED_HITS", "2"))
    except ValueError:
        return 2


def after_forward_hook() -> None:
    """Called after each forward step (outside graph replay). Strong-demand
    filter: a cold expert counts only when its raw sigmoid score reaches
    alpha * (min hot score in the same row). Per-layer GPU counters with EMA
    decay; stage only when counter >= need_hits, rate-limited."""
    if not _STATE["active"]:
        return
    from sglang.srt.layers.moe.topk import (
        consume_cold_dirty,
        get_last_global_ids,
        get_last_scores,
    )

    _stats = _STATE.setdefault(
        "hook_stats", {"calls": 0, "strong": 0, "staged": 0}
    )
    _stats["calls"] += 1
    if _stats["calls"] % 8 != 0 or not consume_cold_dirty():
        return
    alpha = _strong_alpha()
    need = _need_hits()
    staged_now = 0
    for layer_id, pool in _STATE["layers"].items():
        ids = get_last_global_ids(layer_id)
        if ids is None:
            continue
        sc = get_last_scores(layer_id)
        flat = ids.reshape(-1).clone()
        sflat_full = sc.reshape(-1).clone() if sc is not None else None
        n = int((flat >= 0).sum())
        ids.fill_(-1)
        if sc is not None:
            sc.fill_(-1.0)
        if n == 0 or sflat_full is None:
            continue
        flat = flat[: (n // 10) * 10].view(-1, 10)
        sflat = sflat_full[: flat.numel()].view(-1, 10)
        is_cold = pool.cold_mask_gpu[flat]
        hot_min = torch.where(
            ~is_cold, sflat, torch.full_like(sflat, float("inf"))
        ).min(dim=-1, keepdim=True).values
        strong = is_cold & (sflat >= alpha * hot_min)
        if pool.need_counts is None:
            pool.need_counts = torch.zeros(
                pool.num_experts_total, dtype=torch.float32, device=flat.device
            )
        pool.need_counts.mul_(0.75)
        sel = flat[strong]
        if sel.numel() > 0:
            pool.need_counts.index_add_(
                0, sel.reshape(-1), torch.ones(sel.numel(), device=flat.device)
            )
        _stats["strong"] += int(sel.numel())
        cands = pool.cold_gid_gpu[
            pool.need_counts[pool.cold_gid_gpu] >= float(need)
        ]
        if cands.numel() == 0:
            continue
        order = torch.argsort(
            pool.need_counts[cands], descending=True
        )
        cands = cands[order[:4]].tolist()
        pool.tick += 1
        for gid in cands:
            if gid in pool.gid_slot:
                pool.need_counts[gid] = 0.0
                continue
            _stage_expert(pool, gid, quiet=True)
            pool.need_counts[gid] = 0.0
            _stats["staged"] += 1
            staged_now += 1
    if staged_now > 0 and _stats["calls"] % 64 < 8:
        logger.info(
            "[COLD-POOL] strong-filter calls=%d strong_events=%d staged=%d",
            _stats["calls"],
            _stats["strong"],
            _stats["staged"],
        )
