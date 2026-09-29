"""Aggressive GDN stride-N skip: whole linear-attention block (in_proj GEMV +
conv + recurrent kernel + out_proj) is skipped on every Nth decode step; x
passes through the layer untouched.

Two execution regimes:
  * CUDA-graph capture: `set_capture_gdn_skip(True/False, phase)` is flipped
    by the graph runner per capture, so each variant records a different
    kernel sequence ("normal" vs "gdnskip" graphs). Python branches only
    matter at capture time — replays replay the recorded sequence.
  * Eager decode: `gdn_step_begin()` is called once per decode forward from
    the model runner; the per-layer gate reads the cached decision.

Modes (SGLANG_GDN_SKIP_EVERY >= 2 enables):
  * default: every layer skips together on every Nth step (oscillating).
  * SGLANG_GDN_STAGGER=1: alternating-layer stagger — on each step exactly
    half of the GDN layers skip, odd/even parity rotating per step, so every
    token keeps a constant 50% layer coverage (smooth instead of
    all-or-nothing). Requires EVERY == 2.
"""

from __future__ import annotations

import os

import torch

_CAPTURE_SKIP = False
_CAPTURE_PHASE = 0
_STEP_SKIP = False
_EVERY = int(os.environ.get("SGLANG_GDN_SKIP_EVERY", "0") or 0)
_COUNTER = 0
_STAGGER = os.environ.get("SGLANG_GDN_STAGGER", "0") == "1" and _EVERY == 2
_SKIP_LAYERS = frozenset(
    int(x) for x in os.environ.get("SGLANG_GDN_SKIP_LAYERS", "").split(",") if x.strip()
)


def gdn_stride_enabled() -> bool:
    return _EVERY >= 2


def gdn_stride_every() -> int:
    return _EVERY


def gdn_stride_stagger() -> bool:
    return _STAGGER


def set_capture_gdn_skip(value: bool, phase: int = 0) -> None:
    global _CAPTURE_SKIP, _CAPTURE_PHASE
    _CAPTURE_SKIP = bool(value)
    _CAPTURE_PHASE = int(phase) & 1


def get_capture_gdn_skip() -> bool:
    return _CAPTURE_SKIP


def gdn_step_begin() -> bool:
    """Called once per decode forward (eager path). Returns True on skip steps."""
    global _COUNTER, _STEP_SKIP
    if _EVERY < 2 and _REUSE_EVERY < 2:
        _STEP_SKIP = False
        return False
    _COUNTER += 1
    _STEP_SKIP = _EVERY >= 2 and _COUNTER % _EVERY == 0
    return _STEP_SKIP


def should_skip_gdn(layer_idx: int = -1) -> bool:
    if torch.cuda.is_current_stream_capturing():
        if not _CAPTURE_SKIP:
            return False
        if _STAGGER:
            return (layer_idx + _CAPTURE_PHASE) % 2 == 0
        return True
    if _EVERY < 2:
        return False
    if _STAGGER:
        return (layer_idx + _COUNTER) % 2 == 0
    if _SKIP_LAYERS and layer_idx not in _SKIP_LAYERS:
        return False
    return _STEP_SKIP


# lazy-reuse stride (v5): on every Nth decode step the in_proj_qkvz GEMV is
# skipped and the layer reuses the previous step's cached projection (q/k/v/z
# one step stale); in_proj_ba is computed fresh so beta/g (the time signal)
# stay current. conv update + recurrent kernel + out_proj run normally, so
# the state keeps receiving (delayed) write increments.
_REUSE_EVERY = int(os.environ.get("SGLANG_GDN_REUSE_EVERY", "0") or 0)
_QKVZ_CACHE: "dict[int, torch.Tensor]" = {}


def gdn_reuse_enabled() -> bool:
    return _REUSE_EVERY >= 2


def gdn_reuse_every() -> int:
    return _REUSE_EVERY


def gdn_reuse_is_skip() -> bool:
    """True on steps that should reuse the cached qkvz instead of in_proj."""
    if _REUSE_EVERY < 2:
        return False
    if torch.cuda.is_current_stream_capturing():
        return _CAPTURE_SKIP
    return _COUNTER % _REUSE_EVERY == 0


def gdn_qkvz_cache(
    layer_id: int, num_slots: int, dim: int, device, dtype=torch.bfloat16
) -> torch.Tensor:
    """Persistent per-slot cache of the in_proj_qkvz output. Lazily built."""
    t = _QKVZ_CACHE.get(layer_id)
    if t is None:
        t = torch.zeros(num_slots, dim, dtype=dtype, device=device)
        _QKVZ_CACHE[layer_id] = t
    return t


# decay-only stride (time-mock): on every Nth decode step the SSM kernel
# applies only the time decay (S *= exp(g)) and skips the write; readout uses
# the decayed state. Graph-safe: persistent device tensor flipped by
# gdn_decay_tick() outside captured regions.

_DECAY_EVERY = int(os.environ.get("SGLANG_GDN_DECAY_EVERY", "0") or 0)
_DECAY_COUNTER = 0
_DECAY_FLAG: "torch.Tensor | None" = None


def gdn_decay_enabled() -> bool:
    return _DECAY_EVERY >= 2


def gdn_decay_tick(device) -> None:
    """One call per decode forward (graph-external python) flips the flag."""
    global _DECAY_COUNTER, _DECAY_FLAG
    if _DECAY_EVERY < 2:
        return
    flag = gdn_decay_flag(device)
    _DECAY_COUNTER += 1
    val = 1 if _DECAY_COUNTER % _DECAY_EVERY == 0 else 0
    flag.fill_(val)
    if _DECAY_COUNTER <= 40:
        print(f"[gdn-decay-debug] tick={_DECAY_COUNTER} flag={val}", flush=True)


def gdn_decay_flag(device) -> "torch.Tensor":
    global _DECAY_FLAG
    if _DECAY_FLAG is None:
        _DECAY_FLAG = torch.zeros(1, dtype=torch.int32, device=device)
    return _DECAY_FLAG


_GSCALE = float(os.environ.get("SGLANG_GDN_DECAY_GSCALE", "1.0") or 1.0)
_BETA_SCALE = float(os.environ.get("SGLANG_GDN_DECAY_BETA", "1.0") or 1.0)
_GSCALE_T: "torch.Tensor | None" = None
_BETA_SCALE_T: "torch.Tensor | None" = None


def gdn_gscale(device) -> "torch.Tensor":
    global _GSCALE_T
    if _GSCALE_T is None:
        _GSCALE_T = torch.full((1,), _GSCALE, dtype=torch.float32, device=device)
    return _GSCALE_T


def gdn_beta_scale(device) -> "torch.Tensor":
    global _BETA_SCALE_T
    if _BETA_SCALE_T is None:
        _BETA_SCALE_T = torch.full((1,), _BETA_SCALE, dtype=torch.float32, device=device)
    return _BETA_SCALE_T


# ---- activation covariance collector (ASVD probe; env SGLANG_GDN_COV=1) ----
_COV_ON = bool(os.environ.get("SGLANG_GDN_COV", ""))
_COV = {}
_COV_N = {}

def gdn_cov_accum(layer_id: int, x) -> None:
    # x: [..., hidden] on GPU. Accumulate x^T x per layer for spectrum analysis.
    if not _COV_ON:
        return
    import torch
    xf = x.detach().reshape(-1, x.shape[-1]).float()
    if xf.numel() == 0:
        return
    c = _COV.get(layer_id)
    if c is None or c.device != xf.device:
        c = torch.zeros(xf.shape[-1], xf.shape[-1], dtype=torch.float32, device=xf.device)
        _COV[layer_id] = c
        _COV_N[layer_id] = 0
    _COV[layer_id] += xf.T @ xf
    _COV_N[layer_id] += xf.shape[0]
    if _COV_N[layer_id] >= 4096 and _COV_N[layer_id] % 4096 < xf.shape[0]:
        gdn_cov_dump()

def gdn_cov_dump() -> str:
    if not _COV_ON:
        return ""
    import torch
    out = os.environ.get("SGLANG_GDN_COV_OUT", "/tmp/opencode/gdn_cov.pt")
    torch.save({int(k): (v.cpu(), _COV_N.get(k, 0)) for k, v in _COV.items()}, out)
    return out
