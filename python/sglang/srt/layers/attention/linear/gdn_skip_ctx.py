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
    if _EVERY < 2:
        _STEP_SKIP = False
        return False
    _COUNTER += 1
    _STEP_SKIP = _COUNTER % _EVERY == 0
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
    flag.fill_(1 if _DECAY_COUNTER % _DECAY_EVERY == 0 else 0)


def gdn_decay_flag(device) -> "torch.Tensor":
    global _DECAY_FLAG
    if _DECAY_FLAG is None:
        _DECAY_FLAG = torch.zeros(1, dtype=torch.int32, device=device)
    return _DECAY_FLAG
