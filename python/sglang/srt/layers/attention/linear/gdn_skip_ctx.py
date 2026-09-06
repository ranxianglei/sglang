"""Aggressive GDN stride-N skip: whole linear-attention block (in_proj GEMV +
conv + recurrent kernel + out_proj) is skipped on every Nth decode step; x
passes through the layer untouched.

Two execution regimes:
  * CUDA-graph capture: `set_capture_gdn_skip(True/False)` is flipped by the
    graph runner per capture, so each variant records a different kernel
    sequence ("normal" vs "gdnskip" graphs). Python branches only matter at
    capture time — replays replay the recorded sequence.
  * Eager decode: `gdn_step_begin()` is called once per decode forward from
    the model runner; the per-layer gate reads the cached decision.

Enabled via SGLANG_GDN_SKIP_EVERY (>=2) on the server environment.
"""

from __future__ import annotations

import os

import torch

_CAPTURE_SKIP = False
_STEP_SKIP = False
_EVERY = int(os.environ.get("SGLANG_GDN_SKIP_EVERY", "0") or 0)
_COUNTER = 0
_SKIP_LAYERS = frozenset(
    int(x) for x in os.environ.get("SGLANG_GDN_SKIP_LAYERS", "").split(",") if x.strip()
)


def gdn_stride_enabled() -> bool:
    return _EVERY >= 2


def gdn_stride_every() -> int:
    return _EVERY


def set_capture_gdn_skip(value: bool) -> None:
    global _CAPTURE_SKIP
    _CAPTURE_SKIP = bool(value)


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
        return _CAPTURE_SKIP
    if _SKIP_LAYERS and layer_idx not in _SKIP_LAYERS:
        return False
    return _STEP_SKIP
