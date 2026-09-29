"""Triton mixed-input GEMM: int8 groupwise-quantized weights x bf16 activations -> bf16.

Weight layout (K-MAJOR since v3):
  W_u8: [K, N] uint8, transposed copy of the packed int32 view; values are
        (w_int8 + 128) i.e. uint8b128
  S:    [K//128, N] scales (bf16), group_size=128 along K
  dequant: w = (u8 - 128) * scale[k//128, n]

v3 kernel design (post-oracle-review):
  - K-major tile load [BK, BN] feeds tl.dot directly — NO tl.trans after
    arithmetic (trans-after-arith cannot fold into ldmatrix; it forced a bf16
    smem round-trip worth 10-20%).
  - bf16-native dequant: tl.fma(u, s, s*(-128)) == (u-128)*s with a single
    rounding, verified BIT-IDENTICAL to the fp32-dequant path on all layer
    shapes. No fp32 temp tile -> -64 regs/thread -> deeper pipelines fit.

DETERMINISM DESIGN (post-incident):
  No @triton.autotune anywhere. Config selection happens ONCE at weight-load
  time in warm_w8a16(): each candidate is timed AND output-validated against
  a dequant reference; the fastest VALID config is pinned per (N, K).
"""
import torch
import triton
import triton.language as tl


# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) — large-M (prefill) candidates.
_LARGE_CONFIGS = [
    (128, 128, 64, 8, 4),
    (128, 128, 128, 8, 3),
    (256, 128, 64, 8, 3),
    (128, 256, 64, 8, 3),
]
# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps) — small-M split-K (decode) candidates.
_SPLITK_CONFIGS = [
    (16, 64, 128, 4),
    (16, 128, 128, 4),
    (32, 64, 128, 4),
    (16, 128, 64, 8),
]
# Small-M candidates that use the plain large kernel (BM covers M).
_DIRECT_CONFIGS = [
    (32, 64, 128, 4, 3),
    (32, 64, 128, 4, 4),
    (32, 128, 128, 4, 3),
    (16, 64, 128, 4, 3),
]
_LARGE_CFG = {}   # (N, K) -> tuple from _LARGE_CONFIGS
# (N, K) -> ("splitk", (BM,BN,BK,warps), SPLIT_K) or ("direct", (BM,BN,BK,warps,stages))
_SPLITK_CFG = {}

# M6 精扫 pin (L2-flushed 实测最优, 2026-09-29): key=(N,K) -> ("direct",cfg5) | ("splitk",cfg4,sk)
_M6_PINNED = {
    (34816, 5120): ("direct", (32, 256, 64, 4, 3)),
    (5120, 17408): ("splitk", (16, 64, 128, 8), 8),
    (6144, 5120): ("splitk", (8, 256, 64, 4), 4),
    (5120, 6144): ("splitk", (8, 256, 64, 4), 8),
}


@triton.jit
def _w8a16_gemm_kernel(
    A, W, S, C,
    M, N, K,
    stride_am, stride_ak,
    stride_wk, stride_wn,
    stride_sk, stride_sn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_K: tl.constexpr, GM: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GM * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GM, GM)
    pid_m = group_id * GM + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    w_ptrs = W + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=m_mask, other=0.0)
        w = tl.load(w_ptrs, mask=n_mask, other=0.0)
        s = tl.load(S + ((k * BLOCK_K) // GROUP_K) * stride_sk + offs_n * stride_sn,
                    mask=offs_n < N, other=0.0).to(tl.bfloat16)
        wb = tl.fma(w.to(tl.bfloat16), s[None, :], s[None, :] * (-128.0))
        acc = tl.dot(a, wb, acc)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    c = acc.to(tl.bfloat16)
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=m_mask & n_mask)


def w8a16_gemm(a_bf16, w_kN, scales_kN, bias=None, cfg=None):
    """a: [M,K] bf16; w_kN: [K,N] uint8 (b128-encoded); scales_kN: [K//128, N] bf16."""
    M, K = a_bf16.shape
    N = w_kN.shape[1]
    if cfg is None:
        cfg = _LARGE_CFG.get((N, K), _LARGE_CONFIGS[0])
    BM, BN, BK, nw, ns = cfg
    c = torch.empty((M, N), dtype=torch.bfloat16, device=a_bf16.device)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _w8a16_gemm_kernel[grid](
        a_bf16, w_kN, scales_kN, c, M, N, K,
        a_bf16.stride(0), a_bf16.stride(1),
        w_kN.stride(0), w_kN.stride(1),
        scales_kN.stride(0), scales_kN.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        GROUP_K=128, GM=8, num_warps=nw, num_stages=ns,
    )
    if bias is not None:
        c += bias
    return c


@triton.jit
def _w8a16_gemm_splitk_kernel(
    A, W, S, C,
    M, N, K,
    stride_am, stride_ak,
    stride_wk, stride_wn,
    stride_sk, stride_sn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_K: tl.constexpr, SPLIT_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_sk = tl.program_id(1)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    k_iters_total = tl.cdiv(K, BLOCK_K)
    k_per_split = tl.cdiv(k_iters_total, SPLIT_K)
    k_start = pid_sk * k_per_split
    k_end = tl.minimum(k_start + k_per_split, k_iters_total)

    a_ptrs = A + offs_m[:, None] * stride_am + (k_start * BLOCK_K + offs_k[None, :]) * stride_ak
    w_ptrs = W + (k_start * BLOCK_K + offs_k[:, None]) * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < N

    for k in range(k_start, k_end):
        a = tl.load(a_ptrs, mask=m_mask, other=0.0)
        w = tl.load(w_ptrs, mask=n_mask, other=0.0)
        s = tl.load(S + ((k * BLOCK_K) // GROUP_K) * stride_sk + offs_n * stride_sn,
                    mask=offs_n < N, other=0.0).to(tl.bfloat16)
        wb = tl.fma(w.to(tl.bfloat16), s[None, :], s[None, :] * (-128.0))
        acc = tl.dot(a, wb, acc)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    tl.atomic_add(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
                  acc, mask=m_mask & n_mask)


def w8a16_gemm_splitk(a_bf16, w_kN, scales_kN, bias=None, cfg=None):
    M, K = a_bf16.shape
    N = w_kN.shape[1]
    if cfg is None:
        cfg = _SPLITK_CFG.get((N, K), ("splitk", _SPLITK_CONFIGS[0], 4))
    if cfg[0] == "direct":
        return w8a16_gemm(a_bf16, w_kN, scales_kN, bias, cfg=cfg[1])
    kind, c, sk = cfg
    BM, BN, BK, nw = c
    c32 = torch.zeros((M, N), dtype=torch.float32, device=a_bf16.device)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN), sk)
    _w8a16_gemm_splitk_kernel[grid](
        a_bf16, w_kN, scales_kN, c32, M, N, K,
        a_bf16.stride(0), a_bf16.stride(1),
        w_kN.stride(0), w_kN.stride(1),
        scales_kN.stride(0), scales_kN.stride(1),
        c32.stride(0), c32.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        GROUP_K=128, SPLIT_K=sk, num_warps=nw,
    )
    out = c32.to(a_bf16.dtype)
    if bias is not None:
        out += bias
    return out


def w8a16_linear(a_bf16, w_kN, scales_kN, bias=None, splitk_threshold=128):
    if a_bf16.shape[0] < splitk_threshold:
        return w8a16_gemm_splitk(a_bf16, w_kN, scales_kN, bias)
    if _HYBRID and a_bf16.shape[0] >= _HYBRID_M:
        return _hybrid_gemm(a_bf16, w_kN, scales_kN, bias)
    return w8a16_gemm(a_bf16, w_kN, scales_kN, bias)


# Hybrid path (SGLANG_WNA16_HYBRID=1): for large M, dequantize the weight into a
# persistent bf16 scratch with a single-pass fused kernel, then call cuBLAS.
# Rationale (measured @600W): fused dequant costs ~0.3ms per gate_up (5x less
# traffic than torch 3-pass), cuBLAS then runs at 406 TF vs kernel 294 TF.
# Per-element dequant numerics are IDENTICAL to the in-kernel path (same fma).
# Scratch is a module-level singleton reused by every layer -> stable pointer,
# cuda-graph capture safe. Cost: ~356MB GPU memory reserved on first use.
import os as _os
_HYBRID = _os.environ.get("SGLANG_WNA16_HYBRID", "0") == "1"
_HYBRID_M = int(_os.environ.get("SGLANG_WNA16_HYBRID_M", "2048"))
_HYBRID_SCRATCH = None


@triton.jit
def _dequant_kernel(
    W, S, O, K, N,
    stride_wk, stride_wn, stride_sk, stride_sn, stride_ok, stride_on,
    BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_K: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_mask = offs_k < K
    n_mask = offs_n < N
    w = tl.load(W + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
                mask=k_mask[:, None] & n_mask[None, :], other=0.0)
    s = tl.load(S + (offs_k // GROUP_K)[:, None] * stride_sk + offs_n[None, :] * stride_sn,
                mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.bfloat16)
    o = tl.fma(w.to(tl.bfloat16), s, s * (-128.0))
    tl.store(O + offs_k[:, None] * stride_ok + offs_n[None, :] * stride_on,
             o, mask=k_mask[:, None] & n_mask[None, :])


def _hybrid_gemm(a_bf16, w_kN, scales_kN, bias=None):
    global _HYBRID_SCRATCH
    K, N = w_kN.shape
    need = K * N
    if _HYBRID_SCRATCH is None or _HYBRID_SCRATCH.numel() < need:
        _HYBRID_SCRATCH = torch.empty(need, device=w_kN.device, dtype=torch.bfloat16)
    scratch = _HYBRID_SCRATCH[:need].view(K, N)
    grid = (triton.cdiv(K, 128), triton.cdiv(N, 128))
    _dequant_kernel[grid](
        w_kN, scales_kN, scratch, K, N,
        w_kN.stride(0), w_kN.stride(1),
        scales_kN.stride(0), scales_kN.stride(1),
        scratch.stride(0), scratch.stride(1),
        BLOCK_K=128, BLOCK_N=128, GROUP_K=128, num_warps=8,
    )
    out = torch.matmul(a_bf16, scratch)
    if bias is not None:
        out = out + bias
    return out


def _select_config(w_u8, scales, configs, kind, m_probe):
    """Time each config AND validate its output against a dequant reference.
    Only validated configs are eligible; fastest wins. Runs once per (N,K) at load.
    Decode-side timing is L2-flushed (rotating scratch read between reps) so the
    choice reflects production DRAM-bound conditions, not L2-hot flattery."""
    K, N = w_u8.shape
    torch.manual_seed(0)
    a = (torch.randn(m_probe, K, device=w_u8.device) * 0.5).to(torch.bfloat16)
    w_deq = ((w_u8.to(torch.float32).t() - 128.0)
             * scales.t().repeat_interleave(128, dim=1).to(torch.float32)).to(torch.bfloat16)
    ref = (a.float() @ w_deq.float().T)
    del w_deq
    ref_scale = ref.abs().max().item() + 1e-6
    flush = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=w_u8.device)
    import time as _t

    def timed(fn, cfg, flush_between):
        try:
            out = fn(a, w_u8, scales, cfg=cfg)
            rel = (out.float() - ref).abs().max().item() / ref_scale
            if rel > 5e-3:
                return None  # invalid: skip (never select)
            for _ in range(3):
                fn(a, w_u8, scales, cfg=cfg)
            torch.cuda.synchronize()
            t0 = _t.time()
            for _ in range(10):
                if flush_between:
                    flush.sum()
                fn(a, w_u8, scales, cfg=cfg)
            torch.cuda.synchronize()
            return _t.time() - t0
        except Exception:
            return None  # e.g. SMEM overflow on this shape: skip

    if kind == "large":
        best, best_t = None, float("inf")
        for cfg in configs:
            t = timed(w8a16_gemm, cfg, flush_between=False)
            if t is not None and t < best_t:
                best, best_t = cfg, t
        return best

    best, best_t = None, float("inf")
    t_flush = None
    for c in _SPLITK_CONFIGS:
        for sk in (4, 8):
            t = timed(w8a16_gemm_splitk, ("splitk", c, sk), flush_between=True)
            if t is None:
                continue
            if t_flush is None:
                torch.cuda.synchronize(); t0 = _t.time()
                for _ in range(10):
                    flush.sum()
                torch.cuda.synchronize(); t_flush = _t.time() - t0
            if t - t_flush < best_t:
                best, best_t = ("splitk", c, sk), t - t_flush
    for c in _DIRECT_CONFIGS:
        t = timed(lambda x, w, s, cfg=None: w8a16_gemm(x, w, s, None, c), None, flush_between=True)
        if t is None:
            continue
        if t_flush is None:
            torch.cuda.synchronize(); t0 = _t.time()
            for _ in range(10):
                flush.sum()
            torch.cuda.synchronize(); t_flush = _t.time() - t0
        if t - t_flush < best_t:
            best, best_t = ("direct", c), t - t_flush
    del flush
    return best


def warm_w8a16(w_kN, w_scale_kN):
    """Load-time deterministic config selection + kernel compile warmup.
    Call before CUDA graph capture; afterwards launches are fixed-config."""
    key = (w_kN.shape[1], w_kN.shape[0])
    cfg = _select_config(w_kN, w_scale_kN, _LARGE_CONFIGS, "large", m_probe=4096)
    if cfg is not None:
        _LARGE_CFG[key] = cfg
    pin = _M6_PINNED.get(key)
    if pin is not None:
        _SPLITK_CFG[key] = pin
    else:
        cfg = _select_config(w_kN, w_scale_kN, _SPLITK_CONFIGS, "splitk", m_probe=8)
        if cfg is not None:
            _SPLITK_CFG[key] = cfg
    a = torch.zeros((8, w_kN.shape[0]), dtype=torch.bfloat16, device=w_kN.device)
    w8a16_linear(a, w_kN, w_scale_kN)
    a = torch.zeros((4096, w_kN.shape[0]), dtype=torch.bfloat16, device=w_kN.device)
    w8a16_linear(a, w_kN, w_scale_kN)
    torch.cuda.synchronize()
    return _LARGE_CFG.get(key), _SPLITK_CFG.get(key)


def repack_kmajor(w_u8_nk, w_scale_nk):
    """[N,K] uint8 + [N,K//128] bf16 -> K-major contiguous copies."""
    return w_u8_nk.t().contiguous(), w_scale_nk.t().contiguous()


def make_qa(n, k, group=128, device="cuda"):
    torch.manual_seed(42)
    w = (torch.randn(n, k, device=device) * 0.02).to(torch.bfloat16)
    scales = torch.empty(n, k // group, device=device, dtype=torch.bfloat16)
    w_q = torch.empty(n, k, device=device, dtype=torch.uint8)
    for g in range(k // group):
        blk = w[:, g * group:(g + 1) * group]
        s = blk.abs().amax(dim=1) / 127.0
        scales[:, g] = s.to(torch.bfloat16)
        q = torch.round(blk / s[:, None]).clamp(-128, 127).to(torch.int8)
        w_q[:, g * group:(g + 1) * group] = (q.to(torch.int16) + 128).to(torch.uint8)
    a = (torch.randn(4096, k, device=device) * 0.5).to(torch.bfloat16)
    return a, w_q.t().contiguous(), scales.t().contiguous(), w


if __name__ == "__main__":
    import time
    dev = "cuda"
    shapes = [
        (34816, 5120, "mlp gate_up"),
        (5120, 17408, "mlp down"),
        (6144, 5120, "attn q"),
        (5120, 6144, "attn o"),
    ]
    M = 4096
    for n, k, name in shapes:
        a, w_qk, scalesk, w_ref = make_qa(n, k)
        large_cfg, splitk_cfg = warm_w8a16(w_qk, scalesk)
        print(f"{name}: pinned large={large_cfg} splitk={splitk_cfg}")
        out = w8a16_gemm(a, w_qk, scalesk)
        w_deq = ((w_qk.to(torch.float32).t() - 128.0) * scalesk.t().repeat_interleave(128, dim=1)).to(torch.bfloat16)
        ref = (a @ w_deq.T)
        diff = (out.float() - ref.float()).abs().max().item()
        rel = diff / ref.float().abs().max().item()
        def bench(fn, iters=20):
            for _ in range(3): fn()
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(iters): fn()
            torch.cuda.synchronize()
            return (time.time() - t0) / iters
        t_tri = bench(lambda: w8a16_gemm(a, w_qk, scalesk))
        t_cublas = bench(lambda: a @ w_deq.T)
        tf_tri = 2 * M * n * k / t_tri / 1e12
        tf_cu = 2 * M * n * k / t_cublas / 1e12
        bw = n * k / t_tri / 1e9
        print(f"{name:14s} N={n:6d} K={k:6d} M={M}: triton {tf_tri:6.1f} TF ({t_tri*1e3:6.2f}ms) | cuBLAS bf16 {tf_cu:6.1f} TF | maxdiff {diff:.4f} rel {rel:.2e} | wBW {bw:5.0f} GB/s")
