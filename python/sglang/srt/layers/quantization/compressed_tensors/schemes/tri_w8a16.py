"""Triton mixed-input GEMM: int8 groupwise-quantized weights x bf16 activations -> bf16.

Weight layout (compressed-tensors wNa16, little-endian int32 packing):
  W_u8: [N, K] uint8 view of packed int32 [N, K/4]; values are (w_int8 + 128) i.e. uint8b128
  S:    [N, K//128] scales (bf16)
  dequant: w = (u8 - 128) * scale[n, k//128]

Kernel keeps weights as int8 in HBM; dequantizes in registers; tl.dot on bf16.
BLOCK_K <= 128 == group_size -> at most one scale per k-iteration boundary (BK=128: exact).

DETERMINISM DESIGN (post-incident):
  No @triton.autotune anywhere. Config selection happens ONCE at weight-load time in
  warm_w8a16(): each candidate config is timed AND output-validated against a dequant
  reference; the fastest VALID config is pinned per (N, K). Launchers compute the grid
  from the pinned config's own tile sizes, so every config is structurally correct for
  any (M, N) — no reliance on over-coverage luck.
"""
import torch
import triton
import triton.language as tl


# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages) — large-M (prefill) candidates.
# All are structurally safe because the launcher derives the grid from BM/BN.
_LARGE_CONFIGS = [
    (128, 128, 64, 8, 4),
    (128, 256, 64, 8, 3),
    (256, 128, 64, 8, 2),
    (64, 256, 128, 8, 2),
]
# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps) — small-M split-K (decode) candidates.
_SPLITK_CONFIGS = [
    (16, 64, 128, 4),
    (16, 128, 128, 4),
    (32, 64, 128, 4),
    (16, 128, 64, 8),
]
# Small-M candidates that use the plain large kernel (BM covers M, grid = cdiv(N,BN)):
# for bandwidth-bound decode these avoid split-K fp32 atomic contention on some shapes.
_DIRECT_CONFIGS = [
    (32, 64, 128, 4, 3),
    (32, 64, 128, 4, 4),
    (32, 128, 128, 4, 3),
    (16, 64, 128, 4, 3),
]
_LARGE_CFG = {}   # (N, K) -> tuple from _LARGE_CONFIGS
# (N, K) -> ("splitk", (BM,BN,BK,warps), SPLIT_K) or ("direct", (BM,BN,BK,warps,stages))
_SPLITK_CFG = {}


@triton.jit
def _w8a16_gemm_kernel(
    A, W, S, C,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_sm,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    w_ptrs = W + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    m_mask = offs_m[:, None] < M
    n_mask = offs_n[:, None] < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=m_mask, other=0.0)
        w = tl.load(w_ptrs, mask=n_mask, other=0.0)
        s = tl.load(S + offs_n * stride_sm + (k * BLOCK_K) // GROUP_K,
                    mask=offs_n < N, other=0.0)
        wf = (w.to(tl.float32) - 128.0) * s.to(tl.float32)[:, None]
        acc = tl.dot(a, tl.trans(wf.to(tl.bfloat16)), acc)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    c = acc.to(tl.bfloat16)
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def w8a16_gemm(a_bf16, w_u8, scales, bias=None, cfg=None):
    """a: [M,K] bf16; w_u8: [N,K] uint8 (b128-encoded); scales: [N, K//128] bf16."""
    M, K = a_bf16.shape
    N = w_u8.shape[0]
    if cfg is None:
        cfg = _LARGE_CFG.get((N, K), _LARGE_CONFIGS[0])
    BM, BN, BK, nw, ns = cfg
    c = torch.empty((M, N), dtype=torch.bfloat16, device=a_bf16.device)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _w8a16_gemm_kernel[grid](
        a_bf16, w_u8, scales, c, M, N, K,
        a_bf16.stride(0), a_bf16.stride(1),
        w_u8.stride(0), w_u8.stride(1),
        scales.stride(0),
        c.stride(0), c.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        GROUP_K=128, num_warps=nw, num_stages=ns,
    )
    if bias is not None:
        c += bias
    return c


@triton.jit
def _w8a16_gemm_splitk_kernel(
    A, W, S, C,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_sm,
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
    w_ptrs = W + offs_n[:, None] * stride_wn + (k_start * BLOCK_K + offs_k[None, :]) * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    m_mask = offs_m[:, None] < M
    n_mask = offs_n[:, None] < N

    for k in range(k_start, k_end):
        a = tl.load(a_ptrs, mask=m_mask, other=0.0)
        w = tl.load(w_ptrs, mask=n_mask, other=0.0)
        s = tl.load(S + offs_n * stride_sm + (k * BLOCK_K) // GROUP_K,
                    mask=offs_n < N, other=0.0)
        wf = (w.to(tl.float32) - 128.0) * s.to(tl.float32)[:, None]
        acc = tl.dot(a, tl.trans(wf.to(tl.bfloat16)), acc)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    tl.atomic_add(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
                  acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def w8a16_gemm_splitk(a_bf16, w_u8, scales, bias=None, cfg=None):
    M, K = a_bf16.shape
    N = w_u8.shape[0]
    if cfg is None:
        cfg = _SPLITK_CFG.get((N, K), ("splitk", _SPLITK_CONFIGS[0], 4))
    if cfg[0] == "direct":
        return w8a16_gemm(a_bf16, w_u8, scales, bias, cfg=cfg[1])
    kind, c, sk = cfg
    BM, BN, BK, nw = c
    c32 = torch.zeros((M, N), dtype=torch.float32, device=a_bf16.device)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN), sk)
    _w8a16_gemm_splitk_kernel[grid](
        a_bf16, w_u8, scales, c32, M, N, K,
        a_bf16.stride(0), a_bf16.stride(1),
        w_u8.stride(0), w_u8.stride(1),
        scales.stride(0),
        c32.stride(0), c32.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        GROUP_K=128, SPLIT_K=sk, num_warps=nw,
    )
    out = c32.to(a_bf16.dtype)
    if bias is not None:
        out += bias
    return out


def w8a16_linear(a_bf16, w_u8, scales, bias=None, splitk_threshold=128):
    if a_bf16.shape[0] < splitk_threshold:
        return w8a16_gemm_splitk(a_bf16, w_u8, scales, bias)
    return w8a16_gemm(a_bf16, w_u8, scales, bias)


def _select_config(w_u8, scales, configs, kind, m_probe):
    """Time each config AND validate its output against a dequant reference.
    Only validated configs are eligible; fastest wins. Runs once per (N,K) at load.
    Decode-side timing is L2-flushed (rotating scratch read between reps) so the
    choice reflects production DRAM-bound conditions, not L2-hot flattery."""
    N, K = w_u8.shape
    torch.manual_seed(0)
    a = (torch.randn(m_probe, K, device=w_u8.device) * 0.5).to(torch.bfloat16)
    w_deq = ((w_u8.to(torch.float32) - 128.0)
             * scales.repeat_interleave(128, dim=1).to(torch.float32)).to(torch.bfloat16)
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


def warm_w8a16(w_u8, w_scale):
    """Load-time deterministic config selection + kernel compile warmup.
    Call before CUDA graph capture; afterwards launches are fixed-config."""
    key = (w_u8.shape[0], w_u8.shape[1])
    cfg = _select_config(w_u8, w_scale, _LARGE_CONFIGS, "large", m_probe=4096)
    if cfg is not None:
        _LARGE_CFG[key] = cfg
    cfg = _select_config(w_u8, w_scale, _SPLITK_CONFIGS, "splitk", m_probe=8)
    if cfg is not None:
        _SPLITK_CFG[key] = cfg
    a = torch.zeros((8, w_u8.shape[1]), dtype=torch.bfloat16, device=w_u8.device)
    w8a16_linear(a, w_u8, w_scale)
    a = torch.zeros((4096, w_u8.shape[1]), dtype=torch.bfloat16, device=w_u8.device)
    w8a16_linear(a, w_u8, w_scale)
    torch.cuda.synchronize()
    return _LARGE_CFG.get(key), _SPLITK_CFG.get(key)


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
    return a, w_q, scales, w


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
        a, w_q, scales, w_ref = make_qa(n, k)
        large_cfg, splitk_cfg = warm_w8a16(w_q, scales)
        print(f"{name}: pinned large={large_cfg} splitk={splitk_cfg}")
        out = w8a16_gemm(a, w_q, scales)
        w_deq = ((w_q.to(torch.float32) - 128.0) * scales.repeat_interleave(128, dim=1)).to(torch.bfloat16)
        ref = (a @ w_deq.T)
        diff = (out.float() - ref.float()).abs().max().item()
        rel = diff / ref.float().abs().max().item()
        def bench(fn, iters=20):
            for _ in range(3): fn()
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(iters): fn()
            torch.cuda.synchronize()
            return (time.time() - t0) / iters
        t_tri = bench(lambda: w8a16_gemm(a, w_q, scales))
        t_cublas = bench(lambda: a @ w_deq.T)
        tf_tri = 2 * M * n * k / t_tri / 1e12
        tf_cu = 2 * M * n * k / t_cublas / 1e12
        bw = n * k / t_tri / 1e9
        print(f"{name:14s} N={n:6d} K={k:6d} M={M}: triton {tf_tri:6.1f} TF ({t_tri*1e3:6.2f}ms) | cuBLAS bf16 {tf_cu:6.1f} TF | maxdiff {diff:.4f} rel {rel:.2e} | wBW {bw:5.0f} GB/s")
