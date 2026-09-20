"""Triton mixed-input GEMM: int8 groupwise-quantized weights x bf16 activations -> bf16.

Weight layout (compressed-tensors wNa16, little-endian int32 packing):
  W_u8: [N, K] uint8 view of packed int32 [N, K/4]; values are (w_int8 + 128) i.e. uint8b128
  S:    [N, K//128] scales (bf16)
  dequant: w = (u8 - 128) * scale[n, k//128]

Kernel keeps weights as int8 in HBM; dequantizes in registers; tl.dot on bf16.
BLOCK_K == 128 == group_size -> one scale per k-iteration, no gather.
"""
import torch
import triton
import triton.language as tl


_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=4),
]


@triton.autotune(configs=_CONFIGS, key=["M", "N", "K"])
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
    n_groups = K // GROUP_K

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


def w8a16_gemm(a_bf16, w_u8, scales, bias=None):
    """a: [M,K] bf16; w_u8: [N,K] uint8 (b128-encoded); scales: [N, K//128] bf16."""
    M, K = a_bf16.shape
    N = w_u8.shape[0]
    c = torch.empty((M, N), dtype=torch.bfloat16, device=a_bf16.device)
    BM, BN = 128, 128
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _w8a16_gemm_kernel[grid](
        a_bf16, w_u8, scales, c, M, N, K,
        a_bf16.stride(0), a_bf16.stride(1),
        w_u8.stride(0), w_u8.stride(1),
        scales.stride(0),
        c.stride(0), c.stride(1),
        GROUP_K=128,
    )
    if bias is not None:
        c += bias
    return c


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8),
    ],
    key=["N", "K"],
    reset_to_zero=["C"],
)
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


def w8a16_gemm_splitk(a_bf16, w_u8, scales, bias=None):
    M, K = a_bf16.shape
    N = w_u8.shape[0]
    c32 = torch.zeros((M, N), dtype=torch.float32, device=a_bf16.device)
    BM, BN = 16, 64
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN), 4)
    _w8a16_gemm_splitk_kernel[grid](
        a_bf16, w_u8, scales, c32, M, N, K,
        a_bf16.stride(0), a_bf16.stride(1),
        w_u8.stride(0), w_u8.stride(1),
        scales.stride(0),
        c32.stride(0), c32.stride(1),
        GROUP_K=128, SPLIT_K=4,
    )
    out = c32.to(a_bf16.dtype)
    if bias is not None:
        out += bias
    return out


def w8a16_linear(a_bf16, w_u8, scales, bias=None, splitk_threshold=128):
    if a_bf16.shape[0] < splitk_threshold:
        return w8a16_gemm_splitk(a_bf16, w_u8, scales, bias)
    return w8a16_gemm(a_bf16, w_u8, scales, bias)


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


def warm_w8a16(w_u8, w_scale):
    for m in (8, 72, 4096):
        a = torch.zeros((m, w_u8.shape[1]), dtype=torch.bfloat16, device=w_u8.device)
        w8a16_linear(a, w_u8, w_scale)
    torch.cuda.synchronize()

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
