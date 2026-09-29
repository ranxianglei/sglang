# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0

# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import logging
import re as _re
from typing import Callable, Optional

import torch
from compressed_tensors.quantization import ActivationOrdering

# yapf conflicts with isort for this block
# yapf: disable
from sglang.srt.layers.parameter import (
    BasevLLMParameter,
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    PackedColumnParameter,
    PackedvLLMParameter,
    RowvLLMParameter,
    permute_param_layout_,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsLinearScheme,
)
from sglang.srt.layers.quantization.marlin_utils import (
    MarlinLinearLayerConfig,
    apply_gptq_marlin_linear,
    check_marlin_supports_shape,
    marlin_is_k_full,
    marlin_make_empty_g_idx,
    marlin_make_workspace,
    marlin_permute_scales,
    marlin_repeat_scales_on_all_ranks,
    marlin_sort_g_idx,
    marlin_zero_points,
)
from sglang.srt.layers.quantization.utils import (
    get_scalar_types,
    replace_parameter,
    unpack_cols,
)
from sglang.srt.utils import get_bool_env_var, is_cuda

_is_cuda = is_cuda()

if _is_cuda:
    from sglang.kernels.ops.quantization.gptq_marlin_repack import gptq_marlin_repack


ScalarType, scalar_types = get_scalar_types()

# GDN (linear_attn) projections are recurrent-state writers: dequant noise
# compounds over context. Env routes them back to the exact Marlin numerics
# while MLP/attn keep the fast path.
_GDN_PATTERN = None
_GDN_PATTERN_INIT = False


def _gdn_marlin_pattern():
    global _GDN_PATTERN, _GDN_PATTERN_INIT
    if not _GDN_PATTERN_INIT:
        _GDN_PATTERN_INIT = True
        if get_bool_env_var("SGLANG_WNA16_GDN_MARLIN"):
            _GDN_PATTERN = _re.compile(r"linear_attn\.(in_proj_qkv|in_proj_z|out_proj)")
    return _GDN_PATTERN


logger = logging.getLogger(__name__)

__all__ = ["CompressedTensorsWNA16"]
WNA16_SUPPORTED_TYPES_MAP = {
    4: scalar_types.uint4b8,
    8: scalar_types.uint8b128
}
WNA16_ZP_SUPPORTED_TYPES_MAP = {4: scalar_types.uint4, 8: scalar_types.uint8}
WNA16_SUPPORTED_BITS = list(WNA16_SUPPORTED_TYPES_MAP.keys())


class CompressedTensorsWNA16(CompressedTensorsLinearScheme):
    _kernel_backends_being_used: set[str] = set()

    def __init__(self,
                 strategy: str,
                 num_bits: int,
                 group_size: Optional[int] = None,
                 symmetric: Optional[bool] = True,
                 actorder: Optional[ActivationOrdering] = None):

        self.pack_factor = 32 // num_bits
        self.strategy = strategy
        self.symmetric = symmetric
        self.group_size = -1 if group_size is None else group_size
        self.has_g_idx = actorder == ActivationOrdering.GROUP

        if self.group_size == -1 and self.strategy != "channel":
            raise ValueError("Marlin kernels require group quantization or "
                             "channelwise quantization, but found no group "
                             "size and strategy is not channelwise.")

        if num_bits not in WNA16_SUPPORTED_TYPES_MAP:
            raise ValueError(
                f"Unsupported num_bits = {num_bits}. "
                f"Supported num_bits = {WNA16_SUPPORTED_TYPES_MAP.keys()}")

        self.quant_type = (WNA16_ZP_SUPPORTED_TYPES_MAP[num_bits]
                           if not self.symmetric else
                           WNA16_SUPPORTED_TYPES_MAP[num_bits])

        # SGLANG_WNA16_NATIVE_BF16=1: dequantize weights to bf16 at load and
        # use cuBLAS GEMM instead of Marlin (which is decode-optimized and
        # wastes ~half the tensor-core throughput at prefill M>=1k). Cost:
        # weights double in size (KV pool shrinks). Numerics identical.
        self.native_bf16 = get_bool_env_var("SGLANG_WNA16_NATIVE_BF16")
        if self.native_bf16:
            assert self.symmetric, (
                "SGLANG_WNA16_NATIVE_BF16 requires symmetric quantization")
            assert not self.has_g_idx, (
                "SGLANG_WNA16_NATIVE_BF16 does not support actorder/g_idx")

        # SGLANG_WNA16_TRITON=1: keep weights as int8 (uint8b128 packed) and
        # run a custom triton mixed-input GEMM (dequant-on-the-fly, fp32
        # accumulate). Prefill-path GEMM reaches 2-2.5x Marlin throughput
        # while decode still streams int8 weights (no KV/memory cost).
        self.triton_mixed = get_bool_env_var("SGLANG_WNA16_TRITON")
        if self.triton_mixed:
            assert self.symmetric, (
                "SGLANG_WNA16_TRITON requires symmetric quantization")
            assert not self.has_g_idx, (
                "SGLANG_WNA16_TRITON does not support actorder/g_idx")

    @classmethod
    def get_min_capability(cls) -> int:
        # ampere and up
        return 80

    def create_weights(self, layer: torch.nn.Module, output_size: int,
                       input_size: int, output_partition_sizes: list[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):

        output_size_per_partition = sum(output_partition_sizes)

        self.kernel_config = MarlinLinearLayerConfig(
            full_weight_shape=(input_size, output_size),
            partition_weight_shape=(
                input_size_per_partition,
                output_size_per_partition,
            ),
            weight_type=self.quant_type,
            act_type=params_dtype,
            group_size=self.group_size,
            zero_points=not self.symmetric,
            has_g_idx=self.has_g_idx
        )

        # If group_size is -1, we are in channelwise case.
        group_size = self.group_size if self.group_size != -1 else input_size
        row_parallel = (input_size != input_size_per_partition)
        partition_scales = not marlin_repeat_scales_on_all_ranks(
            self.has_g_idx, self.group_size, row_parallel)

        scales_and_zp_size = input_size // group_size

        if partition_scales:
            assert input_size_per_partition % group_size == 0
            scales_and_zp_size = input_size_per_partition // group_size

        weight = PackedvLLMParameter(input_dim=1,
                                     output_dim=0,
                                     weight_loader=weight_loader,
                                     packed_factor=self.pack_factor,
                                     packed_dim=1,
                                     data=torch.empty(
                                         output_size_per_partition,
                                         input_size_per_partition //
                                         self.pack_factor,
                                         dtype=torch.int32,
                                     ))

        weight_scale_args = {
            "weight_loader":
            weight_loader,
            "data":
            torch.empty(
                output_size_per_partition,
                scales_and_zp_size,
                dtype=params_dtype,
            )
        }

        zeros_args = {
            "weight_loader":
            weight_loader,
            "data":
            torch.zeros(
                output_size_per_partition // self.pack_factor,
                scales_and_zp_size,
                dtype=torch.int32,
            )
        }

        if not partition_scales:
            weight_scale = ChannelQuantScaleParameter(output_dim=0,
                                                      **weight_scale_args)

            if not self.symmetric:
                qzeros = PackedColumnParameter(output_dim=0,
                                               packed_dim=0,
                                               packed_factor=self.pack_factor,
                                               **zeros_args)
        else:
            weight_scale = GroupQuantScaleParameter(output_dim=0,
                                                    input_dim=1,
                                                    **weight_scale_args)
            if not self.symmetric:
                qzeros = PackedvLLMParameter(input_dim=1,
                                             output_dim=0,
                                             packed_dim=0,
                                             packed_factor=self.pack_factor,
                                             **zeros_args)

        # A 2D array defining the original shape of the weights
        # before packing
        weight_shape = BasevLLMParameter(data=torch.empty(2,
                                                          dtype=torch.int64),
                                         weight_loader=weight_loader)

        layer.register_parameter("weight_packed", weight)
        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter("weight_shape", weight_shape)

        if not self.symmetric:
            layer.register_parameter("weight_zero_point", qzeros)

        # group index (for activation reordering)
        if self.has_g_idx:
            weight_g_idx = RowvLLMParameter(data=torch.empty(
                input_size_per_partition,
                dtype=torch.int32,
            ),
                                            input_dim=0,
                                            weight_loader=weight_loader)
            layer.register_parameter("weight_g_idx", weight_g_idx)

    # Checkpoints are serialized in compressed-tensors format, which is
    # different from the format the kernel may want. Handle repacking here.
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Default names since marlin requires empty parameters for these,
        # TODO: remove this requirement from marlin (allow optional tensors)
        self.w_q_name = "weight_packed"
        self.w_s_name = "weight_scale"
        self.w_zp_name = "weight_zero_point"
        self.w_gidx_name = "weight_g_idx"

        if self.native_bf16:
            self._process_weights_native_bf16(layer)
            return
        gdn_pat = _gdn_marlin_pattern()
        gdn_marlin = bool(
            gdn_pat is not None
            and gdn_pat.search(getattr(self, "layer_name", "") or "")
        )
        if gdn_marlin:
            layer._wna16_gdn_marlin = True
        if self.triton_mixed and not gdn_marlin:
            self._process_weights_triton(layer)
            return

        device = getattr(layer, self.w_q_name).device
        c = self.kernel_config

        check_marlin_supports_shape(
            c.partition_weight_shape[1],  # out_features
            c.partition_weight_shape[0],  # in_features
            c.full_weight_shape[0],  # in_features
            c.group_size,
        )

        row_parallel = c.partition_weight_shape[0] != c.full_weight_shape[0]
        self.is_k_full = marlin_is_k_full(c.has_g_idx, row_parallel)

        # Allocate marlin workspace.
        self.workspace = marlin_make_workspace(device)

        def _transform_param(
            layer: torch.nn.Module, name: Optional[str], fn: Callable
        ) -> None:
            if name is not None and getattr(layer, name, None) is not None:

                old_param = getattr(layer, name)
                new_param = fn(old_param)
                # replace the parameter with torch.nn.Parameter for TorchDynamo
                # compatibility
                replace_parameter(
                    layer, name, torch.nn.Parameter(new_param.data, requires_grad=False)
                )

        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            x.data = gptq_marlin_repack(
                x.data.contiguous(),
                perm=layer.g_idx_sort_indices,
                size_k=c.partition_weight_shape[0],
                size_n=c.partition_weight_shape[1],
                num_bits=c.weight_type.size_bits,
            )
            return x

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = marlin_permute_scales(
                x.data.contiguous(),
                size_k=c.partition_weight_shape[0],
                size_n=c.partition_weight_shape[1],
                group_size=c.group_size,
            )
            return x

        if c.has_g_idx:
            g_idx, g_idx_sort_indices = marlin_sort_g_idx(
                getattr(layer, self.w_gidx_name)
            )
            _transform_param(layer, self.w_gidx_name, lambda _: g_idx)
            layer.g_idx_sort_indices = g_idx_sort_indices
        else:
            setattr(layer, self.w_gidx_name, marlin_make_empty_g_idx(device))
            layer.g_idx_sort_indices = marlin_make_empty_g_idx(device)

        if c.zero_points:
            grouped_k = (
                c.partition_weight_shape[0] // c.group_size if c.group_size != -1 else 1
            )
            _transform_param(
                layer,
                self.w_zp_name,
                lambda x: marlin_zero_points(
                    unpack_cols(
                        x.t(),
                        c.weight_type.size_bits,
                        grouped_k,
                        c.partition_weight_shape[1],
                    ),
                    size_k=grouped_k,
                    size_n=c.partition_weight_shape[1],
                    num_bits=c.weight_type.size_bits,
                ),
            )
        else:
            setattr(layer, self.w_zp_name, marlin_make_empty_g_idx(device))
        _transform_param(layer, self.w_q_name, transform_w_q)
        _transform_param(layer, self.w_s_name, transform_w_s)

    def _process_weights_native_bf16(self, layer: torch.nn.Module) -> None:
        w_q = getattr(layer, self.w_q_name).data
        w_s = getattr(layer, self.w_s_name).data
        device = w_q.device
        n = w_q.shape[0]
        k = w_q.shape[1] * self.pack_factor
        g = self.group_size
        w = torch.empty(n, k, dtype=torch.bfloat16, device=device)
        chunk = max(1, (1 << 28) // max(k, 1))
        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            t = (
                w_q[i:j].contiguous().view(torch.uint8)[:, :k]
                .to(torch.bfloat16)
                .sub_(128.0)
            )
            if g == -1:
                w[i:j].copy_(t.mul_(w_s[i:j].to(torch.bfloat16)))
            else:
                t = t.view(j - i, k // g, g).mul_(
                    w_s[i:j].to(torch.bfloat16).unsqueeze(-1)
                )
                w[i:j].copy_(t.view(j - i, k))
        layer.weight_native = w
        replace_parameter(
            layer,
            self.w_q_name,
            torch.nn.Parameter(
                torch.empty(0, dtype=torch.int32, device=device),
                requires_grad=False,
            ),
        )

    def _process_weights_triton(self, layer: torch.nn.Module) -> None:
        from sglang.srt.layers.quantization.compressed_tensors.schemes.tri_w8a16 import (
            warm_w8a16,
        )
        w_q = getattr(layer, self.w_q_name).data
        w_s = getattr(layer, self.w_s_name).data
        n = w_q.shape[0]
        k = w_q.shape[1] * self.pack_factor
        g = self.group_size if self.group_size != -1 else k
        w_u8 = w_q.contiguous().view(torch.uint8)[:, :k]
        scale = w_s.to(torch.bfloat16) if w_s.dtype != torch.bfloat16 else w_s
        if scale.shape[1] == 1:
            scale = scale.expand(-1, k // g).contiguous()
        else:
            scale = scale.contiguous()
        layer.w_u8 = w_u8.t().contiguous()
        layer.w_scale = scale.t().contiguous()
        w_u8 = None
        scale = None
        replace_parameter(
            layer,
            self.w_q_name,
            torch.nn.Parameter(
                torch.empty(0, dtype=torch.int32, device=w_q.device),
                requires_grad=False,
            ),
        )
        warm_w8a16(layer.w_u8, layer.w_scale)

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        if getattr(layer, "_wna16_gdn_marlin", False):
            pass
        elif self.native_bf16:
            return torch.nn.functional.linear(x, layer.weight_native, bias)
        elif self.triton_mixed:
            from sglang.srt.layers.quantization.compressed_tensors.schemes.tri_w8a16 import (
                w8a16_linear,
            )
            x2 = x.reshape(-1, x.shape[-1])
            out = w8a16_linear(x2, layer.w_u8, layer.w_scale, bias)
            return out.reshape(*x.shape[:-1], out.shape[-1])
        c = self.kernel_config

        def _get_weight_params(
            layer: torch.nn.Module,
        ) -> tuple[
            torch.Tensor,  # w_q
            torch.Tensor,  # w_s
            Optional[torch.Tensor],  # w_zp,
            Optional[torch.Tensor],  # w_gidx
        ]:
            return (
                getattr(layer, self.w_q_name),
                getattr(layer, self.w_s_name),
                getattr(layer, self.w_zp_name or "", None),
                getattr(layer, self.w_gidx_name or "", None),
            )

        w_q, w_s, w_zp, w_gidx = _get_weight_params(layer)

        # `process_weights_after_loading` will ensure w_zp and w_gidx are not
        #  None for marlin
        return apply_gptq_marlin_linear(
            input=x,
            weight=w_q,
            weight_scale=w_s,
            weight_zp=w_zp,  # type: ignore
            g_idx=w_gidx,  # type: ignore
            g_idx_sort_indices=layer.g_idx_sort_indices,
            workspace=self.workspace,
            wtype=c.weight_type,
            input_size_per_partition=c.partition_weight_shape[0],
            output_size_per_partition=c.partition_weight_shape[1],
            is_k_full=self.is_k_full,
            bias=bias,
        )
