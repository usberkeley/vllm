# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Role-based module replacement for Attention-FFN Disaggregation (AFD).

``apply_afd_roles`` rewrites a fully-built model in place so it plays either the
attention role or the FFN role, without touching any per-model source. It is
model-agnostic: it finds decoder layers by duck-typing (an attention attribute
``self_attn``/``attn`` plus an FFN attribute ``mlp``/``ffn``) and rewrites only
the MoE layers -- those whose FFN block exposes a ``FusedMoE``/``MoERunner`` or a
routed ``experts`` submodule (e.g. DeepSeek-V4's fp4 MegaMoE). The cut stays at
the MoE-module seam (single hidden in, single hidden out). The segmented runtime
validates the surrounding decoder contract at startup and rejects required
model-specific state that activation-only transport cannot reconstruct.

Attention role: keep the real attention (+ KV); replace the MoE block with
``AFDAttnBoundary``, a pure cut marker that exposes the normalized MoE input to
the segmented runtime. The residual stream never leaves this pool.

FFN role: replace attention with ``AFDNoOpAttention`` (identity, no KV) and pop
its ``static_forward_context`` entries so no KV cache is allocated; wrap the real
MoE with ``AFDFfnBoundary``, a pure-compute wrapper used by the FFN event loop.
All transport is owned by the role event loops, never by model ``forward``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.layers.afd import ops as afd_ops  # noqa: F401
from vllm.model_executor.layers.attention.afd_noop import AFDNoOpAttention
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

_LAYER_IDX_RE = re.compile(r"\.(\d+)$")


class AFDAttnBoundary(nn.Module):
    """Attention-role replacement for a decoder layer's MoE block.

    Exposes the normalized post-attention hidden at the cut without performing
    communication. ``AFDSegmentedModel`` stops after this layer and returns the
    value to ``AFDAttentionEventLoop``, the sole owner of A2F/F2A transport.
    """

    def __init__(self, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.emit_cut = True

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # The activation-only segmented runtime rejects models whose real MoE
        # requires additional arguments; optional hints are intentionally dropped.
        if not self.emit_cut:
            return hidden_states
        return torch.ops.vllm.afd_cut(hidden_states, self.layer_id)


# TODO 改为异步发送后，这个类是否可以删除？是否需要这里做 FX 图切割点？
# 正常来说这里是下一层的分界点，不知道 FX 图天生就会在这里与下一层的算子切割开来，
# 如果是就不在需要这个类，也不需要补充自定义算子
class AFDFfnBoundary(nn.Module):
    """FFN-role wrapper around a decoder layer's real MoE block.

    Runs only the local MoE computation. The FFN event loop receives the real
    normalized hidden and sends the result outside model ``forward``.
    """

    def __init__(self, real_mlp: nn.Module, layer_id: int) -> None:
        super().__init__()
        self.mlp = real_mlp
        self.layer_id = layer_id

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Forward any model-specific args (e.g. input_ids) to the real MoE.
        return self.mlp(hidden_states, *args, **kwargs)


# Attention/FFN attribute names, most-standard first so DeepSeek-V2/V3-style
# layers resolve exactly as before and per-model variants (e.g. DeepSeek-V4's
# ``attn``/``ffn``) still match.
_ATTN_ATTRS = ("self_attn", "attn")
_FFN_ATTRS = ("mlp", "ffn")


def _first_attr(module: nn.Module, names: tuple[str, ...]) -> str | None:
    return next((n for n in names if hasattr(module, n)), None)


def _iter_decoder_layers(model: nn.Module):
    """Yield ``(name, module, attn_attr, ffn_attr)`` for decoder-like layers."""
    for name, module in model.named_modules():
        attn_attr = _first_attr(module, _ATTN_ATTRS)
        ffn_attr = _first_attr(module, _FFN_ATTRS)
        if attn_attr is not None and ffn_attr is not None:
            yield name, module, attn_attr, ffn_attr


def _is_moe_layer(ffn_module: nn.Module) -> bool:
    """True if the FFN block is an MoE block (not a dense MLP).

    Recognizes both the shared ``FusedMoE`` (which contains a ``MoERunner``) and
    model-specific MoE blocks that expose a routed ``experts`` submodule without a
    ``MoERunner`` (e.g. DeepSeek-V4's fp4 MegaMoE). Dense MLP blocks expose
    neither and are left to run intact on the attention pool.
    """
    if any(isinstance(m, MoERunner) for m in ffn_module.modules()):
        return True
    return hasattr(ffn_module, "experts")


def _layer_id_from_name(name: str, fallback: int) -> int:
    match = _LAYER_IDX_RE.search(name)
    return int(match.group(1)) if match else fallback


def _pop_attention_kv(static_ctx: dict, attn: nn.Module) -> int:
    """Remove ``real_attn`` and its submodules from the forward context.

    The FFN pool runs no attention, so every ``static_forward_context`` entry the
    real attention owns -- the attention itself plus any KV-bearing submodules it
    nests (e.g. DeepSeek-V4's indexer / compressor / sliding-window caches) --
    must be dropped so the runner allocates zero KV and takes the attention-free
    path. Entries are matched by object identity rather than name prefix: those
    caches register under their own prefixes, which need not sit under the decoder
    layer's name. Returns the number of entries removed.
    """
    attn_module_ids = {id(m) for m in attn.modules()}
    attn_keys = [key for key, mod in static_ctx.items() if id(mod) in attn_module_ids]
    for key in attn_keys:
        static_ctx.pop(key, None)
    return len(attn_keys)


def _assert_supported(layer: nn.Module, name: str, dtype: torch.dtype) -> None:
    if getattr(layer, "use_sequence_parallel_moe", False):
        raise NotImplementedError(
            f"AFD does not support sequence-parallel MoE (layer '{name}'): "
            "it reduce-scatters the residual stream, breaking residual "
            "locality at the seam."
        )
    if dtype == torch.float16:
        raise NotImplementedError(
            "AFD does not support float16: the routed_scaling_factor fp16 "
            "fixup keys on the MoE module type, which module replacement "
            "changes. Use bfloat16."
        )


def apply_afd_roles(model: nn.Module, vllm_config: VllmConfig) -> list[int]:
    """Rewrite ``model`` in place for the configured AFD role.

    Args:
        model: The fully-built (weights-loaded) model to rewrite.
        vllm_config: The engine config; ``afd_config.role`` selects the rewrite.

    Returns:
        The layer ids of the rewritten MoE layers (used by the FFN pool to size
        its per-layer scheduler queues).

    Raises:
        NotImplementedError: If the model exposes no MoE decoder layers, or uses
            a configuration AFD does not support (sequence-parallel MoE, fp16).
    """
    afd_config = vllm_config.afd_config
    assert afd_config is not None

    role = afd_config.role
    dtype = vllm_config.model_config.dtype
    static_ctx = vllm_config.compilation_config.static_forward_context

    moe_layer_ids: list[int] = []
    for seq_idx, (name, layer, attn_attr, ffn_attr) in enumerate(
        _iter_decoder_layers(model)
    ):
        ffn = getattr(layer, ffn_attr)
        if not _is_moe_layer(ffn):
            continue  # dense layers run intact on the attention pool
        _assert_supported(layer, name, dtype)
        layer_id = _layer_id_from_name(name, seq_idx)

        if role == "attention":
            setattr(layer, ffn_attr, AFDAttnBoundary(layer_id))
        elif role == "ffn":
            hidden_size = vllm_config.model_config.get_hidden_size()
            attn = getattr(layer, attn_attr)
            setattr(
                layer,
                attn_attr,
                AFDNoOpAttention(hidden_size, prefix=f"{name}.{attn_attr}"),
            )
            _pop_attention_kv(static_ctx, attn)
            setattr(layer, ffn_attr, AFDFfnBoundary(ffn, layer_id))
        moe_layer_ids.append(layer_id)

    if not moe_layer_ids:
        raise NotImplementedError(
            "AFD found no MoE decoder layers (expected layers exposing an "
            "attention attribute 'self_attn'/'attn' plus an FFN attribute "
            "'mlp'/'ffn' that contains a FusedMoE/MoERunner or a routed "
            "'experts' submodule). AFD only supports MoE models."
        )

    logger.info("AFD applied role '%s' to %d MoE layer(s).", role, len(moe_layer_ids))
    return moe_layer_ids
