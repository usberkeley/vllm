# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from conftest import HIDDEN, FakeMoE, FakeMoEV4, make_vllm_config

from vllm.model_executor.layers.attention.afd_noop import AFDNoOpAttention
from vllm.model_executor.models import afd as afd_mod
from vllm.model_executor.models.afd import (
    AFDAttnBoundary,
    AFDFfnBoundary,
    apply_afd_roles,
)

# Capture the real detector before the autouse monkeypatch rebinds it, so the
# detection test below can exercise the genuine implementation.
_REAL_IS_MOE = afd_mod._is_moe_layer

# (model fixture, attention attr, ffn attr): the standard DeepSeek-V2/V3 shape
# and the DeepSeek-V4 shape must both be rewritten correctly.
_SHAPES = [
    ("fake_model", "self_attn", "mlp"),
    ("fake_model_v4", "attn", "ffn"),
]


@pytest.fixture(autouse=True)
def _treat_all_layers_as_moe(monkeypatch):
    # The fake MoE is not a real MoERunner; force detection for structural tests.
    monkeypatch.setattr(afd_mod, "_is_moe_layer", lambda ffn: True)


@pytest.mark.parametrize("model_fx, attn_attr, ffn_attr", _SHAPES)
def test_attention_role_replaces_ffn(request, model_fx, attn_attr, ffn_attr):
    model = request.getfixturevalue(model_fx)
    apply_afd_roles(model, make_vllm_config("attention"))
    for layer in model.layers:
        assert isinstance(getattr(layer, ffn_attr), AFDAttnBoundary)
        # Attention block is untouched on the attention pool.
        assert not isinstance(getattr(layer, attn_attr), AFDNoOpAttention)
    assert [getattr(layer, ffn_attr).layer_id for layer in model.layers] == [0, 1, 2]


@pytest.mark.parametrize("model_fx, attn_attr, ffn_attr", _SHAPES)
def test_ffn_role_replaces_attn_and_wraps_ffn(request, model_fx, attn_attr, ffn_attr):
    model = request.getfixturevalue(model_fx)
    real_ffns = [getattr(layer, ffn_attr) for layer in model.layers]
    apply_afd_roles(model, make_vllm_config("ffn"))
    for layer, real in zip(model.layers, real_ffns):
        assert isinstance(getattr(layer, attn_attr), AFDNoOpAttention)
        boundary = getattr(layer, ffn_attr)
        assert isinstance(boundary, AFDFfnBoundary)
        assert boundary.mlp is real


@pytest.mark.parametrize("model_fx, attn_attr, ffn_attr", _SHAPES)
def test_ffn_role_pops_owned_attention_kv(request, model_fx, attn_attr, ffn_attr):
    # Register each layer's real attention AND a nested submodule (standing in for
    # DeepSeek-V4's indexer/compressor/SWA caches) under keys that deliberately do
    # NOT prefix-match the layer name -- removal must be by object identity, not
    # name prefix, or those auxiliary KV caches survive and leave the FFN pool
    # with a degenerate hybrid KV config instead of going attention-free.
    model = request.getfixturevalue(model_fx)
    static_ctx = {}
    for i, layer in enumerate(model.layers):
        attn = getattr(layer, attn_attr)
        static_ctx[f"unrelated_name.{i}"] = attn
        static_ctx[f"aux.cache.{i}"] = attn.proj
    survivor = object()
    static_ctx["kept.layer"] = survivor
    cfg = make_vllm_config("ffn", static_ctx=static_ctx)
    apply_afd_roles(model, cfg)
    assert static_ctx == {"kept.layer": survivor}


def test_is_moe_layer_detects_experts_without_moe_runner():
    # DeepSeek-V4's fp4 MegaMoE has no MoERunner; the routed `experts` submodule
    # is the signal. A plain MLP (no experts, no runner) must not match.
    assert _REAL_IS_MOE(FakeMoEV4(HIDDEN)) is True
    assert _REAL_IS_MOE(FakeMoE(HIDDEN)) is False


def test_no_moe_layers_raises(fake_model, monkeypatch):
    monkeypatch.setattr(afd_mod, "_is_moe_layer", lambda ffn: False)
    with pytest.raises(NotImplementedError, match="no MoE decoder layers"):
        apply_afd_roles(fake_model, make_vllm_config("attention"))


def test_sequence_parallel_moe_rejected(fake_model):
    fake_model.layers[0].use_sequence_parallel_moe = True
    with pytest.raises(NotImplementedError, match="sequence-parallel MoE"):
        apply_afd_roles(fake_model, make_vllm_config("attention"))


def test_fp16_rejected(fake_model):
    with pytest.raises(NotImplementedError, match="float16"):
        apply_afd_roles(fake_model, make_vllm_config("attention", dtype=torch.float16))
