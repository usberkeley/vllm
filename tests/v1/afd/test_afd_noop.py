# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.attention.afd_noop import AFDNoOpAttention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase


def test_identity_forward():
    noop = AFDNoOpAttention(hidden_size=8)
    positions = torch.arange(4)
    hidden = torch.randn(4, 8)
    out = noop(positions, hidden)
    assert out is hidden


def test_extra_args_ignored():
    noop = AFDNoOpAttention(hidden_size=8)
    hidden = torch.randn(4, 8)
    # Decoder layers pass model-specific extras (e.g. scaling); must be ignored.
    out = noop(torch.arange(4), hidden, object(), some_kwarg=123)
    assert torch.equal(out, hidden)


def test_not_an_attention_layer_base():
    # Must not be enumerated as a KV-cache-bearing layer.
    assert not issubclass(AFDNoOpAttention, AttentionLayerBase)
