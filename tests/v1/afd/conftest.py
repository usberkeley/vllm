# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared fixtures for AFD tests.

The fake model mirrors the fused-RMSNorm residual-threading structure of real
vLLM MoE decoder layers (e.g. ``DeepseekV2DecoderLayer``): the MoE block
consumes only the normalized hidden and the residual is threaded around it.
This is exactly the structure the AFD seam relies on.

``FakeModelV4`` mirrors the DeepSeek-V4 shape instead: the decoder layer names
its blocks ``attn``/``ffn`` (not ``self_attn``/``mlp``), the FFN exposes a routed
``experts`` submodule (the model-agnostic MoE signal AFD detects, since V4's fp4
MegaMoE has no ``MoERunner``), and the FFN takes an extra ``input_ids`` argument.
The seam is identical -- one hidden in, one hidden out -- so both shapes share the
same correctness invariants.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.config import AFDConfig

HIDDEN = 16
NUM_LAYERS = 3
NUM_TOKENS = 5


class FusedAddRMSNorm(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden))
        self.eps = 1e-6

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight

    def forward(self, x, residual=None):
        if residual is None:
            return self._norm(x)
        residual = x + residual
        return self._norm(residual), residual


class FakeAttn(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, positions, hidden, *args, **kwargs):
        return self.proj(hidden)


class FakeMoE(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.up = nn.Linear(hidden, 4 * hidden)
        self.down = nn.Linear(4 * hidden, hidden)
        self.act = nn.SiLU()

    def forward(self, hidden, **kwargs):
        return self.down(self.act(self.up(hidden)))


class FakeLayer(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.input_layernorm = FusedAddRMSNorm(hidden)
        self.post_attention_layernorm = FusedAddRMSNorm(hidden)
        self.self_attn = FakeAttn(hidden)
        self.mlp = FakeMoE(hidden)

    def forward(self, positions, hidden, residual):
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm(hidden)
        else:
            hidden, residual = self.input_layernorm(hidden, residual)
        hidden = self.self_attn(positions, hidden)
        hidden, residual = self.post_attention_layernorm(hidden, residual)
        hidden = self.mlp(hidden)
        return hidden, residual


class FakeModel(nn.Module):
    def __init__(self, hidden: int, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([FakeLayer(hidden) for _ in range(num_layers)])
        self.norm = FusedAddRMSNorm(hidden)

    def forward(self, positions, hidden):
        residual = None
        for layer in self.layers:
            hidden, residual = layer(positions, hidden, residual)
        hidden, _ = self.norm(hidden, residual)
        return hidden


class FakeExperts(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.up = nn.Linear(hidden, 4 * hidden)
        self.down = nn.Linear(4 * hidden, hidden)
        self.act = nn.SiLU()

    def forward(self, hidden):
        return self.down(self.act(self.up(hidden)))


class FakeMoEV4(nn.Module):
    """DeepSeek-V4-style MoE: extra ``input_ids`` arg + a routed ``experts``.

    ``input_ids`` is accepted but unused so the loopback connector (which runs the
    MoE inline on the attention side and cannot supply pool-local ``input_ids``)
    yields output identical to the monolithic path.
    """

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.experts = FakeExperts(hidden)

    def forward(self, hidden, input_ids=None):
        return self.experts(hidden)


class FakeLayerV4(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.input_layernorm = FusedAddRMSNorm(hidden)
        self.post_attention_layernorm = FusedAddRMSNorm(hidden)
        self.attn = FakeAttn(hidden)
        self.ffn = FakeMoEV4(hidden)

    def forward(self, positions, hidden, input_ids, residual):
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm(hidden)
        else:
            hidden, residual = self.input_layernorm(hidden, residual)
        hidden = self.attn(positions, hidden)
        hidden, residual = self.post_attention_layernorm(hidden, residual)
        hidden = self.ffn(hidden, input_ids)
        return hidden, residual


class FakeModelV4(nn.Module):
    def __init__(self, hidden: int, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([FakeLayerV4(hidden) for _ in range(num_layers)])
        self.norm = FusedAddRMSNorm(hidden)

    def forward(self, positions, hidden, input_ids=None):
        residual = None
        for layer in self.layers:
            hidden, residual = layer(positions, hidden, input_ids, residual)
        hidden, _ = self.norm(hidden, residual)
        return hidden


@pytest.fixture
def fake_model() -> FakeModel:
    torch.manual_seed(0)
    model = FakeModel(HIDDEN, NUM_LAYERS)
    # Randomize params so identity/zero bugs cannot pass by coincidence.
    for p in model.parameters():
        nn.init.normal_(p, std=0.1)
    return model.eval()


@pytest.fixture
def fake_model_v4() -> FakeModelV4:
    torch.manual_seed(0)
    model = FakeModelV4(HIDDEN, NUM_LAYERS)
    for p in model.parameters():
        nn.init.normal_(p, std=0.1)
    return model.eval()


@pytest.fixture
def model_inputs():
    torch.manual_seed(1)
    positions = torch.arange(NUM_TOKENS)
    hidden = torch.randn(NUM_TOKENS, HIDDEN)
    return positions, hidden


@pytest.fixture
def model_inputs_v4():
    torch.manual_seed(1)
    positions = torch.arange(NUM_TOKENS)
    hidden = torch.randn(NUM_TOKENS, HIDDEN)
    input_ids = torch.arange(NUM_TOKENS)
    return positions, hidden, input_ids


def make_vllm_config(
    role: str, dtype: torch.dtype = torch.float32, static_ctx=None
) -> SimpleNamespace:
    """Minimal stand-in exposing only what ``apply_afd_roles`` touches."""
    return SimpleNamespace(
        afd_config=AFDConfig(role=role, afd_connector="LoopbackAFDConnector"),
        model_config=SimpleNamespace(dtype=dtype, get_hidden_size=lambda: HIDDEN),
        compilation_config=SimpleNamespace(
            static_forward_context=static_ctx if static_ctx is not None else {}
        ),
    )
