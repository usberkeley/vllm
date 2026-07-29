# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""No-op attention module for the AFD FFN role.

On the FFN pool the attention block does not run: no KV cache, no attention
FLOPs, no attention kernel. Replacing the real attention module with this
identity (and popping its ``static_forward_context`` entry) removes the block
from the graph entirely, which is why zeroing weights was the wrong idea in the
prior attempts -- a no-op *module* removes the compute, a zero *weight* does not.

This is deliberately NOT an ``AttentionLayerBase`` and does not register into the
forward context, so the model runner never allocates KV for it.
"""

from __future__ import annotations

import torch
from torch import nn


class AFDNoOpAttention(nn.Module):
    """Identity stand-in for a decoder layer's attention block.

    The forward accepts the positional ``positions``/``hidden_states`` arguments
    that decoder layers pass to attention (plus any model-specific extras) and
    returns ``hidden_states`` unchanged. The FFN role's real value arrives over
    the wire at the MoE boundary, so whatever this returns is overwritten there.
    """

    def __init__(self, hidden_size: int, prefix: str = "") -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.prefix = prefix

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        return hidden_states
