# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure graph marker for the Attention-FFN cut.

``afd_cut`` is an aliasing identity: it preserves the activation exactly while
leaving an explicit node in Dynamo/FX graphs. It has no connector access and no
communication side effects. The AFD graph-cut partitioner removes the marker
from executable compute fragments and lets the event loop transport the marker
input to the FFN peer.
"""

from __future__ import annotations

import torch

from vllm.utils.torch_utils import vllm_lib

AFD_SPLITTING_OPS = ["vllm::afd_cut"]


def afd_cut(hidden_states: torch.Tensor, layer_id: int) -> torch.Tensor:
    del layer_id
    return hidden_states


vllm_lib.define(
    "afd_cut(Tensor(a) hidden_states, int layer_id) -> Tensor(a)",
    alias_analysis="FROM_SCHEMA",
)
vllm_lib.impl("afd_cut", afd_cut, dispatch_key="CompositeExplicitAutograd")
vllm_lib._register_fake("afd_cut", afd_cut)
