# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NIXL-backed AFD connector (skeleton).

Intended to reuse the NIXL transport engine's ``agent`` + ``xfer handle``
primitives (single-sided RDMA, ``get_notifs`` for completion) directly -- not the
block/request-oriented NIXL *KV-connector* wrapper (design section 5.7). The
transport body is deferred to a later phase; the class exists so the
factory/registry surface is complete and expresses the async signatures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.distributed.afd_transfer.connector.base import (
    AFDConnectorBase,
    AFDConnectorRole,
    AFDHandle,
    AFDMeta,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_NOT_IMPLEMENTED = (
    "NixlAFDConnector is a skeleton; use LoopbackAFDConnector (testing) or "
    "P2PAFDConnector (NCCL) until the NIXL transport is implemented."
)


class NixlAFDConnector(AFDConnectorBase):
    def __init__(self, vllm_config: VllmConfig, role: AFDConnectorRole) -> None:
        super().__init__(vllm_config, role)
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def isend(self, hidden: torch.Tensor, meta: AFDMeta) -> AFDHandle:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def irecv(self, template: torch.Tensor, meta: AFDMeta) -> AFDHandle:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def poll(self) -> list[AFDHandle]:
        raise NotImplementedError(_NOT_IMPLEMENTED)
