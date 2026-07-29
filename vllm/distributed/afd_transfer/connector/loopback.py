# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""In-process AFD connector for event-loop correctness testing.

When an A2F activation is sent and a real MoE is registered for that
layer, the connector acts as a deterministic FFN peer: it runs the MoE inline
and exposes the matching F2A completion through ``poll``. Communication remains
owned by the Attention event loop; model ``forward`` never calls this connector.
This is the only connector that runs without CUDA or multiple processes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from vllm.distributed.afd_transfer.connector.base import (
    STAGE_A2F,
    STAGE_F2A,
    AFDConnectorBase,
    AFDConnectorRole,
    AFDHandle,
    AFDMeta,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

MoeFn = Callable[[torch.Tensor], torch.Tensor]


class LoopbackAFDConnector(AFDConnectorBase):
    def __init__(self, vllm_config: VllmConfig, role: AFDConnectorRole) -> None:
        super().__init__(vllm_config, role)
        self._moe: dict[int, MoeFn] = {}
        self._staged: dict[tuple[int, int, int], torch.Tensor] = {}
        self._completed: list[AFDHandle] = []

    def register_moe(self, layer_id: int, fn: MoeFn) -> None:
        """Register the real MoE callable for ``layer_id`` (attention-only mode)."""
        self._moe[layer_id] = fn

    def isend(self, hidden: torch.Tensor, meta: AFDMeta) -> AFDHandle:
        if meta.stage == STAGE_A2F and meta.layer_id in self._moe:
            out = self._moe[meta.layer_id](hidden)
            result_meta = AFDMeta(meta.layer_id, STAGE_F2A, meta.transfer_id)
            self._completed.append(AFDHandle(result_meta, out, event=None))
        else:
            self._staged[meta.key()] = hidden
        return AFDHandle(meta, hidden, event=None)

    def irecv(self, template: torch.Tensor, meta: AFDMeta) -> AFDHandle:
        tensor = self._staged.pop(meta.key(), template)
        handle = AFDHandle(meta, tensor, event=None)
        self._completed.append(handle)
        return handle

    def poll(self) -> list[AFDHandle]:
        done, self._completed = self._completed, []
        return done

    def start_listening(self, expected_stage: int) -> None:
        expected = STAGE_F2A if self.role is AFDConnectorRole.ATTENTION else STAGE_A2F
        if expected_stage != expected:
            raise ValueError(
                f"Loopback {self.role.value} connector listens for stage {expected}."
            )
