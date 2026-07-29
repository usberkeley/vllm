# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Interface for Attention-FFN Disaggregation (AFD) connectors.

An AFD connector moves the residual-stream *activations* (not KV) between the
attention pool and the FFN pool at the attention-MoE seam. A single MoE layer
performs two per-step transfers:

- STAGE_A2F: attention pool -> FFN pool (the normalized post-attention hidden,
  the input to the MoE block).
- STAGE_F2A: FFN pool -> attention pool (the MoE output).

The interface is asynchronous: initiation (``isend`` /
``irecv``) is separated from completion (``poll``) so the driver loop never
blocks on a single slow transfer. ``isend`` enqueues a send on the comm stream
and returns immediately; ``irecv`` posts a receive into a preallocated buffer and
returns immediately; ``poll`` harvests whichever in-flight transfers have since
completed. Backend completion (CUDA event / NIXL notif / thread event) is hidden
behind ``AFDHandle.event`` so the driver only ever sees handles.

The abstraction mirrors the KV-connector layout (``AFDConnectorBase`` +
``AFDConnectorFactory``) at the *factory* level so out-of-tree transports register
the same way KV connectors do; the method surface is AFD-specific (activation
ping-pong, not per-request KV block access).
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# Transfer stages. A single MoE layer performs one A2F then one F2A transfer.
STAGE_A2F = 0
STAGE_F2A = 1


class AFDConnectorRole(enum.Enum):
    ATTENTION = "attention"
    FFN = "ffn"


@dataclass
class AFDMeta:
    """Identifies a single activation transfer.

    Attributes:
        layer_id: The decoder layer index the transfer belongs to.
        stage: ``STAGE_A2F`` or ``STAGE_F2A``.
        transfer_id: Identifier for one A2F/F2A round trip. It must be unique
            among in-flight transfers from the same attention rank.
    """

    layer_id: int
    stage: int
    transfer_id: int = 0

    def __post_init__(self) -> None:
        if self.layer_id < 0:
            raise ValueError(f"layer_id must be non-negative, got {self.layer_id}.")
        if self.stage not in (STAGE_A2F, STAGE_F2A):
            raise ValueError(f"unknown AFD transfer stage {self.stage}.")
        if self.transfer_id < 0:
            raise ValueError(
                f"transfer_id must be non-negative, got {self.transfer_id}."
            )

    def key(self) -> tuple[int, int, int]:
        return (self.layer_id, self.stage, self.transfer_id)


class AFDTransferIdAllocator:
    """Thread-safe monotonic transfer-id allocator.

    IDs are not reused during a process lifetime. This keeps routing independent
    of layer execution order and permits multiple batches from the same layer to
    be in flight concurrently.
    """

    def __init__(self, start: int = 0) -> None:
        if start < 0:
            raise ValueError(f"start must be non-negative, got {start}.")
        self._next = start
        self._lock = Lock()

    def next(self) -> int:
        with self._lock:
            transfer_id = self._next
            self._next += 1
        return transfer_id


@dataclass
class AFDHandle:
    """A single in-flight transfer.

    Attributes:
        meta: Which transfer this is (layer, stage, transfer id).
        tensor: The receive destination buffer (``irecv``) or the send source
            buffer (``isend``).
        event: Backend completion token (CUDA event / NIXL notif / thread
            event); ``None`` when the transfer is already complete.
    """

    meta: AFDMeta
    tensor: torch.Tensor
    event: object | None = None


class AFDConnectorBase(ABC):
    """Base class for AFD activation transports.

    Concrete transports implement three non-blocking primitives -- ``isend``
    (initiate a send), ``irecv`` (initiate a receive), and ``poll`` (harvest
    completions). ``all_to_all`` (design section 8.1) is a Phase 2 fusion path.
    """

    def __init__(self, vllm_config: VllmConfig, role: AFDConnectorRole) -> None:
        self.vllm_config = vllm_config
        self.afd_config = vllm_config.afd_config
        self.role = role

    @abstractmethod
    def isend(self, hidden: torch.Tensor, meta: AFDMeta) -> AFDHandle:
        """Non-blocking send: enqueue ``hidden`` for ``meta`` and return a handle.

        Does not wait for the send to land; the returned handle's completion is
        observed via ``poll``.
        """
        raise NotImplementedError

    @abstractmethod
    def irecv(self, template: torch.Tensor, meta: AFDMeta) -> AFDHandle:
        """Non-blocking receive: post a receive for ``meta`` and return a handle.

        ``template`` supplies the expected shape/dtype/device of the incoming
        tensor (fixed-shape scheduling guarantees these match the sender). The
        received tensor lands in ``AFDHandle.tensor`` once ``poll`` reports it.
        """
        raise NotImplementedError

    @abstractmethod
    def poll(self) -> list[AFDHandle]:
        """Return the in-flight receives that have completed since the last call.

        Never blocks. The driver loop calls this each tick to move arrivals into
        its queues (design section 6.3).
        """
        raise NotImplementedError

    def all_to_all(self, x: torch.Tensor, meta: AFDMeta) -> AFDHandle:
        """EP dispatch/combine, folding A->F transfer into the expert all-to-all."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement all_to_all yet."
        )

    def start_listening(self, expected_stage: int) -> None:
        """Enable unsolicited receives for an event-loop driven connector."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support dynamic receives."
        )
