# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Point-to-point AFD connector over ``torch.distributed`` (NCCL).

Transfers the seam activations directly between a paired attention rank and FFN
rank with non-blocking ``isend``/``irecv`` (design section 5.6): sends and
receives are enqueued on the process group and their completion is observed via
``poll`` (``Work.is_completed``). This is the portable default transport.

NOTE: this path requires an initialized process group and real devices; it
cannot be exercised on a CPU-only / single-process host. It is written to be
correct-by-construction and validated on multi-GPU hardware.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from vllm.distributed.afd_transfer.connector.base import (
    AFDConnectorBase,
    AFDConnectorRole,
    AFDHandle,
    AFDMeta,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

_CONTROL_TAG = 2**31 - 1


class P2PAFDConnector(AFDConnectorBase):
    def __init__(self, vllm_config: VllmConfig, role: AFDConnectorRole) -> None:
        super().__init__(vllm_config, role)
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "P2PAFDConnector requires an initialized torch.distributed "
                "process group."
            )
        # The peer rank this instance exchanges activations with. For the v1
        # 1-attention <-> 1-FFN topology this comes from config; a later phase
        # generalizes it to a rank map across the two pools.
        peer = self.afd_config.get_from_extra_config("peer_rank", None)
        if peer is None:
            raise ValueError(
                "P2PAFDConnector needs 'peer_rank' in afd_connector_extra_config."
            )
        self.peer_rank = int(peer)
        self.group = dist.group.WORLD
        world_size = dist.get_world_size(self.group)
        if not 0 <= self.peer_rank < world_size:
            raise ValueError(
                f"AFD peer_rank {self.peer_rank} is outside WORLD size {world_size}."
            )
        if self.peer_rank == dist.get_rank(self.group):
            raise ValueError("AFD peer_rank must identify another process.")
        # Posted receives awaiting completion, harvested by ``poll``. Sends are
        # fire-and-forget: the peer's ``poll`` observes their arrival.
        self._inflight: list[AFDHandle] = []
        self._dynamic_stage: int | None = None
        self._header: torch.Tensor | None = None
        self._header_work = None
        self._dynamic_payload: AFDHandle | None = None
        self._control_sends: list[tuple[torch.Tensor, object]] = []

    @staticmethod
    def _tag(meta: AFDMeta) -> int:
        # A transfer id identifies one round trip, so the stage bit is enough to
        # distinguish its A2F and F2A messages. Layer id is validated by AFDMeta
        # after receipt and need not consume tag bits.
        tag = (meta.transfer_id << 1) | meta.stage
        if tag >= _CONTROL_TAG:
            raise OverflowError(
                f"AFD transfer id {meta.transfer_id} exceeds the P2P tag range."
            )
        return tag

    def isend(self, hidden: torch.Tensor, meta: AFDMeta) -> AFDHandle:
        if self._dynamic_stage is not None:
            if hidden.ndim != 2:
                raise ValueError(
                    "Dynamic AFD P2P only supports 2D (tokens, hidden) tensors."
                )
            header = torch.tensor(
                [
                    meta.layer_id,
                    meta.stage,
                    meta.transfer_id,
                    hidden.shape[0],
                    hidden.shape[1],
                ],
                dtype=torch.int64,
                device=hidden.device,
            )
            control_work = dist.isend(
                header,
                dst=self.peer_rank,
                group=self.group,
                tag=_CONTROL_TAG,
            )
            self._control_sends.append((header, control_work))
        send_tensor = hidden.contiguous()
        work = dist.isend(
            send_tensor,
            dst=self.peer_rank,
            group=self.group,
            tag=self._tag(meta),
        )
        return AFDHandle(meta, send_tensor, event=work)

    def irecv(self, template: torch.Tensor, meta: AFDMeta) -> AFDHandle:
        if self._dynamic_stage is not None:
            raise RuntimeError(
                "Explicit irecv cannot be mixed with dynamic P2P listening."
            )
        buf = torch.empty_like(template)
        work = dist.irecv(
            buf, src=self.peer_rank, group=self.group, tag=self._tag(meta)
        )
        handle = AFDHandle(meta, buf, event=work)
        self._inflight.append(handle)
        return handle

    def _post_header(self) -> None:
        assert self._dynamic_stage is not None
        device = self.vllm_config.device_config.device
        self._header = torch.empty(5, dtype=torch.int64, device=device)
        self._header_work = dist.irecv(
            self._header,
            src=self.peer_rank,
            group=self.group,
            tag=_CONTROL_TAG,
        )

    def start_listening(self, expected_stage: int) -> None:
        if expected_stage not in (0, 1):
            raise ValueError(f"unknown dynamic receive stage {expected_stage}.")
        if self._dynamic_stage is not None:
            if self._dynamic_stage != expected_stage:
                raise RuntimeError("P2P dynamic listener already uses another stage.")
            return
        if self._inflight:
            raise RuntimeError(
                "Cannot start dynamic P2P listening with posted explicit receives."
            )
        self._dynamic_stage = expected_stage
        self._post_header()

    def _poll_dynamic(self) -> list[AFDHandle]:
        if self._dynamic_stage is None:
            return []

        if self._dynamic_payload is not None:
            event = self._dynamic_payload.event
            if event is not None and not event.is_completed():
                return []
            completed = self._dynamic_payload
            self._dynamic_payload = None
            self._post_header()
            return [completed]

        if self._header_work is None or not self._header_work.is_completed():
            return []
        assert self._header is not None
        layer_id, stage, transfer_id, num_tokens, hidden_size = (
            int(value) for value in self._header.tolist()
        )
        if stage != self._dynamic_stage:
            raise RuntimeError(
                f"AFD P2P received stage {stage}, expected {self._dynamic_stage}."
            )
        expected_hidden_size = self.vllm_config.model_config.get_hidden_size()
        if hidden_size != expected_hidden_size or num_tokens < 1:
            raise RuntimeError(
                "Invalid dynamic AFD tensor shape "
                f"({num_tokens}, {hidden_size}); expected hidden size "
                f"{expected_hidden_size}."
            )
        meta = AFDMeta(layer_id, stage, transfer_id)
        tensor = torch.empty(
            (num_tokens, hidden_size),
            dtype=self.vllm_config.model_config.dtype,
            device=self.vllm_config.device_config.device,
        )
        work = dist.irecv(
            tensor,
            src=self.peer_rank,
            group=self.group,
            tag=self._tag(meta),
        )
        self._dynamic_payload = AFDHandle(meta, tensor, event=work)
        self._header = None
        self._header_work = None
        return []

    def poll(self) -> list[AFDHandle]:
        self._control_sends = [
            (tensor, work)
            for tensor, work in self._control_sends
            if not work.is_completed()
        ]
        dynamic = self._poll_dynamic()
        done = [h for h in self._inflight if h.event is None or h.event.is_completed()]
        if done:
            done_ids = {id(h) for h in done}
            self._inflight = [h for h in self._inflight if id(h) not in done_ids]
        return dynamic + done
