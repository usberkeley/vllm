# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-threaded event loops for both AFD roles (design section 3.3).

FFN ticks dynamically select ready layer queues. Attention ticks instead keep
one intact scheduler batch in lockstep, advancing its continuation only after
the current F2A result arrives. ``poll``/``isend`` remain async transport
primitives, and compute and transport use separate CUDA streams.

The loop is expressed against two small protocols (``AFDPollSource``,
``AFDSendSink``) and a ``replay_fn`` callback rather than a concrete
connector, so the scheduling/routing logic is exercised in unit tests with fakes,
independent of NCCL/NIXL and of graph capture. A concrete async connector and the
real per-layer graph replay are wired in by the worker.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch

from vllm.distributed.afd_transfer.connector.base import (
    STAGE_A2F,
    STAGE_F2A,
    AFDHandle,
    AFDMeta,
    AFDTransferIdAllocator,
)
from vllm.distributed.afd_transfer.scheduler import (
    AFDDynamicBatchScheduler,
    QueueItem,
    drop_padding,
    pad_to_bucket,
    pick_bucket,
)

__all__ = [
    "AFDAttentionEventLoop",
    "AFDCreditTracker",
    "AFDHandle",
    "AFDFfnEventLoop",
    "AFDPollSource",
    "AFDSendSink",
]


class AFDPollSource(Protocol):
    """A transport that surfaces completed inbound activations."""

    def poll(self) -> list[AFDHandle]:
        """Return activations that arrived since the last tick (never blocks)."""
        ...


class AFDSendSink(Protocol):
    """A transport that ships an activation to the peer pool."""

    def isend(self, hidden: torch.Tensor, meta: AFDMeta) -> AFDHandle | None: ...


ReplayFn = Callable[[int, torch.Tensor], torch.Tensor]
AttentionReplayFn = Callable[[int, torch.Tensor, list[QueueItem]], torch.Tensor]
CompletionFn = Callable[[torch.Tensor, AFDMeta, object | None], None]


def _handle_complete(handle: AFDHandle) -> bool:
    event = handle.event
    if event is None:
        return True
    if hasattr(event, "is_completed"):
        return bool(event.is_completed())
    if hasattr(event, "query"):
        return bool(event.query())
    return False


class AFDCreditTracker:
    """Bounds concurrent Attention-to-FFN round trips."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"credit capacity must be >= 1, got {capacity}.")
        self.capacity = capacity
        self._in_flight: set[int] = set()

    @property
    def available(self) -> int:
        return self.capacity - len(self._in_flight)

    @property
    def in_flight(self) -> int:
        return len(self._in_flight)

    def acquire(self, transfer_id: int) -> None:
        if transfer_id in self._in_flight:
            raise ValueError(f"transfer id {transfer_id} is already in flight.")
        if self.available == 0:
            raise RuntimeError("AFD attention credits are exhausted.")
        self._in_flight.add(transfer_id)

    def release(self, transfer_id: int) -> None:
        if transfer_id not in self._in_flight:
            raise ValueError(f"unknown completed transfer id {transfer_id}.")
        self._in_flight.remove(transfer_id)


class AFDFfnEventLoop:
    """Drives one FFN worker: poll -> pick -> replay -> send.

    Args:
        scheduler: The per-layer dynamic-batch scheduler.
        source: Transport polled for inbound activations.
        sink: Transport used to return MoE results.
        replay_fn: ``(layer_id, padded_hidden) -> padded_out`` -- replays the
            captured MoE graph for a layer against a bucket-padded batch.
        capture_sizes: Captured cudagraph batch sizes (padding targets).
    """

    def __init__(
        self,
        scheduler: AFDDynamicBatchScheduler,
        source: AFDPollSource,
        sink: AFDSendSink,
        replay_fn: ReplayFn,
        capture_sizes: list[int] | None,
    ) -> None:
        if capture_sizes is not None and not capture_sizes:
            raise ValueError("capture_sizes cannot be empty when padding is enabled.")
        self.scheduler = scheduler
        self.source = source
        self.sink = sink
        self.replay_fn = replay_fn
        self.capture_sizes = capture_sizes
        self._send_handles: list[AFDHandle] = []

    def tick(self, now: float) -> bool:
        """Run one non-blocking loop iteration.

        Returns:
            ``True`` if a layer ran this tick, ``False`` if nothing was ready
            (no batch is launched when idle -- empty-batch safety).
        """
        self._send_handles = [
            handle for handle in self._send_handles if not _handle_complete(handle)
        ]
        for handle in self.source.poll():
            if handle.meta.stage != STAGE_A2F:
                raise ValueError("FFN event loop only accepts completed A2F transfers.")
            self.scheduler.push(QueueItem(handle.tensor, handle.meta, now))

        layer = self.scheduler.pick_ready_layer(now)
        if layer is None:
            return False

        batch = self.scheduler.drain(layer)
        hidden = torch.cat([item.hidden for item in batch], dim=0)
        num_real = hidden.shape[0]

        if self.capture_sizes is None:
            out = self.replay_fn(layer, hidden)
        else:
            bucket = pick_bucket(num_real, self.capture_sizes)
            out = self.replay_fn(layer, pad_to_bucket(hidden, bucket))
            out = drop_padding(out, num_real)

        offset = 0
        for item in batch:
            n = item.num_tokens
            # Return the MoE result to the originating layer as an F2A transfer
            # (the combine step, design section 8.1); addressing is preserved
            # per item so a batch mixing arrivals routes each back correctly.
            ret = AFDMeta(item.meta.layer_id, STAGE_F2A, item.meta.transfer_id)
            send_handle = self.sink.isend(out[offset : offset + n], ret)
            if send_handle is not None and not _handle_complete(send_handle):
                self._send_handles.append(send_handle)
            offset += n
        return True


class AFDAttentionEventLoop:
    """Drives one intact Attention scheduler batch layer by layer.

    New requests enter the first AFD layer through :meth:`submit`. Completed
    F2A transfers advance the same batch to the next AFD layer before another
    submitted batch may run. A completion from the final layer leaves through
    ``completion_fn``. Phase 0 permits only one A2F/F2A round trip at a time.

    ``replay_fn`` executes the local Attention-side segment for one layer and
    must rebuild positions, slot mappings, and backend metadata for the selected
    batch before replaying a graph. Keeping that role-specific work behind the
    callback lets the queue and flow-control logic remain backend independent.
    """

    def __init__(
        self,
        scheduler: AFDDynamicBatchScheduler,
        source: AFDPollSource,
        sink: AFDSendSink,
        replay_fn: AttentionReplayFn,
        completion_fn: CompletionFn,
        capture_sizes: list[int] | None,
        layer_ids: list[int],
        transfer_ids: AFDTransferIdAllocator,
        credit_capacity: int,
    ) -> None:
        if capture_sizes is not None and not capture_sizes:
            raise ValueError("capture_sizes cannot be empty when padding is enabled.")
        if not layer_ids:
            raise ValueError("layer_ids must be non-empty.")
        if layer_ids != sorted(set(layer_ids)):
            raise ValueError("layer_ids must be sorted and unique.")
        if layer_ids[-1] >= len(scheduler.queues):
            raise ValueError("scheduler has no queue for the final layer id.")
        if credit_capacity != 1:
            raise ValueError("Phase 0 Attention scheduling requires credit_capacity=1.")
        self.scheduler = scheduler
        self.source = source
        self.sink = sink
        self.replay_fn = replay_fn
        self.completion_fn = completion_fn
        self.capture_sizes = capture_sizes
        self.layer_ids = layer_ids
        self.transfer_ids = transfer_ids
        self.credits = AFDCreditTracker(credit_capacity)
        self._in_flight_batches: dict[int, list[QueueItem]] = {}
        self._send_handles: dict[int, AFDHandle] = {}
        self._continuation_layer: int | None = None
        self._next_layer = {
            layer: layer_ids[index + 1] if index + 1 < len(layer_ids) else None
            for index, layer in enumerate(layer_ids)
        }

    def submit(
        self,
        hidden: torch.Tensor,
        now: float,
        layer_id: int | None = None,
        context: object | None = None,
    ) -> None:
        """Add newly admitted Attention work to its first executable layer."""
        target = self.layer_ids[0] if layer_id is None else layer_id
        if target not in self._next_layer:
            raise ValueError(f"layer {target} is not an Attention AFD layer.")
        meta = AFDMeta(target, STAGE_F2A)
        self.scheduler.push(QueueItem(hidden, meta, now, context))

    def _validate_f2a_completion(
        self,
        handle: AFDHandle,
    ) -> tuple[QueueItem, int | None]:
        if handle.meta.stage != STAGE_F2A:
            raise ValueError(
                "Attention event loop only accepts completed F2A transfers."
            )

        batch = self._in_flight_batches.get(handle.meta.transfer_id)
        if batch is None:
            raise ValueError(
                f"no local work for transfer id {handle.meta.transfer_id}."
            )
        if len(batch) != 1:
            raise RuntimeError(
                "Phase 0 Attention completion requires exactly one queued item."
            )
        item = batch[0]
        source_layer = item.meta.layer_id
        layer_id = handle.meta.layer_id
        if layer_id != source_layer:
            raise ValueError(
                f"F2A transfer {handle.meta.transfer_id} returned layer "
                f"{layer_id}, expected {source_layer}."
            )
        if layer_id not in self._next_layer:
            raise ValueError(f"unknown completed layer {layer_id}.")

        expected_tokens = item.num_tokens
        if handle.tensor.shape[0] != expected_tokens:
            raise ValueError(
                f"F2A transfer {handle.meta.transfer_id} returned "
                f"{handle.tensor.shape[0]} tokens, expected {expected_tokens}."
            )
        return item, self._next_layer[layer_id]

    def _collect_completions(self, now: float) -> None:
        for handle in self.source.poll():
            item, next_layer = self._validate_f2a_completion(handle)
            transfer_id = handle.meta.transfer_id
            self.credits.release(transfer_id)
            self._send_handles.pop(transfer_id, None)
            self._in_flight_batches.pop(transfer_id)

            hidden = handle.tensor
            if next_layer is None:
                if self.capture_sizes is not None:
                    bucket = pick_bucket(item.num_tokens, self.capture_sizes)
                    hidden = pad_to_bucket(hidden, bucket)
                self.completion_fn(hidden, handle.meta, item.context)
            else:
                meta = AFDMeta(next_layer, STAGE_F2A, transfer_id)
                self.scheduler.push(QueueItem(hidden, meta, now, item.context))
            self._continuation_layer = next_layer

    def tick(self, now: float) -> bool:
        """Advance one intact batch by at most one Attention/FFN layer."""
        self._collect_completions(now)
        if self.credits.in_flight:
            return False

        layer = self._continuation_layer
        if layer is None:
            layer = self.scheduler.pick_ready_layer(now)
        if layer is None:
            return False

        batch = self.scheduler.drain(layer, max_items=1)
        if not batch:
            return False
        self._continuation_layer = None

        hidden = torch.cat([item.hidden for item in batch], dim=0)
        num_real = hidden.shape[0]
        if self.capture_sizes is None:
            out = self.replay_fn(layer, hidden, batch)
        else:
            bucket = pick_bucket(num_real, self.capture_sizes)
            out = self.replay_fn(layer, pad_to_bucket(hidden, bucket), batch)
            out = drop_padding(out, num_real)

        transfer_id = self.transfer_ids.next()
        self.credits.acquire(transfer_id)
        self._in_flight_batches[transfer_id] = batch
        meta = AFDMeta(layer, STAGE_A2F, transfer_id)
        try:
            handle = self.sink.isend(out, meta)
            if handle is not None and not _handle_complete(handle):
                self._send_handles[transfer_id] = handle
        except Exception:
            self._in_flight_batches.pop(transfer_id)
            self._send_handles.pop(transfer_id, None)
            self.credits.release(transfer_id)
            raise
        return True
