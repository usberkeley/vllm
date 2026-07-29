# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-layer out-of-order dynamic-batch scheduler for both AFD roles.

Both pools keep one queue per executable layer and may choose any ready layer,
while preserving each request's A0->F0->A1->F1 dependency chain. Attention uses
a role-specific adapter to rebuild KV metadata for the selected items; this
module only owns role-independent queueing, fairness, and batching policy.

This module is the pure policy + data-structure core (no CUDA, no transport): the
event loop (see ``worker_loop``) drives it. The two properties that matter:

- **Empty-batch safety.** ``pick_ready_layer`` returns ``None`` when nothing is
  ready, so the caller never launches a ``(0, H)`` batch. Combined with
  ``pad_to_bucket`` this removes the ``gridDim=0`` crashes that sank prior
  attempts -- by construction, not as a runtime patch.
- **No starvation.** Scoring folds in queue-head wait time, so a cold layer
  cannot be starved by a hot one (design section 6.2 aging).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import torch

from vllm.distributed.afd_transfer.connector.base import AFDMeta


@dataclass
class QueueItem:
    """One queued unit of AFD work: the activations of a single arrival.

    Attributes:
        hidden: The normed hidden to feed the layer's MoE, shape ``(n, H)``.
        meta: Transfer metadata (layer id, stage, transfer id) used to route the
            result back to the originating attention rank.
        enqueue_ts: Monotonic timestamp when the item was enqueued, used for
            anti-starvation aging.
        context: Role-specific request/attention metadata kept local while the
            activation is in flight.
    """

    hidden: torch.Tensor
    meta: AFDMeta
    enqueue_ts: float
    context: Any = None

    @property
    def num_tokens(self) -> int:
        return self.hidden.shape[0]


class LayerQueue:
    """FIFO queue of pending work for a single model layer."""

    def __init__(self) -> None:
        self._items: deque[QueueItem] = deque()
        self._num_tokens = 0

    def __len__(self) -> int:
        return len(self._items)

    @property
    def size(self) -> int:
        """Total number of tokens currently queued for this layer."""
        return self._num_tokens

    def push(self, item: QueueItem) -> None:
        self._items.append(item)
        self._num_tokens += item.num_tokens

    def head_wait(self, now: float) -> float:
        """Seconds the oldest queued item has waited (0 when empty)."""
        if not self._items:
            return 0.0
        return now - self._items[0].enqueue_ts

    def drain(self, max_items: int | None = None) -> list[QueueItem]:
        """Pop whole scheduler batches without splitting their token ranges."""
        batch: list[QueueItem] = []
        while self._items and (max_items is None or len(batch) < max_items):
            item = self._items.popleft()
            self._num_tokens -= item.num_tokens
            batch.append(item)
        return batch


class AFDDynamicBatchScheduler:
    """Out-of-order, per-layer dynamic-batch scheduler for either AFD role.

    Args:
        num_layers: Number of MoE layers (one queue each).
        age_limit_s: A queue whose head waited this long is prioritized over
            fuller queues (anti-starvation).
    """

    def __init__(
        self,
        num_layers: int,
        age_limit_s: float,
    ) -> None:
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")
        if age_limit_s < 0:
            raise ValueError("age_limit_s must be non-negative.")
        self.queues = [LayerQueue() for _ in range(num_layers)]
        self.age_limit_s = age_limit_s

    def push(self, item: QueueItem) -> None:
        if item.meta.layer_id >= len(self.queues):
            raise ValueError(
                f"layer {item.meta.layer_id} exceeds scheduler queue range."
            )
        self.queues[item.meta.layer_id].push(item)

    @property
    def pending(self) -> int:
        """Total tokens queued across all layers."""
        return sum(q.size for q in self.queues)

    def pick_ready_layer(self, now: float) -> int | None:
        """Return the layer to run next, or ``None`` if nothing is ready.

        Every non-empty queue is ready. An aged head wins first, then the
        largest queue; ties resolve to the lowest layer id.
        """
        ready = [i for i, q in enumerate(self.queues) if q.size]
        if not ready:
            return None

        def score(i: int) -> tuple[bool, int]:
            q = self.queues[i]
            aged = q.head_wait(now) >= self.age_limit_s
            return (aged, q.size)

        return max(ready, key=score)

    def drain(
        self,
        layer: int,
        *,
        max_items: int | None = None,
    ) -> list[QueueItem]:
        """Drain whole arrivals from one layer without token-level slicing."""
        return self.queues[layer].drain(max_items)


def pick_bucket(num_tokens: int, capture_sizes: list[int]) -> int:
    """Smallest captured cudagraph size that fits ``num_tokens``.

    ``capture_sizes`` need not be sorted. Raises if ``num_tokens`` exceeds the
    largest captured size -- silent truncation would drop tokens.
    """
    fits = [s for s in capture_sizes if s >= num_tokens]
    if not fits:
        raise ValueError(
            f"{num_tokens} tokens exceed the largest capture size "
            f"{max(capture_sizes) if capture_sizes else None}."
        )
    return min(fits)


def pad_to_bucket(hidden: torch.Tensor, bucket: int) -> torch.Tensor:
    """Zero-pad ``hidden`` from ``(n, ...)`` to ``(bucket, ...)`` along dim 0."""
    n = hidden.shape[0]
    if n > bucket:
        raise ValueError(f"cannot pad {n} tokens down to bucket {bucket}.")
    if n == bucket:
        return hidden
    pad = hidden.new_zeros((bucket - n, *hidden.shape[1:]))
    return torch.cat([hidden, pad], dim=0)


def drop_padding(padded: torch.Tensor, num_real: int) -> torch.Tensor:
    """Slice the real tokens back out of a bucket-padded tensor."""
    if num_real > padded.shape[0]:
        raise ValueError(
            f"num_real {num_real} exceeds padded length {padded.shape[0]}."
        )
    return padded[:num_real]
