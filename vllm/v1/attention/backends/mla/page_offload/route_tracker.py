# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request route lifecycle for sparse selected-page offload."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TypeVar

ProducerRequestKey = tuple[str, str]
_MAX_GENERATION_HISTORY = 65536
_KeyT = TypeVar("_KeyT", bound=Hashable)


class SparsePageRouteTracker:
    """Track producer generations and consumer route ownership."""

    def __init__(self) -> None:
        self._last_producer_generation: dict[str, int] = {}
        self._active_producer_generation: dict[str, int] = {}
        self._consumer_binding_by_request: dict[
            str, tuple[ProducerRequestKey, int]
        ] = {}
        self._active_generation_by_producer_request: dict[ProducerRequestKey, int] = {}
        self._latest_generation_by_producer_request: dict[ProducerRequestKey, int] = {}

    def validate_new_producer_request(self, request_id: str) -> None:
        if request_id in self._active_producer_generation:
            raise ValueError(
                f"Sparse page request id is already active on producer: {request_id!r}."
            )

    def begin_producer_request(self, request_id: str) -> int:
        self.validate_new_producer_request(request_id)
        generation = self._last_producer_generation.get(request_id, 0) + 1
        self._record(
            self._last_producer_generation,
            request_id,
            generation,
        )
        self._active_producer_generation[request_id] = generation
        return generation

    def get_or_create_producer_generation(self, request_id: str) -> int:
        generation = self._active_producer_generation.get(request_id)
        if generation is not None:
            return generation
        generation = self._last_producer_generation.get(request_id, 0) + 1
        self._record(
            self._last_producer_generation,
            request_id,
            generation,
        )
        return generation

    def validate_new_consumer_request(self, request_id: str) -> None:
        if request_id in self._consumer_binding_by_request:
            raise ValueError(
                f"Sparse page request id is already active on consumer: {request_id!r}."
            )

    def bind_consumer_request(
        self,
        request_id: str,
        producer_request: ProducerRequestKey,
        generation: int,
    ) -> None:
        self.validate_new_consumer_request(request_id)
        active_generation = self._active_generation_by_producer_request.get(
            producer_request
        )
        if active_generation is not None:
            raise ValueError(
                "Sparse page producer request is already active on this DP "
                f"replica: route={producer_request}, "
                f"generation={active_generation}."
            )
        latest_generation = self._latest_generation_by_producer_request.get(
            producer_request, 0
        )
        if generation <= latest_generation:
            raise ValueError(
                "Stale sparse page request generation: "
                f"route={producer_request}, generation={generation}, "
                f"latest={latest_generation}."
            )
        self._consumer_binding_by_request[request_id] = (
            producer_request,
            generation,
        )
        self._active_generation_by_producer_request[producer_request] = generation

    def discard_consumer_request(self, request_id: str) -> None:
        consumer_binding = self._consumer_binding_by_request.pop(request_id, None)
        if consumer_binding is not None:
            self._active_generation_by_producer_request.pop(consumer_binding[0], None)

    def finish_request(self, request_id: str) -> None:
        self._active_producer_generation.pop(request_id, None)
        consumer_binding = self._consumer_binding_by_request.pop(request_id, None)
        if consumer_binding is None:
            return
        producer_request, generation = consumer_binding
        self._active_generation_by_producer_request.pop(producer_request, None)
        self._record(
            self._latest_generation_by_producer_request,
            producer_request,
            max(
                generation,
                self._latest_generation_by_producer_request.get(producer_request, 0),
            ),
        )

    @staticmethod
    def _record(
        history: dict[_KeyT, int],
        key: _KeyT,
        generation: int,
    ) -> None:
        history[key] = generation
        if len(history) > _MAX_GENERATION_HISTORY:
            del history[next(iter(history))]


__all__ = ["SparsePageRouteTracker"]
