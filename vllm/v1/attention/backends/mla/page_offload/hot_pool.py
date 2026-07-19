# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU hot-pool residency policy for sparse selected pages."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from vllm.v1.attention.backends.mla.page_offload.selection import LogicalPage


@dataclass(frozen=True)
class SparsePageStagingPlan:
    """Slot assignments and pages that need data movement."""

    slot_by_page: Mapping[LogicalPage, int]
    miss_pages: tuple[LogicalPage, ...]
    tail_pages_to_refresh: tuple[LogicalPage, ...]

    @property
    def pages_requiring_copy(self) -> tuple[LogicalPage, ...]:
        return (*self.miss_pages, *self.tail_pages_to_refresh)


@dataclass
class SparsePageHotPool:
    """Layer-local hot-pool tensor and LRU residency state."""

    kv_cache: torch.Tensor
    resident_slots: OrderedDict[LogicalPage, int]
    resident_slot_capacity: int
    slots_per_request: int
    tail_slots: dict[LogicalPage, int]
    tail_slot_capacity: int

    @classmethod
    def create(
        cls,
        source_kv_cache: torch.Tensor,
        *,
        resident_slot_capacity: int,
        tail_slot_capacity: int,
        slots_per_request: int,
        use_source_tensor: bool = False,
    ) -> SparsePageHotPool:
        num_slots = resident_slot_capacity + tail_slot_capacity
        if use_source_tensor:
            if source_kv_cache.shape[0] != num_slots:
                raise RuntimeError(
                    "Sparse c4a partial allocation has the wrong number of "
                    f"blocks: got={source_kv_cache.shape[0]} expected={num_slots}."
                )
            pool_tensor = source_kv_cache
        else:
            pool_tensor = torch.empty(
                (num_slots, *source_kv_cache.shape[1:]),
                dtype=source_kv_cache.dtype,
                device=source_kv_cache.device,
            )
        return cls(
            kv_cache=pool_tensor,
            resident_slots=OrderedDict(),
            resident_slot_capacity=resident_slot_capacity,
            slots_per_request=slots_per_request,
            tail_slots={},
            tail_slot_capacity=tail_slot_capacity,
        )

    def is_compatible_with(
        self,
        source_kv_cache: torch.Tensor,
        *,
        tail_slot_capacity: int,
    ) -> bool:
        expected_shape = (
            self.resident_slot_capacity + tail_slot_capacity,
            *source_kv_cache.shape[1:],
        )
        return (
            self.kv_cache.shape == expected_shape
            and self.kv_cache.dtype == source_kv_cache.dtype
            and self.kv_cache.device == source_kv_cache.device
            and self.tail_slot_capacity == tail_slot_capacity
        )

    def build_staging_plan(
        self,
        hot_pages: tuple[LogicalPage, ...],
        tail_pages: tuple[LogicalPage, ...],
    ) -> SparsePageStagingPlan:
        protected_pages = frozenset(hot_pages)
        slot_by_page: dict[LogicalPage, int] = {}
        miss_pages: list[LogicalPage] = []

        for page in hot_pages:
            slot = self.resident_slots.get(page)
            if slot is None:
                slot = self._allocate_resident_slot(
                    protected_pages,
                    request_id=page.request_id,
                )
                self.resident_slots[page] = slot
                miss_pages.append(page)
            else:
                self.resident_slots.move_to_end(page)
            slot_by_page[page] = slot

        for page in tail_pages:
            slot_by_page[page] = self.reserve_tail_slot(page)

        return SparsePageStagingPlan(
            slot_by_page=slot_by_page,
            miss_pages=tuple(miss_pages),
            tail_pages_to_refresh=tail_pages,
        )

    def reserve_tail_slot(self, page: LogicalPage) -> int:
        slot = self.tail_slots.get(page)
        if slot is not None:
            return slot
        occupied = set(self.tail_slots.values())
        for tail_slot in range(self.tail_slot_capacity):
            slot = self.resident_slot_capacity + tail_slot
            if slot not in occupied:
                self.tail_slots[page] = slot
                return slot
        raise RuntimeError("No sparse c4a mutable-tail slot is available.")

    def advance_tail(self, previous: LogicalPage, current: LogicalPage) -> int:
        slot = self.tail_slots.pop(previous, None)
        if slot is None:
            raise KeyError(f"Sparse mutable tail has no GPU slot: {previous!r}.")
        if current in self.tail_slots:
            raise ValueError(f"Sparse mutable tail already exists: {current!r}.")
        self.tail_slots[current] = slot
        return slot

    def cleanup_request(self, request_id: int | str) -> None:
        for page in tuple(self.resident_slots):
            if page.request_id == request_id:
                del self.resident_slots[page]
        for page in tuple(self.tail_slots):
            if page.request_id == request_id:
                del self.tail_slots[page]

    def _allocate_resident_slot(
        self,
        protected_pages: frozenset[LogicalPage],
        *,
        request_id: int | str,
    ) -> int:
        request_resident_slots = tuple(
            (page, slot)
            for page, slot in self.resident_slots.items()
            if page.request_id == request_id
        )
        if len(request_resident_slots) >= self.slots_per_request:
            for page, slot in request_resident_slots:
                if page in protected_pages:
                    continue
                del self.resident_slots[page]
                return slot

        occupied_slots = set(self.resident_slots.values())
        for slot in range(self.resident_slot_capacity):
            if slot not in occupied_slots:
                return slot

        for page, slot in tuple(self.resident_slots.items()):
            if page in protected_pages:
                continue
            del self.resident_slots[page]
            return slot
        raise RuntimeError("No evictable sparse page hot-pool slot is available.")


__all__ = ["SparsePageHotPool", "SparsePageStagingPlan"]
