# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metrics for sparse selected-page staging."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass

from vllm.v1.attention.backends.mla.page_offload.selection import (
    LogicalPage,
    SparsePageSelection,
)


@dataclass(frozen=True)
class SparsePageSelectionStats:
    """Selected-page statistics for one sparse layer staging call."""

    layer_name: str
    observation_index: int
    decode_step: int
    num_unique_pages: int
    num_tail_pages: int
    same_layer_cross_decode_step_miss_pages: int
    same_layer_cross_decode_step_estimated_bytes: int
    reference_layer_name: str | None = None
    same_decode_step_reference_missing_pages: int = 0
    same_decode_step_reference_missing_estimated_bytes: int = 0
    num_same_decode_step_reference_reused_pages: int = 0


class SparsePageSelectionCollector:
    """Collect reuse signals from sparse page staging calls."""

    def __init__(
        self,
        hot_page_capacity: int,
        page_size_bytes: int,
    ) -> None:
        self.hot_page_capacity = hot_page_capacity
        self.page_size_bytes = page_size_bytes
        self._observation_index = 0
        self._decode_step = -1
        self._per_layer_hot_pages: dict[str, OrderedDict[LogicalPage, None]] = (
            defaultdict(OrderedDict)
        )
        self._previous_layer_idx: int | None = None
        self._decode_step_reference_layer_name: str | None = None
        self._decode_step_reference_pages: frozenset[tuple[int | str, int]] = (
            frozenset()
        )
        self.records: list[SparsePageSelectionStats] = []

    def record(
        self,
        layer_name: str,
        selection: SparsePageSelection,
    ) -> SparsePageSelectionStats:
        self._observation_index += 1
        non_tail_pages = [
            page for page in selection.unique_pages if page not in selection.tail_pages
        ]
        layer_agnostic_pages = [
            self._to_layer_agnostic_key(page) for page in non_tail_pages
        ]

        # Same-layer cross-decode-step miss(H): reuse from previous decode steps.
        cross_decode_step_miss_pages = self._simulate_hot_pool(
            layer_name,
            non_tail_pages,
        )
        cross_decode_step_estimated_bytes = (
            cross_decode_step_miss_pages * self.page_size_bytes
        )

        current_layer_idx = self._extract_layer_index(layer_name)
        # A non-monotonic layer id starts a new decode sweep. The first observed
        # sparse layer in that sweep becomes the reference for same-step reuse.
        if self._starts_decode_step(current_layer_idx):
            self._decode_step += 1
            self._decode_step_reference_layer_name = layer_name
            self._decode_step_reference_pages = frozenset(layer_agnostic_pages)

        # Same-step reference reuse: compare request/page ids while ignoring the
        # layer component of LogicalPage.
        selected_pages = frozenset(layer_agnostic_pages)
        reference_reused_pages = selected_pages & self._decode_step_reference_pages
        reference_missing_pages = selected_pages - self._decode_step_reference_pages
        stats = SparsePageSelectionStats(
            layer_name=layer_name,
            observation_index=self._observation_index,
            decode_step=self._decode_step,
            num_unique_pages=len(selection.unique_pages),
            num_tail_pages=len(selection.tail_pages),
            same_layer_cross_decode_step_miss_pages=cross_decode_step_miss_pages,
            same_layer_cross_decode_step_estimated_bytes=(
                cross_decode_step_estimated_bytes
            ),
            reference_layer_name=self._decode_step_reference_layer_name,
            same_decode_step_reference_missing_pages=len(reference_missing_pages),
            same_decode_step_reference_missing_estimated_bytes=(
                len(reference_missing_pages) * self.page_size_bytes
            ),
            num_same_decode_step_reference_reused_pages=len(reference_reused_pages),
        )
        self.records.append(stats)
        self._previous_layer_idx = current_layer_idx
        return stats

    def _simulate_hot_pool(
        self,
        layer_name: str,
        candidate_pages: list[LogicalPage],
    ) -> int:
        hot_page_capacity = self.hot_page_capacity
        if hot_page_capacity == 0:
            return len(candidate_pages)

        hot_pages = self._per_layer_hot_pages[layer_name]
        miss_count = 0
        for page in candidate_pages:
            if page in hot_pages:
                hot_pages.move_to_end(page)
                continue
            miss_count += 1
            hot_pages[page] = None
            while len(hot_pages) > hot_page_capacity:
                hot_pages.popitem(last=False)
        return miss_count

    @staticmethod
    def _to_layer_agnostic_key(page: LogicalPage) -> tuple[int | str, int]:
        return (page.request_id, page.page_idx)

    @staticmethod
    def _extract_layer_index(layer_name: str) -> int | None:
        int_vals: list[int] = []
        for part in layer_name.split("."):
            try:
                int_vals.append(int(part))
            except ValueError:
                continue
        if len(int_vals) != 1:
            return None
        return int_vals[0]

    def _starts_decode_step(self, current_layer_idx: int | None) -> bool:
        if self._decode_step_reference_layer_name is None:
            return True
        return (
            current_layer_idx is not None
            and self._previous_layer_idx is not None
            and current_layer_idx <= self._previous_layer_idx
        )

    @property
    def latest(self) -> SparsePageSelectionStats | None:
        return self.records[-1] if self.records else None
