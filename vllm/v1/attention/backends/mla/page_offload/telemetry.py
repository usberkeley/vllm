# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Observe-only telemetry for sparse selected pages."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass

from vllm.v1.attention.backends.mla.page_offload.selected_pages import (
    LogicalPage,
    SparsePageSelection,
)


@dataclass(frozen=True)
class SparseSelectionStats:
    """Per-layer selected-page statistics for one observe-only step."""

    layer_name: str
    step: int
    num_unique_pages: int
    num_tail_pages: int
    simulated_miss_pages: dict[int, int]
    estimated_bytes: dict[int, int]


class SparseSelectionCollector:
    """Simulate per-layer hot pools without changing model execution."""

    def __init__(
        self,
        hot_pool_blocks: tuple[int, ...],
        page_size_bytes: int,
    ) -> None:
        self.hot_pool_blocks = tuple(sorted(set(hot_pool_blocks)))
        self.page_size_bytes = page_size_bytes
        self._step = 0
        self._hot_pages: dict[
            tuple[str, int], OrderedDict[LogicalPage, None]
        ] = defaultdict(OrderedDict)
        self.records: list[SparseSelectionStats] = []

    def record(
        self,
        layer_name: str,
        selection: SparsePageSelection,
    ) -> SparseSelectionStats:
        self._step += 1
        candidates = [
            page for page in selection.unique_pages if page not in selection.tail_pages
        ]
        simulated_miss_pages: dict[int, int] = {}
        estimated_bytes: dict[int, int] = {}

        for hot_pool_blocks in self.hot_pool_blocks:
            miss_pages = self._simulate_hot_pool(
                layer_name,
                hot_pool_blocks,
                candidates,
            )
            simulated_miss_pages[hot_pool_blocks] = miss_pages
            estimated_bytes[hot_pool_blocks] = miss_pages * self.page_size_bytes

        stats = SparseSelectionStats(
            layer_name=layer_name,
            step=self._step,
            num_unique_pages=len(selection.unique_pages),
            num_tail_pages=len(selection.tail_pages),
            simulated_miss_pages=simulated_miss_pages,
            estimated_bytes=estimated_bytes,
        )
        self.records.append(stats)
        return stats

    def _simulate_hot_pool(
        self,
        layer_name: str,
        hot_pool_blocks: int,
        candidates: list[LogicalPage],
    ) -> int:
        if hot_pool_blocks == 0:
            return len(candidates)

        hot_pages = self._hot_pages[(layer_name, hot_pool_blocks)]
        misses = 0
        for page in candidates:
            if page in hot_pages:
                hot_pages.move_to_end(page)
                continue
            misses += 1
            hot_pages[page] = None
            while len(hot_pages) > hot_pool_blocks:
                hot_pages.popitem(last=False)
        return misses

    @property
    def latest(self) -> SparseSelectionStats | None:
        return self.records[-1] if self.records else None
