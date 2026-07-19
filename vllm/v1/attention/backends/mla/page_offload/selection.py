# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Selected-page data structures and adapter contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

import torch


@dataclass(frozen=True, order=True)
class LogicalPage:
    """A compressed sparse MLA page scoped to a request row and layer."""

    request_id: int | str
    layer_name: str
    page_idx: int
    generation: int = 0


@dataclass(frozen=True)
class SelectedPage:
    """A logical page selected by a layer's sparse top-k rows."""

    logical_page: LogicalPage
    request_row: int
    token_rows: tuple[int, ...] = ()
    is_tail: bool = False


@dataclass(frozen=True)
class SparsePageSelection:
    """Deduplicated selected pages for one sparse MLA layer invocation."""

    selected_pages: tuple[SelectedPage, ...]
    unique_pages: tuple[LogicalPage, ...]
    tail_pages: frozenset[LogicalPage]
    miss_pages: tuple[LogicalPage, ...] = ()
    current_tail_pages: tuple[SelectedPage, ...] = ()


class SparsePageAdapter(ABC):
    """Model-specific sparse page extraction."""

    page_size_bytes: int
    storage_block_size: int
    compress_ratio: int

    @abstractmethod
    def supports_layer(
        self,
        layer_name: str,
        kv_cache_dtype: str,
    ) -> bool:
        raise NotImplementedError

    @abstractmethod
    def extract_selection(
        self,
        *,
        layer_name: str,
        req_id_per_token: torch.Tensor,
        topk_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        request_identities: Mapping[int, tuple[int | str, int]] | None = None,
    ) -> SparsePageSelection:
        raise NotImplementedError
