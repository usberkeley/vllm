# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 c4a sparse page adapter."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.page_offload.config import (
    SparsePageOffloadConfig,
    extract_sparse_page_layer_index,
)
from vllm.v1.attention.backends.mla.page_offload.selection import (
    LogicalPage,
    SelectedPage,
    SparsePageAdapter,
    SparsePageSelection,
)

logger = init_logger(__name__)


class DeepSeekV4C4AAdapter(SparsePageAdapter):
    """Extract selected c4a compressed pages from DeepSeek V4 sparse top-k."""

    compress_ratio: int = 4
    storage_block_size: int = 64
    entry_size_bytes: int = 584
    page_size_bytes: int = entry_size_bytes * storage_block_size

    def __init__(
        self,
        hf_config: Any,
        sparse_page_config: SparsePageOffloadConfig,
    ) -> None:
        self.hf_config = hf_config
        self.config = sparse_page_config
        self.model_type = getattr(hf_config, "model_type", None)
        self.compress_ratios = tuple(getattr(hf_config, "compress_ratios", ()))

    def supports_layer(
        self,
        layer_name: str,
        kv_cache_dtype: str,
    ) -> bool:
        reason = self.support_failure_reason(layer_name, kv_cache_dtype)
        if reason is not None:
            logger.info_once(
                "Sparse page offload skipped layer=%s: %s",
                layer_name,
                reason,
            )
            return False
        return True

    def support_failure_reason(
        self,
        layer_name: str,
        kv_cache_dtype: str,
    ) -> str | None:
        if not self.config.enabled:
            return "sparse page config is disabled"
        if not self.config.includes_layer(layer_name):
            return "layer is excluded by sparse_page_offload_layers"
        if self.model_type != "deepseek_v4":
            return f"model_type is {self.model_type!r}, expected 'deepseek_v4'"
        if kv_cache_dtype != "fp8_ds_mla":
            return f"kv_cache_dtype is {kv_cache_dtype!r}, expected 'fp8_ds_mla'"
        layer_idx = extract_sparse_page_layer_index(layer_name)
        if layer_idx is None:
            return "could not extract layer index from layer_name"
        if not 0 <= layer_idx < len(self.compress_ratios):
            return (
                f"layer index {layer_idx} is outside compress_ratios length "
                f"{len(self.compress_ratios)}"
            )
        compress_ratio = int(self.compress_ratios[layer_idx])
        if compress_ratio != self.compress_ratio:
            return f"compress_ratio is {compress_ratio}, expected {self.compress_ratio}"
        return None

    def extract_selection(
        self,
        *,
        layer_name: str,
        req_id_per_token: torch.Tensor,
        topk_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        request_identities: Mapping[int, tuple[int | str, int]] | None = None,
    ) -> SparsePageSelection:
        request_rows = req_id_per_token.detach().to("cpu", dtype=torch.int64)
        topk_indices_cpu = topk_indices.detach().to("cpu", dtype=torch.int64)
        sequence_lengths_cpu = seq_lens.detach().to("cpu", dtype=torch.int64)

        token_rows_by_page: dict[LogicalPage, set[int]] = defaultdict(set)
        request_row_by_page: dict[LogicalPage, int] = {}
        tail_pages: set[LogicalPage] = set()
        current_tail_pages: list[SelectedPage] = []

        def request_identity(request_row: int) -> tuple[int | str, int]:
            if request_identities is None:
                return request_row, 0
            return request_identities.get(request_row, (request_row, 0))

        for request_row, sequence_length in enumerate(sequence_lengths_cpu.tolist()):
            if sequence_length <= 0:
                continue
            compressed_len = (int(sequence_length) + self.compress_ratio - 1) // (
                self.compress_ratio
            )
            if compressed_len == 0:
                continue
            tail_page_idx = (compressed_len - 1) // self.storage_block_size
            request_id, generation = request_identity(request_row)
            tail_page = LogicalPage(
                request_id=request_id,
                layer_name=layer_name,
                page_idx=tail_page_idx,
                generation=generation,
            )
            tail_pages.add(tail_page)
            current_tail_pages.append(
                SelectedPage(
                    logical_page=tail_page,
                    request_row=request_row,
                    is_tail=True,
                )
            )

        for token_row, request_row in enumerate(request_rows.tolist()):
            if request_row < 0:
                continue
            valid_tokens = topk_indices_cpu[token_row][topk_indices_cpu[token_row] >= 0]
            request_id, generation = request_identity(int(request_row))
            for page_idx in torch.unique(valid_tokens // self.storage_block_size):
                logical_page = LogicalPage(
                    request_id=request_id,
                    layer_name=layer_name,
                    page_idx=int(page_idx.item()),
                    generation=generation,
                )
                token_rows_by_page[logical_page].add(token_row)
                request_row_by_page[logical_page] = int(request_row)

        unique_pages = tuple(sorted(token_rows_by_page))
        selected_pages = tuple(
            SelectedPage(
                logical_page=page,
                request_row=request_row_by_page[page],
                token_rows=tuple(sorted(token_rows_by_page[page])),
                is_tail=page in tail_pages,
            )
            for page in unique_pages
        )

        return SparsePageSelection(
            selected_pages=selected_pages,
            unique_pages=unique_pages,
            tail_pages=frozenset(page for page in unique_pages if page in tail_pages),
            current_tail_pages=tuple(current_tail_pages),
        )
