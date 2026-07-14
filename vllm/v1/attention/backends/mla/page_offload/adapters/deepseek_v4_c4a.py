# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 c4a sparse page adapter."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from vllm.v1.attention.backends.mla.page_offload.config import (
    SparsePageOffloadConfig,
    extract_sparse_page_layer_index,
)
from vllm.v1.attention.backends.mla.page_offload.selected_pages import (
    LogicalPage,
    SelectedPage,
    SparsePageAdapter,
    SparsePageSelection,
)


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
        if not self.config.enabled or not self.config.includes_layer(layer_name):
            return False
        if self.model_type != "deepseek_v4" or kv_cache_dtype != "fp8_ds_mla":
            return False
        layer_idx = extract_sparse_page_layer_index(layer_name)
        if layer_idx is None:
            return False
        if not 0 <= layer_idx < len(self.compress_ratios):
            return False
        return int(self.compress_ratios[layer_idx]) == self.compress_ratio

    def extract_selection(
        self,
        *,
        layer_name: str,
        req_id_per_token: torch.Tensor,
        topk_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> SparsePageSelection:
        req_rows = req_id_per_token.detach().to("cpu", dtype=torch.int64)
        topk = topk_indices.detach().to("cpu", dtype=torch.int64)
        seq_lens_cpu = seq_lens.detach().to("cpu", dtype=torch.int64)

        page_to_token_rows: dict[LogicalPage, set[int]] = defaultdict(set)
        page_to_req_row: dict[LogicalPage, int] = {}
        tail_pages: set[LogicalPage] = set()

        for req_row, seq_len in enumerate(seq_lens_cpu.tolist()):
            if seq_len <= 0:
                continue
            compressed_len = (int(seq_len) + self.compress_ratio - 1) // (
                self.compress_ratio
            )
            if compressed_len == 0:
                continue
            tail_page_idx = (compressed_len - 1) // self.storage_block_size
            tail_pages.add(
                LogicalPage(
                    request_id=req_row,
                    layer_name=layer_name,
                    page_idx=tail_page_idx,
                )
            )

        for token_row, req_row in enumerate(req_rows.tolist()):
            if req_row < 0:
                continue
            valid_tokens = topk[token_row][topk[token_row] >= 0]
            for page_idx in torch.unique(valid_tokens // self.storage_block_size):
                logical = LogicalPage(
                    request_id=int(req_row),
                    layer_name=layer_name,
                    page_idx=int(page_idx.item()),
                )
                page_to_token_rows[logical].add(token_row)
                page_to_req_row[logical] = int(req_row)

        unique_pages = tuple(sorted(page_to_token_rows))
        selected_pages = tuple(
            SelectedPage(
                logical=page,
                req_row=page_to_req_row[page],
                token_rows=tuple(sorted(page_to_token_rows[page])),
                is_tail=page in tail_pages,
            )
            for page in unique_pages
        )

        return SparsePageSelection(
            selected_pages=selected_pages,
            unique_pages=unique_pages,
            tail_pages=frozenset(page for page in unique_pages if page in tail_pages),
        )
