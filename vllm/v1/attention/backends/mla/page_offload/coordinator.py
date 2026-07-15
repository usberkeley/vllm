# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Coordinator for sparse MLA selected-page offload."""

from __future__ import annotations

from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.page_offload.adapters.deepseek_v4_c4a import (
    DeepSeekV4C4AAdapter,
)
from vllm.v1.attention.backends.mla.page_offload.config import (
    SparsePageOffloadConfig,
)
from vllm.v1.attention.backends.mla.page_offload.selected_pages import (
    SparsePageAdapter,
)
from vllm.v1.attention.backends.mla.page_offload.telemetry import (
    SparseSelectionCollector,
    SparseSelectionStats,
)

logger = init_logger(__name__)


class SparsePageOffloadCoordinator:
    """Phase 1 observe-only coordinator for sparse selected pages."""

    def __init__(
        self,
        config: SparsePageOffloadConfig,
        adapter: SparsePageAdapter,
    ) -> None:
        self.config = config
        self.adapter = adapter
        self.collector = SparseSelectionCollector(
            hot_pool_blocks=config.hot_pool_blocks,
            page_size_bytes=adapter.page_size_bytes,
        )

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
    ) -> SparsePageOffloadCoordinator | None:
        hf_config = vllm_config.model_config.hf_text_config
        config = SparsePageOffloadConfig.from_hf_config(hf_config)
        if not config.enabled:
            return None
        adapter = DeepSeekV4C4AAdapter(hf_config, config)
        return cls(config, adapter)

    def observe_layer(
        self,
        *,
        layer_name: str,
        kv_cache_dtype: str,
        req_id_per_token: torch.Tensor,
        topk_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> SparseSelectionStats | None:
        if not self.adapter.supports_layer(layer_name, kv_cache_dtype):
            return None
        selection = self.adapter.extract_selection(
            layer_name=layer_name,
            req_id_per_token=req_id_per_token,
            topk_indices=topk_indices,
            seq_lens=seq_lens,
        )
        stats = self.collector.record(layer_name, selection)
        logger.info(
            "Sparse page observe-only stats: layer=%s step=%d "
            "unique_pages=%d tail_pages=%d simulated_miss_pages=%s "
            "estimated_bytes=%s",
            stats.layer_name,
            stats.step,
            stats.num_unique_pages,
            stats.num_tail_pages,
            stats.simulated_miss_pages,
            stats.estimated_bytes,
        )
        return stats


def maybe_create_sparse_page_offload_coordinator(
    vllm_config: Any | None,
) -> SparsePageOffloadCoordinator | None:
    if vllm_config is None or vllm_config.model_config is None:
        return None
    return SparsePageOffloadCoordinator.from_vllm_config(vllm_config)
