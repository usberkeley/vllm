# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Coordinator for sparse MLA selected-page offload."""

from __future__ import annotations

import weakref
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.page_offload.adapters.deepseek_v4_c4a import (
    DeepSeekV4C4AAdapter,
)
from vllm.v1.attention.backends.mla.page_offload.config import (
    SparsePageOffloadConfig,
    SparsePageParallelTopology,
)
from vllm.v1.attention.backends.mla.page_offload.protocol import (
    SparsePagePrefillSideband,
    SparsePagePrefillWorkerMetadata,
    SparsePageReference,
)
from vllm.v1.attention.backends.mla.page_offload.selection import (
    LogicalPage,
    SparsePageAdapter,
)
from vllm.v1.attention.backends.mla.page_offload.selection_metrics import (
    SparsePageSelectionCollector,
    SparsePageSelectionStats,
)
from vllm.v1.attention.backends.mla.page_offload.staging import (
    SparsePageStagingManager,
    SparsePageStagingResult,
)

logger = init_logger(__name__)

# All DeepSeek V4 attention layers are built from the same VllmConfig. Keep the
# coordinator shared there so telemetry and sync hot pools span sparse layers
# instead of resetting state for every attention module.
_COORDINATOR_CACHE: dict[int, SparsePageOffloadCoordinator | None] = {}
_COORDINATOR_FINALIZERS: dict[int, weakref.finalize] = {}
_COORDINATOR_STRONG_OWNERS: dict[int, Any] = {}


class SparsePageOffloadCoordinator:
    """Worker-local coordinator for sparse selected pages."""

    def __init__(
        self,
        config: SparsePageOffloadConfig,
        adapter: SparsePageAdapter,
        *,
        max_concurrent_requests: int = 1,
    ) -> None:
        self.config = config
        self.adapter = adapter
        self.selection_collector = SparsePageSelectionCollector(
            hot_page_capacity=config.hot_pages_per_request,
            page_size_bytes=adapter.page_size_bytes,
        )
        self.staging_manager = SparsePageStagingManager(
            hot_pages_per_request=config.hot_pages_per_request,
            max_tail_pages=max_concurrent_requests,
            require_authoritative_cpu_pages=config.can_stage_decode,
            allocate_partial=config.allocate_partial,
            cpu_pool_size_bytes=(
                int(config.cpu_pool_size_gib * 1024**3)
                if config.cpu_pool_size_gib > 0
                else None
            ),
        )
        self._pending_sideband_by_request_row: dict[int, SparsePagePrefillSideband] = {}
        self._request_identity_by_row: dict[int, tuple[int | str, int]] = {}

    def bind_request_rows(
        self,
        request_identities: Sequence[tuple[int | str, int]],
    ) -> None:
        """Bind model-runner batch rows to stable request identities."""
        self._request_identity_by_row = dict(enumerate(request_identities))

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
    ) -> SparsePageOffloadCoordinator | None:
        kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
        config = SparsePageOffloadConfig.from_kv_transfer_config(kv_transfer_config)
        if not config.enabled:
            logger.info_once(
                "Sparse page offload coordinator disabled: kv_connector=%r kv_role=%r",
                getattr(kv_transfer_config, "kv_connector", None),
                getattr(kv_transfer_config, "kv_role", None),
            )
            return None
        topology = SparsePageParallelTopology.from_vllm_config(vllm_config)
        topology.validate()
        hf_config = vllm_config.model_config.hf_text_config
        adapter = DeepSeekV4C4AAdapter(hf_config, config)
        logger.info_once(
            "Sparse page offload coordinator enabled: model_type=%r "
            "role=%r hot_pages_per_request=%d layer_ids=%s page_size_bytes=%d "
            "dp_rank=%d dp_size=%d tp_size=%d expert_parallel=%r",
            getattr(hf_config, "model_type", None),
            config.role,
            config.hot_pages_per_request,
            config.layer_ids,
            adapter.page_size_bytes,
            topology.dp_rank,
            topology.dp_size,
            topology.tp_size,
            topology.expert_parallel,
        )
        scheduler_config = getattr(vllm_config, "scheduler_config", None)
        max_concurrent_requests = int(getattr(scheduler_config, "max_num_seqs", 1))
        return cls(
            config,
            adapter,
            max_concurrent_requests=max_concurrent_requests,
        )

    def stage_decode_layer(
        self,
        *,
        layer_name: str,
        kv_cache_dtype: str,
        req_id_per_token: torch.Tensor,
        topk_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_cache: torch.Tensor,
        source_block_table: torch.Tensor,
        is_decode_only: bool,
    ) -> SparsePageStagingResult:
        if not self.config.can_stage_decode:
            return SparsePageStagingResult(
                kv_cache=kv_cache,
                block_table=source_block_table,
                enabled=False,
            )
        if not self.adapter.supports_layer(layer_name, kv_cache_dtype):
            return SparsePageStagingResult(
                kv_cache=kv_cache,
                block_table=source_block_table,
                enabled=False,
            )
        selection = self.adapter.extract_selection(
            layer_name=layer_name,
            req_id_per_token=req_id_per_token,
            topk_indices=topk_indices,
            seq_lens=seq_lens,
            request_identities=self._request_identity_by_row,
        )
        selection_stats = self.selection_collector.record(layer_name, selection)
        self._log_selection_stats(selection_stats)
        if not is_decode_only:
            logger.warning_once(
                "Sparse page offload behavior disabled for layer=%s: "
                "Phase 1 only supports pure decode.",
                layer_name,
            )
            return SparsePageStagingResult(
                kv_cache=kv_cache,
                block_table=source_block_table,
                enabled=False,
            )
        staging_result = self.staging_manager.stage_decode_pages(
            layer_name=layer_name,
            selection=selection,
            kv_cache=kv_cache,
            source_block_table=source_block_table,
        )
        logger.info(
            "Sparse page sync offload stats: layer=%s observation_index=%d "
            "decode_step=%d enabled=%r miss_pages=%d unique_pages=%d "
            "tail_pages=%d",
            selection_stats.layer_name,
            selection_stats.observation_index,
            selection_stats.decode_step,
            staging_result.enabled,
            len(staging_result.miss_pages),
            selection_stats.num_unique_pages,
            selection_stats.num_tail_pages,
        )
        return staging_result

    def seal_prefill_request(
        self,
        *,
        layer_name: str,
        kv_cache_dtype: str,
        req_id_per_token: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_cache: torch.Tensor,
        source_block_table: torch.Tensor,
    ) -> None:
        if not self.config.can_seal_prefill:
            return
        if not self.adapter.supports_layer(layer_name, kv_cache_dtype):
            return
        num_requests = req_id_per_token.shape[0]
        if num_requests == 0:
            return

        request_ids_cpu = req_id_per_token.detach().to("cpu", dtype=torch.int64)
        sequence_lengths_cpu = seq_lens.detach().to("cpu", dtype=torch.int64)

        enabled = False
        num_tail_pages = 0
        for request_id in request_ids_cpu.tolist():
            if request_id < 0 or request_id >= sequence_lengths_cpu.shape[0]:
                # todo log warning
                continue
            seal_result = self.staging_manager.seal_prefill_request(
                layer_name=layer_name,
                request_id=request_id,
                request_row=request_id,
                sequence_length=int(sequence_lengths_cpu[request_id].item()),
                kv_cache=kv_cache,
                source_block_table=source_block_table,
                storage_block_size=self.adapter.storage_block_size,
                compress_ratio=getattr(self.adapter, "compress_ratio", 1),
            )
            if seal_result.enabled:
                enabled = True
                num_tail_pages += len(seal_result.tail_pages)
                self._merge_pending_prefill_sideband(
                    request_row=request_id,
                    sideband=SparsePagePrefillSideband(
                        tail_pages=tuple(
                            SparsePageReference(
                                layer_name=page.layer_name,
                                page_idx=page.page_idx,
                            )
                            for page in seal_result.tail_pages
                        ),
                    ),
                )

        if enabled:
            logger.info(
                "Sparse page prefill seal: layer=%s requests=%d tail_pages=%d",
                layer_name,
                num_requests,
                num_tail_pages,
            )

    def pop_prefill_worker_metadata(
        self,
        request_ids_by_row: Sequence[str],
    ) -> SparsePagePrefillWorkerMetadata | None:
        request_sidebands: dict[str, SparsePagePrefillSideband] = {}
        for request_row, request_id in enumerate(request_ids_by_row):
            sideband = self._pending_sideband_by_request_row.pop(request_row, None)
            if sideband is not None:
                request_sidebands[str(request_id)] = sideband

        if not request_sidebands:
            return None
        return SparsePagePrefillWorkerMetadata(request_sidebands=request_sidebands)

    def get_prefill_cpu_pages(
        self,
        request_row: int,
        request_id: str,
    ) -> tuple[tuple[LogicalPage, torch.Tensor], ...]:
        """Return P-side pages sealed for one current prefill batch row."""
        return self.staging_manager.rekey_cpu_pages(request_row, request_id)

    def reserve_received_cpu_pages(
        self,
        *,
        request_id: str,
        generation: int,
        pages: Iterable[SparsePageReference],
        kv_caches: Mapping[str, torch.Tensor],
    ) -> dict[LogicalPage, torch.Tensor]:
        """Reserve D-side CPU destinations for an incoming sparse transfer."""
        destinations: dict[LogicalPage, torch.Tensor] = {}
        try:
            for page_ref in pages:
                kv_cache = kv_caches.get(page_ref.layer_name)
                if kv_cache is None or kv_cache.shape[0] == 0:
                    raise ValueError(
                        "Sparse page receive could not resolve c4a KV cache for "
                        f"layer={page_ref.layer_name!r}."
                    )
                page = LogicalPage(
                    request_id=request_id,
                    layer_name=page_ref.layer_name,
                    page_idx=page_ref.page_idx,
                    generation=generation,
                )
                destinations[page] = self.staging_manager.reserve_cpu_page(
                    page, kv_cache[0]
                )
        except Exception:
            self.staging_manager.cleanup_request(request_id)
            raise
        return destinations

    def mark_received_cpu_pages_ready(
        self,
        pages: Iterable[LogicalPage],
    ) -> None:
        self.staging_manager.mark_cpu_pages_ready(pages)

    def prepare_tail_slot_mapping(
        self,
        *,
        layer_name: str,
        positions: torch.Tensor,
        request_rows: torch.Tensor,
        original_slot_mapping: torch.Tensor,
        kv_cache: torch.Tensor,
    ) -> torch.Tensor:
        """Return physical slots for D-side mutable c4a tail writes."""
        if not self.config.allocate_partial:
            return original_slot_mapping
        return self.staging_manager.prepare_tail_slot_mapping(
            layer_name=layer_name,
            positions=positions,
            request_rows=request_rows,
            request_identities=self._request_identity_by_row,
            original_slot_mapping=original_slot_mapping,
            kv_cache=kv_cache,
            compress_ratio=self.adapter.compress_ratio,
            storage_block_size=self.adapter.storage_block_size,
        )

    def restore_received_tail(
        self,
        page: LogicalPage,
        cpu_tensor: torch.Tensor,
        kv_cache: torch.Tensor,
    ) -> None:
        self.staging_manager.restore_received_tail(page, cpu_tensor, kv_cache)

    def cleanup_request(self, request_id: str) -> None:
        self.staging_manager.cleanup_request(request_id)

    def _merge_pending_prefill_sideband(
        self,
        *,
        request_row: int,
        sideband: SparsePagePrefillSideband,
    ) -> None:
        current = self._pending_sideband_by_request_row.get(request_row)
        self._pending_sideband_by_request_row[request_row] = (
            sideband if current is None else current.merge(sideband)
        )

    @staticmethod
    def _log_selection_stats(stats: SparsePageSelectionStats) -> None:
        logger.info(
            "Sparse page stats: layer=%s observation_index=%d "
            "decode_step=%d unique_pages=%d tail_pages=%d "
            "same_layer_cross_decode_step_miss_pages=%d "
            "same_layer_cross_decode_step_estimated_bytes=%d "
            "reference_layer=%s "
            "same_decode_step_reference_missing_pages=%d "
            "same_decode_step_reference_missing_estimated_bytes=%d "
            "same_decode_step_reference_reused_pages=%d",
            stats.layer_name,
            stats.observation_index,
            stats.decode_step,
            stats.num_unique_pages,
            stats.num_tail_pages,
            stats.same_layer_cross_decode_step_miss_pages,
            stats.same_layer_cross_decode_step_estimated_bytes,
            stats.reference_layer_name,
            stats.same_decode_step_reference_missing_pages,
            stats.same_decode_step_reference_missing_estimated_bytes,
            stats.num_same_decode_step_reference_reused_pages,
        )


def _drop_sparse_page_offload_coordinator(cache_key: int) -> None:
    _COORDINATOR_CACHE.pop(cache_key, None)
    _COORDINATOR_FINALIZERS.pop(cache_key, None)
    _COORDINATOR_STRONG_OWNERS.pop(cache_key, None)


def get_sparse_page_offload_coordinator(
    vllm_config: Any | None,
) -> SparsePageOffloadCoordinator | None:
    if vllm_config is None or vllm_config.model_config is None:
        logger.info_once(
            "Sparse page offload coordinator not created: "
            "current VllmConfig or model_config is unavailable."
        )
        return None

    cache_key = id(vllm_config)
    if cache_key in _COORDINATOR_CACHE:
        return _COORDINATOR_CACHE[cache_key]

    coordinator = SparsePageOffloadCoordinator.from_vllm_config(vllm_config)
    _COORDINATOR_CACHE[cache_key] = coordinator
    try:
        _COORDINATOR_FINALIZERS[cache_key] = weakref.finalize(
            vllm_config,
            _drop_sparse_page_offload_coordinator,
            cache_key,
        )
    except TypeError:
        # Lightweight test configs such as SimpleNamespace cannot be weakly
        # referenced. Retain their identity so Python cannot recycle the id
        # and return another config's coordinator.
        _COORDINATOR_STRONG_OWNERS[cache_key] = vllm_config
    return coordinator
