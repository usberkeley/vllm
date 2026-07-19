# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-authoritative sparse selected-page staging."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.page_offload.block_table_cache import (
    LayerLocalBlockTableCache,
)
from vllm.v1.attention.backends.mla.page_offload.hot_pool import (
    SparsePageHotPool,
    SparsePageStagingPlan,
)
from vllm.v1.attention.backends.mla.page_offload.selection import (
    LogicalPage,
    SparsePageSelection,
)

logger = init_logger(__name__)


@dataclass(frozen=True)
class SparsePageStagingResult:
    """Inputs to use for sparse MLA after synchronous page staging."""

    kv_cache: torch.Tensor
    block_table: torch.Tensor
    enabled: bool
    miss_pages: tuple[LogicalPage, ...] = ()


@dataclass(frozen=True)
class SparsePageSealResult:
    """Result of sealing a prefill request into CPU authoritative pages.

    The prefill instance frees its GPU KV as a whole at ``request_finished``, so
    no per-page GPU block release is reported here.
    """

    enabled: bool
    tail_pages: tuple[LogicalPage, ...] = ()


class SparsePageStagingManager:
    """Own worker-local CPU pages and a layer-local GPU hot pool.

    On the decode side, immutable pages must arrive in the CPU pool before they
    can be staged. The mutable tail is restored to its request block and then
    refreshed into its reserved hot-pool slot on every decode step.
    """

    def __init__(
        self,
        *,
        hot_pages_per_request: int,
        max_tail_pages: int = 1,
        require_authoritative_cpu_pages: bool = False,
        allocate_partial: bool = False,
        cpu_pool_size_bytes: int | None = None,
        block_table_cache: LayerLocalBlockTableCache | None = None,
    ) -> None:
        if max_tail_pages <= 0:
            raise ValueError("max_tail_pages must be greater than 0.")
        self.hot_pages_per_request = hot_pages_per_request
        self.max_tail_pages = max_tail_pages
        self.require_authoritative_cpu_pages = require_authoritative_cpu_pages
        self.allocate_partial = allocate_partial
        self.cpu_pool_size_bytes = cpu_pool_size_bytes
        self.block_table_cache = block_table_cache or LayerLocalBlockTableCache()
        self._cpu_page_store: dict[LogicalPage, torch.Tensor] = {}
        self._cpu_ready_pages: set[LogicalPage] = set()
        self._cpu_pool_bytes = 0
        self._hot_pool_by_layer: dict[str, SparsePageHotPool] = {}
        self._mutable_tail_by_request_layer: dict[
            tuple[int | str, str, int], LogicalPage
        ] = {}

    def stage_decode_pages(
        self,
        *,
        layer_name: str,
        selection: SparsePageSelection,
        kv_cache: torch.Tensor,
        source_block_table: torch.Tensor,
    ) -> SparsePageStagingResult:
        if self.hot_pages_per_request <= 0:
            logger.warning_once(
                "Sparse page offload disabled for layer=%s: hot pool is empty.",
                layer_name,
            )
            return SparsePageStagingResult(
                kv_cache=kv_cache,
                block_table=source_block_table,
                enabled=False,
            )
        self._seal_rolled_over_tails(
            selection=selection,
            kv_cache=kv_cache,
            source_block_table=source_block_table,
        )

        selected_logical_pages = tuple(selection.unique_pages)
        tail_pages = tuple(
            page for page in selected_logical_pages if page in selection.tail_pages
        )
        hot_pages = tuple(
            page for page in selected_logical_pages if page not in selection.tail_pages
        )
        if self.require_authoritative_cpu_pages:
            missing_pages = tuple(
                page for page in hot_pages if page not in self._cpu_page_store
            )
            if missing_pages:
                raise RuntimeError(
                    "Sparse page transfer did not provide authoritative CPU pages: "
                    f"{missing_pages!r}."
                )
            unready_pages = tuple(
                page for page in hot_pages if page not in self._cpu_ready_pages
            )
            if unready_pages:
                raise RuntimeError(
                    "Sparse page transfer is not ready for selected CPU pages: "
                    f"{unready_pages!r}."
                )
        if not selected_logical_pages:
            return SparsePageStagingResult(
                kv_cache=kv_cache,
                block_table=source_block_table,
                enabled=True,
            )
        selected_hot_page_count_by_request = Counter(
            page.request_id for page in hot_pages
        )
        over_quota = {
            request_id: count
            for request_id, count in selected_hot_page_count_by_request.items()
            if count > self.hot_pages_per_request
        }
        if over_quota:
            logger.warning_once(
                "Sparse page offload disabled for layer=%s: selected pages "
                "exceed per-request hot pool quota %d: %s.",
                layer_name,
                self.hot_pages_per_request,
                tuple(sorted(over_quota.items(), key=lambda item: str(item[0]))),
            )
            return SparsePageStagingResult(
                kv_cache=kv_cache,
                block_table=source_block_table,
                enabled=False,
            )
        if len(tail_pages) > self.max_tail_pages:
            logger.warning_once(
                "Sparse page offload disabled for layer=%s: selected %d tail "
                "pages but tail pool has %d blocks.",
                layer_name,
                len(tail_pages),
                self.max_tail_pages,
            )
            return SparsePageStagingResult(
                kv_cache=kv_cache,
                block_table=source_block_table,
                enabled=False,
            )

        hot_pool = self._get_or_create_hot_pool(layer_name, kv_cache)
        staging_plan = hot_pool.build_staging_plan(hot_pages, tail_pages)
        self._execute_staging_plan(
            staging_plan=staging_plan,
            selection=selection,
            hot_pool=hot_pool,
            kv_cache=kv_cache,
            source_block_table=source_block_table,
        )

        self._synchronize_device(hot_pool.kv_cache.device)
        block_table = self.block_table_cache.get_or_create(
            layer_name, source_block_table
        )
        self._patch_block_table(block_table, selection, staging_plan.slot_by_page)
        return SparsePageStagingResult(
            kv_cache=hot_pool.kv_cache,
            block_table=block_table,
            enabled=True,
            miss_pages=staging_plan.miss_pages,
        )

    def _seal_rolled_over_tails(
        self,
        *,
        selection: SparsePageSelection,
        kv_cache: torch.Tensor,
        source_block_table: torch.Tensor,
    ) -> None:
        if self.allocate_partial:
            if not selection.current_tail_pages:
                return
            layer_name = selection.current_tail_pages[0].logical_page.layer_name
            hot_pool = self._get_or_create_hot_pool(layer_name, kv_cache)
            for current_tail in selection.current_tail_pages:
                self._prepare_partial_tail(
                    current_tail.logical_page,
                    hot_pool,
                )
            return
        pages_to_seal: list[tuple[LogicalPage, int]] = []
        for current_tail in selection.current_tail_pages:
            page = current_tail.logical_page
            key = (page.request_id, page.layer_name, page.generation)
            previous = self._mutable_tail_by_request_layer.get(key)
            if previous is None:
                if self.require_authoritative_cpu_pages and (
                    page not in self._cpu_page_store
                    or page not in self._cpu_ready_pages
                ):
                    raise RuntimeError(
                        "Sparse page transfer did not provide the initial tail: "
                        f"{page!r}."
                    )
                self._mutable_tail_by_request_layer[key] = page
                continue
            if page.page_idx < previous.page_idx:
                raise RuntimeError(
                    "Sparse mutable tail moved backwards: "
                    f"previous={previous!r} current={page!r}."
                )
            if page.page_idx == previous.page_idx:
                continue
            if current_tail.request_row >= source_block_table.shape[0]:
                raise ValueError(
                    "Sparse tail request row is outside the block table: "
                    f"row={current_tail.request_row}."
                )
            for page_idx in range(previous.page_idx, page.page_idx):
                if page_idx >= source_block_table.shape[1]:
                    raise ValueError(
                        "Sparse sealed tail is outside the block table: "
                        f"page={page_idx}."
                    )
                sealed_page = LogicalPage(
                    request_id=page.request_id,
                    layer_name=page.layer_name,
                    page_idx=page_idx,
                    generation=page.generation,
                )
                block_id = int(
                    source_block_table[current_tail.request_row, page_idx].item()
                )
                pages_to_seal.append((sealed_page, block_id))
            self._mutable_tail_by_request_layer[key] = page

        if not pages_to_seal:
            return
        block_ids = self._make_index_tensor(
            (block_id for _, block_id in pages_to_seal),
            kv_cache.device,
        )
        cpu_pages = kv_cache.index_select(0, block_ids).detach().to("cpu", copy=True)
        for (page, _), cpu_page in zip(
            pages_to_seal,
            cpu_pages.unbind(0),
            strict=True,
        ):
            self.install_cpu_page(page, cpu_page, ready=True)

    def _execute_staging_plan(
        self,
        *,
        staging_plan: SparsePageStagingPlan,
        selection: SparsePageSelection,
        hot_pool: SparsePageHotPool,
        kv_cache: torch.Tensor,
        source_block_table: torch.Tensor,
    ) -> None:
        source_block_by_page = self._resolve_source_block_ids(
            source_block_table,
            selection,
            staging_plan.pages_requiring_copy,
        )
        uncached_miss_pages = tuple(
            page for page in staging_plan.miss_pages if page not in self._cpu_page_store
        )
        if uncached_miss_pages:
            if self.require_authoritative_cpu_pages:
                raise RuntimeError(
                    "Sparse page transfer did not provide authoritative CPU pages: "
                    f"{uncached_miss_pages!r}."
                )
            source_block_ids = self._make_index_tensor(
                (source_block_by_page[page] for page in uncached_miss_pages),
                kv_cache.device,
            )
            cpu_page_batch = (
                kv_cache.index_select(0, source_block_ids).detach().to("cpu", copy=True)
            )
            for page, cpu_page in zip(
                uncached_miss_pages,
                cpu_page_batch.unbind(0),
                strict=True,
            ):
                self.install_cpu_page(page, cpu_page.clone(), ready=True)

        unready_miss_pages = tuple(
            page
            for page in staging_plan.miss_pages
            if page not in self._cpu_ready_pages
        )
        if unready_miss_pages:
            raise RuntimeError(
                "Sparse page transfer is not ready for selected CPU pages: "
                f"{unready_miss_pages!r}."
            )

        if staging_plan.miss_pages:
            cpu_page_batch = torch.stack(
                [self._cpu_page_store[page] for page in staging_plan.miss_pages]
            )
            hot_pool.kv_cache.index_copy_(
                0,
                self._make_index_tensor(
                    (
                        staging_plan.slot_by_page[page]
                        for page in staging_plan.miss_pages
                    ),
                    hot_pool.kv_cache.device,
                ),
                cpu_page_batch.to(hot_pool.kv_cache.device, non_blocking=False),
            )

        if staging_plan.tail_pages_to_refresh:
            if self.allocate_partial:
                return
            hot_pool.kv_cache.index_copy_(
                0,
                self._make_index_tensor(
                    (
                        staging_plan.slot_by_page[page]
                        for page in staging_plan.tail_pages_to_refresh
                    ),
                    hot_pool.kv_cache.device,
                ),
                kv_cache.index_select(
                    0,
                    self._make_index_tensor(
                        (
                            source_block_by_page[page]
                            for page in staging_plan.tail_pages_to_refresh
                        ),
                        kv_cache.device,
                    ),
                ),
            )

    def prepare_tail_slot_mapping(
        self,
        *,
        layer_name: str,
        positions: torch.Tensor,
        request_rows: torch.Tensor,
        request_identities: Mapping[int, tuple[int | str, int]],
        original_slot_mapping: torch.Tensor,
        kv_cache: torch.Tensor,
        compress_ratio: int,
        storage_block_size: int,
    ) -> torch.Tensor:
        """Map D-side c4a compressor writes into persistent tail slots."""
        if not self.allocate_partial:
            return original_slot_mapping
        if positions.shape[0] != request_rows.shape[0]:
            raise ValueError("Sparse tail positions and request rows must align.")
        if original_slot_mapping.shape[0] < positions.shape[0]:
            raise ValueError("Sparse tail slot mapping is shorter than positions.")

        hot_pool = self._get_or_create_hot_pool(layer_name, kv_cache)
        mapped = original_slot_mapping.clone()
        positions_cpu = positions.detach().to("cpu", dtype=torch.int64)
        rows_cpu = request_rows.detach().to("cpu", dtype=torch.int64)
        current_page_by_request: dict[tuple[int | str, int], LogicalPage] = {}
        mappings: list[tuple[int, int]] = []
        for token_row, (position, request_row) in enumerate(
            zip(positions_cpu.tolist(), rows_cpu.tolist(), strict=True)
        ):
            if request_row < 0:
                continue
            request_id, generation = request_identities.get(
                request_row,
                (request_row, 0),
            )
            compressed_idx = int(position) // compress_ratio
            page = LogicalPage(
                request_id=request_id,
                layer_name=layer_name,
                page_idx=compressed_idx // storage_block_size,
                generation=generation,
            )
            request_key = (request_id, generation)
            previous_current = current_page_by_request.get(request_key)
            if previous_current is not None and previous_current != page:
                raise RuntimeError(
                    "Sparse c4a partial allocation does not support one request "
                    "crossing a page boundary within one decode batch."
                )
            current_page_by_request[request_key] = page
            slot = self._prepare_partial_tail(page, hot_pool)
            page_offset = compressed_idx % storage_block_size
            mappings.append((token_row, slot * storage_block_size + page_offset))

        if mappings:
            token_rows = self._make_index_tensor(
                (token_row for token_row, _ in mappings),
                mapped.device,
            )
            physical_slots = torch.tensor(
                [physical_slot for _, physical_slot in mappings],
                dtype=mapped.dtype,
                device=mapped.device,
            )
            mapped.index_copy_(0, token_rows, physical_slots)
        return mapped

    def restore_received_tail(
        self,
        page: LogicalPage,
        cpu_tensor: torch.Tensor,
        kv_cache: torch.Tensor,
    ) -> None:
        """Restore an incoming mutable tail into its partial GPU slot."""
        if not self.allocate_partial:
            raise RuntimeError("Sparse tail restore requires partial allocation.")
        hot_pool = self._get_or_create_hot_pool(page.layer_name, kv_cache)
        slot = hot_pool.reserve_tail_slot(page)
        hot_pool.kv_cache[slot].copy_(cpu_tensor)
        key = (page.request_id, page.layer_name, page.generation)
        self._mutable_tail_by_request_layer[key] = page

    def _prepare_partial_tail(
        self,
        page: LogicalPage,
        hot_pool: SparsePageHotPool,
    ) -> int:
        key = (page.request_id, page.layer_name, page.generation)
        previous = self._mutable_tail_by_request_layer.get(key)
        if previous is None:
            cpu_page = self._cpu_page_store.get(page)
            if cpu_page is None or page not in self._cpu_ready_pages:
                raise RuntimeError(
                    "Sparse page transfer did not provide the initial tail: "
                    f"{page!r}."
                )
            slot = hot_pool.reserve_tail_slot(page)
            hot_pool.kv_cache[slot].copy_(cpu_page)
            self._mutable_tail_by_request_layer[key] = page
            return slot
        if page.page_idx < previous.page_idx:
            raise RuntimeError(
                "Sparse mutable tail moved backwards: "
                f"previous={previous!r} current={page!r}."
            )
        if page.page_idx == previous.page_idx:
            return hot_pool.reserve_tail_slot(page)
        if page.page_idx != previous.page_idx + 1:
            raise RuntimeError(
                "Sparse mutable tail skipped a page: "
                f"previous={previous!r} current={page!r}."
            )
        previous_slot = hot_pool.tail_slots.get(previous)
        if previous_slot is None:
            raise RuntimeError(f"Sparse mutable tail has no GPU slot: {previous!r}.")
        sealed_page = hot_pool.kv_cache[previous_slot].detach().to("cpu", copy=True)
        self.install_cpu_page(previous, sealed_page, ready=True)
        slot = hot_pool.advance_tail(previous, page)
        self._mutable_tail_by_request_layer[key] = page
        return slot

    def seal_prefill_request(
        self,
        *,
        layer_name: str,
        request_id: int | str,
        request_row: int,
        sequence_length: int,
        kv_cache: torch.Tensor,
        source_block_table: torch.Tensor,
        storage_block_size: int,
        compress_ratio: int,
    ) -> SparsePageSealResult:
        """Seal a finished prefill request into producer-side CPU pages.

        The connector registers and transfers these pages to the decode CPU
        pool. GPU KV is released as a whole by the prefill scheduler, so this
        path performs no per-page GPU free and never populates the decode-side
        hot pool.
        """
        if self.hot_pages_per_request <= 0 or sequence_length <= 0:
            return SparsePageSealResult(enabled=False)

        compressed_length = (sequence_length + compress_ratio - 1) // compress_ratio
        if compressed_length <= 0:
            return SparsePageSealResult(enabled=False)

        tail_page_index = (compressed_length - 1) // storage_block_size
        sealed_pages = tuple(
            LogicalPage(request_id, layer_name, page_idx)
            for page_idx in range(tail_page_index)
        )
        tail_page = LogicalPage(request_id, layer_name, tail_page_index)
        transferred_pages = (*sealed_pages, tail_page)

        transferred_page_indices = torch.arange(
            tail_page_index + 1,
            dtype=torch.long,
            device=source_block_table.device,
        )
        source_block_ids = (
            source_block_table[request_row]
            .index_select(0, transferred_page_indices)
            .to(
                device=kv_cache.device,
                dtype=torch.long,
            )
        )
        cpu_page_batch = (
            kv_cache.index_select(0, source_block_ids).detach().to("cpu", copy=True)
        )
        for page, cpu_page in zip(
            transferred_pages,
            cpu_page_batch.unbind(0),
            strict=True,
        ):
            self.install_cpu_page(page, cpu_page, ready=True)

        return SparsePageSealResult(
            enabled=True,
            tail_pages=(tail_page,),
        )

    def install_cpu_page(
        self,
        page: LogicalPage,
        tensor: torch.Tensor,
        *,
        ready: bool,
    ) -> torch.Tensor:
        """Install one CPU authoritative page and account for its capacity."""
        if tensor.device.type != "cpu":
            raise ValueError("Sparse authoritative pages must reside on CPU.")
        existing = self._cpu_page_store.get(page)
        if existing is not None:
            if existing.shape != tensor.shape or existing.dtype != tensor.dtype:
                raise ValueError(f"Conflicting sparse CPU page layout for {page!r}.")
            if existing.data_ptr() != tensor.data_ptr():
                existing.copy_(tensor)
            if ready:
                self._cpu_ready_pages.add(page)
            return existing

        page_bytes = tensor.nbytes
        new_total = self._cpu_pool_bytes + page_bytes
        if (
            self.cpu_pool_size_bytes is not None
            and new_total > self.cpu_pool_size_bytes
        ):
            raise MemoryError(
                "Sparse authoritative CPU pool capacity exceeded: "
                f"requested={new_total} limit={self.cpu_pool_size_bytes}."
            )
        self._cpu_page_store[page] = tensor
        self._cpu_pool_bytes = new_total
        if ready:
            self._cpu_ready_pages.add(page)
        return tensor

    def reserve_cpu_page(
        self,
        page: LogicalPage,
        page_template: torch.Tensor,
    ) -> torch.Tensor:
        """Reserve an unready CPU destination page for connector receive."""
        existing = self._cpu_page_store.get(page)
        if existing is not None:
            return existing
        tensor = torch.empty_like(page_template, device="cpu")
        return self.install_cpu_page(page, tensor, ready=False)

    def mark_cpu_pages_ready(self, pages: Iterable[LogicalPage]) -> None:
        for page in pages:
            if page not in self._cpu_page_store:
                raise KeyError(f"Unknown sparse CPU page: {page!r}.")
            self._cpu_ready_pages.add(page)

    def iter_cpu_pages(
        self,
        request_id: int | str,
        *,
        generation: int = 0,
    ) -> tuple[tuple[LogicalPage, torch.Tensor], ...]:
        return tuple(
            (page, tensor)
            for page, tensor in self._cpu_page_store.items()
            if page.request_id == request_id and page.generation == generation
        )

    def rekey_cpu_pages(
        self,
        old_request_id: int | str,
        new_request_id: int | str,
    ) -> tuple[tuple[LogicalPage, torch.Tensor], ...]:
        """Move prefill row-scoped pages onto a stable engine request id."""
        moved: list[tuple[LogicalPage, torch.Tensor]] = []
        for old_page, tensor in tuple(self._cpu_page_store.items()):
            if old_page.request_id != old_request_id:
                continue
            new_page = LogicalPage(
                request_id=new_request_id,
                layer_name=old_page.layer_name,
                page_idx=old_page.page_idx,
                generation=old_page.generation,
            )
            if new_page in self._cpu_page_store:
                raise ValueError(f"Sparse CPU page already exists: {new_page!r}.")
            del self._cpu_page_store[old_page]
            self._cpu_page_store[new_page] = tensor
            if old_page in self._cpu_ready_pages:
                self._cpu_ready_pages.remove(old_page)
                self._cpu_ready_pages.add(new_page)
            moved.append((new_page, tensor))
        return tuple(moved)

    def cleanup_request(self, request_id: int | str) -> None:
        removed_pages = tuple(
            page for page in self._cpu_page_store if page.request_id == request_id
        )
        for page in removed_pages:
            self._cpu_pool_bytes -= self._cpu_page_store.pop(page).nbytes
            self._cpu_ready_pages.discard(page)
        for hot_pool in self._hot_pool_by_layer.values():
            hot_pool.cleanup_request(request_id)
        for key in tuple(self._mutable_tail_by_request_layer):
            if key[0] == request_id:
                del self._mutable_tail_by_request_layer[key]

    def _get_or_create_hot_pool(
        self,
        layer_name: str,
        kv_cache: torch.Tensor,
    ) -> SparsePageHotPool:
        hot_pool = self._hot_pool_by_layer.get(layer_name)
        if hot_pool is None or not hot_pool.is_compatible_with(
            kv_cache,
            tail_slot_capacity=self.max_tail_pages,
        ):
            hot_pool = SparsePageHotPool.create(
                kv_cache,
                resident_slot_capacity=self.resident_slot_capacity,
                tail_slot_capacity=self.max_tail_pages,
                slots_per_request=self.hot_pages_per_request,
                use_source_tensor=self.allocate_partial,
            )
            self._hot_pool_by_layer[layer_name] = hot_pool
        return hot_pool

    @property
    def resident_slot_capacity(self) -> int:
        return self.hot_pages_per_request * self.max_tail_pages

    @classmethod
    def _resolve_source_block_ids(
        cls,
        block_table: torch.Tensor,
        selection: SparsePageSelection,
        logical_pages: Iterable[LogicalPage],
    ) -> dict[LogicalPage, int]:
        logical_pages = tuple(logical_pages)
        if not logical_pages:
            return {}
        selection_by_page = {
            selected_page.logical_page: selected_page
            for selected_page in selection.selected_pages
        }
        rows = cls._make_index_tensor(
            (selection_by_page[page].request_row for page in logical_pages),
            block_table.device,
        )
        columns = cls._make_index_tensor(
            (page.page_idx for page in logical_pages),
            block_table.device,
        )
        source_block_ids = block_table[rows, columns].to(device="cpu", dtype=torch.long)
        return dict(zip(logical_pages, source_block_ids.tolist(), strict=True))

    @staticmethod
    def _make_index_tensor(values: Iterable[int], device: torch.device) -> torch.Tensor:
        return torch.tensor(tuple(values), dtype=torch.long, device=device)

    @staticmethod
    def _patch_block_table(
        block_table: torch.Tensor,
        selection: SparsePageSelection,
        slot_by_page: Mapping[LogicalPage, int],
    ) -> None:
        selected_page_records = tuple(selection.selected_pages)
        if not selected_page_records:
            return
        rows = SparsePageStagingManager._make_index_tensor(
            (page.request_row for page in selected_page_records),
            block_table.device,
        )
        columns = SparsePageStagingManager._make_index_tensor(
            (page.logical_page.page_idx for page in selected_page_records),
            block_table.device,
        )
        slots = torch.tensor(
            tuple(slot_by_page[page.logical_page] for page in selected_page_records),
            dtype=block_table.dtype,
            device=block_table.device,
        )
        block_table[rows, columns] = slots

    @staticmethod
    def _synchronize_device(device: torch.device) -> None:
        if device.type == "cuda":
            torch.cuda.current_stream(device).synchronize()
