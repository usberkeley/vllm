# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire protocol types for sparse selected-page offload."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorWorkerMetadata,
)
from vllm.v1.attention.backends.mla.page_offload.config import (
    SparsePageParallelTopology,
)

SPARSE_PAGE_SIDEBAND_VERSION = 4


@dataclass(frozen=True)
class SparsePageRoute:
    """Stable producer route, optionally bound to one consumer DP replica."""

    producer_engine_id: str
    producer_dp_rank: int
    producer_tp_size: int
    generation: int
    producer_expert_parallel: bool = False
    consumer_engine_id: str | None = None
    consumer_dp_rank: int | None = None
    consumer_tp_size: int | None = None
    consumer_expert_parallel: bool | None = None

    @classmethod
    def from_producer(
        cls,
        producer_topology: SparsePageParallelTopology,
        generation: int,
    ) -> SparsePageRoute:
        producer_topology.validate()
        if generation <= 0:
            raise ValueError("Sparse page request generation must be greater than 0.")
        return cls(
            producer_engine_id=producer_topology.engine_id,
            producer_dp_rank=producer_topology.dp_rank,
            producer_tp_size=producer_topology.tp_size,
            producer_expert_parallel=producer_topology.expert_parallel,
            generation=generation,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SparsePageRoute:
        try:
            return cls(
                producer_engine_id=str(payload["producer_engine_id"]),
                producer_dp_rank=int(payload["producer_dp_rank"]),
                producer_tp_size=int(payload["producer_tp_size"]),
                producer_expert_parallel=bool(
                    payload.get("producer_expert_parallel", False)
                ),
                generation=int(payload["generation"]),
                consumer_engine_id=(
                    str(payload["consumer_engine_id"])
                    if payload.get("consumer_engine_id") is not None
                    else None
                ),
                consumer_dp_rank=(
                    int(payload["consumer_dp_rank"])
                    if payload.get("consumer_dp_rank") is not None
                    else None
                ),
                consumer_tp_size=(
                    int(payload["consumer_tp_size"])
                    if payload.get("consumer_tp_size") is not None
                    else None
                ),
                consumer_expert_parallel=(
                    bool(payload["consumer_expert_parallel"])
                    if payload.get("consumer_expert_parallel") is not None
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid sparse page route metadata.") from exc

    def bind_consumer(
        self,
        consumer_topology: SparsePageParallelTopology,
        *,
        remote_engine_id: str | None,
    ) -> SparsePageRoute:
        consumer_topology.validate()
        consumer_topology.validate_remote_tp_size(self.producer_tp_size)
        if self.generation <= 0:
            raise ValueError("Sparse page request generation must be greater than 0.")
        if remote_engine_id != self.producer_engine_id:
            raise ValueError(
                "Sparse page producer engine does not match NIXL route: "
                f"sparse={self.producer_engine_id!r}, "
                f"nixl={remote_engine_id!r}."
            )
        expected_consumer = (
            consumer_topology.engine_id,
            consumer_topology.dp_rank,
            consumer_topology.tp_size,
        )
        existing_consumer = (
            self.consumer_engine_id,
            self.consumer_dp_rank,
            self.consumer_tp_size,
        )
        if any(value is not None for value in existing_consumer) and (
            existing_consumer != expected_consumer
        ):
            raise ValueError(
                "Sparse page request is already bound to a different "
                f"consumer: existing={existing_consumer}, "
                f"local={expected_consumer}."
            )
        return replace(
            self,
            consumer_engine_id=consumer_topology.engine_id,
            consumer_dp_rank=consumer_topology.dp_rank,
            consumer_tp_size=consumer_topology.tp_size,
            consumer_expert_parallel=consumer_topology.expert_parallel,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "producer_engine_id": self.producer_engine_id,
            "producer_dp_rank": self.producer_dp_rank,
            "producer_tp_size": self.producer_tp_size,
            "producer_expert_parallel": self.producer_expert_parallel,
            "generation": self.generation,
            "consumer_engine_id": self.consumer_engine_id,
            "consumer_dp_rank": self.consumer_dp_rank,
            "consumer_tp_size": self.consumer_tp_size,
            "consumer_expert_parallel": self.consumer_expert_parallel,
        }


@dataclass(frozen=True, order=True)
class SparsePageReference:
    """A sparse page reference scoped by layer."""

    layer_name: str
    page_idx: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_name": self.layer_name,
            "page_idx": self.page_idx,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SparsePageReference:
        try:
            return cls(
                layer_name=str(payload["layer_name"]),
                page_idx=int(payload["page_idx"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid sparse page reference metadata.") from exc


@dataclass(frozen=True, order=True)
class SparsePageTransferPage:
    """One P-side registered DRAM page exposed to a D worker."""

    layer_name: str
    page_idx: int
    source_address: int
    page_size_bytes: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SparsePageTransferPage:
        try:
            page = cls(
                layer_name=str(payload["layer_name"]),
                page_idx=int(payload["page_idx"]),
                source_address=int(payload["source_address"]),
                page_size_bytes=int(payload["page_size_bytes"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid sparse page transfer metadata.") from exc
        if page.source_address <= 0 or page.page_size_bytes <= 0:
            raise ValueError("Sparse page transfer address and size must be positive.")
        return page

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_name": self.layer_name,
            "page_idx": self.page_idx,
            "source_address": self.source_address,
            "page_size_bytes": self.page_size_bytes,
        }


@dataclass(frozen=True, order=True)
class SparsePageRankTransfer:
    """Registered source pages owned by one producer TP rank."""

    tp_rank: int
    pages: tuple[SparsePageTransferPage, ...]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SparsePageRankTransfer:
        try:
            transfer = cls(
                tp_rank=int(payload["tp_rank"]),
                pages=tuple(
                    SparsePageTransferPage.from_dict(page)
                    for page in payload.get("pages", ())
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid sparse TP-rank transfer metadata.") from exc
        if transfer.tp_rank < 0:
            raise ValueError("Sparse TP rank must be non-negative.")
        return transfer

    def to_dict(self) -> dict[str, Any]:
        return {
            "tp_rank": self.tp_rank,
            "pages": [page.to_dict() for page in self.pages],
        }


@dataclass(frozen=True)
class SparsePagePrefillSideband:
    """Sparse prefill sideband for one request."""

    tail_pages: tuple[SparsePageReference, ...] = ()
    rank_transfers: tuple[SparsePageRankTransfer, ...] = ()

    def merge(self, other: SparsePagePrefillSideband) -> SparsePagePrefillSideband:
        return SparsePagePrefillSideband(
            tail_pages=_merge_pages(self.tail_pages, other.tail_pages),
            rank_transfers=_merge_rank_transfers(
                self.rank_transfers, other.rank_transfers
            ),
        )

    @classmethod
    def from_kv_transfer_params(
        cls,
        payload: dict[str, Any],
    ) -> SparsePagePrefillSideband:
        return cls(
            tail_pages=tuple(
                SparsePageReference.from_dict(page)
                for page in payload.get("tail_pages", ())
            ),
            rank_transfers=tuple(
                SparsePageRankTransfer.from_dict(transfer)
                for transfer in payload.get("rank_transfers", ())
            ),
        )

    def to_kv_transfer_params(self) -> dict[str, Any]:
        params = {
            "version": SPARSE_PAGE_SIDEBAND_VERSION,
            "tail_pages": [page.to_dict() for page in self.tail_pages],
        }
        if self.rank_transfers:
            params["rank_transfers"] = [
                transfer.to_dict() for transfer in self.rank_transfers
            ]
        return params


@dataclass(frozen=True)
class SparsePagePrefillWorkerMetadata(KVConnectorWorkerMetadata):
    """Worker-to-scheduler sparse page prefill sideband metadata."""

    request_sidebands: dict[str, SparsePagePrefillSideband]

    def aggregate(self, other: KVConnectorWorkerMetadata) -> KVConnectorWorkerMetadata:
        assert isinstance(other, SparsePagePrefillWorkerMetadata)
        merged = dict(self.request_sidebands)
        for req_id, sideband in other.request_sidebands.items():
            current = merged.get(req_id)
            merged[req_id] = sideband if current is None else current.merge(sideband)
        return SparsePagePrefillWorkerMetadata(request_sidebands=merged)


def _merge_pages(
    existing_pages: tuple[SparsePageReference, ...],
    new_pages: tuple[SparsePageReference, ...],
) -> tuple[SparsePageReference, ...]:
    return tuple(sorted(set(existing_pages) | set(new_pages)))


def _merge_rank_transfers(
    existing: tuple[SparsePageRankTransfer, ...],
    new: tuple[SparsePageRankTransfer, ...],
) -> tuple[SparsePageRankTransfer, ...]:
    pages_by_rank: dict[int, dict[tuple[str, int], SparsePageTransferPage]] = {}
    for transfer in (*existing, *new):
        rank_pages = pages_by_rank.setdefault(transfer.tp_rank, {})
        for page in transfer.pages:
            key = (page.layer_name, page.page_idx)
            previous = rank_pages.get(key)
            if previous is not None and previous != page:
                raise ValueError(
                    "Conflicting sparse page transfer descriptor for "
                    f"tp_rank={transfer.tp_rank} page={key}."
                )
            rank_pages[key] = page
    return tuple(
        SparsePageRankTransfer(
            tp_rank=tp_rank,
            pages=tuple(sorted(pages.values())),
        )
        for tp_rank, pages in sorted(pages_by_rank.items())
    )
