# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Receiver-pull staging protocol and local resource management."""

from __future__ import annotations

import math
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from functools import partial
from typing import Any, Generic, Literal, Protocol, TypeAlias, TypeVar

import msgspec
import torch

STAGING_PROTOCOL_VERSION = 1
STAGE_NOTIF_PREFIX = b"NIXL_STAGE_V1:"

_DEFAULT_SLOT_BYTES = 256 * 1024 * 1024
_SLOT_ALIGNMENT = 1

ReadyIdentity: TypeAlias = tuple[str, int, str, str, int, str, str, int, int, int]
ConsumerIdentity: TypeAlias = tuple[str, int, str]
ProducerIdentity: TypeAlias = tuple[str, int, str]
SourceSlotKey: TypeAlias = tuple[str, int, str, int]
TransferEdgeKey: TypeAlias = tuple[str, str, int, str, int]


def transfer_edge_key(
    value: StagingTransferIntent | StageReady,
) -> TransferEdgeKey:
    """Return the request transfer plus concrete TP edge identity."""
    return (
        value.transfer_id,
        value.producer_engine_id,
        value.producer_rank,
        value.consumer_engine_id,
        value.consumer_rank,
    )


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed <= 0 or parsed != value:
        raise ValueError(f"{name} must be positive")
    return parsed


def _non_negative_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < 0 or parsed != value:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _positive_float(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


@dataclass(frozen=True)
class StagingConfig:
    """Validated per-worker receiver-pull staging configuration."""

    enabled: bool = False
    buffer_bytes: int = 0
    buffer_fraction: float = 0.0
    slot_bytes: int = _DEFAULT_SLOT_BYTES
    max_inflight: int = 1
    max_inflight_per_peer: int = 1
    max_ready_per_request: int = 1
    ready_retry_interval: float = 0.5
    transfer_timeout: float = 30.0
    quarantine_max_bytes: int = 0
    fallback: Literal["direct", "fail"] = "direct"

    @classmethod
    def from_extra_config(cls, extra_config: dict[str, Any]) -> StagingConfig:
        enabled_value = extra_config.get("staging_enabled", False)
        if isinstance(enabled_value, bool):
            enabled = enabled_value
        elif isinstance(enabled_value, str) and enabled_value.lower() in (
            "true",
            "false",
        ):
            enabled = enabled_value.lower() == "true"
        else:
            raise ValueError("staging_enabled must be a boolean")
        buffer_bytes = _non_negative_int(
            "staging_buffer_bytes", extra_config.get("staging_buffer_bytes", 0)
        )
        fraction_value = extra_config.get("staging_buffer_fraction", 0.0)
        if isinstance(fraction_value, bool):
            raise ValueError("staging_buffer_fraction must be a number")
        try:
            buffer_fraction = float(fraction_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("staging_buffer_fraction must be a number") from exc
        slot_bytes = _positive_int(
            "staging_slot_bytes",
            extra_config.get("staging_slot_bytes", _DEFAULT_SLOT_BYTES),
        )
        max_inflight = _positive_int(
            "staging_max_inflight", extra_config.get("staging_max_inflight", 1)
        )
        max_inflight_per_peer = _positive_int(
            "staging_max_inflight_per_peer",
            extra_config.get("staging_max_inflight_per_peer", 1),
        )
        max_ready_per_request = _positive_int(
            "staging_max_ready_per_request",
            extra_config.get("staging_max_ready_per_request", 1),
        )
        ready_retry_interval = _positive_float(
            "staging_ready_retry_interval",
            extra_config.get("staging_ready_retry_interval", 0.5),
        )
        transfer_timeout = _positive_float(
            "staging_transfer_timeout",
            extra_config.get("staging_transfer_timeout", 30.0),
        )
        quarantine_max_bytes = _non_negative_int(
            "staging_quarantine_max_bytes",
            extra_config.get("staging_quarantine_max_bytes", 0),
        )
        fallback = extra_config.get("staging_fallback", "direct")

        if fallback not in ("direct", "fail"):
            raise ValueError("staging_fallback must be 'direct' or 'fail'")
        if not 0 <= buffer_fraction < 1:
            raise ValueError("staging_buffer_fraction must be in [0, 1)")
        if enabled and buffer_bytes <= 0 and buffer_fraction <= 0:
            raise ValueError(
                "staging_buffer_bytes or staging_buffer_fraction must be positive "
                "when staging is enabled"
            )
        if slot_bytes < _SLOT_ALIGNMENT:
            raise ValueError(
                f"staging_slot_bytes must be at least {_SLOT_ALIGNMENT} bytes"
            )
        slot_bytes -= slot_bytes % _SLOT_ALIGNMENT
        if enabled and buffer_bytes and buffer_bytes < slot_bytes:
            raise ValueError("staging_buffer_bytes must be at least staging_slot_bytes")

        return cls(
            enabled=enabled,
            buffer_bytes=buffer_bytes,
            buffer_fraction=buffer_fraction,
            slot_bytes=slot_bytes,
            max_inflight=max_inflight,
            max_inflight_per_peer=max_inflight_per_peer,
            max_ready_per_request=max_ready_per_request,
            ready_retry_interval=ready_retry_interval,
            transfer_timeout=transfer_timeout,
            quarantine_max_bytes=quarantine_max_bytes,
            fallback=fallback,
        )

    @property
    def slot_count(self) -> int:
        return self.buffer_bytes // self.slot_bytes

    def resolve_buffer_bytes(self, total_device_bytes: int) -> StagingConfig:
        """Resolve a fractional allocation and align it to complete slots."""
        if total_device_bytes <= 0:
            raise ValueError("total_device_bytes must be positive")
        if not self.enabled:
            return self
        requested = self.buffer_bytes or int(total_device_bytes * self.buffer_fraction)
        resolved = requested - requested % self.slot_bytes
        if resolved < self.slot_bytes:
            raise ValueError(
                "staging buffer must be at least one complete staging slot"
            )
        return replace(
            self,
            buffer_bytes=resolved,
            max_inflight=min(self.max_inflight, resolved // self.slot_bytes),
        )


def resolve_staging_config(vllm_config: Any, total_device_bytes: int) -> StagingConfig:
    """Resolve and persist the per-worker NIXL pull staging reservation."""
    transfer_config = vllm_config.kv_transfer_config
    if transfer_config is None or transfer_config.kv_connector not in (
        "NixlConnector",
        "NixlPullConnector",
    ):
        return StagingConfig()
    extra_config = transfer_config.kv_connector_extra_config
    config = StagingConfig.from_extra_config(extra_config)
    if not config.enabled:
        return config
    resolved = config.resolve_buffer_bytes(total_device_bytes)
    previous = extra_config.get("_staging_resolved_buffer_bytes")
    if previous is not None and previous != resolved.buffer_bytes:
        raise ValueError(
            "staging buffer resolved to different sizes in memory planning and "
            "connector initialization"
        )
    extra_config["_staging_resolved_buffer_bytes"] = resolved.buffer_bytes
    return resolved


def load_resolved_staging_config(vllm_config: Any) -> StagingConfig:
    """Load the staging config after GPU memory planning resolved its size."""
    transfer_config = vllm_config.kv_transfer_config
    if transfer_config is None or transfer_config.kv_connector not in (
        "NixlConnector",
        "NixlPullConnector",
    ):
        return StagingConfig()
    extra_config = transfer_config.kv_connector_extra_config
    config = StagingConfig.from_extra_config(extra_config)
    if not config.enabled:
        return config
    resolved = extra_config.get("_staging_resolved_buffer_bytes")
    if resolved is None:
        raise RuntimeError(
            "NIXL staging memory was not reserved before connector initialization"
        )
    return replace(
        config, buffer_bytes=_positive_int("resolved staging bytes", resolved)
    )


class StageReady(  # type: ignore[call-arg]
    msgspec.Struct, frozen=True, tag="stage_ready"
):
    protocol_version: int
    producer_generation: str
    consumer_generation: str
    transfer_id: str
    request_id: str
    chunk_index: int
    source_slot_id: int
    source_slot_epoch: int
    valid_bytes: int
    plan_id: str
    producer_engine_id: str
    producer_rank: int
    consumer_engine_id: str
    consumer_rank: int


class StageReadComplete(  # type: ignore[call-arg]
    msgspec.Struct, frozen=True, tag="stage_read_complete"
):
    protocol_version: int
    producer_generation: str
    consumer_generation: str
    transfer_id: str
    chunk_index: int
    source_slot_id: int
    source_slot_epoch: int
    producer_engine_id: str
    producer_rank: int
    consumer_engine_id: str
    consumer_rank: int


class StageCancel(  # type: ignore[call-arg]
    msgspec.Struct, frozen=True, tag="stage_cancel"
):
    protocol_version: int
    producer_generation: str
    consumer_generation: str
    transfer_id: str
    request_id: str
    producer_request_id: str
    consumer_request_id: str
    consumer_engine_id: str
    consumer_rank: int


class StageStatusQuery(  # type: ignore[call-arg]
    msgspec.Struct, frozen=True, tag="stage_status_query"
):
    protocol_version: int
    producer_generation: str
    consumer_generation: str
    transfer_id: str
    chunk_index: int
    source_slot_id: int
    source_slot_epoch: int
    producer_engine_id: str
    producer_rank: int
    consumer_engine_id: str
    consumer_rank: int


SafeStatus = Literal[
    "safe_not_submitted",
    "inflight",
    "safe_complete",
    "safe_retired",
    "unknown",
]


class StageStatusReply(  # type: ignore[call-arg]
    msgspec.Struct, frozen=True, tag="stage_status_reply"
):
    protocol_version: int
    producer_generation: str
    consumer_generation: str
    transfer_id: str
    chunk_index: int
    source_slot_id: int
    source_slot_epoch: int
    status: SafeStatus
    producer_engine_id: str
    producer_rank: int
    consumer_engine_id: str
    consumer_rank: int


StageMessage: TypeAlias = (
    StageReady | StageReadComplete | StageCancel | StageStatusQuery | StageStatusReply
)
_STAGE_MESSAGE_DECODER = msgspec.msgpack.Decoder(StageMessage)


class StagingTransferIntent(msgspec.Struct, frozen=True):  # type: ignore[call-arg]
    """Router-selected immutable producer/consumer pairing."""

    protocol_version: int
    producer_generation: str
    consumer_generation: str
    transfer_id: str
    producer_request_id: str
    consumer_request_id: str
    producer_engine_id: str
    producer_rank: int
    producer_tp_size: int
    consumer_engine_id: str
    consumer_rank: int
    consumer_tp_size: int
    producer_host: str
    producer_port: int
    consumer_host: str
    consumer_port: int
    plan_id: str
    mode_attempt: int
    source_ranges_by_group: tuple[tuple[tuple[int, int], ...], ...] = ()

    @property
    def dedup_key(self) -> tuple[str, str, str, int, str, int, str]:
        return (
            self.producer_generation,
            self.consumer_generation,
            self.producer_engine_id,
            self.producer_rank,
            self.consumer_engine_id,
            self.consumer_rank,
            self.transfer_id,
        )


class StageModePrepared(msgspec.Struct, frozen=True):  # type: ignore[call-arg]
    transfer_id: str
    mode_attempt: int
    engine_id: str
    rank: int
    generation: str
    peer_engine_id: str
    peer_rank: int
    slot_bytes: int
    supported: bool
    reason: str = ""


class StageModeCommit(msgspec.Struct, frozen=True):  # type: ignore[call-arg]
    transfer_id: str
    mode_attempt: int
    mode: Literal["staged", "direct", "fail"]
    # (producer engine, producer rank, consumer engine, consumer rank, bytes)
    edges: tuple[tuple[str, int, str, int, int], ...]


class StageModeAbort(msgspec.Struct, frozen=True):  # type: ignore[call-arg]
    transfer_id: str
    mode_attempt: int
    reason: str = ""


class StageModeQuery(msgspec.Struct, frozen=True):  # type: ignore[call-arg]
    transfer_id: str
    mode_attempt: int


ModeDecision: TypeAlias = StageModeCommit | StageModeAbort


def _validate_mode_decision(decision: ModeDecision) -> None:
    if not decision.transfer_id:
        raise ValueError("Mode decision transfer_id must not be empty")
    if decision.mode_attempt < 0:
        raise ValueError("Mode decision attempt must be non-negative")
    if not isinstance(decision, StageModeCommit):
        return
    if not decision.edges:
        raise ValueError("Mode commit edges must not be empty")
    identities = [edge[:4] for edge in decision.edges]
    if len(identities) != len(set(identities)):
        raise ValueError("Mode commit edges must be unique")
    for (
        producer_engine,
        producer_rank,
        consumer_engine,
        consumer_rank,
        size,
    ) in decision.edges:
        if not producer_engine or not consumer_engine:
            raise ValueError("Mode commit edges must identify both engines")
        if min(producer_rank, consumer_rank) < 0 or size <= 0:
            raise ValueError("Mode commit edge ranks and size are invalid")


class ModeDecisionLedger:
    """Worker-side idempotent acceptance of durable Router decisions."""

    def __init__(self) -> None:
        self._decisions: dict[tuple[str, int], ModeDecision] = {}
        self._latest_attempt: dict[str, int] = {}
        self._committed: dict[str, StageModeCommit] = {}

    def accept(self, decision: ModeDecision) -> bool:
        _validate_mode_decision(decision)
        committed = self._committed.get(decision.transfer_id)
        if committed is not None:
            if committed == decision:
                return False
            raise RuntimeError("Conflicting staging mode decision after commit")
        latest = self._latest_attempt.get(decision.transfer_id, -1)
        if decision.mode_attempt < latest:
            return False
        key = (decision.transfer_id, decision.mode_attempt)
        current = self._decisions.get(key)
        if current is not None:
            if current != decision:
                raise RuntimeError("Conflicting staging mode decision")
            return False
        if decision.mode_attempt == latest:
            raise RuntimeError("Conflicting staging mode decision")
        self._latest_attempt[decision.transfer_id] = decision.mode_attempt
        self._decisions[key] = decision
        if isinstance(decision, StageModeCommit):
            self._committed[decision.transfer_id] = decision
        return True

    def get(self, transfer_id: str, mode_attempt: int) -> ModeDecision | None:
        return self._decisions.get((transfer_id, mode_attempt))


class IntentLedger:
    """Idempotently retain immutable Router intents and cancellation tombstones."""

    def __init__(self) -> None:
        self._intents: dict[
            tuple[str, str, str, int, str, int, str], StagingTransferIntent
        ] = {}
        self._cancelled: set[tuple[str, str, str, int, str, int, str]] = set()

    def accept(self, intent: StagingTransferIntent) -> bool:
        validate_intent(intent)
        current = self._intents.get(intent.dedup_key)
        if current is not None:
            if current != intent:
                raise RuntimeError("Conflicting staging transfer intent")
            return False
        if intent.dedup_key in self._cancelled:
            return False
        self._intents[intent.dedup_key] = intent
        return True

    def cancel(self, intent: StagingTransferIntent) -> None:
        self._intents.pop(intent.dedup_key, None)
        self._cancelled.add(intent.dedup_key)

    def get(
        self, key: tuple[str, str, str, int, str, int, str]
    ) -> StagingTransferIntent | None:
        return self._intents.get(key)


def validate_intent(intent: StagingTransferIntent) -> None:
    if intent.protocol_version != STAGING_PROTOCOL_VERSION:
        raise ValueError("Unsupported staging intent protocol version")
    for field in (
        "producer_generation",
        "consumer_generation",
        "transfer_id",
        "producer_request_id",
        "consumer_request_id",
        "producer_engine_id",
        "consumer_engine_id",
        "producer_host",
        "consumer_host",
        "plan_id",
    ):
        if not getattr(intent, field):
            raise ValueError(f"Staging intent {field} must not be empty")
    for field in ("producer_rank", "consumer_rank", "mode_attempt"):
        if getattr(intent, field) < 0:
            raise ValueError(f"Staging intent {field} must be non-negative")
    for field in ("producer_tp_size", "consumer_tp_size"):
        _positive_int(field, getattr(intent, field))
    for field in ("producer_port", "consumer_port"):
        port = getattr(intent, field)
        if not 0 < port <= 65535:
            raise ValueError(f"Staging intent {field} must be a valid TCP port")
    for ranges in intent.source_ranges_by_group:
        previous_end = -1
        for start, count in ranges:
            if start < 0 or count <= 0 or start < previous_end:
                raise ValueError(
                    "Staging source ranges must be positive, sorted, and disjoint"
                )
            previous_end = start + count


class DurableModeStore(Protocol):
    def load(self, transfer_id: str, mode_attempt: int) -> ModeDecision | None: ...

    def save(self, decision: ModeDecision) -> None: ...


class InMemoryModeStore:
    """Test/reference store; production Routers provide replicated storage."""

    def __init__(self) -> None:
        self._decisions: dict[tuple[str, int], ModeDecision] = {}

    def load(self, transfer_id: str, mode_attempt: int) -> ModeDecision | None:
        return self._decisions.get((transfer_id, mode_attempt))

    def save(self, decision: ModeDecision) -> None:
        _validate_mode_decision(decision)
        key = (decision.transfer_id, decision.mode_attempt)
        current = self._decisions.get(key)
        if current is not None and current != decision:
            raise RuntimeError("Conflicting durable staging mode decision")
        self._decisions[key] = decision


class ModeCoordinator:
    """Router-side all-edge PREPARED aggregation and durable decision replay."""

    def __init__(self, store: DurableModeStore) -> None:
        self._store = store
        self._prepared: dict[
            tuple[str, int], dict[tuple[str, int, str, int], StageModePrepared]
        ] = {}

    def record_prepared(self, prepared: StageModePrepared) -> bool:
        if (
            not prepared.transfer_id
            or not prepared.engine_id
            or not prepared.generation
            or not prepared.peer_engine_id
            or min(prepared.mode_attempt, prepared.rank, prepared.peer_rank) < 0
            or prepared.slot_bytes <= 0
        ):
            raise ValueError("Invalid STAGE_MODE_PREPARED")
        decision = self._store.load(prepared.transfer_id, prepared.mode_attempt)
        if decision is not None:
            return False
        key = (prepared.transfer_id, prepared.mode_attempt)
        edge_endpoint = (
            prepared.engine_id,
            prepared.rank,
            prepared.peer_engine_id,
            prepared.peer_rank,
        )
        records = self._prepared.setdefault(key, {})
        current = records.get(edge_endpoint)
        if current is not None:
            if current != prepared:
                raise RuntimeError("Conflicting STAGE_MODE_PREPARED")
            return False
        records[edge_endpoint] = prepared
        return True

    def decide(
        self,
        transfer_id: str,
        mode_attempt: int,
        edges: tuple[tuple[str, int, str, int], ...],
        fallback: Literal["direct", "fail"],
    ) -> StageModeCommit:
        if not edges or len(set(edges)) != len(edges):
            raise ValueError("Mode decision edges must be non-empty and unique")
        existing = self._store.load(transfer_id, mode_attempt)
        if existing is not None:
            if not isinstance(existing, StageModeCommit):
                raise RuntimeError("Mode attempt was already aborted")
            return existing
        records = self._prepared.get((transfer_id, mode_attempt), {})
        committed_edges: list[tuple[str, int, str, int, int]] = []
        all_supported = True
        for producer_engine, producer_rank, consumer_engine, consumer_rank in edges:
            producer = records.get(
                (producer_engine, producer_rank, consumer_engine, consumer_rank)
            )
            consumer = records.get(
                (consumer_engine, consumer_rank, producer_engine, producer_rank)
            )
            if producer is None or consumer is None:
                raise RuntimeError("Not all staging participants are PREPARED")
            all_supported &= producer.supported and consumer.supported
            committed_edges.append(
                (
                    producer_engine,
                    producer_rank,
                    consumer_engine,
                    consumer_rank,
                    min(producer.slot_bytes, consumer.slot_bytes),
                )
            )
        mode: Literal["staged", "direct", "fail"] = (
            "staged" if all_supported else fallback
        )
        decision = StageModeCommit(
            transfer_id, mode_attempt, mode, tuple(committed_edges)
        )
        self._store.save(decision)
        return decision

    def abort(self, abort: StageModeAbort) -> StageModeAbort:
        existing = self._store.load(abort.transfer_id, abort.mode_attempt)
        if existing is not None:
            if existing != abort:
                raise RuntimeError("Mode attempt already has a durable decision")
            return abort
        self._store.save(abort)
        return abort

    def query(self, query: StageModeQuery) -> ModeDecision | None:
        return self._store.load(query.transfer_id, query.mode_attempt)


def encode_stage_message(message: StageMessage) -> bytes:
    """Encode a typed staging notification with a disjoint wire prefix."""
    _validate_stage_message(message)
    return STAGE_NOTIF_PREFIX + msgspec.msgpack.encode(message)


def decode_stage_message(payload: bytes) -> StageMessage:
    """Decode and validate a staging notification."""
    if not payload.startswith(STAGE_NOTIF_PREFIX):
        raise ValueError("Not a NIXL staging notification")
    encoded = payload[len(STAGE_NOTIF_PREFIX) :]
    try:
        message = _STAGE_MESSAGE_DECODER.decode(encoded)
    except (msgspec.DecodeError, msgspec.ValidationError) as exc:
        raise ValueError("Invalid NIXL staging notification") from exc
    if message.protocol_version != STAGING_PROTOCOL_VERSION:
        raise ValueError(
            f"Unsupported NIXL staging protocol version: {message.protocol_version}"
        )
    _validate_stage_message(message)
    return message


def _validate_stage_message(message: StageMessage) -> None:
    for field in ("producer_generation", "consumer_generation", "transfer_id"):
        if not getattr(message, field):
            raise ValueError(f"Staging message {field} must not be empty")
    if isinstance(message, StageCancel):
        if not message.producer_request_id or not message.consumer_request_id:
            raise ValueError("STAGE_CANCEL must identify both requests")
        return
    for field in ("producer_engine_id", "consumer_engine_id"):
        if not getattr(message, field):
            raise ValueError(f"Staging message {field} must not be empty")
    for field in ("producer_rank", "consumer_rank", "chunk_index", "source_slot_id"):
        if getattr(message, field) < 0:
            raise ValueError(f"Staging message {field} must be non-negative")
    if message.source_slot_epoch <= 0:
        raise ValueError("Staging message source_slot_epoch must be positive")
    if isinstance(message, StageReady):
        if message.valid_bytes <= 0:
            raise ValueError("STAGE_READY valid_bytes must be positive")
        if not message.plan_id or not message.request_id:
            raise ValueError("STAGE_READY must identify its request and plan")


def ready_identity(message: StageReady) -> ReadyIdentity:
    """Return the complete immutable identity of a READY occupant."""
    return (
        message.producer_engine_id,
        message.producer_rank,
        message.producer_generation,
        message.consumer_engine_id,
        message.consumer_rank,
        message.consumer_generation,
        message.transfer_id,
        message.chunk_index,
        message.source_slot_id,
        message.source_slot_epoch,
    )


def completion_matches_ready(
    completion: StageReadComplete | StageStatusReply,
    ready: StageReady,
) -> bool:
    """Check a release proof against every field in the READY identity."""
    return (
        completion.producer_engine_id == ready.producer_engine_id
        and completion.producer_rank == ready.producer_rank
        and completion.producer_generation == ready.producer_generation
        and completion.consumer_engine_id == ready.consumer_engine_id
        and completion.consumer_rank == ready.consumer_rank
        and completion.consumer_generation == ready.consumer_generation
        and completion.transfer_id == ready.transfer_id
        and completion.chunk_index == ready.chunk_index
        and completion.source_slot_id == ready.source_slot_id
        and completion.source_slot_epoch == ready.source_slot_epoch
    )


@dataclass(frozen=True)
class CopySegment:
    """A contiguous byte range copied between one KV page and a chunk."""

    region_index: int
    block_id: int
    block_offset: int
    chunk_offset: int
    length: int
    scatter: bool = True


@dataclass(frozen=True)
class StagingChunk:
    index: int
    valid_bytes: int
    segments: tuple[CopySegment, ...]


@dataclass(frozen=True)
class StagingCopyPlan:
    """Compact deterministic mapping between KV pages and staging chunks."""

    plan_id: str
    total_bytes: int
    slot_bytes: int
    chunks: tuple[StagingChunk, ...]

    @classmethod
    def from_wire_segments(
        cls,
        plan_id: str,
        wire_segments: tuple[tuple[int, int, int, int, bool], ...],
        slot_bytes: int,
    ) -> StagingCopyPlan:
        """Build chunks from topology-produced byte ranges.

        Each input tuple is ``(region, block, offset, length, scatter)``. This
        is the common entry point for heterogeneous TP and HMA/Mamba plans,
        whose owned head/state slices may be smaller than a physical page.
        """
        if not plan_id:
            raise ValueError("plan_id must not be empty")
        _positive_int("slot_bytes", slot_bytes)
        chunks: list[StagingChunk] = []
        chunk_segments: list[CopySegment] = []
        chunk_offset = 0
        total_bytes = 0
        for region, block, block_offset, length, should_scatter in wire_segments:
            if min(region, block, block_offset) < 0 or length <= 0:
                raise ValueError("Wire segment indices and lengths are invalid")
            consumed = 0
            while consumed < length:
                part = min(length - consumed, slot_bytes - chunk_offset)
                chunk_segments.append(
                    CopySegment(
                        region,
                        block,
                        block_offset + consumed,
                        chunk_offset,
                        part,
                        should_scatter,
                    )
                )
                consumed += part
                chunk_offset += part
                total_bytes += part
                if chunk_offset == slot_bytes:
                    chunks.append(
                        StagingChunk(len(chunks), chunk_offset, tuple(chunk_segments))
                    )
                    chunk_segments = []
                    chunk_offset = 0
        if chunk_segments:
            chunks.append(
                StagingChunk(len(chunks), chunk_offset, tuple(chunk_segments))
            )
        return cls(plan_id, total_bytes, slot_bytes, tuple(chunks))

    @classmethod
    def build(
        cls,
        plan_id: str,
        region_block_ids: tuple[tuple[int, ...], ...],
        region_block_bytes: tuple[int, ...],
        slot_bytes: int,
        scatter_block_ids: tuple[frozenset[int], ...] | None = None,
    ) -> StagingCopyPlan:
        if not plan_id:
            raise ValueError("plan_id must not be empty")
        if slot_bytes <= 0:
            raise ValueError("slot_bytes must be positive")
        if len(region_block_ids) != len(region_block_bytes):
            raise ValueError("Each staging region must have a block size")
        if any(size <= 0 for size in region_block_bytes):
            raise ValueError("Region block sizes must be positive")
        if any(block < 0 for blocks in region_block_ids for block in blocks):
            raise ValueError("block IDs must be non-negative")
        if scatter_block_ids is not None and len(scatter_block_ids) != len(
            region_block_ids
        ):
            raise ValueError("Each staging region must have a scatter block set")

        chunks: list[StagingChunk] = []
        segments: list[CopySegment] = []
        chunk_offset = 0
        total_bytes = 0

        def finish_chunk() -> None:
            nonlocal segments, chunk_offset
            if chunk_offset:
                chunks.append(StagingChunk(len(chunks), chunk_offset, tuple(segments)))
                segments = []
                chunk_offset = 0

        for region_index, (blocks, block_bytes) in enumerate(
            zip(region_block_ids, region_block_bytes)
        ):
            for block_id in blocks:
                block_offset = 0
                while block_offset < block_bytes:
                    length = min(block_bytes - block_offset, slot_bytes - chunk_offset)
                    segments.append(
                        CopySegment(
                            region_index=region_index,
                            block_id=block_id,
                            block_offset=block_offset,
                            chunk_offset=chunk_offset,
                            length=length,
                            scatter=(
                                scatter_block_ids is None
                                or block_id in scatter_block_ids[region_index]
                            ),
                        )
                    )
                    block_offset += length
                    chunk_offset += length
                    total_bytes += length
                    if chunk_offset == slot_bytes:
                        finish_chunk()
        finish_chunk()
        return cls(plan_id, total_bytes, slot_bytes, tuple(chunks))


class StagingTransferSession:
    """Freeze one transfer's mode and local copy plan before data movement."""

    def __init__(self, intent: StagingTransferIntent) -> None:
        validate_intent(intent)
        self.intent = intent
        self._decision: ModeDecision | None = None
        self._plan: StagingCopyPlan | None = None

    @property
    def edge_key(self) -> TransferEdgeKey:
        return transfer_edge_key(self.intent)

    @property
    def decision(self) -> ModeDecision | None:
        return self._decision

    @property
    def plan(self) -> StagingCopyPlan | None:
        return self._plan

    def accept_decision(self, decision: ModeDecision) -> bool:
        """Accept one immutable Router decision for the intent's attempt."""
        _validate_mode_decision(decision)
        if (
            decision.transfer_id != self.intent.transfer_id
            or decision.mode_attempt != self.intent.mode_attempt
        ):
            raise RuntimeError("Mode decision does not match the transfer intent")
        if self._decision is not None:
            if self._decision != decision:
                raise RuntimeError("Conflicting staging mode decision")
            return False
        self._decision = decision
        return True

    def freeze_plan(self, plan: StagingCopyPlan) -> bool:
        """Freeze chunk geometry after a staged commit, before gather or READ."""
        decision = self._decision
        if not isinstance(decision, StageModeCommit) or decision.mode != "staged":
            raise RuntimeError("A staged MODE_COMMIT is required before planning")
        if plan.plan_id != self.intent.plan_id:
            raise RuntimeError("Copy plan ID does not match the transfer intent")
        wire_chunk_bytes = committed_wire_chunk_bytes(self.intent, decision)
        if plan.slot_bytes != wire_chunk_bytes:
            raise RuntimeError("Copy plan does not use the committed chunk geometry")
        if self._plan is not None:
            if self._plan != plan:
                raise RuntimeError("Copy plan changed after it was frozen")
            return False
        self._plan = plan
        return True

    def validate_ready(self, ready: StageReady) -> StagingChunk:
        """Validate READY against the frozen intent, edge, and chunk plan."""
        _validate_stage_message(ready)
        plan = self._plan
        if plan is None:
            raise RuntimeError("READY arrived before the copy plan was frozen")
        intent = self.intent
        if (
            ready.producer_generation != intent.producer_generation
            or ready.consumer_generation != intent.consumer_generation
            or ready.transfer_id != intent.transfer_id
            or ready.request_id != intent.producer_request_id
            or ready.plan_id != intent.plan_id
            or ready.producer_engine_id != intent.producer_engine_id
            or ready.producer_rank != intent.producer_rank
            or ready.consumer_engine_id != intent.consumer_engine_id
            or ready.consumer_rank != intent.consumer_rank
        ):
            raise RuntimeError("READY does not match the committed transfer intent")
        if ready.chunk_index >= len(plan.chunks):
            raise RuntimeError("READY chunk index is outside the copy plan")
        chunk = plan.chunks[ready.chunk_index]
        if ready.valid_bytes != chunk.valid_bytes:
            raise RuntimeError("READY valid length does not match the copy plan")
        return chunk


class StagingSessionRegistry:
    """Bound worker-local registry for committed transfer sessions."""

    def __init__(self) -> None:
        self._sessions: dict[TransferEdgeKey, StagingTransferSession] = {}
        self._terminal: dict[TransferEdgeKey, tuple[str, str]] = {}

    def register(self, intent: StagingTransferIntent) -> StagingTransferSession:
        validate_intent(intent)
        key = transfer_edge_key(intent)
        current = self._sessions.get(key)
        if current is not None:
            if current.intent != intent:
                raise RuntimeError("A transfer ID cannot be rebound to another peer")
            return current
        if key in self._terminal:
            raise RuntimeError("A terminal transfer edge cannot be reused")
        session = StagingTransferSession(intent)
        self._sessions[key] = session
        return session

    def get(self, transfer_id: str) -> StagingTransferSession | None:
        matches = self.for_transfer(transfer_id)
        return matches[0] if len(matches) == 1 else None

    def get_for_ready(self, ready: StageReady) -> StagingTransferSession | None:
        return self._sessions.get(transfer_edge_key(ready))

    def for_transfer(self, transfer_id: str) -> tuple[StagingTransferSession, ...]:
        return tuple(
            session for key, session in self._sessions.items() if key[0] == transfer_id
        )

    def require_ready(self, ready: StageReady) -> StagingChunk:
        session = self.get_for_ready(ready)
        if session is None:
            raise RuntimeError("READY belongs to an unknown transfer")
        return session.validate_ready(ready)

    def retire_edge(self, key: TransferEdgeKey) -> None:
        session = self._sessions.pop(key, None)
        if session is not None:
            self._terminal[key] = (
                session.intent.producer_generation,
                session.intent.consumer_generation,
            )

    def retire(self, transfer_id: str) -> None:
        for session in self.for_transfer(transfer_id):
            self.retire_edge(session.edge_key)

    def retire_generation(
        self, producer_generation: str, consumer_generation: str
    ) -> None:
        """Bound intent tombstones after a certified generation teardown."""
        generations = (producer_generation, consumer_generation)
        for key, session in list(self._sessions.items()):
            if (
                session.intent.producer_generation,
                session.intent.consumer_generation,
            ) == generations:
                del self._sessions[key]
        for key, terminal_generations in list(self._terminal.items()):
            if terminal_generations == generations:
                del self._terminal[key]


class TransferCompletionTracker:
    """Report receive completion only after every planned scatter finishes."""

    def __init__(self) -> None:
        self._expected: dict[str, int] = {}
        self._scattered: dict[str, set[int]] = {}
        self._reported: set[str] = set()
        self._failed: set[str] = set()

    def register(self, transfer_id: str, chunk_count: int) -> None:
        _positive_int("chunk_count", chunk_count)
        current = self._expected.get(transfer_id)
        if current is not None and current != chunk_count:
            raise RuntimeError("Transfer chunk count changed after registration")
        self._expected[transfer_id] = chunk_count
        self._scattered.setdefault(transfer_id, set())

    def fail(self, transfer_id: str) -> None:
        self._failed.add(transfer_id)

    def retire(self, transfer_id: str) -> None:
        self._expected.pop(transfer_id, None)
        self._scattered.pop(transfer_id, None)
        self._reported.discard(transfer_id)
        self._failed.discard(transfer_id)

    def is_registered(self, transfer_id: str) -> bool:
        return transfer_id in self._expected

    def observe_scatter(self, ready: StageReady, tracker_id: str | None = None) -> bool:
        tracker_id = ready.transfer_id if tracker_id is None else tracker_id
        expected = self._expected.get(tracker_id)
        if expected is None:
            raise RuntimeError("Scatter belongs to an unregistered transfer")
        if not 0 <= ready.chunk_index < expected:
            raise RuntimeError("Scatter chunk index is outside the transfer plan")
        self._scattered[tracker_id].add(ready.chunk_index)
        if tracker_id in self._failed:
            return False
        complete = len(self._scattered[tracker_id]) == expected
        if not complete or tracker_id in self._reported:
            return False
        self._reported.add(tracker_id)
        return True


def committed_wire_chunk_bytes(
    intent: StagingTransferIntent,
    commit: StageModeCommit,
) -> int:
    """Return the Router-frozen edge geometry or fail closed."""
    if commit.transfer_id != intent.transfer_id:
        raise RuntimeError("Mode commit belongs to another transfer")
    if commit.mode_attempt != intent.mode_attempt:
        raise RuntimeError("Mode commit attempt does not match the intent")
    if commit.mode != "staged":
        raise RuntimeError(f"Transfer mode is {commit.mode}, not staged")
    edge = (
        intent.producer_engine_id,
        intent.producer_rank,
        intent.consumer_engine_id,
        intent.consumer_rank,
    )
    matches = [item[4] for item in commit.edges if item[:4] == edge]
    if len(matches) != 1 or matches[0] <= 0:
        raise RuntimeError("Mode commit does not contain one valid transfer edge")
    return matches[0]


def validate_preflight(
    intent: StagingTransferIntent,
    config: StagingConfig,
    *,
    engine_id: str,
    rank: int,
    generation: str,
    pipeline_parallel_size: int,
    backend_capabilities: BackendSafetyCapabilities | None = None,
) -> StageModePrepared:
    """Validate local request capability without authorizing data movement."""
    validate_intent(intent)
    if pipeline_parallel_size != 1:
        supported, reason = False, "pipeline parallelism is unsupported"
    elif backend_capabilities is None or not backend_capabilities.supports_staging:
        supported, reason = False, "backend safety contract is not verified"
    elif intent.protocol_version != STAGING_PROTOCOL_VERSION:
        supported, reason = False, "protocol version mismatch"
    elif not config.enabled or config.slot_count == 0:
        supported, reason = False, "staging pool is disabled or empty"
    else:
        is_producer = (engine_id, rank, generation) == (
            intent.producer_engine_id,
            intent.producer_rank,
            intent.producer_generation,
        )
        is_consumer = (engine_id, rank, generation) == (
            intent.consumer_engine_id,
            intent.consumer_rank,
            intent.consumer_generation,
        )
        supported = is_producer or is_consumer
        reason = "" if supported else "intent does not identify this worker"
    peer_engine_id = (
        intent.consumer_engine_id
        if engine_id == intent.producer_engine_id
        else intent.producer_engine_id
    )
    peer_rank = (
        intent.consumer_rank
        if engine_id == intent.producer_engine_id
        else intent.producer_rank
    )
    return StageModePrepared(
        transfer_id=intent.transfer_id,
        mode_attempt=intent.mode_attempt,
        engine_id=engine_id,
        rank=rank,
        generation=generation,
        peer_engine_id=peer_engine_id,
        peer_rank=peer_rank,
        slot_bytes=config.slot_bytes,
        supported=supported,
        reason=reason,
    )


def gather_chunk(
    regions: tuple[torch.Tensor, ...],
    slot: torch.Tensor,
    chunk: StagingChunk,
) -> None:
    """Gather one copy-plan chunk into a byte-addressed staging slot."""
    if slot.dtype != torch.uint8 or slot.numel() < chunk.valid_bytes:
        raise ValueError("staging slot is smaller than the chunk")
    for segment in chunk.segments:
        page = regions[segment.region_index][segment.block_id]
        if not page.is_contiguous():
            raise ValueError("staging requires contiguous KV pages")
        page_bytes = page.view(torch.uint8).reshape(-1)
        slot.narrow(0, segment.chunk_offset, segment.length).copy_(
            page_bytes.narrow(0, segment.block_offset, segment.length)
        )


def scatter_chunk(
    regions: tuple[torch.Tensor, ...],
    slot: torch.Tensor,
    chunk: StagingChunk,
) -> None:
    """Scatter one copy-plan chunk from a byte-addressed staging slot."""
    if slot.dtype != torch.uint8 or slot.numel() < chunk.valid_bytes:
        raise ValueError("staging slot is smaller than the chunk")
    for segment in chunk.segments:
        if not segment.scatter:
            continue
        page = regions[segment.region_index][segment.block_id]
        if not page.is_contiguous():
            raise ValueError("staging requires contiguous KV pages")
        page.view(torch.uint8).reshape(-1).narrow(
            0, segment.block_offset, segment.length
        ).copy_(slot.narrow(0, segment.chunk_offset, segment.length))


class ProducerSlotState(str, Enum):
    FREE = "free"
    PACKING = "packing"
    READY_LOCAL = "ready_local"
    READY = "ready_local"
    EXPOSED = "exposed"
    READING = "exposed"
    QUARANTINED = "quarantined"


class ConsumerSlotState(str, Enum):
    FREE = "free"
    READING = "reading"
    SCATTERING = "scattering"
    QUARANTINED = "quarantined"


class RemoteChunkState(str, Enum):
    NOT_SEEN = "not_seen"
    QUEUED = "queued"
    POSTING = "posting"
    INFLIGHT = "inflight"
    DONE = "done"
    ABORTED = "aborted"
    UNKNOWN = "unknown"


SlotState: TypeAlias = ProducerSlotState | ConsumerSlotState
SlotOwner: TypeAlias = tuple[Hashable, int]


@dataclass
class StagingSlot:
    slot_id: int
    state: SlotState
    owner: SlotOwner | None = None
    epoch: int = 0
    pending_consumers: set[ConsumerIdentity] | None = None


class StagingSlotPool:
    """Single-thread-owned contiguous staging allocation and slot state."""

    def __init__(
        self,
        slot_count: int,
        slot_bytes: int = 1,
        device: torch.device | str | None = None,
        producer: bool = True,
    ) -> None:
        _positive_int("slot_count", slot_count)
        _positive_int("slot_bytes", slot_bytes)
        self.slot_bytes = slot_bytes
        self.producer = producer
        self._buffer = (
            torch.empty(slot_count * slot_bytes, dtype=torch.uint8, device=device)
            if device is not None
            else None
        )
        free = ProducerSlotState.FREE if producer else ConsumerSlotState.FREE
        self.slots = [StagingSlot(slot_id=i, state=free) for i in range(slot_count)]

    @property
    def usable_bytes(self) -> int:
        return len(self.slots) * self.slot_bytes

    @property
    def quarantine_bytes(self) -> int:
        return sum(
            self.slot_bytes
            for slot in self.slots
            if slot.state
            in (ProducerSlotState.QUARANTINED, ConsumerSlotState.QUARANTINED)
        )

    def pool_registration_region(self, device_id: int) -> tuple[int, int, int, str]:
        if self._buffer is None:
            raise RuntimeError("The staging pool has no backing tensor")
        return self._buffer.data_ptr(), self.usable_bytes, device_id, ""

    def slot_transfer_regions(self, device_id: int) -> list[tuple[int, int, int]]:
        if self._buffer is None:
            raise RuntimeError("The staging pool has no backing tensor")
        base = self._buffer.data_ptr()
        return [
            (base + slot.slot_id * self.slot_bytes, self.slot_bytes, device_id)
            for slot in self.slots
        ]

    def slot_view(self, slot_id: int, valid_bytes: int | None = None) -> torch.Tensor:
        if self._buffer is None:
            raise RuntimeError("The staging pool has no backing tensor")
        self._get(slot_id)
        length = self.slot_bytes if valid_bytes is None else valid_bytes
        if not 0 <= length <= self.slot_bytes:
            raise ValueError(
                f"valid_bytes must be in [0, {self.slot_bytes}], got {length}"
            )
        return self._buffer.narrow(0, slot_id * self.slot_bytes, length)

    def acquire(self, owner: SlotOwner) -> StagingSlot | None:
        free = ProducerSlotState.FREE if self.producer else ConsumerSlotState.FREE
        initial = (
            ProducerSlotState.PACKING if self.producer else ConsumerSlotState.READING
        )
        for slot in self.slots:
            if slot.state == free:
                slot.state = initial
                slot.owner = owner
                slot.epoch += 1
                if self.producer:
                    slot.pending_consumers = set()
                return slot
        return None

    def transition(
        self, slot_id: int, expected: SlotState, new_state: SlotState
    ) -> None:
        slot = self._get(slot_id)
        if slot.state != expected:
            raise RuntimeError(
                f"Slot {slot_id} is {slot.state.value}, expected {expected.value}"
            )
        allowed = self._allowed_transitions()
        if new_state not in allowed[expected]:
            raise RuntimeError(
                f"Invalid slot transition {expected.value} -> {new_state.value}"
            )
        slot.state = new_state

    def expose(self, slot_id: int, consumers: set[ConsumerIdentity]) -> None:
        if not consumers:
            raise ValueError("An exposed producer slot needs at least one consumer")
        slot = self._get(slot_id)
        self.transition(
            slot_id, ProducerSlotState.READY_LOCAL, ProducerSlotState.EXPOSED
        )
        slot.pending_consumers = set(consumers)

    def complete_consumer(
        self,
        slot_id: int,
        epoch: int,
        consumer: ConsumerIdentity,
    ) -> bool:
        slot = self._get(slot_id)
        if slot.epoch != epoch or slot.state != ProducerSlotState.EXPOSED:
            return False
        assert slot.pending_consumers is not None
        if consumer not in slot.pending_consumers:
            return False
        slot.pending_consumers.remove(consumer)
        if slot.pending_consumers:
            return False
        self.release(slot_id)
        return True

    def release(self, slot_id: int) -> None:
        slot = self._get(slot_id)
        quarantined = (
            ProducerSlotState.QUARANTINED
            if self.producer
            else ConsumerSlotState.QUARANTINED
        )
        if slot.state == quarantined:
            raise RuntimeError(f"Quarantined slot {slot_id} cannot be reused")
        if (
            self.producer
            and slot.state == ProducerSlotState.EXPOSED
            and slot.pending_consumers
        ):
            raise RuntimeError(f"Exposed slot {slot_id} still has consumers")
        slot.state = ProducerSlotState.FREE if self.producer else ConsumerSlotState.FREE
        slot.owner = None
        slot.pending_consumers = None

    def quarantine(self, slot_id: int) -> None:
        slot = self._get(slot_id)
        slot.state = (
            ProducerSlotState.QUARANTINED
            if self.producer
            else ConsumerSlotState.QUARANTINED
        )

    def retire_quarantined(self, slot_id: int) -> None:
        """Reuse a quarantined slot after the caller completed a safe barrier."""
        slot = self._get(slot_id)
        quarantined = (
            ProducerSlotState.QUARANTINED
            if self.producer
            else ConsumerSlotState.QUARANTINED
        )
        if slot.state != quarantined:
            raise RuntimeError(f"Slot {slot_id} is not quarantined")
        slot.state = ProducerSlotState.FREE if self.producer else ConsumerSlotState.FREE
        slot.owner = None
        slot.pending_consumers = None

    def _get(self, slot_id: int) -> StagingSlot:
        if not 0 <= slot_id < len(self.slots):
            raise IndexError(f"Invalid staging slot ID: {slot_id}")
        return self.slots[slot_id]

    def _allowed_transitions(self) -> dict[SlotState, set[SlotState]]:
        if self.producer:
            return {
                ProducerSlotState.FREE: {ProducerSlotState.PACKING},
                ProducerSlotState.PACKING: {
                    ProducerSlotState.READY_LOCAL,
                    ProducerSlotState.QUARANTINED,
                },
                ProducerSlotState.READY_LOCAL: {
                    ProducerSlotState.EXPOSED,
                    ProducerSlotState.FREE,
                    ProducerSlotState.QUARANTINED,
                },
                ProducerSlotState.EXPOSED: {
                    ProducerSlotState.FREE,
                    ProducerSlotState.QUARANTINED,
                },
                ProducerSlotState.QUARANTINED: set(),
            }
        return {
            ConsumerSlotState.FREE: {ConsumerSlotState.READING},
            ConsumerSlotState.READING: {
                ConsumerSlotState.SCATTERING,
                ConsumerSlotState.QUARANTINED,
            },
            ConsumerSlotState.SCATTERING: {
                ConsumerSlotState.FREE,
                ConsumerSlotState.QUARANTINED,
            },
            ConsumerSlotState.QUARANTINED: set(),
        }


# Name used by the design and by early tests.
StagingSlotManager = StagingSlotPool


class FairReadyQueue:
    """Round-robin READY queue across producers, then requests."""

    def __init__(self, max_ready_per_request: int | None = None) -> None:
        if max_ready_per_request is not None:
            _positive_int("max_ready_per_request", max_ready_per_request)
        self.max_ready_per_request = max_ready_per_request
        self._queues: OrderedDict[Hashable, OrderedDict[str, deque[StageReady]]] = (
            OrderedDict()
        )
        self._keys: set[tuple[Any, ...]] = set()

    @staticmethod
    def identity(message: StageReady) -> tuple[Any, ...]:
        return ready_identity(message)

    def push(self, producer: Hashable, message: StageReady) -> bool:
        key = self.identity(message)
        if key in self._keys:
            return False
        requests = self._queues.setdefault(producer, OrderedDict())
        messages = requests.setdefault(message.request_id, deque())
        if (
            self.max_ready_per_request is not None
            and len(messages) >= self.max_ready_per_request
        ):
            return False
        messages.append(message)
        self._keys.add(key)
        return True

    def remove_transfer(self, transfer_id: str) -> list[StageReady]:
        """Remove queued chunks for a cancelled transfer."""
        removed: list[StageReady] = []
        for producer, requests in list(self._queues.items()):
            for request_id, messages in list(requests.items()):
                kept: deque[StageReady] = deque()
                for message in messages:
                    if message.transfer_id == transfer_id:
                        self._keys.remove(self.identity(message))
                        removed.append(message)
                    else:
                        kept.append(message)
                if kept:
                    requests[request_id] = kept
                else:
                    del requests[request_id]
            if not requests:
                del self._queues[producer]
        return removed

    def pop(self) -> tuple[Hashable, StageReady] | None:
        if not self._queues:
            return None
        producer, requests = self._queues.popitem(last=False)
        request_id, messages = requests.popitem(last=False)
        message = messages.popleft()
        self._keys.remove(self.identity(message))
        if messages:
            requests[request_id] = messages
        if requests:
            self._queues[producer] = requests
        return producer, message

    def __len__(self) -> int:
        return sum(
            len(messages)
            for requests in self._queues.values()
            for messages in requests.values()
        )


@dataclass(frozen=True)
class _ChunkEntry:
    epoch: int
    identity: ReadyIdentity
    state: RemoteChunkState


class ChunkLedger:
    """Bounded per-source-slot deduplication and safe-status ledger."""

    _TERMINAL = frozenset({RemoteChunkState.DONE, RemoteChunkState.ABORTED})

    def __init__(self) -> None:
        self._entries: dict[SourceSlotKey, _ChunkEntry] = {}

    @staticmethod
    def _slot_key(message: StageReady | StageStatusQuery) -> SourceSlotKey:
        return (
            message.producer_engine_id,
            message.producer_rank,
            message.producer_generation,
            message.source_slot_id,
        )

    def observe_ready(self, message: StageReady) -> RemoteChunkState:
        slot_key = self._slot_key(message)
        identity = ready_identity(message)
        current = self._entries.get(slot_key)
        if current is None:
            self._entries[slot_key] = _ChunkEntry(
                message.source_slot_epoch, identity, RemoteChunkState.QUEUED
            )
            return RemoteChunkState.QUEUED
        if message.source_slot_epoch < current.epoch:
            return RemoteChunkState.ABORTED
        if message.source_slot_epoch == current.epoch:
            if current.identity != identity:
                raise RuntimeError("Conflicting READY identity for source slot epoch")
            return current.state
        if current.state not in self._TERMINAL:
            raise RuntimeError("A source slot advanced before reaching a safe terminal")
        self._entries[slot_key] = _ChunkEntry(
            message.source_slot_epoch, identity, RemoteChunkState.QUEUED
        )
        return RemoteChunkState.QUEUED

    def transition(
        self,
        message: StageReady,
        expected: RemoteChunkState,
        new_state: RemoteChunkState,
    ) -> None:
        slot_key = self._slot_key(message)
        current = self._entries.get(slot_key)
        if (
            current is None
            or current.epoch != message.source_slot_epoch
            or current.identity != ready_identity(message)
            or current.state != expected
        ):
            raise RuntimeError(f"Unexpected chunk state: {current}")
        allowed = {
            RemoteChunkState.QUEUED: {
                RemoteChunkState.POSTING,
                RemoteChunkState.ABORTED,
            },
            RemoteChunkState.POSTING: {
                RemoteChunkState.INFLIGHT,
                RemoteChunkState.ABORTED,
                RemoteChunkState.UNKNOWN,
            },
            RemoteChunkState.INFLIGHT: {
                RemoteChunkState.DONE,
                RemoteChunkState.UNKNOWN,
            },
            RemoteChunkState.UNKNOWN: {RemoteChunkState.DONE},
        }
        if new_state not in allowed.get(expected, set()):
            raise RuntimeError(
                f"Invalid chunk transition {expected.value} -> {new_state.value}"
            )
        self._entries[slot_key] = replace(current, state=new_state)

    def abort_queued(self, message: StageReady) -> bool:
        """Install the ABORTED tombstone before acknowledging cancellation."""
        if self.state(message) != RemoteChunkState.QUEUED:
            return False
        self.transition(message, RemoteChunkState.QUEUED, RemoteChunkState.ABORTED)
        return True

    def status(self, query: StageStatusQuery) -> SafeStatus:
        slot_key = self._slot_key(query)
        current = self._entries.get(slot_key)
        if current is None:
            return "unknown"
        if query.source_slot_epoch < current.epoch:
            return "safe_retired"
        if query.source_slot_epoch != current.epoch:
            return "unknown"
        identity = (
            query.producer_engine_id,
            query.producer_rank,
            query.producer_generation,
            query.consumer_engine_id,
            query.consumer_rank,
            query.consumer_generation,
            query.transfer_id,
            query.chunk_index,
            query.source_slot_id,
            query.source_slot_epoch,
        )
        if identity != current.identity:
            return "unknown"
        statuses: dict[RemoteChunkState, SafeStatus] = {
            RemoteChunkState.ABORTED: "safe_not_submitted",
            RemoteChunkState.DONE: "safe_complete",
            RemoteChunkState.INFLIGHT: "inflight",
            RemoteChunkState.POSTING: "unknown",
            RemoteChunkState.UNKNOWN: "unknown",
            RemoteChunkState.QUEUED: "unknown",
            RemoteChunkState.NOT_SEEN: "unknown",
        }
        return statuses[current.state]

    def state(self, message: StageReady) -> RemoteChunkState | None:
        """Return the state for the exact source-slot occupant."""
        slot_key = self._slot_key(message)
        current = self._entries.get(slot_key)
        if (
            current is None
            or current.epoch != message.source_slot_epoch
            or current.identity != ready_identity(message)
        ):
            return None
        return current.state

    def retire_generation(self, producer_engine_id: str, generation: str) -> None:
        """Drop watermarks only after the caller completed a backend barrier."""
        for key in list(self._entries):
            if key[0] == producer_engine_id and key[2] == generation:
                del self._entries[key]


@dataclass
class _PendingReadComplete:
    message: StageReadComplete
    handle_released: bool = False


class ReadCompletionOutbox:
    """Retry explicit READ_COMPLETE only after recording safe NIXL DONE."""

    def __init__(self, ledger: ChunkLedger) -> None:
        self._ledger = ledger
        self._pending: OrderedDict[tuple[Any, ...], _PendingReadComplete] = (
            OrderedDict()
        )
        self._notified_epoch: dict[tuple[str, int, str, int], int] = {}

    @staticmethod
    def _identity(message: StageReady) -> tuple[Any, ...]:
        return FairReadyQueue.identity(message)

    @staticmethod
    def _slot_key(message: StageReady) -> tuple[str, int, str, int]:
        return (
            message.producer_engine_id,
            message.producer_rank,
            message.producer_generation,
            message.source_slot_id,
        )

    @staticmethod
    def _completion(message: StageReady) -> StageReadComplete:
        return StageReadComplete(
            protocol_version=message.protocol_version,
            producer_generation=message.producer_generation,
            transfer_id=message.transfer_id,
            chunk_index=message.chunk_index,
            source_slot_id=message.source_slot_id,
            consumer_engine_id=message.consumer_engine_id,
            consumer_rank=message.consumer_rank,
            consumer_generation=message.consumer_generation,
            source_slot_epoch=message.source_slot_epoch,
            producer_engine_id=message.producer_engine_id,
            producer_rank=message.producer_rank,
        )

    def observe_done(
        self, message: StageReady, release_handle: Callable[[], None]
    ) -> bool:
        """Record DONE, release its handle, then make completion sendable.

        Returns whether a new completion became sendable. If handle release raises,
        the DONE tombstone remains installed and a later call retries the release.
        """
        key = self._identity(message)
        slot_key = self._slot_key(message)
        if self._notified_epoch.get(slot_key, -1) >= message.source_slot_epoch:
            return False

        state = self._ledger.state(message)
        if state in (RemoteChunkState.INFLIGHT, RemoteChunkState.UNKNOWN):
            self._ledger.transition(message, state, RemoteChunkState.DONE)
        elif state != RemoteChunkState.DONE:
            raise RuntimeError(f"Cannot complete READ from state {state}")

        pending = self._pending.setdefault(
            key, _PendingReadComplete(self._completion(message))
        )
        if pending.handle_released:
            return False
        release_handle()
        pending.handle_released = True
        return True

    def send_next(self, send_notification: Callable[[bytes], None]) -> bool:
        """Send one completion, retaining it when delivery raises."""
        for key, pending in self._pending.items():
            if not pending.handle_released:
                continue
            send_notification(encode_stage_message(pending.message))
            slot_key = (
                pending.message.producer_engine_id,
                pending.message.producer_rank,
                pending.message.producer_generation,
                pending.message.source_slot_id,
            )
            self._notified_epoch[slot_key] = pending.message.source_slot_epoch
            del self._pending[key]
            return True
        return False

    def __len__(self) -> int:
        return len(self._pending)


@dataclass(frozen=True)
class BackendSafetyCapabilities:
    """Evidence-backed memory-safety contract for one NIXL deployment."""

    definitely_not_submitted: bool
    safe_done: bool
    explicit_completion_control: bool
    drain_barrier: bool
    teardown_barrier: bool

    @property
    def supports_staging(self) -> bool:
        return (
            self.definitely_not_submitted
            and self.safe_done
            and self.explicit_completion_control
            and (self.drain_barrier or self.teardown_barrier)
        )


BackendKey: TypeAlias = tuple[str, str, str]


class BackendSafetyRegistry:
    """Capability matrix keyed by NIXL version, transport, and memory type."""

    def __init__(
        self,
        entries: Mapping[BackendKey, BackendSafetyCapabilities] | None = None,
    ) -> None:
        self._entries = dict(entries or {})

    def register(
        self, key: BackendKey, capabilities: BackendSafetyCapabilities
    ) -> None:
        current = self._entries.get(key)
        if current is not None and current != capabilities:
            raise RuntimeError(f"Conflicting backend safety evidence for {key}")
        self._entries[key] = capabilities

    def require_staging(self, key: BackendKey) -> BackendSafetyCapabilities:
        capabilities = self._entries.get(key)
        if capabilities is None:
            raise RuntimeError(f"No staging safety evidence for backend {key}")
        if not capabilities.supports_staging:
            raise RuntimeError(f"Backend {key} does not safely support staging")
        return capabilities


CacheKeyT = TypeVar("CacheKeyT", bound=Hashable)
CacheValueT = TypeVar("CacheValueT")


class DescriptorCache(Generic[CacheKeyT, CacheValueT]):
    """Bounded LRU for prepared full-slot and tail descriptor handles."""

    def __init__(
        self,
        capacity: int,
        release: Callable[[CacheValueT], None],
    ) -> None:
        _positive_int("descriptor cache capacity", capacity)
        self._capacity = capacity
        self._release = release
        self._entries: OrderedDict[CacheKeyT, CacheValueT] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get_or_create(
        self, key: CacheKeyT, create: Callable[[], CacheValueT]
    ) -> CacheValueT:
        value = self._entries.pop(key, None)
        if value is not None:
            self.hits += 1
            self._entries[key] = value
            return value
        self.misses += 1
        value = create()
        self._entries[key] = value
        if len(self._entries) > self._capacity:
            _, evicted = self._entries.popitem(last=False)
            self._release(evicted)
        return value

    def clear(self) -> None:
        for value in self._entries.values():
            self._release(value)
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


@dataclass
class _ExposedChunk:
    ready: StageReady
    first_ready_attempt: float | None = None
    last_ready_attempt: float | None = None
    inflight_proven: bool = False


class ProducerProgress:
    """Single-thread-owned producer slot and release-proof state machine."""

    _SAFE_RELEASE = frozenset({"safe_not_submitted", "safe_complete", "safe_retired"})

    def __init__(
        self,
        pool: StagingSlotPool,
        producer_engine_id: str,
        producer_rank: int,
        producer_generation: str,
        ready_retry_interval: float,
        sessions: StagingSessionRegistry | None = None,
    ) -> None:
        if not pool.producer:
            raise ValueError("ProducerProgress requires a producer slot pool")
        if not producer_generation:
            raise ValueError("producer_generation must not be empty")
        self.pool = pool
        self.producer_engine_id = producer_engine_id
        self.producer_rank = producer_rank
        self.producer_generation = producer_generation
        self.ready_retry_interval = ready_retry_interval
        self.sessions = sessions
        self._chunks: dict[ReadyIdentity, _ExposedChunk] = {}
        self._slot_chunks: dict[int, set[ReadyIdentity]] = {}
        self._slot_consumers: dict[int, set[ConsumerIdentity]] = {}

    def add_ready(self, ready_messages: tuple[StageReady, ...]) -> None:
        """Freeze READY identities for one gathered source slot."""
        if not ready_messages:
            raise ValueError("At least one READY is required")
        first = ready_messages[0]
        slot = self.pool._get(first.source_slot_id)
        if slot.state != ProducerSlotState.READY_LOCAL:
            raise RuntimeError("Source slot gather has not completed")
        if slot.epoch != first.source_slot_epoch:
            raise RuntimeError("READY epoch does not match source slot occupant")
        consumers: set[ConsumerIdentity] = set()
        keys: set[ReadyIdentity] = set()
        for ready in ready_messages:
            _validate_stage_message(ready)
            if self.sessions is not None:
                self.sessions.require_ready(ready)
            if (
                ready.producer_engine_id != self.producer_engine_id
                or ready.producer_rank != self.producer_rank
                or ready.producer_generation != self.producer_generation
                or ready.source_slot_id != first.source_slot_id
                or ready.source_slot_epoch != first.source_slot_epoch
                or ready.valid_bytes != first.valid_bytes
                or ready.plan_id != first.plan_id
                or ready.chunk_index != first.chunk_index
            ):
                raise RuntimeError("Consumers cannot share different chunk geometry")
            consumer = (
                ready.consumer_engine_id,
                ready.consumer_rank,
                ready.consumer_generation,
            )
            if consumer in consumers:
                raise RuntimeError("Duplicate consumer for producer source slot")
            consumers.add(consumer)
            key = ready_identity(ready)
            if key in self._chunks:
                raise RuntimeError("READY is already registered")
            keys.add(key)
        if first.source_slot_id in self._slot_chunks:
            raise RuntimeError("Source slot already has registered READY messages")
        for ready in ready_messages:
            key = ready_identity(ready)
            self._chunks[key] = _ExposedChunk(ready)
        self._slot_chunks[first.source_slot_id] = keys
        self._slot_consumers[first.source_slot_id] = consumers

    def send_ready(
        self,
        send: Callable[[StageReady, bytes], None],
        now: float | None = None,
    ) -> int:
        """Attempt eligible READY sends; send failures leave slots EXPOSED."""
        now = time.monotonic() if now is None else now
        attempts = 0
        for exposed in self._chunks.values():
            if exposed.inflight_proven or (
                exposed.last_ready_attempt is not None
                and now - exposed.last_ready_attempt < self.ready_retry_interval
            ):
                continue
            slot = self.pool._get(exposed.ready.source_slot_id)
            if slot.state == ProducerSlotState.READY_LOCAL:
                self.pool.expose(
                    exposed.ready.source_slot_id,
                    self._slot_consumers[exposed.ready.source_slot_id],
                )
            if exposed.first_ready_attempt is None:
                exposed.first_ready_attempt = now
            exposed.last_ready_attempt = now
            attempts += 1
            send(exposed.ready, encode_stage_message(exposed.ready))
        return attempts

    def accept_read_complete(self, message: StageReadComplete) -> bool:
        for key, exposed in list(self._chunks.items()):
            if completion_matches_ready(message, exposed.ready):
                return self._release_consumer(key, exposed.ready)
        return False

    def accept_status_reply(self, message: StageStatusReply) -> bool:
        for key, exposed in list(self._chunks.items()):
            if not completion_matches_ready(message, exposed.ready):
                continue
            if message.status == "inflight":
                exposed.inflight_proven = True
                return False
            if message.status in self._SAFE_RELEASE:
                return self._release_consumer(key, exposed.ready)
            return False
        return False

    def status_queries(
        self,
        now: float | None = None,
        timeout: float | None = None,
    ) -> tuple[StageStatusQuery, ...]:
        """Build reconciliation queries, optionally only for timed-out chunks."""
        if (now is None) != (timeout is None):
            raise ValueError("now and timeout must be provided together")
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive")
        filter_timed_out = now is not None
        query_now = 0.0 if now is None else now
        query_timeout = 0.0 if timeout is None else timeout
        return tuple(
            StageStatusQuery(
                protocol_version=ready.protocol_version,
                producer_generation=ready.producer_generation,
                consumer_generation=ready.consumer_generation,
                transfer_id=ready.transfer_id,
                chunk_index=ready.chunk_index,
                source_slot_id=ready.source_slot_id,
                source_slot_epoch=ready.source_slot_epoch,
                producer_engine_id=ready.producer_engine_id,
                producer_rank=ready.producer_rank,
                consumer_engine_id=ready.consumer_engine_id,
                consumer_rank=ready.consumer_rank,
            )
            for chunk in self._chunks.values()
            if not filter_timed_out
            or (
                chunk.first_ready_attempt is not None
                and query_now - chunk.first_ready_attempt >= query_timeout
            )
            for ready in (chunk.ready,)
        )

    def cancel_transfer(self, transfer_id: str) -> tuple[StageStatusQuery, ...]:
        """Cancel unexposed chunks locally and reconcile every exposed chunk."""
        exposed_queries: list[StageStatusQuery] = []
        slot_ids = {
            chunk.ready.source_slot_id
            for chunk in self._chunks.values()
            if chunk.ready.transfer_id == transfer_id
        }
        for slot_id in slot_ids:
            slot = self.pool._get(slot_id)
            keys = self._slot_chunks[slot_id]
            if slot.state == ProducerSlotState.READY_LOCAL:
                for key in tuple(keys):
                    del self._chunks[key]
                del self._slot_chunks[slot_id]
                del self._slot_consumers[slot_id]
                self.pool.release(slot_id)
                continue
            for key in keys:
                ready = self._chunks[key].ready
                exposed_queries.append(
                    StageStatusQuery(
                        protocol_version=ready.protocol_version,
                        producer_generation=ready.producer_generation,
                        consumer_generation=ready.consumer_generation,
                        transfer_id=ready.transfer_id,
                        chunk_index=ready.chunk_index,
                        source_slot_id=ready.source_slot_id,
                        source_slot_epoch=ready.source_slot_epoch,
                        producer_engine_id=ready.producer_engine_id,
                        producer_rank=ready.producer_rank,
                        consumer_engine_id=ready.consumer_engine_id,
                        consumer_rank=ready.consumer_rank,
                    )
                )
        return tuple(exposed_queries)

    def cleanup_consumer_generation(
        self,
        consumer_engine_id: str,
        consumer_generation: str,
        teardown_barrier: Callable[[], None],
    ) -> int:
        """Release a lost consumer only after a certified teardown barrier."""
        teardown_barrier()
        released_slots = 0
        matching = [
            (key, exposed.ready)
            for key, exposed in self._chunks.items()
            if exposed.ready.consumer_engine_id == consumer_engine_id
            and exposed.ready.consumer_generation == consumer_generation
            and self.pool._get(exposed.ready.source_slot_id).state
            == ProducerSlotState.EXPOSED
        ]
        for key, ready in matching:
            released_slots += self._release_consumer(key, ready)
        return released_slots

    def _release_consumer(self, key: ReadyIdentity, ready: StageReady) -> bool:
        consumer = (
            ready.consumer_engine_id,
            ready.consumer_rank,
            ready.consumer_generation,
        )
        released = self.pool.complete_consumer(
            ready.source_slot_id, ready.source_slot_epoch, consumer
        )
        del self._chunks[key]
        slot_keys = self._slot_chunks[ready.source_slot_id]
        slot_keys.remove(key)
        if not slot_keys:
            del self._slot_chunks[ready.source_slot_id]
            del self._slot_consumers[ready.source_slot_id]
        return released

    def has_transfer(self, transfer_id: str) -> bool:
        """Whether a transfer still owns an exposed source-slot reference."""
        return any(
            chunk.ready.transfer_id == transfer_id for chunk in self._chunks.values()
        )


@dataclass
class _ProducerGather:
    session: StagingTransferSession
    chunk: StagingChunk
    slot_id: int
    done: Callable[[], bool]


class ProducerPipeline:
    """Fairly gather committed transfers into a bounded producer outbox."""

    def __init__(
        self,
        pool: StagingSlotPool,
        config: StagingConfig,
        producer_engine_id: str,
        producer_rank: int,
        producer_generation: str,
        sessions: StagingSessionRegistry | None = None,
    ) -> None:
        if not pool.producer:
            raise ValueError("ProducerPipeline requires a producer slot pool")
        self.pool = pool
        self.config = config
        self.sessions = sessions
        self.progress = ProducerProgress(
            pool,
            producer_engine_id,
            producer_rank,
            producer_generation,
            config.ready_retry_interval,
            sessions,
        )
        self._pending: OrderedDict[TransferEdgeKey, deque[int]] = OrderedDict()
        self._sessions: dict[TransferEdgeKey, StagingTransferSession] = {}
        self._packing: dict[int, _ProducerGather] = {}
        self._gathered: dict[TransferEdgeKey, set[int]] = {}
        self._stable_edges: set[TransferEdgeKey] = set()
        self._reported_stable: set[str] = set()
        self._cancelled: set[str] = set()
        self.failed_transfers: set[str] = set()

    def register_transfer(self, session: StagingTransferSession) -> None:
        """Queue every chunk of a frozen, single-edge producer plan."""
        plan = session.plan
        if plan is None:
            raise RuntimeError("Transfer plan must be frozen before registration")
        intent = session.intent
        if (
            intent.producer_engine_id != self.progress.producer_engine_id
            or intent.producer_rank != self.progress.producer_rank
            or intent.producer_generation != self.progress.producer_generation
        ):
            raise RuntimeError("Transfer intent does not identify this producer")
        edge_key = session.edge_key
        current = self._sessions.get(edge_key)
        if current is not None:
            if current is not session and current.intent != intent:
                raise RuntimeError("Conflicting producer transfer session")
            return
        if not plan.chunks:
            self._stable_edges.add(edge_key)
            self._sessions[edge_key] = session
            return
        if self.sessions is not None:
            registered = self.sessions.register(intent)
            if registered is not session:
                decision = session.decision
                assert decision is not None
                registered.accept_decision(decision)
                registered.freeze_plan(plan)
                session = registered
        edge_key = session.edge_key
        self._sessions[edge_key] = session
        self._pending[edge_key] = deque(range(len(plan.chunks)))
        self._gathered[edge_key] = set()

    def _active_count(self, transfer_id: str) -> int:
        return sum(
            slot.owner is not None
            and isinstance(slot.owner[0], tuple)
            and slot.owner[0][0] == transfer_id
            for slot in self.pool.slots
        )

    def start_available(
        self,
        start_gather: Callable[
            [StagingTransferSession, StagingChunk, int], Callable[[], bool]
        ],
    ) -> int:
        """Start fair gathers until the pool or per-request limit is full."""
        started = 0
        candidates = len(self._pending)
        while self._pending and candidates:
            edge_key, chunks = self._pending.popitem(last=False)
            transfer_id = edge_key[0]
            candidates -= 1
            if transfer_id in self._cancelled:
                continue
            if self._active_count(transfer_id) >= self.config.max_ready_per_request:
                self._pending[edge_key] = chunks
                continue
            chunk_index = chunks.popleft()
            session = self._sessions[edge_key]
            plan = session.plan
            assert plan is not None
            chunk = plan.chunks[chunk_index]
            slot = self.pool.acquire((edge_key, chunk_index))
            if slot is None:
                chunks.appendleft(chunk_index)
                self._pending[edge_key] = chunks
                break
            try:
                done = start_gather(session, chunk, slot.slot_id)
            except Exception:
                self.pool.quarantine(slot.slot_id)
                self.failed_transfers.add(transfer_id)
                self._cancelled.add(transfer_id)
                continue
            self._packing[slot.slot_id] = _ProducerGather(
                session, chunk, slot.slot_id, done
            )
            if chunks:
                self._pending[edge_key] = chunks
            started += 1
            candidates = max(candidates, len(self._pending))
        return started

    def poll_gathers(self) -> tuple[StageReady, ...]:
        """Publish READY only after gather completion; drain cancelled gathers."""
        ready_messages: list[StageReady] = []
        for slot_id, gather in list(self._packing.items()):
            try:
                done = gather.done()
            except Exception:
                self.pool.quarantine(slot_id)
                del self._packing[slot_id]
                transfer_id = gather.session.intent.transfer_id
                self.failed_transfers.add(transfer_id)
                self._cancelled.add(transfer_id)
                continue
            if not done:
                continue
            del self._packing[slot_id]
            intent = gather.session.intent
            if intent.transfer_id in self._cancelled:
                self.pool.release(slot_id)
                if not any(
                    pending.session.intent.transfer_id == intent.transfer_id
                    for pending in self._packing.values()
                ):
                    for edge_key in self._sessions:
                        if edge_key[0] == intent.transfer_id:
                            self._stable_edges.add(edge_key)
                continue
            self.pool.transition(
                slot_id,
                ProducerSlotState.PACKING,
                ProducerSlotState.READY_LOCAL,
            )
            slot = self.pool._get(slot_id)
            ready = StageReady(
                protocol_version=intent.protocol_version,
                producer_generation=intent.producer_generation,
                consumer_generation=intent.consumer_generation,
                transfer_id=intent.transfer_id,
                request_id=intent.producer_request_id,
                chunk_index=gather.chunk.index,
                source_slot_id=slot_id,
                source_slot_epoch=slot.epoch,
                valid_bytes=gather.chunk.valid_bytes,
                plan_id=intent.plan_id,
                producer_engine_id=intent.producer_engine_id,
                producer_rank=intent.producer_rank,
                consumer_engine_id=intent.consumer_engine_id,
                consumer_rank=intent.consumer_rank,
            )
            self.progress.add_ready((ready,))
            ready_messages.append(ready)
            edge_key = gather.session.edge_key
            gathered = self._gathered[edge_key]
            gathered.add(gather.chunk.index)
            plan = gather.session.plan
            assert plan is not None
            if len(gathered) == len(plan.chunks):
                self._stable_edges.add(edge_key)
        return tuple(ready_messages)

    def send_ready(
        self,
        send: Callable[[StageReady, bytes], None],
        now: float | None = None,
    ) -> int:
        return self.progress.send_ready(send, now)

    def cancel_transfer(self, transfer_id: str) -> tuple[StageStatusQuery, ...]:
        """Cancel queued work and safely reconcile exposed source slots."""
        self._cancelled.add(transfer_id)
        for edge_key in tuple(self._pending):
            if edge_key[0] == transfer_id:
                del self._pending[edge_key]
        if not any(
            gather.session.intent.transfer_id == transfer_id
            for gather in self._packing.values()
        ):
            self._stable_edges.update(
                key for key in self._sessions if key[0] == transfer_id
            )
        return self.progress.cancel_transfer(transfer_id)

    def pop_source_stable(self) -> set[str]:
        """Return requests whose source KV lease no longer blocks new gathers."""
        stable = {
            key[0]
            for key in self._stable_edges
            if key[0] not in self._reported_stable
            and all(
                candidate in self._stable_edges
                for candidate in self._sessions
                if candidate[0] == key[0]
            )
        }
        self._reported_stable.update(stable)
        return stable

    def pop_completed_transfers(self) -> set[str]:
        """Retire transfers after gathers and all remote reads are safe."""
        completed_edges = {
            edge_key
            for edge_key in self._sessions
            if edge_key in self._stable_edges
            and edge_key not in self._pending
            and not self.progress.has_transfer(edge_key[0])
            and not any(
                gather.session.edge_key == edge_key for gather in self._packing.values()
            )
        }
        candidate_transfers = {key[0] for key in completed_edges}
        for edge_key in completed_edges:
            del self._sessions[edge_key]
            self._gathered.pop(edge_key, None)
            self._stable_edges.discard(edge_key)
            if self.sessions is not None:
                self.sessions.retire_edge(edge_key)
        return {
            transfer_id
            for transfer_id in candidate_transfers
            if not any(key[0] == transfer_id for key in self._sessions)
        }


class DefinitelyNotSubmittedError(RuntimeError):
    """A READ post failed with proof that the backend never submitted it."""


class PossiblySubmittedError(RuntimeError):
    """A READ post failed after creating a handle; submission is unknown."""

    def __init__(self, handle: Hashable, message: str) -> None:
        super().__init__(message)
        self.handle = handle


class ReadBackend(Protocol):
    def post_read(self, ready: StageReady, local_slot_id: int) -> Hashable: ...

    def check_read(self, handle: Hashable) -> str: ...

    def release_read(self, handle: Hashable) -> None: ...


@dataclass
class _ConsumerRead:
    ready: StageReady
    local_slot_id: int
    handle: Hashable


@dataclass
class _ConsumerScatter:
    ready: StageReady
    local_slot_id: int
    done: Callable[[], bool]


class ConsumerProgress:
    """Receiver-pull READ, completion, scatter, and quarantine state machine."""

    def __init__(
        self,
        pool: StagingSlotPool,
        config: StagingConfig,
        consumer_engine_id: str,
        consumer_rank: int,
        consumer_generation: str,
        sessions: StagingSessionRegistry | None = None,
    ) -> None:
        if pool.producer:
            raise ValueError("ConsumerProgress requires a consumer slot pool")
        self.pool = pool
        self.config = config
        self.consumer_engine_id = consumer_engine_id
        self.consumer_rank = consumer_rank
        self.consumer_generation = consumer_generation
        self.sessions = sessions
        self.ready = FairReadyQueue(config.max_ready_per_request)
        self.ledger = ChunkLedger()
        self.completions = ReadCompletionOutbox(self.ledger)
        self._reads: dict[Hashable, _ConsumerRead] = {}
        self._unknown_reads: dict[Hashable, _ConsumerRead] = {}
        self._scatters: dict[int, _ConsumerScatter] = {}
        self._inflight_per_peer: dict[ProducerIdentity, int] = {}
        self._quarantined_by_peer: dict[str, set[int]] = {}
        self._cancelled_transfers: set[str] = set()
        self.unavailable_peers: set[str] = set()
        self.completion_tracker = TransferCompletionTracker()
        self.failed_transfers: set[str] = set()
        self._completed_edges: set[TransferEdgeKey] = set()

    @staticmethod
    def _peer_identity(message: StageReady) -> ProducerIdentity:
        return (
            message.producer_engine_id,
            message.producer_rank,
            message.producer_generation,
        )

    def register_transfer(self, session: StagingTransferSession) -> None:
        """Register the frozen destination plan used for request completion."""
        plan = session.plan
        if plan is None:
            raise RuntimeError("Transfer plan must be frozen before registration")
        if self.sessions is not None:
            registered = self.sessions.register(session.intent)
            if registered is not session:
                if registered.intent != session.intent:
                    raise RuntimeError("Conflicting transfer session")
                decision = session.decision
                assert decision is not None
                registered.accept_decision(decision)
                registered.freeze_plan(plan)
        if plan.chunks:
            self.completion_tracker.register(repr(session.edge_key), len(plan.chunks))
        else:
            self._completed_edges.add(session.edge_key)

    def receive_ready(self, message: StageReady) -> bool:
        _validate_stage_message(message)
        if self.sessions is not None:
            self.sessions.require_ready(message)
        if (
            message.consumer_engine_id != self.consumer_engine_id
            or message.consumer_rank != self.consumer_rank
            or message.consumer_generation != self.consumer_generation
            or message.producer_engine_id in self.unavailable_peers
            or message.valid_bytes > self.pool.slot_bytes
        ):
            return False
        state = self.ledger.observe_ready(message)
        if message.transfer_id in self._cancelled_transfers:
            if state == RemoteChunkState.QUEUED:
                self.ledger.abort_queued(message)
            return False
        if state != RemoteChunkState.QUEUED:
            return False
        return self.ready.push(self._peer_identity(message), message)

    def post_available(self, backend: ReadBackend) -> int:
        posted = 0
        candidates = len(self.ready)
        while len(self._reads) < self.config.max_inflight and candidates:
            queued = self.ready.pop()
            if queued is None:
                break
            candidates -= 1
            peer_key, message = queued
            assert isinstance(peer_key, tuple) and len(peer_key) == 3
            peer = message.producer_engine_id
            if (
                self._inflight_per_peer.get(peer_key, 0)
                >= self.config.max_inflight_per_peer
            ):
                self.ready.push(peer_key, message)
                continue
            slot = self.pool.acquire((message.transfer_id, message.chunk_index))
            if slot is None:
                self.ready.push(peer_key, message)
                break
            self.ledger.transition(
                message, RemoteChunkState.QUEUED, RemoteChunkState.POSTING
            )
            try:
                handle = backend.post_read(message, slot.slot_id)
            except DefinitelyNotSubmittedError:
                self.ledger.transition(
                    message, RemoteChunkState.POSTING, RemoteChunkState.ABORTED
                )
                self.pool.release(slot.slot_id)
                continue
            except PossiblySubmittedError as exc:
                self.ledger.transition(
                    message, RemoteChunkState.POSTING, RemoteChunkState.UNKNOWN
                )
                self.pool.quarantine(slot.slot_id)
                self._unknown_reads[exc.handle] = _ConsumerRead(
                    message, slot.slot_id, exc.handle
                )
                self._quarantined_by_peer.setdefault(peer, set()).add(slot.slot_id)
                self.failed_transfers.add(message.transfer_id)
                self.completion_tracker.fail(repr(transfer_edge_key(message)))
                self._mark_peer_unavailable(peer)
                continue
            except Exception:
                self.ledger.transition(
                    message, RemoteChunkState.POSTING, RemoteChunkState.UNKNOWN
                )
                self.pool.quarantine(slot.slot_id)
                self._quarantined_by_peer.setdefault(peer, set()).add(slot.slot_id)
                self.failed_transfers.add(message.transfer_id)
                self.completion_tracker.fail(repr(transfer_edge_key(message)))
                self._mark_peer_unavailable(peer)
                continue
            self.ledger.transition(
                message, RemoteChunkState.POSTING, RemoteChunkState.INFLIGHT
            )
            try:
                duplicate_handle = handle in self._reads
            except TypeError:
                duplicate_handle = True
            if duplicate_handle:
                self.ledger.transition(
                    message, RemoteChunkState.INFLIGHT, RemoteChunkState.UNKNOWN
                )
                self.pool.quarantine(slot.slot_id)
                self._quarantined_by_peer.setdefault(peer, set()).add(slot.slot_id)
                self._mark_peer_unavailable(peer)
                continue
            self._reads[handle] = _ConsumerRead(message, slot.slot_id, handle)
            self._inflight_per_peer[peer_key] = (
                self._inflight_per_peer.get(peer_key, 0) + 1
            )
            posted += 1
        return posted

    def poll_reads(
        self,
        backend: ReadBackend,
        start_scatter: Callable[[StageReady, int], Callable[[], bool]],
    ) -> int:
        completed = 0
        for handle, read in list(self._reads.items()):
            try:
                state = backend.check_read(handle)
            except Exception:
                state = "UNKNOWN"
            if state in ("PROC", "INFLIGHT"):
                continue
            peer = read.ready.producer_engine_id
            peer_key = self._peer_identity(read.ready)
            if state != "DONE":
                self._inflight_per_peer[peer_key] -= 1
                del self._reads[handle]
                self._unknown_reads[handle] = read
                self.ledger.transition(
                    read.ready, RemoteChunkState.INFLIGHT, RemoteChunkState.UNKNOWN
                )
                self.pool.quarantine(read.local_slot_id)
                self._quarantined_by_peer.setdefault(peer, set()).add(
                    read.local_slot_id
                )
                self.failed_transfers.add(read.ready.transfer_id)
                self.completion_tracker.fail(repr(transfer_edge_key(read.ready)))
                self._mark_peer_unavailable(peer)
                continue
            try:
                self.completions.observe_done(
                    read.ready, partial(backend.release_read, handle)
                )
            except Exception:
                continue
            self._inflight_per_peer[peer_key] -= 1
            del self._reads[handle]
            self.pool.transition(
                read.local_slot_id,
                ConsumerSlotState.READING,
                ConsumerSlotState.SCATTERING,
            )
            try:
                done = start_scatter(read.ready, read.local_slot_id)
            except Exception:
                self.pool.quarantine(read.local_slot_id)
                self._quarantined_by_peer.setdefault(peer, set()).add(
                    read.local_slot_id
                )
                self.failed_transfers.add(read.ready.transfer_id)
                self.completion_tracker.fail(repr(transfer_edge_key(read.ready)))
                continue
            self._scatters[read.local_slot_id] = _ConsumerScatter(
                read.ready, read.local_slot_id, done
            )
            completed += 1
        return completed

    def poll_unknown_reads(self, backend: ReadBackend) -> int:
        """Reconcile handles whose transfer submission result was unknown."""
        safely_retired = 0
        for handle, read in list(self._unknown_reads.items()):
            try:
                state = backend.check_read(handle)
            except Exception:
                continue
            if state != "DONE":
                continue
            try:
                self.completions.observe_done(
                    read.ready, partial(backend.release_read, handle)
                )
            except Exception:
                continue
            del self._unknown_reads[handle]
            self.pool.retire_quarantined(read.local_slot_id)
            self._quarantined_by_peer[read.ready.producer_engine_id].discard(
                read.local_slot_id
            )
            safely_retired += 1
        return safely_retired

    def poll_scatters(self) -> tuple[StageReady, ...]:
        completed: list[StageReady] = []
        for slot_id, scatter in list(self._scatters.items()):
            try:
                done = scatter.done()
            except Exception:
                self.pool.quarantine(slot_id)
                del self._scatters[slot_id]
                peer = scatter.ready.producer_engine_id
                self._quarantined_by_peer.setdefault(peer, set()).add(slot_id)
                self.failed_transfers.add(scatter.ready.transfer_id)
                self.completion_tracker.fail(repr(transfer_edge_key(scatter.ready)))
                continue
            if not done:
                continue
            self.pool.release(slot_id)
            del self._scatters[slot_id]
            if scatter.ready.transfer_id not in self._cancelled_transfers:
                completed.append(scatter.ready)
                edge_key = transfer_edge_key(scatter.ready)
                tracker_key = repr(edge_key)
                if self.completion_tracker.is_registered(
                    tracker_key
                ) and self.completion_tracker.observe_scatter(
                    scatter.ready, tracker_key
                ):
                    self._completed_edges.add(edge_key)
        return tuple(completed)

    def pop_completed_transfers(self) -> set[str]:
        """Return transfers whose complete destination plans were scattered."""
        completed = self._completed_edges
        self._completed_edges = set()
        transfer_ids = {key[0] for key in completed}
        for edge_key in completed:
            self.completion_tracker.retire(repr(edge_key))
            if self.sessions is not None:
                self.sessions.retire_edge(edge_key)
        if self.sessions is None:
            return transfer_ids
        return {
            transfer_id
            for transfer_id in transfer_ids
            if not self.sessions.for_transfer(transfer_id)
        }

    def cancel_transfer(self, transfer_id: str) -> tuple[StageStatusReply, ...]:
        self._cancelled_transfers.add(transfer_id)
        replies: list[StageStatusReply] = []
        for message in self.ready.remove_transfer(transfer_id):
            self.ledger.abort_queued(message)
            replies.append(self._status_reply(message, "safe_not_submitted"))
        for read in self._reads.values():
            if read.ready.transfer_id == transfer_id:
                replies.append(self._status_reply(read.ready, "inflight"))
        for read in self._unknown_reads.values():
            if read.ready.transfer_id == transfer_id:
                replies.append(self._status_reply(read.ready, "unknown"))
        return tuple(replies)

    def reply_status(self, query: StageStatusQuery) -> StageStatusReply:
        return StageStatusReply(
            protocol_version=query.protocol_version,
            producer_generation=query.producer_generation,
            consumer_generation=query.consumer_generation,
            transfer_id=query.transfer_id,
            chunk_index=query.chunk_index,
            source_slot_id=query.source_slot_id,
            source_slot_epoch=query.source_slot_epoch,
            status=self.ledger.status(query),
            producer_engine_id=query.producer_engine_id,
            producer_rank=query.producer_rank,
            consumer_engine_id=query.consumer_engine_id,
            consumer_rank=query.consumer_rank,
        )

    def _status_reply(self, ready: StageReady, status: SafeStatus) -> StageStatusReply:
        return StageStatusReply(
            protocol_version=ready.protocol_version,
            producer_generation=ready.producer_generation,
            consumer_generation=ready.consumer_generation,
            transfer_id=ready.transfer_id,
            chunk_index=ready.chunk_index,
            source_slot_id=ready.source_slot_id,
            source_slot_epoch=ready.source_slot_epoch,
            status=status,
            producer_engine_id=ready.producer_engine_id,
            producer_rank=ready.producer_rank,
            consumer_engine_id=ready.consumer_engine_id,
            consumer_rank=ready.consumer_rank,
        )

    def _mark_peer_unavailable(self, peer: str) -> None:
        if (
            self.config.quarantine_max_bytes == 0
            or self.pool.quarantine_bytes >= self.config.quarantine_max_bytes
        ):
            self.unavailable_peers.add(peer)

    def cleanup_peer_generation(
        self,
        peer: str,
        generation: str,
        teardown_barrier: Callable[[], None],
    ) -> None:
        """Retire unknown DMA state only after a certified backend barrier."""
        teardown_barrier()
        for slot_id in self._quarantined_by_peer.pop(peer, set()):
            self.pool.retire_quarantined(slot_id)
        self.ledger.retire_generation(peer, generation)
        self.unavailable_peers.discard(peer)


class StagingNotificationHandler:
    """Route typed staging notifications without legacy string fallback."""

    def __init__(
        self,
        producer: ProducerProgress | None = None,
        consumer: ConsumerProgress | None = None,
    ) -> None:
        if producer is None and consumer is None:
            raise ValueError("A staging notification handler needs a local role")
        self.producer = producer
        self.consumer = consumer
        self.invalid_notifications = 0
        self.ignored_notifications = 0

    def receive(self, payload: bytes) -> tuple[StageMessage, ...]:
        """Apply one notification and return any required control replies."""
        try:
            message = decode_stage_message(payload)
        except ValueError:
            self.invalid_notifications += 1
            return ()

        if isinstance(message, StageReady):
            if self.consumer is None:
                self.ignored_notifications += 1
            else:
                self.consumer.receive_ready(message)
            return ()
        if isinstance(message, StageReadComplete):
            if self.producer is None:
                self.ignored_notifications += 1
            else:
                self.producer.accept_read_complete(message)
            return ()
        if isinstance(message, StageStatusReply):
            if self.producer is None:
                self.ignored_notifications += 1
            else:
                self.producer.accept_status_reply(message)
            return ()
        if isinstance(message, StageStatusQuery):
            if self.consumer is None:
                self.ignored_notifications += 1
                return ()
            return (self.consumer.reply_status(message),)

        replies: list[StageMessage] = []
        if self.consumer is not None:
            replies.extend(self.consumer.cancel_transfer(message.transfer_id))
        if self.producer is not None:
            replies.extend(self.producer.cancel_transfer(message.transfer_id))
        return tuple(replies)
