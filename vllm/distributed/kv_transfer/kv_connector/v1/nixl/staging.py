# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded GPU staging primitives for NIXL push transfers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig

STAGING_PROTOCOL_VERSION = 1

STAGE_DATA_NOTIF_PREFIX = b"STAGE_DATA:"
STAGE_ACK_NOTIF_PREFIX = b"STAGE_ACK:"
STAGE_RELEASE_NOTIF_PREFIX = b"STAGE_RELEASE:"

_DEFAULT_SLOT_BYTES = 256 * 1024 * 1024
_SLOT_ALIGNMENT = 256
_MIN_PIPELINE_SLOTS = 3


def _as_non_negative_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} must be an integer, got {value!r}") from e
    if parsed < 0 or parsed != value:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return parsed


@dataclass(frozen=True)
class NixlStagingConfig:
    """Validated, per-GPU NIXL staging configuration."""

    requested: bool
    active: bool
    pool_bytes: int
    slot_bytes: int
    slot_count: int
    max_inflight: int
    max_inflight_per_request: int
    fallback: Literal["direct", "fail"]

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        total_memory: int,
    ) -> NixlStagingConfig:
        kv_config = vllm_config.kv_transfer_config
        if kv_config is None or kv_config.kv_connector != "NixlPushConnector":
            return cls.disabled()

        extra = kv_config.kv_connector_extra_config
        fallback = extra.get("staging_fallback", "direct")
        if fallback not in ("direct", "fail"):
            raise ValueError(
                f"staging_fallback must be either 'direct' or 'fail', got {fallback!r}"
            )

        fraction_value = extra.get("staging_buffer_fraction", 0.0)
        if isinstance(fraction_value, bool):
            raise ValueError("staging_buffer_fraction must be a number")
        try:
            fraction = float(fraction_value)
        except (TypeError, ValueError) as e:
            raise ValueError("staging_buffer_fraction must be a number") from e
        if not 0.0 <= fraction < 1.0:
            raise ValueError(
                f"staging_buffer_fraction must be in [0, 1), got {fraction_value!r}"
            )

        if "staging_buffer_bytes" in extra:
            requested_bytes = _as_non_negative_int(
                "staging_buffer_bytes", extra["staging_buffer_bytes"]
            )
        else:
            requested_bytes = int(total_memory * fraction)

        requested = requested_bytes > 0
        if not requested:
            return cls.disabled(fallback=fallback)

        slot_bytes = _as_non_negative_int(
            "staging_slot_bytes",
            extra.get("staging_slot_bytes", _DEFAULT_SLOT_BYTES),
        )
        slot_bytes -= slot_bytes % _SLOT_ALIGNMENT
        if slot_bytes == 0:
            raise ValueError(
                f"staging_slot_bytes must be at least {_SLOT_ALIGNMENT} bytes"
            )

        slot_count = requested_bytes // slot_bytes
        pool_bytes = slot_count * slot_bytes
        active = slot_count >= _MIN_PIPELINE_SLOTS
        if not active and fallback == "fail":
            raise ValueError(
                "NIXL GPU staging requires at least "
                f"{_MIN_PIPELINE_SLOTS} slots, but buffer={requested_bytes} and "
                f"slot={slot_bytes} provide {slot_count}"
            )

        configured_max_inflight = _as_non_negative_int(
            "staging_max_inflight", extra.get("staging_max_inflight", 0)
        )
        max_inflight = configured_max_inflight or slot_count
        max_inflight = min(max_inflight, slot_count)
        max_per_request = _as_non_negative_int(
            "staging_max_inflight_per_request",
            extra.get("staging_max_inflight_per_request", 2),
        )
        if max_per_request == 0:
            raise ValueError("staging_max_inflight_per_request must be positive")
        max_per_request = min(max_per_request, max_inflight)

        return cls(
            requested=True,
            active=active,
            pool_bytes=pool_bytes,
            slot_bytes=slot_bytes,
            slot_count=slot_count,
            max_inflight=max_inflight,
            max_inflight_per_request=max_per_request,
            fallback=fallback,
        )

    @classmethod
    def disabled(
        cls, fallback: Literal["direct", "fail"] = "direct"
    ) -> NixlStagingConfig:
        return cls(
            requested=False,
            active=False,
            pool_bytes=0,
            slot_bytes=0,
            slot_count=0,
            max_inflight=0,
            max_inflight_per_request=0,
            fallback=fallback,
        )


def get_nixl_staging_buffer_bytes(vllm_config: VllmConfig, total_memory: int) -> int:
    """Return bytes that must be withheld from automatic KV-cache sizing."""
    config = NixlStagingConfig.from_vllm_config(vllm_config, total_memory)
    return config.pool_bytes if config.active else 0


def gather_staging_blocks(
    caches: tuple[torch.Tensor, ...],
    slot: torch.Tensor,
    block_ids: torch.Tensor,
) -> int:
    """Pack selected block-major cache pages contiguously into a slot."""
    block_count = block_ids.numel()
    offset = 0
    for cache in caches:
        page_bytes = cache[0].numel() * cache.element_size()
        region_bytes = block_count * page_bytes
        if offset + region_bytes > slot.numel():
            raise ValueError("staging slot is too small for selected KV blocks")
        target = (
            slot.narrow(0, offset, region_bytes)
            .view(cache.dtype)
            .view(block_count, *cache.shape[1:])
        )
        torch.index_select(cache, 0, block_ids, out=target)
        offset += region_bytes
    return offset


def scatter_staging_blocks(
    caches: tuple[torch.Tensor, ...],
    slot: torch.Tensor,
    block_ids: torch.Tensor,
) -> int:
    """Scatter a contiguous staging frame into block-major cache pages."""
    block_count = block_ids.numel()
    offset = 0
    for cache in caches:
        page_bytes = cache[0].numel() * cache.element_size()
        region_bytes = block_count * page_bytes
        if offset + region_bytes > slot.numel():
            raise ValueError("staging frame is smaller than its KV block mapping")
        source = (
            slot.narrow(0, offset, region_bytes)
            .view(cache.dtype)
            .view(block_count, *cache.shape[1:])
        )
        cache.index_copy_(0, block_ids, source)
        offset += region_bytes
    return offset


@dataclass(frozen=True)
class StagingCredit:
    slot_id: int
    epoch: int


@dataclass(frozen=True)
class RemoteStagingRegion:
    base_addr: int
    pool_bytes: int
    slot_bytes: int
    slot_count: int
    device_id: int
    protocol_version: int

    @property
    def enabled(self) -> bool:
        return (
            self.protocol_version == STAGING_PROTOCOL_VERSION
            and self.base_addr > 0
            and self.pool_bytes > 0
            and self.slot_bytes > 0
            and self.slot_count >= _MIN_PIPELINE_SLOTS
            and self.pool_bytes >= self.slot_count * self.slot_bytes
        )


class StagingSlotPool:
    """Single-thread-owned allocator over one contiguous GPU tensor."""

    def __init__(self, tensor: torch.Tensor, slot_bytes: int):
        if (
            slot_bytes <= 0
            or tensor.ndim != 1
            or tensor.element_size() != 1
            or not tensor.is_contiguous()
            or tensor.numel() % slot_bytes != 0
        ):
            raise ValueError("staging tensor must contain an integer number of slots")
        self.tensor = tensor
        self.slot_bytes = slot_bytes
        self.slot_count = tensor.numel() // slot_bytes
        self._free = deque(range(self.slot_count))
        self._in_use: set[int] = set()

    @property
    def num_free(self) -> int:
        return len(self._free)

    def acquire(self) -> int | None:
        if not self._free:
            return None
        slot_id = self._free.popleft()
        self._in_use.add(slot_id)
        return slot_id

    def release(self, slot_id: int) -> None:
        if slot_id not in self._in_use:
            raise RuntimeError(f"staging slot {slot_id} is not in use")
        self._in_use.remove(slot_id)
        self._free.append(slot_id)

    def address(self, slot_id: int) -> int:
        self._validate_slot(slot_id)
        return self.tensor.data_ptr() + slot_id * self.slot_bytes

    def view(self, slot_id: int, valid_bytes: int | None = None) -> torch.Tensor:
        self._validate_slot(slot_id)
        length = self.slot_bytes if valid_bytes is None else valid_bytes
        if not 0 <= length <= self.slot_bytes:
            raise ValueError(
                f"valid_bytes must be in [0, {self.slot_bytes}], got {length}"
            )
        return self.tensor.narrow(0, slot_id * self.slot_bytes, length)

    def _validate_slot(self, slot_id: int) -> None:
        if not 0 <= slot_id < self.slot_count:
            raise IndexError(f"invalid staging slot {slot_id}")
