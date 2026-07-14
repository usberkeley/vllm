# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for sparse MLA selected-page offload."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
    return bool(value)


def _parse_layer_ids(value: Any) -> frozenset[int] | None:
    if value is None or value == "auto":
        return None
    if isinstance(value, int):
        return frozenset((value,))
    if isinstance(value, (list, tuple, set, frozenset)):
        return frozenset(int(v) for v in value)
    if not isinstance(value, str):
        raise ValueError(
            "sparse_page_offload_layers must be 'auto', a layer id, "
            "a list of layer ids, or a comma-separated layer range string."
        )

    layer_ids: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            start_id = int(start)
            end_id = int(end)
            if end_id < start_id:
                raise ValueError(
                    f"Invalid sparse_page_offload_layers range: {part!r}"
                )
            layer_ids.update(range(start_id, end_id + 1))
        else:
            layer_ids.add(int(part))
    return frozenset(layer_ids)


def _parse_hot_pool_blocks(value: Any) -> tuple[int, ...]:
    if value is None:
        return (512,)
    if isinstance(value, int):
        return (value,)
    if isinstance(value, str):
        return tuple(int(v.strip()) for v in value.split(",") if v.strip())
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(int(v) for v in value)
    raise ValueError(
        "sparse_page_hot_pool_blocks must be an int or a comma-separated/list "
        "of ints."
    )


def extract_sparse_page_layer_index(layer_name: str) -> int | None:
    int_vals: list[int] = []
    for part in layer_name.split("."):
        try:
            int_vals.append(int(part))
        except ValueError:
            continue
    if len(int_vals) != 1:
        return None
    return int_vals[0]


@dataclass(frozen=True)
class SparsePageOffloadConfig:
    """Runtime config for sparse selected-page offload.

    Phase 1 implements observe-only collection. If behavior offload is requested
    without ``sparse_page_observe_only``, it is explicitly downgraded to
    observe-only instead of touching the logits path.
    """

    enabled: bool = False
    observe_only: bool = True
    hot_pool_blocks: tuple[int, ...] = (512,)
    prefetch_lookahead: int = 0
    cpu_pool_size_gib: float = 0.0
    transfer_backend: str = "native_cpu"
    layer_ids: frozenset[int] | None = None

    @classmethod
    def from_hf_config(cls, hf_config: Any) -> SparsePageOffloadConfig:
        sparse_page_offload = _as_bool(
            getattr(hf_config, "sparse_page_offload", False)
        )
        observe_only = _as_bool(
            getattr(hf_config, "sparse_page_observe_only", False)
        )
        enabled = sparse_page_offload or observe_only
        if not enabled:
            return cls()

        if sparse_page_offload and not observe_only:
            logger.warning_once(
                "sparse_page_offload behavior is not implemented yet; "
                "running DeepSeek V4 sparse page offload in observe-only mode."
            )
            observe_only = True

        hot_pool_blocks = _parse_hot_pool_blocks(
            getattr(hf_config, "sparse_page_hot_pool_blocks", 512)
        )
        if not hot_pool_blocks or any(blocks < 0 for blocks in hot_pool_blocks):
            raise ValueError("sparse_page_hot_pool_blocks must be non-negative.")

        layer_ids = _parse_layer_ids(
            getattr(hf_config, "sparse_page_offload_layers", "auto")
        )
        return cls(
            enabled=enabled,
            observe_only=observe_only,
            hot_pool_blocks=hot_pool_blocks,
            prefetch_lookahead=int(
                getattr(hf_config, "sparse_page_prefetch_lookahead", 0)
            ),
            cpu_pool_size_gib=float(
                getattr(hf_config, "sparse_page_cpu_pool_size_gib", 0.0)
            ),
            transfer_backend=str(
                getattr(hf_config, "sparse_page_transfer_backend", "native_cpu")
            ),
            layer_ids=layer_ids,
        )

    def includes_layer(self, layer_name: str) -> bool:
        if self.layer_ids is None:
            return True
        layer_idx = extract_sparse_page_layer_index(layer_name)
        if layer_idx is None:
            return False
        return layer_idx in self.layer_ids
