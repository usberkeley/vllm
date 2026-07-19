# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for sparse MLA selected-page offload."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import KVTransferConfig, VllmConfig

logger = init_logger(__name__)

SPARSE_PAGE_CONNECTOR = "SparsePageConnector"


@dataclass(frozen=True)
class SparsePageParallelTopology:
    """Parallel identity and constraints for one P or D engine."""

    engine_id: str
    tp_size: int = 1
    dp_rank: int = 0
    dp_size: int = 1
    expert_parallel: bool = False
    dcp_size: int = 1
    elastic_ep: bool = False

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
    ) -> SparsePageParallelTopology:
        parallel_config = getattr(vllm_config, "parallel_config", None)
        kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
        return cls(
            engine_id=str(getattr(kv_transfer_config, "engine_id", "") or ""),
            tp_size=int(getattr(parallel_config, "tensor_parallel_size", 1)),
            dp_rank=int(getattr(parallel_config, "data_parallel_rank", 0)),
            dp_size=int(getattr(parallel_config, "data_parallel_size", 1)),
            expert_parallel=bool(
                getattr(parallel_config, "enable_expert_parallel", False)
            ),
            dcp_size=int(getattr(parallel_config, "decode_context_parallel_size", 1)),
            elastic_ep=bool(getattr(parallel_config, "enable_elastic_ep", False)),
        )

    def validate(self) -> None:
        if not self.engine_id:
            raise ValueError("Sparse page offload requires a non-empty engine id.")
        if self.tp_size <= 0:
            raise ValueError("Sparse page offload requires TP size greater than 0.")
        if self.dp_size <= 0:
            raise ValueError("Sparse page offload requires DP size greater than 0.")
        if not 0 <= self.dp_rank < self.dp_size:
            raise ValueError(
                "Sparse page offload requires data_parallel_rank in "
                f"[0, {self.dp_size}), got {self.dp_rank}."
            )
        if self.dcp_size != 1:
            raise NotImplementedError(
                "Sparse page offload does not support decode context "
                f"parallelism (DCP={self.dcp_size})."
            )
        if self.elastic_ep:
            raise NotImplementedError(
                "Sparse page offload does not support elastic expert "
                "parallel resize while requests are active."
            )

    def validate_remote_tp_size(self, remote_tp_size: int) -> None:
        if remote_tp_size <= 0:
            raise ValueError(
                "Sparse page offload requires remote TP size greater than 0."
            )
        larger = max(self.tp_size, remote_tp_size)
        smaller = min(self.tp_size, remote_tp_size)
        if larger % smaller != 0:
            raise ValueError(
                "Sparse page offload requires producer and consumer TP sizes "
                "to be equal or integer multiples, got "
                f"local_tp={self.tp_size}, remote_tp={remote_tp_size}."
            )


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
                raise ValueError(f"Invalid sparse_page_offload_layers range: {part!r}")
            layer_ids.update(range(start_id, end_id + 1))
        else:
            layer_ids.add(int(part))
    return frozenset(layer_ids)


def _parse_hot_pages_per_request(value: Any) -> int:
    if value is None:
        return 512
    if isinstance(value, bool):
        raise ValueError("sparse_page_hot_pool_blocks must be an int.")
    if isinstance(value, int):
        page_count = value
    elif isinstance(value, str):
        page_count = int(value.strip())
    else:
        raise ValueError("sparse_page_hot_pool_blocks must be an int.")
    if page_count < 0:
        raise ValueError("sparse_page_hot_pool_blocks must be non-negative.")
    return page_count


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

    Enablement is derived from the KV transfer setup. A prefill instance using
    ``SparsePageConnector`` as a producer may seal prefill sideband, while a
    decode instance using it as a consumer may prepare selected-page decode.
    All knobs come from the connector's ``kv_connector_extra_config`` rather
    than the model config.
    """

    enabled: bool = False
    role: str | None = None
    hot_pages_per_request: int = 512
    prefetch_lookahead: int = 0
    cpu_pool_size_gib: float = 0.0
    transfer_backend: str = "nixl"
    layer_ids: frozenset[int] | None = None
    allocate_partial: bool = False

    @classmethod
    def from_kv_transfer_config(
        cls,
        kv_transfer_config: KVTransferConfig | None,
    ) -> SparsePageOffloadConfig:
        if (
            kv_transfer_config is None
            or getattr(kv_transfer_config, "kv_connector", None)
            != SPARSE_PAGE_CONNECTOR
        ):
            return cls()

        role = getattr(kv_transfer_config, "kv_role", None)
        if role not in ("kv_producer", "kv_consumer"):
            return cls(role=role)

        extra = getattr(kv_transfer_config, "kv_connector_extra_config", None) or {}
        return cls(
            enabled=True,
            role=role,
            hot_pages_per_request=_parse_hot_pages_per_request(
                extra.get("sparse_page_hot_pool_blocks", 512)
            ),
            prefetch_lookahead=int(extra.get("sparse_page_prefetch_lookahead", 0)),
            cpu_pool_size_gib=float(extra.get("sparse_page_cpu_pool_size_gib", 0.0)),
            transfer_backend=str(extra.get("sparse_page_transfer_backend", "nixl")),
            layer_ids=_parse_layer_ids(extra.get("sparse_page_offload_layers", "auto")),
            allocate_partial=role == "kv_consumer",
        )

    def includes_layer(self, layer_name: str) -> bool:
        if self.layer_ids is None:
            return True
        layer_idx = extract_sparse_page_layer_index(layer_name)
        if layer_idx is None:
            return False
        return layer_idx in self.layer_ids

    @property
    def can_stage_decode(self) -> bool:
        return self.enabled and self.role == "kv_consumer"

    @property
    def can_seal_prefill(self) -> bool:
        return self.enabled and self.role == "kv_producer"
