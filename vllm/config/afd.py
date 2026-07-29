# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import field
from typing import Any, Literal, get_args

from vllm.config.utils import config
from vllm.utils.hashing import safe_hash

AFDRole = Literal["attention", "ffn", "none"]


@config
class AFDConfig:
    """Configuration for Attention-FFN Disaggregation (AFD).

    AFD runs the memory-bound attention block and the compute-bound FFN/MoE
    block of a MoE model on separately scaled GPU pools, connected by a
    per-decode-step transfer of the residual-stream hidden states at the
    attention-MoE seam. See ``AFD_GRAPH_CUT_DESIGN`` for the full design.
    """

    role: AFDRole = "none"
    """The AFD role of this vLLM instance. 'attention' runs attention + KV
    cache, 'ffn' runs the FFN/MoE experts, 'none' disables AFD."""

    afd_connector: str | None = None
    """Name of the AFD connector used to transmit activations between the
    attention and FFN pools (e.g. 'P2PAFDConnector', 'LoopbackAFDConnector')."""

    afd_connector_module_path: str | None = None
    """Python module path to dynamically load an out-of-tree AFD connector
    from. Takes priority over the built-in registry."""

    afd_connector_extra_config: dict[str, Any] = field(default_factory=dict)
    """Any extra config the AFD connector may need (e.g. NIC name, ports)."""

    handshake_timeout_s: float = 30.0
    """Timeout in seconds for the initial connector handshake."""

    transfer_timeout_s: float = 30.0
    """Timeout in seconds for a single activation transfer."""

    max_inflight_batches: int = 1
    """Maximum Attention-to-FFN round trips awaiting F2A completion.

    Phase 0 advances one intact scheduler batch synchronously, so this must be
    one. A larger value is reserved for a future Attention scheduling phase.
    """

    age_limit_ms: float = 10.0
    """Anti-starvation threshold: a layer whose queue head has waited at least
    this many milliseconds is prioritized over fuller layers (design section
    6.2 aging)."""

    def compute_hash(self) -> str:
        """Provide a hash of graph-affecting AFD fields.

        The AFD role changes the module graph (attention vs FFN modules are
        replaced), so the role must contribute to the compilation cache key.
        Connector/transport fields do not affect the graph.
        """
        factors: list[Any] = [self.role]
        return safe_hash(str(factors).encode(), usedforsecurity=False).hexdigest()

    def __post_init__(self) -> None:
        if self.role not in get_args(AFDRole):
            raise ValueError(
                f"Unsupported AFD role: {self.role}. "
                f"Supported roles are {get_args(AFDRole)}"
            )

        if self.role != "none" and self.afd_connector is None:
            raise ValueError("afd_connector must be set when AFD role is not 'none'.")

        if self.max_inflight_batches != 1:
            raise ValueError(
                "max_inflight_batches must be 1 for synchronous Phase 0 "
                f"Attention scheduling, got {self.max_inflight_batches}."
            )
        if self.age_limit_ms < 0:
            raise ValueError("age_limit_ms must be non-negative.")

    @property
    def is_attention(self) -> bool:
        return self.role == "attention"

    @property
    def is_ffn(self) -> bool:
        return self.role == "ffn"

    def get_from_extra_config(self, key: str, default: Any) -> Any:
        return self.afd_connector_extra_config.get(key, default)
