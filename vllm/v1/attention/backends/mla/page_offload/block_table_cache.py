# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layer-local block table cache for sparse selected-page offload."""

from __future__ import annotations

import torch


class LayerLocalBlockTableCache:
    """Own fixed-address block table buffers keyed by layer and source shape."""

    def __init__(self) -> None:
        self._buffers: dict[tuple[str, torch.device, torch.dtype], torch.Tensor] = {}

    def get_or_create(
        self,
        layer_name: str,
        source_block_table: torch.Tensor,
    ) -> torch.Tensor:
        key = (layer_name, source_block_table.device, source_block_table.dtype)
        buffer = self._buffers.get(key)
        if (
            buffer is None
            or buffer.shape != source_block_table.shape
            or buffer.stride() != source_block_table.stride()
        ):
            buffer = torch.empty_strided(
                source_block_table.shape,
                source_block_table.stride(),
                dtype=source_block_table.dtype,
                device=source_block_table.device,
            )
            self._buffers[key] = buffer
        buffer.copy_(source_block_table)
        return buffer
