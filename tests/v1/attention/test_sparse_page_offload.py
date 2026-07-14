# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.v1.attention.backends.mla.page_offload.adapters.deepseek_v4_c4a import (
    DeepSeekV4C4AAdapter,
)
from vllm.v1.attention.backends.mla.page_offload.config import (
    SparsePageOffloadConfig,
)
from vllm.v1.attention.backends.mla.page_offload.selected_pages import (
    LogicalPage,
    SparsePageSelection,
)
from vllm.v1.attention.backends.mla.page_offload.telemetry import (
    SparseSelectionCollector,
)


def _hf_config(**overrides):
    values = {
        "model_type": "deepseek_v4",
        "compress_ratios": [1, 4, 128, 4],
        "sparse_page_observe_only": True,
        "sparse_page_hot_pool_blocks": [1, 2],
        "sparse_page_offload_layers": "auto",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_config_parses_layer_ranges_and_hot_pool_blocks():
    config = SparsePageOffloadConfig.from_hf_config(
        _hf_config(
            sparse_page_hot_pool_blocks="0,4,16",
            sparse_page_offload_layers="1,3-5",
        )
    )

    assert config.enabled
    assert config.observe_only
    assert config.hot_pool_blocks == (0, 4, 16)
    assert config.includes_layer("model.layers.1.self_attn.attn")
    assert config.includes_layer("model.layers.4.self_attn.attn")
    assert not config.includes_layer("model.layers.2.self_attn.attn")


def test_deepseek_v4_c4a_adapter_supports_only_c4a_fp8_ds_mla_layers():
    config = SparsePageOffloadConfig.from_hf_config(_hf_config())
    adapter = DeepSeekV4C4AAdapter(_hf_config(), config)

    assert adapter.supports_layer("model.layers.1.self_attn.attn", "fp8_ds_mla")
    assert adapter.supports_layer("model.layers.3.self_attn.attn", "fp8_ds_mla")
    assert not adapter.supports_layer("model.layers.0.self_attn.attn", "fp8_ds_mla")
    assert not adapter.supports_layer("model.layers.2.self_attn.attn", "fp8_ds_mla")
    assert not adapter.supports_layer("model.layers.1.self_attn.attn", "auto")


def test_deepseek_v4_c4a_adapter_extracts_unique_pages_and_tail_pages():
    hf_config = _hf_config(compress_ratios=[4])
    config = SparsePageOffloadConfig.from_hf_config(hf_config)
    adapter = DeepSeekV4C4AAdapter(hf_config, config)

    selection = adapter.extract_selection(
        layer_name="model.layers.0.self_attn.attn",
        req_id_per_token=torch.tensor([0, 1], dtype=torch.int32),
        topk_indices=torch.tensor(
            [
                [0, 1, 63, 64, 128, -1],
                [64, 65, 127, 256, 320, -1],
            ],
            dtype=torch.int32,
        ),
        seq_lens=torch.tensor([260, 1028], dtype=torch.int32),
    )

    assert selection.unique_pages == (
        LogicalPage(0, "model.layers.0.self_attn.attn", 0),
        LogicalPage(0, "model.layers.0.self_attn.attn", 1),
        LogicalPage(0, "model.layers.0.self_attn.attn", 2),
        LogicalPage(1, "model.layers.0.self_attn.attn", 1),
        LogicalPage(1, "model.layers.0.self_attn.attn", 4),
        LogicalPage(1, "model.layers.0.self_attn.attn", 5),
    )
    assert selection.tail_pages == frozenset(
        {
            LogicalPage(0, "model.layers.0.self_attn.attn", 1),
            LogicalPage(1, "model.layers.0.self_attn.attn", 4),
        }
    )
    assert [page.is_tail for page in selection.selected_pages] == [
        False,
        True,
        False,
        False,
        True,
        False,
    ]
    assert selection.selected_pages[0].token_rows == (0,)
    assert selection.selected_pages[3].token_rows == (1,)


def test_sparse_selection_collector_simulates_layer_local_miss_curve():
    layer_name = "model.layers.0.self_attn.attn"
    page0 = LogicalPage(0, layer_name, 0)
    page1 = LogicalPage(0, layer_name, 1)
    page2 = LogicalPage(0, layer_name, 2)
    collector = SparseSelectionCollector(hot_pool_blocks=(1, 2), page_size_bytes=10)

    stats0 = collector.record(
        layer_name,
        SparsePageSelection(
            selected_pages=(),
            unique_pages=(page0, page1),
            tail_pages=frozenset(),
        ),
    )
    stats1 = collector.record(
        layer_name,
        SparsePageSelection(
            selected_pages=(),
            unique_pages=(page1, page2),
            tail_pages=frozenset({page2}),
        ),
    )

    assert stats0.simulated_miss_pages == {1: 2, 2: 2}
    assert stats0.estimated_bytes == {1: 20, 2: 20}
    assert stats1.simulated_miss_pages == {1: 0, 2: 0}
    assert stats1.estimated_bytes == {1: 0, 2: 0}
    assert stats1.num_tail_pages == 1
