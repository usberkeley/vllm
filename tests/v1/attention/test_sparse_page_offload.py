# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import ReqMeta
from vllm.distributed.kv_transfer.kv_connector.v1.sparse_page_connector import (
    SparsePageConnector,
    SparsePageNixlPullConnectorWorker,
    _merge_sideband_params,
)
from vllm.v1.attention.backends.mla.page_offload.adapters.deepseek_v4_c4a import (
    DeepSeekV4C4AAdapter,
)
from vllm.v1.attention.backends.mla.page_offload.config import (
    SparsePageOffloadConfig,
    SparsePageParallelTopology,
)
from vllm.v1.attention.backends.mla.page_offload.coordinator import (
    SparsePageOffloadCoordinator,
    get_sparse_page_offload_coordinator,
)
from vllm.v1.attention.backends.mla.page_offload.protocol import (
    SPARSE_PAGE_SIDEBAND_VERSION,
    SparsePagePrefillSideband,
    SparsePagePrefillWorkerMetadata,
    SparsePageRankTransfer,
    SparsePageReference,
    SparsePageRoute,
    SparsePageTransferPage,
)
from vllm.v1.attention.backends.mla.page_offload.route_tracker import (
    SparsePageRouteTracker,
)
from vllm.v1.attention.backends.mla.page_offload.selection import (
    LogicalPage,
    SelectedPage,
    SparsePageSelection,
)
from vllm.v1.attention.backends.mla.page_offload.selection_metrics import (
    SparsePageSelectionCollector,
)
from vllm.v1.attention.backends.mla.page_offload.staging import (
    SparsePageStagingManager,
)


def _hf_config(**overrides):
    values = {
        "model_type": "deepseek_v4",
        "compress_ratios": [1, 4, 128, 4],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _kv_transfer_config(
    kv_role="kv_consumer",
    connector="SparsePageConnector",
    **extra,
):
    return KVTransferConfig(
        kv_connector=connector,
        kv_role=kv_role,
        kv_connector_extra_config=extra,
    )


def test_config_parses_layer_ranges_and_hot_pool_blocks():
    config = SparsePageOffloadConfig.from_kv_transfer_config(
        _kv_transfer_config(
            sparse_page_hot_pool_blocks=16,
            sparse_page_offload_layers="1,3-5",
        )
    )

    assert config.enabled
    assert config.hot_pages_per_request == 16
    assert config.includes_layer("model.layers.1.self_attn.attn")
    assert config.includes_layer("model.layers.4.self_attn.attn")
    assert not config.includes_layer("model.layers.2.self_attn.attn")


def test_config_roles_enable_only_their_side_of_offload():
    assert not SparsePageOffloadConfig.from_kv_transfer_config(None).enabled
    producer = SparsePageOffloadConfig.from_kv_transfer_config(
        _kv_transfer_config(kv_role="kv_producer")
    )
    assert producer.enabled
    assert producer.can_seal_prefill
    assert not producer.can_stage_decode

    consumer = SparsePageOffloadConfig.from_kv_transfer_config(
        _kv_transfer_config(kv_role="kv_consumer")
    )
    assert consumer.enabled
    assert consumer.can_stage_decode
    assert not consumer.can_seal_prefill

    assert not SparsePageOffloadConfig.from_kv_transfer_config(
        _kv_transfer_config(kv_role="kv_both")
    ).enabled
    assert not SparsePageOffloadConfig.from_kv_transfer_config(
        _kv_transfer_config(connector="NixlConnector")
    ).enabled


def test_parallel_topology_supports_tp_and_ep_and_rejects_unsupported_modes():
    topology = SparsePageParallelTopology(
        engine_id="decode",
        tp_size=2,
        dp_rank=1,
        dp_size=4,
        expert_parallel=True,
    )

    topology.validate()
    topology.validate_remote_tp_size(1)
    topology.validate_remote_tp_size(2)
    topology.validate_remote_tp_size(4)
    with pytest.raises(ValueError, match="integer multiples"):
        topology.validate_remote_tp_size(3)
    with pytest.raises(NotImplementedError, match="DCP=2"):
        SparsePageParallelTopology(
            engine_id="decode",
            dcp_size=2,
        ).validate()
    with pytest.raises(NotImplementedError, match="elastic expert parallel"):
        SparsePageParallelTopology(
            engine_id="decode",
            elastic_ep=True,
        ).validate()


def test_sparse_page_route_binds_dp_owner_without_coupling_ep_topologies():
    producer = SparsePageParallelTopology(
        engine_id="prefill",
        tp_size=1,
        dp_rank=2,
        dp_size=4,
        expert_parallel=True,
    )
    consumer = SparsePageParallelTopology(
        engine_id="decode",
        tp_size=2,
        dp_rank=3,
        dp_size=8,
        expert_parallel=False,
    )

    route = SparsePageRoute.from_producer(producer, generation=7)
    bound = SparsePageRoute.from_dict(route.to_dict()).bind_consumer(
        consumer,
        remote_engine_id="prefill",
    )

    assert bound.producer_dp_rank == 2
    assert bound.producer_tp_size == 1
    assert bound.producer_expert_parallel
    assert bound.consumer_engine_id == "decode"
    assert bound.consumer_dp_rank == 3
    assert bound.consumer_tp_size == 2
    assert not bound.consumer_expert_parallel

    with pytest.raises(ValueError, match="different consumer"):
        bound.bind_consumer(
            SparsePageParallelTopology(
                engine_id="decode",
                tp_size=2,
                dp_rank=4,
                dp_size=8,
            ),
            remote_engine_id="prefill",
        )


def test_sparse_page_route_tracker_owns_producer_generation_lifecycle():
    tracker = SparsePageRouteTracker()

    assert tracker.begin_producer_request("req-a") == 1
    assert tracker.get_or_create_producer_generation("req-a") == 1
    with pytest.raises(ValueError, match="already active on producer"):
        tracker.begin_producer_request("req-a")

    tracker.finish_request("req-a")

    assert tracker.begin_producer_request("req-a") == 2


def test_sparse_page_connector_is_registered():
    assert (
        KVConnectorFactory.get_connector_class_by_name("SparsePageConnector")
        is SparsePageConnector
    )


def test_deepseek_v4_c4a_adapter_supports_only_c4a_fp8_ds_mla_layers():
    config = SparsePageOffloadConfig(enabled=True)
    adapter = DeepSeekV4C4AAdapter(_hf_config(), config)

    assert adapter.supports_layer("model.layers.1.self_attn.attn", "fp8_ds_mla")
    assert adapter.supports_layer("model.layers.3.self_attn.attn", "fp8_ds_mla")
    assert not adapter.supports_layer("model.layers.0.self_attn.attn", "fp8_ds_mla")
    assert not adapter.supports_layer("model.layers.2.self_attn.attn", "fp8_ds_mla")
    assert not adapter.supports_layer("model.layers.1.self_attn.attn", "auto")


def test_deepseek_v4_c4a_adapter_extracts_unique_pages_and_tail_pages():
    hf_config = _hf_config(compress_ratios=[4])
    config = SparsePageOffloadConfig(enabled=True)
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


def test_sparse_selection_collector_simulates_same_layer_cross_decode_step_miss_curve():
    layer_name = "model.layers.0.self_attn.attn"
    page0 = LogicalPage(0, layer_name, 0)
    page1 = LogicalPage(0, layer_name, 1)
    page2 = LogicalPage(0, layer_name, 2)
    collector = SparsePageSelectionCollector(hot_page_capacity=2, page_size_bytes=10)

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

    assert stats0.same_layer_cross_decode_step_miss_pages == 2
    assert stats0.same_layer_cross_decode_step_estimated_bytes == 20
    assert stats1.same_layer_cross_decode_step_miss_pages == 0
    assert stats1.same_layer_cross_decode_step_estimated_bytes == 0
    assert stats0.decode_step == 0
    assert stats1.decode_step == 1
    assert stats1.num_tail_pages == 1


def test_sparse_selection_collector_tracks_same_step_reference_missing_pages():
    reference_layer = "model.layers.0.self_attn.attn"
    layer1 = "model.layers.1.self_attn.attn"
    collector = SparsePageSelectionCollector(hot_page_capacity=2, page_size_bytes=10)

    stats0 = collector.record(
        reference_layer,
        SparsePageSelection(
            selected_pages=(),
            unique_pages=(
                LogicalPage(0, reference_layer, 0),
                LogicalPage(0, reference_layer, 1),
            ),
            tail_pages=frozenset(),
        ),
    )
    stats1 = collector.record(
        layer1,
        SparsePageSelection(
            selected_pages=(),
            unique_pages=(
                LogicalPage(0, layer1, 1),
                LogicalPage(0, layer1, 2),
            ),
            tail_pages=frozenset(),
        ),
    )

    assert stats0.reference_layer_name == reference_layer
    assert stats0.same_decode_step_reference_missing_pages == 0
    assert stats0.decode_step == 0
    assert stats1.same_layer_cross_decode_step_miss_pages == 2
    assert stats1.reference_layer_name == reference_layer
    assert stats1.same_decode_step_reference_missing_pages == 1
    assert stats1.same_decode_step_reference_missing_estimated_bytes == 10
    assert stats1.num_same_decode_step_reference_reused_pages == 1
    assert stats1.decode_step == 0


def test_sparse_selection_collector_increments_decode_step_once_per_layer_sweep():
    reference_layer = "model.layers.0.self_attn.attn"
    layer1 = "model.layers.1.self_attn.attn"
    collector = SparsePageSelectionCollector(hot_page_capacity=1, page_size_bytes=10)

    stats0 = collector.record(
        reference_layer,
        SparsePageSelection(
            selected_pages=(),
            unique_pages=(LogicalPage(0, reference_layer, 0),),
            tail_pages=frozenset(),
        ),
    )
    stats1 = collector.record(
        layer1,
        SparsePageSelection(
            selected_pages=(),
            unique_pages=(LogicalPage(0, layer1, 0),),
            tail_pages=frozenset(),
        ),
    )
    stats2 = collector.record(
        reference_layer,
        SparsePageSelection(
            selected_pages=(),
            unique_pages=(LogicalPage(0, reference_layer, 0),),
            tail_pages=frozenset(),
        ),
    )

    assert stats0.observation_index == 1
    assert stats1.observation_index == 2
    assert stats2.observation_index == 3
    assert stats0.decode_step == 0
    assert stats1.decode_step == 0
    assert stats2.decode_step == 1


def test_sparse_page_offload_coordinator_is_shared_per_vllm_config():
    hf_config = _hf_config(compress_ratios=[4])
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=hf_config),
        kv_transfer_config=_kv_transfer_config(sparse_page_hot_pool_blocks=2),
    )

    coordinator0 = get_sparse_page_offload_coordinator(vllm_config)
    coordinator1 = get_sparse_page_offload_coordinator(vllm_config)

    assert coordinator0 is not None
    assert coordinator0 is coordinator1


def test_coordinator_prepares_multiple_requests_on_one_dp_replica():
    layer_name = "model.layers.0.self_attn.attn"
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=_hf_config(compress_ratios=[4])),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2,
            data_parallel_rank=1,
            data_parallel_size=2,
            enable_expert_parallel=True,
            decode_context_parallel_size=1,
            enable_elastic_ep=False,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        kv_transfer_config=_kv_transfer_config(sparse_page_hot_pool_blocks=1),
    )
    coordinator = get_sparse_page_offload_coordinator(vllm_config)
    assert coordinator is not None
    kv_cache = torch.empty((4, 2, 3), dtype=torch.int32)
    block_table = torch.tensor([[2, 3], [6, 7]], dtype=torch.int32)
    for request_id in range(2):
        for page_idx in range(2):
            page = LogicalPage(request_id, layer_name, page_idx)
            coordinator.staging_manager.install_cpu_page(
                page,
                torch.full(
                    (2, 3),
                    request_id * 10 + page_idx,
                    dtype=torch.int32,
                ),
                ready=True,
            )

    result = coordinator.stage_decode_layer(
        layer_name=layer_name,
        kv_cache_dtype="fp8_ds_mla",
        req_id_per_token=torch.tensor([0, 1], dtype=torch.int32),
        topk_indices=torch.tensor([[0, 64], [0, 64]], dtype=torch.int32),
        seq_lens=torch.tensor([512, 512], dtype=torch.int32),
        kv_cache=kv_cache,
        source_block_table=block_table,
        is_decode_only=True,
    )

    assert result.enabled
    assert result.kv_cache.shape == (4, 2, 3)
    assert torch.equal(
        result.block_table,
        torch.tensor([[0, 2], [1, 3]], dtype=torch.int32),
    )


def test_staging_manager_stages_pages_and_patches_block_table():
    layer_name = "model.layers.0.self_attn.attn"
    page0 = LogicalPage(0, layer_name, 0)
    page1 = LogicalPage(0, layer_name, 1)
    kv_cache = torch.arange(4 * 2 * 3, dtype=torch.int32).reshape(4, 2, 3)
    block_table = torch.tensor([[2, 3, 1, 0]], dtype=torch.int32)
    manager = SparsePageStagingManager(hot_pages_per_request=2)

    result = manager.stage_decode_pages(
        layer_name=layer_name,
        selection=SparsePageSelection(
            selected_pages=(
                SelectedPage(page0, request_row=0, token_rows=(0,)),
                SelectedPage(page1, request_row=0, token_rows=(0,)),
            ),
            unique_pages=(page0, page1),
            tail_pages=frozenset({page1}),
        ),
        kv_cache=kv_cache,
        source_block_table=block_table,
    )

    assert result.enabled
    assert result.kv_cache.shape == (3, 2, 3)
    assert torch.equal(result.kv_cache[0], kv_cache[2])
    assert torch.equal(result.kv_cache[2], kv_cache[3])
    assert result.block_table is not block_table
    assert torch.equal(block_table, torch.tensor([[2, 3, 1, 0]], dtype=torch.int32))
    assert torch.equal(
        result.block_table,
        torch.tensor([[0, 2, 1, 0]], dtype=torch.int32),
    )
    assert result.miss_pages == (page0,)

    hot_pool_data_ptr = result.kv_cache.data_ptr()
    block_table_data_ptr = result.block_table.data_ptr()
    resident_page = result.kv_cache[0].clone()
    manager._cpu_page_store[page0].fill_(-1)
    kv_cache[2].fill_(-2)
    result1 = manager.stage_decode_pages(
        layer_name=layer_name,
        selection=SparsePageSelection(
            selected_pages=(SelectedPage(page0, request_row=0, token_rows=(0,)),),
            unique_pages=(page0,),
            tail_pages=frozenset(),
        ),
        kv_cache=kv_cache,
        source_block_table=block_table,
    )

    assert result1.kv_cache.data_ptr() == hot_pool_data_ptr
    assert result1.block_table.data_ptr() == block_table_data_ptr
    assert result1.miss_pages == ()
    assert torch.equal(result1.kv_cache[0], resident_page)


def test_staging_manager_handles_multiple_requests_per_dp_rank():
    layer_name = "model.layers.0.self_attn.attn"
    req0_hot = LogicalPage(0, layer_name, 0)
    req0_tail = LogicalPage(0, layer_name, 1)
    req1_hot = LogicalPage(1, layer_name, 0)
    req1_tail = LogicalPage(1, layer_name, 1)
    kv_cache = torch.arange(8 * 2 * 3, dtype=torch.int32).reshape(8, 2, 3)
    block_table = torch.tensor(
        [
            [2, 3],
            [6, 7],
        ],
        dtype=torch.int32,
    )
    manager = SparsePageStagingManager(
        hot_pages_per_request=1,
        max_tail_pages=2,
    )

    result = manager.stage_decode_pages(
        layer_name=layer_name,
        selection=SparsePageSelection(
            selected_pages=(
                SelectedPage(req0_hot, request_row=0, token_rows=(0,)),
                SelectedPage(req0_tail, request_row=0, token_rows=(0,)),
                SelectedPage(req1_hot, request_row=1, token_rows=(1,)),
                SelectedPage(req1_tail, request_row=1, token_rows=(1,)),
            ),
            unique_pages=(req0_hot, req0_tail, req1_hot, req1_tail),
            tail_pages=frozenset({req0_tail, req1_tail}),
        ),
        kv_cache=kv_cache,
        source_block_table=block_table,
    )

    assert result.enabled
    assert result.kv_cache.shape == (4, 2, 3)
    assert result.miss_pages == (req0_hot, req1_hot)
    assert torch.equal(result.kv_cache[0], kv_cache[2])
    assert torch.equal(result.kv_cache[1], kv_cache[6])
    assert torch.equal(result.kv_cache[2], kv_cache[3])
    assert torch.equal(result.kv_cache[3], kv_cache[7])
    assert torch.equal(
        result.block_table,
        torch.tensor([[0, 2], [1, 3]], dtype=torch.int32),
    )


def test_staging_manager_falls_back_when_hot_pool_is_too_small():
    layer_name = "model.layers.0.self_attn.attn"
    page0 = LogicalPage(0, layer_name, 0)
    page1 = LogicalPage(0, layer_name, 1)
    kv_cache = torch.arange(4 * 2 * 3, dtype=torch.int32).reshape(4, 2, 3)
    block_table = torch.tensor([[2, 3, 1, 0]], dtype=torch.int32)
    manager = SparsePageStagingManager(hot_pages_per_request=1)

    result = manager.stage_decode_pages(
        layer_name=layer_name,
        selection=SparsePageSelection(
            selected_pages=(
                SelectedPage(page0, request_row=0, token_rows=(0,)),
                SelectedPage(page1, request_row=0, token_rows=(0,)),
            ),
            unique_pages=(page0, page1),
            tail_pages=frozenset(),
        ),
        kv_cache=kv_cache,
        source_block_table=block_table,
    )

    assert not result.enabled
    assert result.kv_cache is kv_cache
    assert result.block_table is block_table


def test_consumer_staging_requires_ready_authoritative_cpu_pages():
    layer_name = "model.layers.0.self_attn.attn"
    page = LogicalPage("req-a", layer_name, 0, generation=2)
    kv_cache = torch.full((4, 2, 3), -1, dtype=torch.int32)
    block_table = torch.tensor([[2, 3]], dtype=torch.int32)
    selection = SparsePageSelection(
        selected_pages=(SelectedPage(page, request_row=0, token_rows=(0,)),),
        unique_pages=(page,),
        tail_pages=frozenset(),
    )
    manager = SparsePageStagingManager(
        hot_pages_per_request=1,
        require_authoritative_cpu_pages=True,
    )

    with pytest.raises(RuntimeError, match="did not provide authoritative"):
        manager.stage_decode_pages(
            layer_name=layer_name,
            selection=selection,
            kv_cache=kv_cache,
            source_block_table=block_table,
        )

    destination = manager.reserve_cpu_page(page, kv_cache[0])
    destination.fill_(7)
    with pytest.raises(RuntimeError, match="is not ready"):
        manager.stage_decode_pages(
            layer_name=layer_name,
            selection=selection,
            kv_cache=kv_cache,
            source_block_table=block_table,
        )

    manager.mark_cpu_pages_ready((page,))
    result = manager.stage_decode_pages(
        layer_name=layer_name,
        selection=selection,
        kv_cache=kv_cache,
        source_block_table=block_table,
    )

    assert result.enabled
    assert torch.equal(result.kv_cache[0], torch.full((2, 3), 7, dtype=torch.int32))


def test_consumer_seals_mutable_tail_into_cpu_pool_on_page_rollover():
    layer_name = "model.layers.0.self_attn.attn"
    page0 = LogicalPage("req-a", layer_name, 0, generation=2)
    page1 = LogicalPage("req-a", layer_name, 1, generation=2)
    kv_cache = torch.full((4, 2, 3), -1, dtype=torch.int32)
    kv_cache[2].fill_(1)
    block_table = torch.tensor([[2, 3]], dtype=torch.int32)
    manager = SparsePageStagingManager(
        hot_pages_per_request=1,
        require_authoritative_cpu_pages=True,
    )
    manager.install_cpu_page(
        page0,
        torch.full((2, 3), 1, dtype=torch.int32),
        ready=True,
    )
    manager.stage_decode_pages(
        layer_name=layer_name,
        selection=SparsePageSelection(
            selected_pages=(
                SelectedPage(page0, request_row=0, token_rows=(0,), is_tail=True),
            ),
            unique_pages=(page0,),
            tail_pages=frozenset((page0,)),
            current_tail_pages=(
                SelectedPage(page0, request_row=0, is_tail=True),
            ),
        ),
        kv_cache=kv_cache,
        source_block_table=block_table,
    )

    kv_cache[2].fill_(7)
    result = manager.stage_decode_pages(
        layer_name=layer_name,
        selection=SparsePageSelection(
            selected_pages=(
                SelectedPage(page0, request_row=0, token_rows=(0,)),
            ),
            unique_pages=(page0,),
            tail_pages=frozenset(),
            current_tail_pages=(
                SelectedPage(page1, request_row=0, is_tail=True),
            ),
        ),
        kv_cache=kv_cache,
        source_block_table=block_table,
    )

    assert result.enabled
    assert torch.equal(result.kv_cache[0], torch.full((2, 3), 7, dtype=torch.int32))


def test_partial_allocation_maps_tail_writes_and_never_reads_logical_gpu_block():
    layer_name = "model.layers.0.self_attn.attn"
    page0 = LogicalPage("req-a", layer_name, 0, generation=2)
    page1 = LogicalPage("req-a", layer_name, 1, generation=2)
    kv_cache = torch.empty((2, 2, 3), dtype=torch.int32)
    manager = SparsePageStagingManager(
        hot_pages_per_request=1,
        max_tail_pages=1,
        require_authoritative_cpu_pages=True,
        allocate_partial=True,
    )
    initial_tail = torch.full((2, 3), 1, dtype=torch.int32)
    manager.install_cpu_page(page0, initial_tail.clone(), ready=True)
    manager.restore_received_tail(page0, initial_tail, kv_cache)

    mapped = manager.prepare_tail_slot_mapping(
        layer_name=layer_name,
        positions=torch.tensor([255], dtype=torch.int64),
        request_rows=torch.tensor([0], dtype=torch.int32),
        request_identities={0: ("req-a", 2)},
        original_slot_mapping=torch.tensor([999], dtype=torch.int64),
        kv_cache=kv_cache,
        compress_ratio=4,
        storage_block_size=64,
    )
    assert mapped.tolist() == [127]

    kv_cache[1].fill_(7)
    rollover_mapping = manager.prepare_tail_slot_mapping(
        layer_name=layer_name,
        positions=torch.tensor([256], dtype=torch.int64),
        request_rows=torch.tensor([0], dtype=torch.int32),
        request_identities={0: ("req-a", 2)},
        original_slot_mapping=torch.tensor([1000], dtype=torch.int64),
        kv_cache=kv_cache,
        compress_ratio=4,
        storage_block_size=64,
    )
    assert rollover_mapping.tolist() == [64]

    result = manager.stage_decode_pages(
        layer_name=layer_name,
        selection=SparsePageSelection(
            selected_pages=(
                SelectedPage(page0, request_row=0, token_rows=(0,)),
            ),
            unique_pages=(page0,),
            tail_pages=frozenset(),
            current_tail_pages=(
                SelectedPage(page1, request_row=0, is_tail=True),
            ),
        ),
        kv_cache=kv_cache,
        source_block_table=torch.tensor([[500, 501]], dtype=torch.int32),
    )

    assert result.kv_cache.data_ptr() == kv_cache.data_ptr()
    assert result.block_table.tolist() == [[0, 501]]
    assert torch.equal(result.kv_cache[0], torch.full((2, 3), 7, dtype=torch.int32))


def test_staging_manager_seals_prefill_to_cpu_pool():
    layer_name = "model.layers.0.self_attn.attn"
    pages = tuple(LogicalPage(0, layer_name, idx) for idx in range(4))
    kv_cache = torch.arange(6 * 2 * 3, dtype=torch.int32).reshape(6, 2, 3)
    block_table = torch.tensor([[2, 3, 4, 5, 1, 0]], dtype=torch.int32)
    manager = SparsePageStagingManager(hot_pages_per_request=2)

    result = manager.seal_prefill_request(
        layer_name=layer_name,
        request_id=0,
        request_row=0,
        sequence_length=4 * 4 * 64,
        kv_cache=kv_cache,
        source_block_table=block_table,
        storage_block_size=64,
        compress_ratio=4,
    )

    # Seal sends the immutable pages to the CPU authoritative pool. It must not
    # touch the decode-side hot pool or free any GPU block.
    assert result.enabled
    assert result.tail_pages == (pages[3],)
    assert torch.equal(manager._cpu_page_store[pages[0]], kv_cache[2])
    assert torch.equal(manager._cpu_page_store[pages[1]], kv_cache[3])
    assert torch.equal(manager._cpu_page_store[pages[2]], kv_cache[4])
    assert layer_name not in manager._hot_pool_by_layer

    decode = manager.stage_decode_pages(
        layer_name=layer_name,
        selection=SparsePageSelection(
            selected_pages=(
                SelectedPage(pages[0], request_row=0, token_rows=(0,)),
                SelectedPage(pages[2], request_row=0, token_rows=(0,)),
            ),
            unique_pages=(pages[0], pages[2]),
            tail_pages=frozenset(),
        ),
        kv_cache=kv_cache,
        source_block_table=block_table,
    )

    # Decode owns the hot pool; both selected pages miss it and page in from the
    # CPU pool (page 2 was sealed there, page 0 is staged on demand).
    assert decode.enabled
    assert decode.miss_pages == (pages[0], pages[2])
    assert torch.equal(decode.kv_cache[0], kv_cache[2])
    assert torch.equal(decode.kv_cache[1], kv_cache[4])
    assert torch.equal(
        decode.block_table,
        torch.tensor([[0, 3, 1, 5, 1, 0]], dtype=torch.int32),
    )


def test_coordinator_seal_prefill_handles_batched_requests():
    layer_name = "model.layers.0.self_attn.attn"
    hf_config = _hf_config(compress_ratios=[4])
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=hf_config),
        kv_transfer_config=_kv_transfer_config(
            kv_role="kv_producer",
            sparse_page_hot_pool_blocks=4,
        ),
    )
    coordinator = get_sparse_page_offload_coordinator(vllm_config)
    assert coordinator is not None

    # Two prefill requests. seq_len 1024 -> compressed_len 256 -> tail page 3,
    # sealed pages {0, 1, 2}.
    seq_lens = torch.tensor([1024, 1024], dtype=torch.int32)
    req_id_per_token = torch.tensor([0, 1], dtype=torch.int32)
    kv_cache = torch.arange(30 * 2 * 3, dtype=torch.int32).reshape(30, 2, 3)
    source_block_table = torch.tensor(
        [
            [10, 11, 12, 13],
            [20, 21, 22, 23],
        ],
        dtype=torch.int32,
    )

    coordinator.seal_prefill_request(
        layer_name=layer_name,
        kv_cache_dtype="fp8_ds_mla",
        req_id_per_token=req_id_per_token,
        seq_lens=seq_lens,
        kv_cache=kv_cache,
        source_block_table=source_block_table,
    )

    metadata = coordinator.pop_prefill_worker_metadata(("req-0", "req-1"))
    assert metadata == SparsePagePrefillWorkerMetadata(
        request_sidebands={
            "req-0": SparsePagePrefillSideband(
                tail_pages=(SparsePageReference(layer_name, 3),),
            ),
            "req-1": SparsePagePrefillSideband(
                tail_pages=(SparsePageReference(layer_name, 3),),
            ),
        }
    )
    # Both requests seal pages {0, 1, 2} into the CPU pool via their own block
    # rows; no GPU block is freed here.
    cpu_page_store = coordinator.staging_manager._cpu_page_store
    for req, base in ((0, 10), (1, 20)):
        for page_idx in range(3):
            page = LogicalPage(req, layer_name, page_idx)
            assert torch.equal(cpu_page_store[page], kv_cache[base + page_idx])


def test_coordinator_drains_prefill_sideband_as_worker_metadata():
    layer_name = "model.layers.0.self_attn.attn"
    hf_config = _hf_config(compress_ratios=[4])
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=hf_config),
        kv_transfer_config=_kv_transfer_config(
            kv_role="kv_producer",
            sparse_page_hot_pool_blocks=4,
        ),
    )
    coordinator = get_sparse_page_offload_coordinator(vllm_config)
    assert coordinator is not None

    coordinator.seal_prefill_request(
        layer_name=layer_name,
        kv_cache_dtype="fp8_ds_mla",
        req_id_per_token=torch.tensor([0], dtype=torch.int32),
        seq_lens=torch.tensor([1024], dtype=torch.int32),
        kv_cache=torch.arange(20 * 2 * 3, dtype=torch.int32).reshape(20, 2, 3),
        source_block_table=torch.tensor([[10, 11, 12, 13]], dtype=torch.int32),
    )

    metadata = coordinator.pop_prefill_worker_metadata(("req-a",))
    assert metadata == SparsePagePrefillWorkerMetadata(
        request_sidebands={
            "req-a": SparsePagePrefillSideband(
                tail_pages=(SparsePageReference(layer_name, 3),),
            )
        }
    )
    assert coordinator.pop_prefill_worker_metadata(("req-a",)) is None


def test_sparse_page_prefill_worker_metadata_aggregates_and_serializes():
    layer_name = "model.layers.0.self_attn.attn"
    metadata0 = SparsePagePrefillWorkerMetadata(
        request_sidebands={
            "req-a": SparsePagePrefillSideband(
                tail_pages=(SparsePageReference(layer_name, 3),),
            )
        }
    )
    metadata1 = SparsePagePrefillWorkerMetadata(
        request_sidebands={
            "req-a": SparsePagePrefillSideband(
                tail_pages=(SparsePageReference(layer_name, 4),),
            )
        }
    )

    merged = metadata0.aggregate(metadata1)

    assert isinstance(merged, SparsePagePrefillWorkerMetadata)
    sideband = merged.request_sidebands["req-a"]
    assert sideband.to_kv_transfer_params() == {
        "version": SPARSE_PAGE_SIDEBAND_VERSION,
        "tail_pages": [
            {"layer_name": layer_name, "page_idx": 3},
            {"layer_name": layer_name, "page_idx": 4},
        ],
    }


def test_sparse_page_prefill_worker_metadata_preserves_per_rank_page_descriptors():
    layer_name = "model.layers.0.self_attn.attn"
    transfers = tuple(
        SparsePageRankTransfer(
            tp_rank=rank,
            pages=(
                SparsePageTransferPage(
                    layer_name=layer_name,
                    page_idx=0,
                    source_address=1000 + rank * 100,
                    page_size_bytes=128,
                ),
            ),
        )
        for rank in range(2)
    )
    metadata0 = SparsePagePrefillWorkerMetadata(
        request_sidebands={
            "req-a": SparsePagePrefillSideband(rank_transfers=(transfers[0],))
        }
    )
    metadata1 = SparsePagePrefillWorkerMetadata(
        request_sidebands={
            "req-a": SparsePagePrefillSideband(rank_transfers=(transfers[1],))
        }
    )

    merged = metadata0.aggregate(metadata1)
    params = merged.request_sidebands["req-a"].to_kv_transfer_params()
    round_trip = SparsePagePrefillSideband.from_kv_transfer_params(params)

    assert round_trip.rank_transfers == transfers


class _FakeNixlAgent:
    def __init__(self):
        self.registered = []
        self.deregistered = []
        self.transfers = []
        self.released_dlists = []
        self.notifications = []

    @staticmethod
    def get_reg_descs(memory_data, memory_type):
        return (memory_type, tuple(memory_data))

    def register_memory(self, descs, backends):
        self.registered.append((descs, tuple(backends)))

    def deregister_memory(self, descs):
        self.deregistered.append(descs)

    @staticmethod
    def get_xfer_descs(memory_data, memory_type):
        return (memory_type, tuple(memory_data))

    @staticmethod
    def prep_xfer_dlist(agent, descs):
        return (agent, descs)

    def make_prepped_xfer(self, operation, local, local_ids, remote, remote_ids):
        handle = (operation, local, tuple(local_ids), remote, tuple(remote_ids))
        self.transfers.append(handle)
        return handle

    @staticmethod
    def transfer(handle):
        return handle

    def release_dlist_handle(self, handle):
        self.released_dlists.append(handle)

    def send_notif(self, agent, notif_msg):
        self.notifications.append((agent, notif_msg))


def test_sparse_page_worker_registers_prefill_cpu_pages_per_tp_rank():
    layer_name = "model.layers.0.self_attn.attn"
    page0 = LogicalPage("req-a", layer_name, 0)
    page1 = LogicalPage("req-a", layer_name, 1)
    tensor0 = torch.arange(6, dtype=torch.int32).reshape(2, 3)
    tensor1 = tensor0 + 10
    worker = object.__new__(SparsePageNixlPullConnectorWorker)
    worker.tp_rank = 1
    worker.nixl_wrapper = _FakeNixlAgent()
    worker.nixl_backends = ["UCX"]
    worker._sparse_source_registrations = {}

    transfer = worker.register_sparse_source_pages(
        request_id="req-a",
        generation=3,
        pages=((page1, tensor1), (page0, tensor0)),
    )

    assert transfer.tp_rank == 1
    assert tuple((page.layer_name, page.page_idx) for page in transfer.pages) == (
        (layer_name, 0),
        (layer_name, 1),
    )
    assert transfer.pages[0].source_address == tensor0.data_ptr()
    assert transfer.pages[0].page_size_bytes == tensor0.nbytes
    assert len(worker.nixl_wrapper.registered) == 1


def test_sparse_page_worker_reads_registered_prefill_pages_into_consumer_pool():
    layer_name = "model.layers.0.self_attn.attn"
    config = SparsePageOffloadConfig(
        enabled=True,
        role="kv_consumer",
        hot_pages_per_request=1,
    )
    coordinator = SparsePageOffloadCoordinator(
        config,
        DeepSeekV4C4AAdapter(_hf_config(compress_ratios=[4]), config),
    )
    kv_cache = torch.full((4, 2, 3), -1, dtype=torch.int32)
    page_size_bytes = kv_cache[0].nbytes
    route = SparsePageRoute.from_producer(
        SparsePageParallelTopology(engine_id="prefill", tp_size=1),
        generation=3,
    )
    sideband = SparsePagePrefillSideband(
        rank_transfers=(
            SparsePageRankTransfer(
                tp_rank=0,
                pages=(
                    SparsePageTransferPage(
                        layer_name=layer_name,
                        page_idx=0,
                        source_address=123456,
                        page_size_bytes=page_size_bytes,
                    ),
                    SparsePageTransferPage(
                        layer_name=layer_name,
                        page_idx=1,
                        source_address=123456 + page_size_bytes,
                        page_size_bytes=page_size_bytes,
                    ),
                ),
            ),
        ),
        tail_pages=(SparsePageReference(layer_name, 1),),
    )
    params = sideband.to_kv_transfer_params()
    params["route"] = route.to_dict()
    meta = ReqMeta(
        local_block_ids=([2, 3],),
        local_physical_block_ids=([2, 3],),
        tp_size=1,
    )
    worker = object.__new__(SparsePageNixlPullConnectorWorker)
    worker._coordinator = coordinator
    worker._remote_agents = {"prefill": {(0, 0): "prefill-rank-0"}}
    worker.tp_mappings = {"prefill": SimpleNamespace(all_source_ranks=(0,))}
    worker.device_kv_caches = {layer_name: kv_cache}
    worker.kv_cache_config = SimpleNamespace(
        kv_cache_groups=(SimpleNamespace(layer_names=(layer_name,)),)
    )
    worker.nixl_wrapper = _FakeNixlAgent()
    worker.nixl_backends = ["UCX"]
    worker._recving_transfers = defaultdict(list)
    worker._sparse_receive_jobs = {}
    worker.world_size = 1
    worker.tp_rank = 0

    assert worker._try_start_sparse_receive("req-a", params, meta)
    assert len(worker.nixl_wrapper.transfers) == 1
    assert worker.nixl_wrapper.transfers[0][0] == "READ"

    job = worker._sparse_receive_jobs.pop("req-a")
    for page_tensor in job.page_tensors.values():
        page_tensor.fill_(9)
    worker._finish_sparse_receive("req-a", job)
    coordinator.bind_request_rows((('req-a', 3),))
    result = coordinator.stage_decode_layer(
        layer_name=layer_name,
        kv_cache_dtype="fp8_ds_mla",
        req_id_per_token=torch.tensor([0], dtype=torch.int32),
        topk_indices=torch.tensor([[0]], dtype=torch.int32),
        seq_lens=torch.tensor([512], dtype=torch.int32),
        kv_cache=kv_cache,
        source_block_table=torch.tensor([[2, 3]], dtype=torch.int32),
        is_decode_only=True,
    )

    assert result.enabled
    assert torch.equal(result.kv_cache[0], torch.full((2, 3), 9, dtype=torch.int32))
    assert len(worker.nixl_wrapper.notifications) == 1


def test_sparse_page_params_merge_preserves_existing_sideband():
    layer_name = "model.layers.0.self_attn.attn"
    merged = _merge_sideband_params(
        {
            "version": 1,
            "tail_pages": [{"layer_name": layer_name, "page_idx": 2}],
        },
        SparsePagePrefillSideband(
            tail_pages=(SparsePageReference(layer_name, 3),),
        ).to_kv_transfer_params(),
    )

    assert merged == {
        "version": SPARSE_PAGE_SIDEBAND_VERSION,
        "tail_pages": [
            {"layer_name": layer_name, "page_idx": 2},
            {"layer_name": layer_name, "page_idx": 3},
        ],
    }


def test_sparse_page_connector_adds_producer_route_and_generation():
    layer_name = "model.layers.0.self_attn.attn"
    connector = object.__new__(SparsePageConnector)
    connector._topology = SparsePageParallelTopology(
        engine_id="prefill",
        tp_size=2,
        dp_rank=1,
        dp_size=4,
        expert_parallel=True,
    )
    connector._pending_sideband_by_request = {
        "req-a": SparsePagePrefillSideband(
            tail_pages=(SparsePageReference(layer_name, 3),),
        )
    }
    connector._route_tracker = SparsePageRouteTracker()
    for _ in range(3):
        connector._route_tracker.begin_producer_request("req-a")
        connector._route_tracker.finish_request("req-a")
    connector._route_tracker.begin_producer_request("req-a")

    params = connector._merge_prefill_sideband(
        "req-a",
        {"remote_engine_id": "prefill", "tp_size": 2},
    )

    assert params is not None
    sparse_params = params["sparse_page_offload"]
    assert sparse_params["version"] == SPARSE_PAGE_SIDEBAND_VERSION
    assert SparsePageRoute.from_dict(sparse_params["route"]) == SparsePageRoute(
        producer_engine_id="prefill",
        producer_dp_rank=1,
        producer_tp_size=2,
        producer_expert_parallel=True,
        generation=4,
    )


def test_sparse_page_connector_binds_one_dp_owner_and_rejects_stale_generation():
    producer = SparsePageParallelTopology(
        engine_id="prefill",
        tp_size=1,
        dp_rank=0,
        dp_size=2,
    )
    route = SparsePageRoute.from_producer(producer, generation=3)
    connector = object.__new__(SparsePageConnector)
    connector._topology = SparsePageParallelTopology(
        engine_id="decode",
        tp_size=2,
        dp_rank=1,
        dp_size=4,
        expert_parallel=True,
    )
    connector._route_tracker = SparsePageRouteTracker()
    params = {
        "remote_engine_id": "prefill",
        "remote_request_id": "remote-a",
        "tp_size": 1,
        "sparse_page_offload": {
            "version": SPARSE_PAGE_SIDEBAND_VERSION,
            "route": route.to_dict(),
            "tail_pages": [],
        },
    }
    request = SimpleNamespace(request_id="local-a", kv_transfer_params=params)

    connector._validate_and_bind_consumer_route(request)

    bound = SparsePageRoute.from_dict(params["sparse_page_offload"]["route"])
    assert bound.consumer_engine_id == "decode"
    assert bound.consumer_dp_rank == 1
    assert bound.consumer_tp_size == 2
    assert bound.consumer_expert_parallel

    duplicate_request = SimpleNamespace(
        request_id="local-duplicate",
        kv_transfer_params={
            **params,
            "sparse_page_offload": {
                **params["sparse_page_offload"],
                "route": route.to_dict(),
            },
        },
    )
    with pytest.raises(ValueError, match="already active"):
        connector._validate_and_bind_consumer_route(duplicate_request)

    reused_local_request = SimpleNamespace(
        request_id="local-a",
        kv_transfer_params={
            **params,
            "remote_request_id": "remote-b",
            "sparse_page_offload": {
                **params["sparse_page_offload"],
                "route": route.to_dict(),
            },
        },
    )
    with pytest.raises(ValueError, match="already active on consumer"):
        connector._validate_and_bind_consumer_route(reused_local_request)

    connector._route_tracker.finish_request("local-a")
    stale_request = SimpleNamespace(
        request_id="local-b",
        kv_transfer_params={
            **params,
            "sparse_page_offload": {
                **params["sparse_page_offload"],
                "route": route.to_dict(),
            },
        },
    )
    with pytest.raises(ValueError, match="Stale sparse page request"):
        connector._validate_and_bind_consumer_route(stale_request)


def test_staging_manager_cleanup_drops_request_state():
    layer_name = "model.layers.0.self_attn.attn"
    page0 = LogicalPage(0, layer_name, 0)
    kv_cache = torch.arange(4 * 2 * 3, dtype=torch.int32).reshape(4, 2, 3)
    block_table = torch.tensor([[2, 3, 1, 0]], dtype=torch.int32)
    manager = SparsePageStagingManager(hot_pages_per_request=1)

    first = manager.stage_decode_pages(
        layer_name=layer_name,
        selection=SparsePageSelection(
            selected_pages=(SelectedPage(page0, request_row=0, token_rows=(0,)),),
            unique_pages=(page0,),
            tail_pages=frozenset(),
        ),
        kv_cache=kv_cache,
        source_block_table=block_table,
    )
    manager.cleanup_request(0)
    second = manager.stage_decode_pages(
        layer_name=layer_name,
        selection=SparsePageSelection(
            selected_pages=(SelectedPage(page0, request_row=0, token_rows=(0,)),),
            unique_pages=(page0,),
            tail_pages=frozenset(),
        ),
        kv_cache=kv_cache,
        source_block_table=block_table,
    )

    assert first.miss_pages == (page0,)
    assert second.miss_pages == (page0,)
