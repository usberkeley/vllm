# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from vllm.compilation.backends import split_graph
from vllm.config import AFDConfig
from vllm.distributed.afd_transfer.connector import (
    STAGE_F2A,
    AFDConnectorRole,
    AFDTransferIdAllocator,
)
from vllm.distributed.afd_transfer.connector.loopback import LoopbackAFDConnector
from vllm.distributed.afd_transfer.graph_cut import (
    AFDFXAttentionContext,
    AFDFXAttentionExecutor,
    AFDGraphBoundary,
    is_afd_cut_node,
    partition_afd_graph,
)
from vllm.distributed.afd_transfer.scheduler import AFDDynamicBatchScheduler
from vllm.distributed.afd_transfer.worker_loop import AFDAttentionEventLoop
from vllm.model_executor.layers.afd.ops import AFD_SPLITTING_OPS


def _graph(x, residual):
    saved_residual = residual * 2
    hidden = torch.ops.aten.add.Tensor(x, 1)
    hidden = torch.ops.vllm.afd_cut(hidden, 2)
    hidden = torch.ops.aten.add.Tensor(hidden, saved_residual)
    hidden = torch.ops.aten.mul.Tensor(hidden, 3)
    hidden = torch.ops.vllm.afd_cut(hidden, 5)
    return torch.ops.aten.add.Tensor(hidden, saved_residual)


def _program_inputs():
    torch.manual_seed(0)
    x = torch.randn(4, 8)
    residual = torch.randn(4, 8)
    return x, residual


def _expected(x, residual):
    saved_residual = residual * 2
    first_moe = (x + 1) * 10
    second_moe_input = (first_moe + saved_residual) * 3
    return second_moe_input - 4 + saved_residual


def test_pure_marker_is_preserved_as_an_fx_split_point():
    x, residual = _program_inputs()
    graph_module = make_fx(_graph)(x, residual)

    cuts = [node for node in graph_module.graph.nodes if is_afd_cut_node(node)]
    assert [node.args[1] for node in cuts] == [2, 5]

    _, split_items = split_graph(graph_module, AFD_SPLITTING_OPS)
    splitting_graphs = [item for item in split_items if item.is_splitting_graph]
    assert len(splitting_graphs) == 2


def test_graph_cut_program_injects_moe_outputs_and_preserves_continuation():
    x, residual = _program_inputs()
    program = partition_afd_graph(make_fx(_graph)(x, residual))

    first = program.start(x, residual)
    assert isinstance(first, AFDGraphBoundary)
    assert first.layer_id == 2
    torch.testing.assert_close(first.hidden_states, x + 1)

    second = program.resume(first, first.hidden_states * 10)
    assert isinstance(second, AFDGraphBoundary)
    assert second.layer_id == 5
    torch.testing.assert_close(
        second.hidden_states,
        ((x + 1) * 10 + residual * 2) * 3,
    )

    actual = program.resume(second, second.hidden_states - 4)
    torch.testing.assert_close(actual, _expected(x, residual))

    assert all(
        not any(is_afd_cut_node(node) for node in segment.graph_module.graph.nodes)
        for segment in program.segments
    )
    assert program.segments[0].output_names


def test_graph_cut_program_validates_remote_output_contract():
    x, residual = _program_inputs()
    program = partition_afd_graph(make_fx(_graph)(x, residual))
    boundary = program.start(x, residual)
    assert isinstance(boundary, AFDGraphBoundary)

    with pytest.raises(ValueError, match="shape, dtype, and device"):
        program.resume(boundary, torch.randn(1, 8))


def test_graph_cut_program_rejects_a_stale_boundary():
    x, residual = _program_inputs()
    program = partition_afd_graph(make_fx(_graph)(x, residual))
    first = program.start(x, residual)
    assert isinstance(first, AFDGraphBoundary)
    second = program.resume(first, first.hidden_states * 10)
    assert isinstance(second, AFDGraphBoundary)

    with pytest.raises(RuntimeError, match="stale"):
        program.resume(first, first.hidden_states * 10)


def test_graph_cut_program_supports_concurrent_continuations():
    x, residual = _program_inputs()
    program = partition_afd_graph(make_fx(_graph)(x, residual))

    one = program.start(x, residual)
    two = program.start(x + 1, residual + 1)
    assert isinstance(one, AFDGraphBoundary)
    assert isinstance(two, AFDGraphBoundary)

    one = program.resume(one, one.hidden_states * 10)
    two = program.resume(two, two.hidden_states * 10)
    assert isinstance(one, AFDGraphBoundary)
    assert isinstance(two, AFDGraphBoundary)

    actual_one = program.resume(one, one.hidden_states - 4)
    actual_two = program.resume(two, two.hidden_states - 4)
    torch.testing.assert_close(actual_one, _expected(x, residual))
    torch.testing.assert_close(actual_two, _expected(x + 1, residual + 1))


def test_value_used_only_by_a_later_cut_stays_live():
    def graph_with_late_cut(x):
        saved = x * 2
        hidden = torch.ops.vllm.afd_cut(x + 1, 1)
        hidden = hidden + 3
        late_hidden = torch.ops.vllm.afd_cut(saved, 4)
        return hidden + late_hidden

    x = torch.randn(3, 4)
    program = partition_afd_graph(make_fx(graph_with_late_cut)(x))
    first = program.start(x)
    assert isinstance(first, AFDGraphBoundary)
    second = program.resume(first, first.hidden_states * 10)
    assert isinstance(second, AFDGraphBoundary)
    torch.testing.assert_close(second.hidden_states, x * 2)

    actual = program.resume(second, second.hidden_states - 1)
    torch.testing.assert_close(actual, (x + 1) * 10 + 3 + x * 2 - 1)


def test_attention_event_loop_drives_fx_continuations():
    x, residual = _program_inputs()
    program = partition_afd_graph(make_fx(_graph)(x, residual))
    executor = AFDFXAttentionExecutor(program)
    config = SimpleNamespace(
        afd_config=AFDConfig(
            role="attention",
            afd_connector="LoopbackAFDConnector",
        )
    )
    connector = LoopbackAFDConnector(config, AFDConnectorRole.ATTENTION)
    connector.register_moe(2, lambda hidden: hidden * 10)
    connector.register_moe(5, lambda hidden: hidden - 4)
    connector.start_listening(STAGE_F2A)

    scheduler = AFDDynamicBatchScheduler(
        num_layers=6,
        age_limit_s=0.0,
    )
    completed = []

    context = AFDFXAttentionContext((x, residual))

    def replay(layer_id, hidden, items):
        item_context = items[0].context
        return executor.run_attention_segment(layer_id, hidden, item_context)

    def finish(hidden, _meta, item_context):
        completed.append(executor.finalize_attention(hidden, item_context))

    loop = AFDAttentionEventLoop(
        scheduler=scheduler,
        source=connector,
        sink=connector,
        replay_fn=replay,
        completion_fn=finish,
        capture_sizes=None,
        layer_ids=executor.layer_ids,
        transfer_ids=AFDTransferIdAllocator(),
        credit_capacity=1,
    )
    loop.submit(x, now=0.0, context=context)

    for tick in range(5):
        loop.tick(float(tick))
        if completed:
            break

    assert len(completed) == 1
    torch.testing.assert_close(completed[0], _expected(x, residual))
