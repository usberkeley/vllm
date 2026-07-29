# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from conftest import make_vllm_config
from torch import nn

from vllm.config import CUDAGraphMode
from vllm.distributed.afd_transfer.connector.base import (
    STAGE_A2F,
    STAGE_F2A,
    AFDHandle,
    AFDMeta,
    AFDTransferIdAllocator,
)
from vllm.distributed.afd_transfer.graph_cut import is_afd_cut_node
from vllm.distributed.afd_transfer.runtime import (
    AFDAttentionContext,
    AFDRuntime,
    AFDSegmentedModel,
    _AFDCUDAGraphSegmentRunner,
)
from vllm.distributed.afd_transfer.scheduler import AFDDynamicBatchScheduler
from vllm.forward_context import (
    BatchDescriptor,
    ForwardContext,
    override_forward_context,
)
from vllm.model_executor.layers.attention.afd_noop import AFDNoOpAttention
from vllm.model_executor.models.afd import AFDAttnBoundary, AFDFfnBoundary


class Scale(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, hidden_states):
        return hidden_states * self.factor


class DecoderLayer(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp

    def forward(self, positions, hidden_states, residual, optional=None):
        del positions, optional
        if residual is None:
            residual = hidden_states
        residual = residual + 0.25
        hidden_states = self.mlp(hidden_states + residual)
        return hidden_states, residual


class FinalNorm(nn.Module):
    def forward(self, hidden_states, residual):
        return hidden_states + residual, None


class Backbone(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.start_layer = 0
        self.end_layer = len(layers)
        self.norm = FinalNorm()

    def embed_input_ids(self, input_ids):
        return input_ids.float().unsqueeze(-1)


class CausalModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.model = backbone


def _full_forward(model, hidden, positions):
    residual = None
    for layer in model.model.layers:
        hidden, residual = layer(positions, hidden, residual)
    return model.model.norm(hidden, residual)[0]


def test_segmented_attention_matches_full_residual_decoder():
    baseline = CausalModel(
        Backbone(
            [
                DecoderLayer(Scale(2)),
                DecoderLayer(Scale(3)),
                DecoderLayer(Scale(5)),
                DecoderLayer(Scale(4)),
                DecoderLayer(Scale(6)),
            ]
        )
    )
    attention = CausalModel(
        Backbone(
            [
                DecoderLayer(Scale(2)),
                DecoderLayer(AFDAttnBoundary(1)),
                DecoderLayer(Scale(5)),
                DecoderLayer(AFDAttnBoundary(3)),
                DecoderLayer(Scale(6)),
            ]
        )
    )
    adapter = AFDSegmentedModel(attention, "attention")
    assert adapter.fx_executor is not None
    assert all(
        not any(is_afd_cut_node(node) for node in segment.graph_module.graph.nodes)
        for segment in adapter.fx_executor.program.segments
    )
    assert all(
        not boundary.emit_cut
        for _, boundary in adapter.boundaries.values()
        if isinstance(boundary, AFDAttnBoundary)
    )
    positions = torch.arange(4)
    hidden = torch.arange(4, dtype=torch.float32).unsqueeze(-1)
    expected = _full_forward(baseline, hidden, positions)
    context = AFDAttentionContext(positions, None, None)

    hidden = adapter.run_attention_segment(1, hidden, context)
    hidden = hidden * 3
    hidden = adapter.run_attention_segment(3, hidden, context)
    hidden = hidden * 4
    actual = adapter.finalize_attention(hidden, context)

    torch.testing.assert_close(actual, expected)


def test_segmented_ffn_calls_real_boundary_mlp():
    model = CausalModel(
        Backbone(
            [
                DecoderLayer(AFDFfnBoundary(Scale(3), 0)),
                DecoderLayer(AFDFfnBoundary(Scale(5), 1)),
            ]
        )
    )
    adapter = AFDSegmentedModel(model, "ffn")
    hidden = torch.ones(2, 4)
    torch.testing.assert_close(adapter.run_ffn(1, hidden), hidden * 5)


def test_segmented_runtime_rejects_unknown_layer_contract():
    class BadLayer(nn.Module):
        def forward(self, hidden_states):
            return hidden_states

    model = CausalModel(Backbone([BadLayer()]))
    model.model.layers[0].mlp = AFDAttnBoundary(0)
    with pytest.raises(NotImplementedError, match="positions"):
        AFDSegmentedModel(model, "attention")


def test_segmented_runtime_rejects_required_optional_layer_state():
    model = CausalModel(Backbone([DecoderLayer(AFDAttnBoundary(0))]))
    model.model.config = type("Config", (), {"llama_4_scaling": {"beta": 1}})()
    with pytest.raises(NotImplementedError, match="llama_4_scaling"):
        AFDSegmentedModel(model, "attention")


def test_afd_runtime_owns_ffn_role_setup(fake_model_v4):
    config = make_vllm_config("ffn")
    config.parallel_config = SimpleNamespace(
        pipeline_parallel_size=1,
        use_ubatching=False,
    )
    config.speculative_config = None
    config.lora_config = None

    runtime = AFDRuntime.create(
        config,
        fake_model_v4,
        supports_mm_inputs=False,
        is_pooling_model=False,
    )

    assert runtime is not None
    assert runtime.segmented.model is fake_model_v4
    assert all(
        isinstance(layer.attn, AFDNoOpAttention) for layer in fake_model_v4.layers
    )


def test_attention_runtime_configures_smallest_capture_bucket():
    config = make_vllm_config("attention")
    config.compilation_config.cudagraph_capture_sizes = [4, 8, 16]
    runtime = AFDRuntime(
        config,
        segmented=None,
        connector=None,
        scheduler=None,
        transfer_ids=None,
    )

    mode, descriptor, should_ubatch, tokens_across_dp = (
        runtime.configure_batch_execution(
            num_tokens=5,
            num_reqs=2,
            cudagraph_mode=CUDAGraphMode.FULL,
        )
    )

    assert mode == CUDAGraphMode.FULL
    assert descriptor.num_tokens == 8
    assert descriptor.num_reqs == 2
    assert should_ubatch is False
    assert tokens_across_dp is None


def test_attention_runtime_keeps_oversized_batch_unpadded():
    config = make_vllm_config("attention")
    config.compilation_config.cudagraph_capture_sizes = [4, 8]
    runtime = AFDRuntime(
        config,
        segmented=None,
        connector=None,
        scheduler=None,
        transfer_ids=None,
    )

    _, descriptor, _, _ = runtime.configure_batch_execution(
        num_tokens=9,
        num_reqs=2,
        cudagraph_mode=CUDAGraphMode.FULL,
    )

    assert descriptor.num_tokens == 9


def test_attention_runtime_executes_padded_bucket_and_returns_real_tokens():
    class LoopbackFfn:
        def __init__(self):
            self.pending: list[AFDHandle] = []
            self.sent_shapes: list[torch.Size] = []

        def isend(self, hidden, meta):
            assert meta.stage == STAGE_A2F
            self.sent_shapes.append(hidden.shape)
            self.pending.append(
                AFDHandle(
                    AFDMeta(meta.layer_id, STAGE_F2A, meta.transfer_id),
                    hidden * 3,
                )
            )

        def poll(self):
            pending, self.pending = self.pending, []
            return pending

    baseline = CausalModel(Backbone([DecoderLayer(Scale(3))]))
    attention = CausalModel(Backbone([DecoderLayer(AFDAttnBoundary(0))]))
    config = make_vllm_config("attention")
    config.compilation_config.cudagraph_capture_sizes = [4, 8]
    config.compilation_config.fast_moe_cold_start = False
    config.parallel_config = SimpleNamespace(
        data_parallel_size=1,
        use_sequence_parallel_moe=False,
        is_moe_model=True,
    )
    connector = LoopbackFfn()
    runtime = AFDRuntime(
        config,
        AFDSegmentedModel(attention, "attention"),
        connector,
        AFDDynamicBatchScheduler(1, age_limit_s=0.0),
        AFDTransferIdAllocator(),
    )
    input_ids = torch.tensor([1, 2, 3, 99])

    actual = runtime.execute_attention(
        input_ids=input_ids,
        inputs_embeds=None,
        positions=torch.arange(4),
        attn_metadata=None,
        slot_mappings=None,
        num_tokens_unpadded=3,
        cudagraph_mode=CUDAGraphMode.FULL,
        batch_descriptor=BatchDescriptor(num_tokens=4, num_reqs=1),
    )
    expected = _full_forward(
        baseline,
        input_ids[:3].float().unsqueeze(-1),
        torch.arange(3),
    )

    assert connector.sent_shapes == [torch.Size([3, 1])]
    assert actual.shape == (3, 1)
    torch.testing.assert_close(actual, expected)


def test_afd_cudagraph_segment_runner_reuses_static_input_address():
    input_addresses = []

    def runnable(hidden):
        return hidden + 1

    def wrapper(hidden):
        input_addresses.append(hidden.data_ptr())
        return runnable(hidden)

    runner = _AFDCUDAGraphSegmentRunner(
        runnable,
        vllm_config=None,
        wrapper=wrapper,
    )
    descriptor = BatchDescriptor(num_tokens=4, num_reqs=1)
    context = ForwardContext(
        no_compile_layers={},
        all_moe_layers=None,
        attn_metadata={},
        slot_mapping={},
        cudagraph_runtime_mode=CUDAGraphMode.FULL,
        batch_descriptor=descriptor,
    )

    with override_forward_context(context):
        first = runner(torch.ones(4, 2))
        second = runner(torch.full((4, 2), 3.0))

    assert input_addresses[0] == input_addresses[1]
    torch.testing.assert_close(first, torch.full((4, 2), 2.0))
    torch.testing.assert_close(second, torch.full((4, 2), 4.0))
