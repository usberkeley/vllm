# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end AFD seam correctness through the production event-loop path."""

import time

import torch
from conftest import HIDDEN, make_vllm_config

from vllm.distributed.afd_transfer.connector import (
    STAGE_F2A,
    AFDConnectorRole,
    AFDTransferIdAllocator,
)
from vllm.distributed.afd_transfer.connector.loopback import LoopbackAFDConnector
from vllm.distributed.afd_transfer.runtime import (
    AFDAttentionContext,
    AFDSegmentedModel,
)
from vllm.distributed.afd_transfer.scheduler import AFDDynamicBatchScheduler
from vllm.distributed.afd_transfer.worker_loop import AFDAttentionEventLoop
from vllm.model_executor.models import afd as afd_mod
from vllm.model_executor.models.afd import (
    AFDAttnBoundary,
    AFDFfnBoundary,
    apply_afd_roles,
)


def _drive_attention_loop(model, connector, hidden, positions):
    segmented = AFDSegmentedModel(model, "attention")
    scheduler = AFDDynamicBatchScheduler(
        num_layers=max(segmented.layer_ids) + 1,
        age_limit_s=0.0,
    )
    completed = []

    def replay(layer_id, activation, items):
        context = items[0].context
        assert isinstance(context, AFDAttentionContext)
        return segmented.run_attention_segment(layer_id, activation, context)

    def finish(activation, _meta, context):
        assert isinstance(context, AFDAttentionContext)
        completed.append(segmented.finalize_attention(activation, context))

    loop = AFDAttentionEventLoop(
        scheduler=scheduler,
        source=connector,
        sink=connector,
        replay_fn=replay,
        completion_fn=finish,
        capture_sizes=None,
        layer_ids=segmented.layer_ids,
        transfer_ids=AFDTransferIdAllocator(),
        credit_capacity=1,
    )
    loop.submit(
        hidden,
        now=time.monotonic(),
        context=AFDAttentionContext(positions, None, None),
    )
    for _ in range(2 * len(segmented.layer_ids) + 1):
        loop.tick(time.monotonic())
        if completed:
            return completed[0]
    raise AssertionError("AFD loopback did not complete.")


def test_loopback_event_loop_bitwise_matches_monolithic(
    fake_model, model_inputs, monkeypatch
):
    positions, hidden = model_inputs
    expected = fake_model(positions, hidden.clone())
    real_mlps = {i: layer.mlp for i, layer in enumerate(fake_model.layers)}

    monkeypatch.setattr(afd_mod, "_is_moe_layer", lambda ffn: True)
    apply_afd_roles(fake_model, make_vllm_config("attention"))
    fake_model.embed_input_ids = lambda input_ids: input_ids

    connector = LoopbackAFDConnector(
        make_vllm_config("attention"), AFDConnectorRole.ATTENTION
    )
    for layer_id, mlp in real_mlps.items():
        connector.register_moe(layer_id, mlp)
    connector.start_listening(STAGE_F2A)

    actual = _drive_attention_loop(fake_model, connector, hidden.clone(), positions)

    assert torch.equal(expected, actual)


def test_attention_boundary_is_a_pure_cut():
    boundary = AFDAttnBoundary(layer_id=3)
    hidden = torch.randn(4, HIDDEN)

    assert boundary(hidden, torch.arange(4)) is hidden


def test_ffn_boundary_is_pure_compute_and_forwards_model_args(fake_model_v4):
    boundary = AFDFfnBoundary(fake_model_v4.layers[0].ffn, layer_id=0)
    hidden = torch.randn(4, HIDDEN)
    input_ids = torch.arange(4)

    expected = boundary.mlp(hidden, input_ids)
    actual = boundary(hidden, input_ids)

    torch.testing.assert_close(actual, expected)
