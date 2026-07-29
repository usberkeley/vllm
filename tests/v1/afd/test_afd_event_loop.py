# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the AFD event-loop ticks (design section 3.3).

Driven with a fake poll-source / send-sink / replay callback so the
poll->pick->replay->send wiring, bucket padding, and per-item routing are
verified without NCCL/NIXL or graph capture.
"""

import pytest
import torch

from vllm.distributed.afd_transfer.connector.base import (
    STAGE_A2F,
    STAGE_F2A,
    AFDMeta,
    AFDTransferIdAllocator,
)
from vllm.distributed.afd_transfer.scheduler import AFDDynamicBatchScheduler
from vllm.distributed.afd_transfer.worker_loop import (
    AFDAttentionEventLoop,
    AFDFfnEventLoop,
    AFDHandle,
)

HIDDEN = 8
CAPTURE_SIZES = [4, 8, 16]
Completion = tuple[torch.Tensor, AFDMeta, object | None]


class FakeSource:
    """Yields a preset batch of arrivals on the first poll, then nothing."""

    def __init__(self, handles):
        self._pending = list(handles)

    def poll(self):
        out, self._pending = self._pending, []
        return out


class ManualSource(FakeSource):
    def add(self, handle):
        self._pending.append(handle)


class FakeSink:
    def __init__(self):
        self.sent: list[tuple[torch.Tensor, AFDMeta]] = []

    def isend(self, hidden, meta):
        self.sent.append((hidden, meta))


def _replay(layer_id, hidden):
    # Distinct per-layer transform so routing/layer mixups are detectable.
    return hidden * 10.0 + layer_id


def _handle(layer, num_tokens, transfer_id=0):
    return AFDHandle(
        meta=AFDMeta(
            layer_id=layer,
            stage=STAGE_A2F,
            transfer_id=transfer_id,
        ),
        tensor=torch.randn(num_tokens, HIDDEN),
    )


def _loop(source, sink, num_layers=3):
    sched = AFDDynamicBatchScheduler(num_layers, age_limit_s=0.0)
    return AFDFfnEventLoop(sched, source, sink, _replay, CAPTURE_SIZES)


def test_idle_tick_sends_nothing():
    # No arrivals -> no ready layer -> no batch launched (empty-batch safety).
    sink = FakeSink()
    loop = _loop(FakeSource([]), sink)
    assert loop.tick(now=0.0) is False
    assert sink.sent == []


def test_tick_replays_ready_layer_and_returns_result():
    h = _handle(layer=1, num_tokens=3, transfer_id=17)
    sink = FakeSink()
    loop = _loop(FakeSource([h]), sink)

    assert loop.tick(now=0.0) is True
    assert len(sink.sent) == 1
    out, meta = sink.sent[0]
    assert meta.layer_id == 1
    assert meta.stage == STAGE_F2A  # result returns as the F2A (combine) transfer
    assert meta.transfer_id == 17
    # Padding is dropped: the result has exactly the real token count, and equals
    # the replay applied to the real tokens.
    assert out.shape == (3, HIDDEN)
    torch.testing.assert_close(out, h.tensor * 10.0 + 1)


def test_padding_is_applied_then_dropped():
    captured = {}

    def replay(layer_id, hidden):
        captured["padded_len"] = hidden.shape[0]
        return hidden

    h = _handle(layer=0, num_tokens=3)
    sched = AFDDynamicBatchScheduler(1, age_limit_s=0.0)
    sink = FakeSink()
    loop = AFDFfnEventLoop(sched, FakeSource([h]), sink, replay, CAPTURE_SIZES)

    loop.tick(now=0.0)
    assert captured["padded_len"] == 4  # 3 tokens padded up to smallest bucket
    assert sink.sent[0][0].shape == (3, HIDDEN)  # padding dropped on the way out


def test_multiple_arrivals_same_layer_batched_and_split_back():
    a = _handle(layer=2, num_tokens=2)
    b = _handle(layer=2, num_tokens=3)
    sink = FakeSink()
    loop = _loop(FakeSource([a, b]), sink)

    assert loop.tick(now=0.0) is True
    # One replay over the concatenated batch, split back per originating arrival.
    assert [t.shape[0] for t, _ in sink.sent] == [2, 3]
    torch.testing.assert_close(sink.sent[0][0], a.tensor * 10.0 + 2)
    torch.testing.assert_close(sink.sent[1][0], b.tensor * 10.0 + 2)


def test_poll_runs_any_non_empty_layer_without_waiting_to_fill():
    sched = AFDDynamicBatchScheduler(2, age_limit_s=10.0)
    sink = FakeSink()
    src = FakeSource([_handle(layer=0, num_tokens=5)])
    loop = AFDFfnEventLoop(sched, src, sink, _replay, CAPTURE_SIZES)

    assert loop.tick(now=0.0) is True
    assert sched.pending == 0 and len(sink.sent) == 1


def test_stage_f2a_constant_available():
    # Guard the return-stage constant the sink uses when addressing F->A.
    assert STAGE_F2A != STAGE_A2F


def _attention_loop(source, sink, completions, layer_ids, credit_capacity=1):
    sched = AFDDynamicBatchScheduler(max(layer_ids) + 1, age_limit_s=1.0)
    loop = AFDAttentionEventLoop(
        scheduler=sched,
        source=source,
        sink=sink,
        replay_fn=lambda layer, hidden, items: _replay(layer, hidden),
        completion_fn=lambda hidden, meta, context: completions.append(
            (hidden, meta, context)
        ),
        capture_sizes=CAPTURE_SIZES,
        layer_ids=layer_ids,
        transfer_ids=AFDTransferIdAllocator(),
        credit_capacity=credit_capacity,
    )
    return loop, sched


def test_attention_rejects_multiple_phase0_credits():
    with pytest.raises(ValueError, match="credit_capacity=1"):
        _attention_loop(FakeSource([]), FakeSink(), [], [0], credit_capacity=2)


def _f2a(sent):
    hidden, meta = sent
    return AFDHandle(
        meta=AFDMeta(meta.layer_id, STAGE_F2A, meta.transfer_id),
        tensor=hidden,
    )


def test_attention_loop_advances_across_layers():
    source = ManualSource([])
    sink = FakeSink()
    completions: list[Completion] = []
    loop, _ = _attention_loop(source, sink, completions, layer_ids=[0, 2])
    original = torch.randn(2, HIDDEN)

    context = {"request_id": "request-0"}
    loop.submit(original, now=0.0, context=context)
    assert loop.tick(now=0.0) is True
    first = sink.sent[-1]
    assert first[1].layer_id == 0
    assert first[1].stage == STAGE_A2F

    source.add(_f2a(first))
    assert loop.tick(now=1.0) is True
    second = sink.sent[-1]
    assert second[1].layer_id == 2
    assert second[1].transfer_id != first[1].transfer_id

    source.add(_f2a(second))
    assert loop.tick(now=2.0) is False
    assert len(completions) == 1
    assert completions[0][1].layer_id == 2
    assert completions[0][2] is context


def test_attention_blocks_until_f2a_return():
    source = ManualSource([])
    sink = FakeSink()
    completions: list[Completion] = []
    loop, sched = _attention_loop(
        source, sink, completions, layer_ids=[0], credit_capacity=1
    )
    loop.submit(torch.randn(2, HIDDEN), now=0.0)
    loop.submit(torch.randn(2, HIDDEN), now=0.0)

    assert loop.tick(now=0.0) is True
    assert loop.credits.available == 0
    assert sched.pending == 2
    assert loop.tick(now=0.5) is False

    first = sink.sent[0]
    source.add(_f2a(first))
    assert loop.tick(now=1.0) is True
    assert len(sink.sent) == 2
    assert sink.sent[1][1].transfer_id != first[1].transfer_id
    assert len(completions) == 1


def test_attention_sends_whole_submitted_batches_one_at_a_time():
    source = ManualSource([])
    sink = FakeSink()
    loop, sched = _attention_loop(source, sink, [], layer_ids=[0])
    for _ in range(3):
        loop.submit(torch.randn(2, HIDDEN), now=0.0)

    assert loop.tick(now=0.0) is True
    assert len(sink.sent) == 1
    assert sched.pending == 4


def test_attention_pads_final_continuation_to_capture_size():
    source = ManualSource([])
    sink = FakeSink()
    completed: list[Completion] = []
    loop, _ = _attention_loop(source, sink, completed, layer_ids=[0])
    original = torch.randn(3, HIDDEN)

    loop.submit(original, now=0.0)
    assert loop.tick(now=0.0) is True
    sent = sink.sent[-1]
    assert sent[0].shape == (3, HIDDEN)

    source.add(_f2a(sent))
    assert loop.tick(now=1.0) is False
    assert completed[0][0].shape == (4, HIDDEN)


def test_attention_rejects_f2a_with_wrong_token_count():
    source = ManualSource([])
    sink = FakeSink()
    loop, _ = _attention_loop(source, sink, [], layer_ids=[0])
    loop.submit(torch.randn(2, HIDDEN), now=0.0)
    assert loop.tick(now=0.0) is True

    _, sent_meta = sink.sent[-1]
    source.add(
        AFDHandle(
            meta=AFDMeta(sent_meta.layer_id, STAGE_F2A, sent_meta.transfer_id),
            tensor=torch.randn(3, HIDDEN),
        )
    )
    with pytest.raises(ValueError, match="returned 3 tokens, expected 2"):
        loop.tick(now=1.0)
    assert loop.credits.in_flight == 1


def test_attention_finishes_each_batch_before_starting_the_next():
    source = ManualSource([])
    sink = FakeSink()
    completions: list[Completion] = []
    loop, _ = _attention_loop(source, sink, completions, layer_ids=[0, 1])
    first = torch.randn(1, HIDDEN)
    second = torch.randn(1, HIDDEN)
    loop.submit(first, now=0.0, context="first")
    loop.submit(second, now=0.0, context="second")

    assert loop.tick(now=0.0) is True
    assert len(sink.sent) == 1
    sent, _ = sink.sent[0]
    assert sent.shape == (1, HIDDEN)

    assert loop.tick(now=0.5) is False
    assert len(sink.sent) == 1

    source.add(_f2a(sink.sent[0]))
    assert loop.tick(now=1.0) is True
    assert len(sink.sent) == 2
    assert sink.sent[1][1].layer_id == 1

    source.add(_f2a(sink.sent[1]))
    assert loop.tick(now=2.0) is True
    assert [context for _, _, context in completions] == ["first"]
    assert sink.sent[2][1].layer_id == 0
