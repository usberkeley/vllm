# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the AFD cross-layer scheduler (design section 6.2)."""

import pytest
import torch

from vllm.distributed.afd_transfer.connector.base import STAGE_A2F, AFDMeta
from vllm.distributed.afd_transfer.scheduler import (
    AFDDynamicBatchScheduler,
    LayerQueue,
    QueueItem,
    drop_padding,
    pad_to_bucket,
    pick_bucket,
)

HIDDEN = 8


def _item(layer: int, num_tokens: int, ts: float) -> QueueItem:
    return QueueItem(
        hidden=torch.randn(num_tokens, HIDDEN),
        meta=AFDMeta(layer_id=layer, stage=STAGE_A2F),
        enqueue_ts=ts,
    )


def _sched(num_layers=4, age_limit_s=5.0):
    return AFDDynamicBatchScheduler(num_layers, age_limit_s)


# ---- LayerQueue ----


def test_queue_size_tracks_tokens_not_items():
    q = LayerQueue()
    q.push(_item(0, 3, 0.0))
    q.push(_item(0, 5, 0.0))
    assert q.size == 8
    assert len(q) == 2


def test_queue_head_wait_empty_is_zero():
    assert LayerQueue().head_wait(now=100.0) == 0.0


def test_queue_drain_preserves_whole_items():
    q = LayerQueue()
    q.push(_item(0, 10, 0.0))
    q.push(_item(0, 50, 0.0))
    batch = q.drain()
    assert [item.num_tokens for item in batch] == [10, 50]
    assert q.size == 0


def test_queue_drain_respects_item_credit_limit():
    q = LayerQueue()
    for _ in range(3):
        q.push(_item(0, 2, 0.0))
    batch = q.drain(max_items=2)
    assert len(batch) == 2
    assert q.size == 2


# ---- readiness ----


def test_non_empty_queue_is_ready_immediately():
    s = _sched()
    s.push(_item(0, 4, ts=0.0))
    assert s.pick_ready_layer(now=0.0) == 0


def test_empty_scheduler_picks_nothing():
    # Empty-batch safety: nothing ready -> None, so no (0, H) batch is launched.
    assert _sched().pick_ready_layer(now=1e9) is None


# ---- scoring: aging beats size ----


def test_aging_preempts_a_fuller_layer():
    s = _sched(age_limit_s=5.0)
    s.push(_item(0, 4, ts=0.0))  # small but old
    s.push(_item(3, 30, ts=99.5))  # large but fresh
    assert s.pick_ready_layer(now=100.0) == 0


def test_larger_batch_wins_when_neither_aged():
    s = _sched(age_limit_s=100.0)
    s.push(_item(1, 10, ts=0.0))
    s.push(_item(2, 20, ts=0.0))
    assert s.pick_ready_layer(now=0.0) == 2  # neither aged -> fuller layer 2


def test_tie_breaks_to_lowest_layer_id():
    s = _sched(age_limit_s=100.0)
    s.push(_item(3, 10, ts=0.0))
    s.push(_item(1, 10, ts=0.0))
    assert s.pick_ready_layer(now=0.0) == 1  # equal score -> lowest id


def test_drain_routes_to_the_selected_layer_only():
    s = _sched()
    s.push(_item(1, 8, ts=0.0))
    s.push(_item(2, 8, ts=0.0))
    batch = s.drain(1)
    assert all(i.meta.layer_id == 1 for i in batch)
    assert s.queues[2].size == 8  # other layer untouched


def test_drain_selected_layer_aggregates_whole_arrivals():
    s = _sched()
    s.push(_item(0, 8, ts=0.0))
    s.push(_item(0, 13, ts=0.0))
    batch = s.drain(0)
    assert [item.num_tokens for item in batch] == [8, 13]
    assert s.pending == 0


# ---- padding helpers ----


def test_pick_bucket_smallest_fit():
    assert pick_bucket(5, [1, 8, 16, 32]) == 8
    assert pick_bucket(8, [1, 8, 16]) == 8


def test_pick_bucket_rejects_overflow():
    with pytest.raises(ValueError, match="exceed"):
        pick_bucket(40, [1, 8, 16, 32])


def test_pad_then_drop_is_identity():
    x = torch.randn(5, HIDDEN)
    padded = pad_to_bucket(x, 8)
    assert padded.shape == (8, HIDDEN)
    assert torch.equal(padded[5:], torch.zeros(3, HIDDEN))
    assert torch.equal(drop_padding(padded, 5), x)


def test_pad_to_exact_bucket_is_noop():
    x = torch.randn(8, HIDDEN)
    assert pad_to_bucket(x, 8) is x


def test_pad_rejects_shrink():
    with pytest.raises(ValueError, match="cannot pad"):
        pad_to_bucket(torch.randn(9, HIDDEN), 8)
