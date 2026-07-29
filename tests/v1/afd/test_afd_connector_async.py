# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The async connector interface (design section 5.6).

Exercises the ``isend``/``irecv``/``poll`` primitives directly on the loopback
connector, and confirms a connector plugs straight into the FFN event loop as its
poll-source / send-sink (design section 6.3) -- no blocking send/recv path.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from vllm.config import AFDConfig
from vllm.distributed.afd_transfer.connector import (
    STAGE_A2F,
    STAGE_F2A,
    AFDConnectorRole,
    AFDHandle,
    AFDMeta,
    AFDTransferIdAllocator,
)
from vllm.distributed.afd_transfer.connector.loopback import LoopbackAFDConnector
from vllm.distributed.afd_transfer.connector.p2p import P2PAFDConnector

HIDDEN = 8


def _loopback():
    cfg = SimpleNamespace(
        afd_config=AFDConfig(role="attention", afd_connector="LoopbackAFDConnector")
    )
    return LoopbackAFDConnector(cfg, AFDConnectorRole.ATTENTION)


def test_isend_returns_handle_not_tensor():
    conn = _loopback()
    h = conn.isend(torch.randn(3, HIDDEN), AFDMeta(0, STAGE_A2F))
    assert isinstance(h, AFDHandle)
    assert h.meta.stage == STAGE_A2F


def test_inline_moe_round_trip_via_poll():
    # isend(A2F) runs the registered MoE inline; poll delivers the F2A result
    # exactly as the production Attention event loop expects.
    conn = _loopback()
    conn.register_moe(0, lambda x: x * 3.0)
    x = torch.randn(4, HIDDEN)

    conn.isend(x, AFDMeta(0, STAGE_A2F))
    done = conn.poll()

    assert [d.meta.key() for d in done] == [AFDMeta(0, STAGE_F2A).key()]
    torch.testing.assert_close(done[0].tensor, x * 3.0)


def test_poll_drains_only_once():
    conn = _loopback()
    conn.irecv(torch.randn(2, HIDDEN), AFDMeta(1, STAGE_A2F))
    assert len(conn.poll()) == 1
    assert conn.poll() == []  # nothing left pending after draining


def test_same_layer_concurrent_transfers_do_not_overwrite():
    conn = _loopback()
    first = torch.randn(2, HIDDEN)
    second = torch.randn(2, HIDDEN)
    conn.isend(first, AFDMeta(0, STAGE_A2F, transfer_id=10))
    conn.isend(second, AFDMeta(0, STAGE_A2F, transfer_id=11))

    one = conn.irecv(first, AFDMeta(0, STAGE_A2F, transfer_id=10))
    two = conn.irecv(second, AFDMeta(0, STAGE_A2F, transfer_id=11))
    assert torch.equal(one.tensor, first)
    assert torch.equal(two.tensor, second)


def test_transfer_id_allocator_is_monotonic():
    ids = AFDTransferIdAllocator(start=7)
    assert [ids.next(), ids.next(), ids.next()] == [7, 8, 9]


def test_p2p_tag_uses_transfer_id_and_stage():
    a2f = P2PAFDConnector._tag(AFDMeta(3, STAGE_A2F, transfer_id=9))
    f2a = P2PAFDConnector._tag(AFDMeta(3, STAGE_F2A, transfer_id=9))
    next_transfer = P2PAFDConnector._tag(AFDMeta(3, STAGE_A2F, transfer_id=10))
    assert len({a2f, f2a, next_transfer}) == 3


class _Work:
    def __init__(self, completed=False):
        self.completed = completed

    def is_completed(self):
        return self.completed


def _p2p(monkeypatch, peer_rank=1):
    receives = []
    sends = []

    def irecv(tensor, **kwargs):
        work = _Work()
        receives.append((tensor, kwargs, work))
        return work

    def isend(tensor, **kwargs):
        work = _Work(completed=True)
        sends.append((tensor, kwargs, work))
        return work

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(dist, "irecv", irecv)
    monkeypatch.setattr(dist, "isend", isend)
    cfg = SimpleNamespace(
        afd_config=AFDConfig(
            role="attention",
            afd_connector="P2PAFDConnector",
            afd_connector_extra_config={"peer_rank": peer_rank},
        ),
        device_config=SimpleNamespace(device=torch.device("cpu")),
        model_config=SimpleNamespace(
            dtype=torch.float32,
            get_hidden_size=lambda: HIDDEN,
        ),
    )
    return P2PAFDConnector(cfg, AFDConnectorRole.ATTENTION), receives, sends


def test_p2p_dynamic_listener_receives_header_then_payload(monkeypatch):
    conn, receives, _ = _p2p(monkeypatch)
    conn.start_listening(STAGE_F2A)
    header, _, header_work = receives[-1]
    header.copy_(torch.tensor([2, STAGE_F2A, 17, 3, HIDDEN]))
    header_work.completed = True

    assert conn.poll() == []
    payload, _, payload_work = receives[-1]
    assert payload.shape == (3, HIDDEN)
    payload.fill_(4)
    payload_work.completed = True

    [completed] = conn.poll()
    assert completed.meta == AFDMeta(2, STAGE_F2A, 17)
    torch.testing.assert_close(completed.tensor, torch.full((3, HIDDEN), 4.0))
    assert len(receives) == 3  # next control receive is already posted


def test_p2p_dynamic_isend_sends_shape_header(monkeypatch):
    conn, _, sends = _p2p(monkeypatch)
    conn.start_listening(STAGE_F2A)
    hidden = torch.randn(3, HIDDEN)
    conn.isend(hidden, AFDMeta(2, STAGE_A2F, 17))

    assert len(sends) == 2
    assert sends[0][0].tolist() == [2, STAGE_A2F, 17, 3, HIDDEN]
    torch.testing.assert_close(sends[1][0], hidden)


def test_p2p_rejects_invalid_peer_rank(monkeypatch):
    with pytest.raises(ValueError, match="outside WORLD"):
        _p2p(monkeypatch, peer_rank=2)


def test_connector_drives_the_event_loop():
    # A loopback connector is both a poll-source and a send-sink, so it plugs
    # into the FFN event loop directly. Feed one A2F arrival, tick, and the MoE
    # result is sent back as F2A.
    from vllm.distributed.afd_transfer.scheduler import AFDDynamicBatchScheduler
    from vllm.distributed.afd_transfer.worker_loop import AFDFfnEventLoop

    conn = _loopback()
    x = torch.randn(3, HIDDEN)
    # Stage an inbound A2F arrival for layer 0 (as if it came off the wire).
    conn.irecv(x, AFDMeta(0, STAGE_A2F))

    sched = AFDDynamicBatchScheduler(num_layers=1, age_limit_s=0.0)
    loop = AFDFfnEventLoop(
        sched, conn, conn, replay_fn=lambda layer, h: h + layer, capture_sizes=[4]
    )

    assert loop.tick(now=0.0) is True
    # The result was sent back; drain the staged F2A to confirm routing.
    sent = conn._staged[(0, STAGE_F2A, 0)]
    torch.testing.assert_close(sent, x + 0)
