# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace

import msgspec
import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlAgentMetadata,
    decode_staging_request_metadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    _NixlStagingReadBackend,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.staging import (
    STAGE_NOTIF_PREFIX,
    STAGING_PROTOCOL_VERSION,
    BackendSafetyCapabilities,
    BackendSafetyRegistry,
    ChunkLedger,
    ConsumerProgress,
    ConsumerSlotState,
    DefinitelyNotSubmittedError,
    DescriptorCache,
    FairReadyQueue,
    InMemoryModeStore,
    ModeCoordinator,
    ModeDecisionLedger,
    ProducerPipeline,
    ProducerProgress,
    ProducerSlotState,
    ReadCompletionOutbox,
    RemoteChunkState,
    StageCancel,
    StageModeAbort,
    StageModeCommit,
    StageModePrepared,
    StageModeQuery,
    StageReadComplete,
    StageReady,
    StageStatusQuery,
    StageStatusReply,
    StagingConfig,
    StagingCopyPlan,
    StagingNotificationHandler,
    StagingSessionRegistry,
    StagingSlotPool,
    StagingTransferIntent,
    StagingTransferSession,
    TransferCompletionTracker,
    committed_wire_chunk_bytes,
    decode_stage_message,
    encode_stage_message,
    gather_chunk,
    resolve_staging_config,
    scatter_chunk,
    validate_preflight,
)


def _ready(request_id: str, chunk_index: int, epoch: int = 1) -> StageReady:
    return StageReady(
        protocol_version=STAGING_PROTOCOL_VERSION,
        producer_generation="generation-1",
        transfer_id=f"transfer-{request_id}",
        request_id=request_id,
        chunk_index=chunk_index,
        source_slot_id=chunk_index,
        valid_bytes=64,
        plan_id=f"plan-{request_id}",
        consumer_generation="consumer-generation-1",
        source_slot_epoch=epoch,
        producer_engine_id="producer",
        producer_rank=0,
        consumer_engine_id="consumer",
        consumer_rank=0,
    )


def test_staging_config_resolves_fraction_and_validates_limits():
    config = StagingConfig.from_extra_config(
        {
            "staging_enabled": "true",
            "staging_buffer_fraction": 0.25,
            "staging_slot_bytes": 128,
            "staging_max_inflight": 2,
        }
    ).resolve_buffer_bytes(2048)
    assert config.buffer_bytes == 512
    assert config.slot_count == 4
    assert config.max_inflight == 2

    with pytest.raises(ValueError, match="staging_fallback"):
        StagingConfig.from_extra_config({"staging_fallback": "unsafe"})
    with pytest.raises(ValueError, match="must be at least"):
        StagingConfig.from_extra_config(
            {
                "staging_enabled": True,
                "staging_buffer_bytes": 64,
                "staging_slot_bytes": 128,
            }
        )
    with pytest.raises(ValueError, match="staging_slot_bytes"):
        StagingConfig.from_extra_config({"staging_slot_bytes": True})
    with pytest.raises(ValueError, match="staging_transfer_timeout"):
        StagingConfig.from_extra_config({"staging_transfer_timeout": float("nan")})


def test_staging_reservation_is_persisted_for_connector_allocation():
    extra_config = {
        "staging_enabled": True,
        "staging_buffer_fraction": 0.25,
        "staging_slot_bytes": 128,
    }
    config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector="NixlPullConnector",
            kv_connector_extra_config=extra_config,
        )
    )
    resolved = resolve_staging_config(config, 2048)
    assert resolved.buffer_bytes == 512
    assert extra_config["_staging_resolved_buffer_bytes"] == 512


def test_router_staging_metadata_round_trips_typed_values():
    intent = _intent()
    commit = StageModeCommit(
        intent.transfer_id,
        intent.mode_attempt,
        "staged",
        (("producer", 0, "consumer", 0, 64),),
    )
    intents, decoded_commit = decode_staging_request_metadata(
        {
            "staging_transfer_intents": [msgspec.to_builtins(intent)],
            "staging_mode_commit": msgspec.msgpack.encode(commit),
        }
    )
    assert intents == (intent,)
    assert decoded_commit == commit


def test_copy_plan_chunks_regions_and_preserves_tail():
    plan = StagingCopyPlan.build(
        plan_id="request-1",
        region_block_ids=((2, 4), (7,)),
        region_block_bytes=(6, 4),
        slot_bytes=8,
    )
    assert plan.total_bytes == 16
    assert [chunk.valid_bytes for chunk in plan.chunks] == [8, 8]
    assert plan.chunks[0].segments[0].block_id == 2
    assert plan.chunks[0].segments[1].block_id == 4
    assert plan.chunks[1].segments[-1].region_index == 1

    tail = StagingCopyPlan.build("tail", ((1,),), (10,), slot_bytes=6)
    assert [chunk.valid_bytes for chunk in tail.chunks] == [6, 4]


def test_copy_plan_gather_scatter_is_byte_exact():
    source = (torch.arange(24, dtype=torch.uint8).reshape(3, 8),)
    destination = (torch.zeros_like(source[0]),)
    plan = StagingCopyPlan.build("copy", ((0, 2),), (8,), slot_bytes=10)
    for chunk in plan.chunks:
        slot = torch.empty(10, dtype=torch.uint8)
        gather_chunk(source, slot, chunk)
        scatter_chunk(destination, slot, chunk)
    assert torch.equal(destination[0][0], source[0][0])
    assert torch.equal(destination[0][2], source[0][2])
    assert torch.count_nonzero(destination[0][1]) == 0


def test_slot_pool_checks_epochs_references_and_quarantine():
    producer = StagingSlotPool(slot_count=1, producer=True)
    slot = producer.acquire(("transfer", 0))
    assert slot is not None and slot.epoch == 1
    producer.transition(
        slot.slot_id, ProducerSlotState.PACKING, ProducerSlotState.READY_LOCAL
    )
    consumer = ("decode", 0, "generation")
    producer.expose(slot.slot_id, {consumer})
    assert not producer.complete_consumer(slot.slot_id, 0, consumer)
    assert producer.complete_consumer(slot.slot_id, 1, consumer)

    slot = producer.acquire(("other", 0))
    assert slot is not None and slot.epoch == 2
    producer.quarantine(slot.slot_id)
    assert producer.acquire(("next", 0)) is None
    with pytest.raises(RuntimeError, match="cannot be reused"):
        producer.release(slot.slot_id)

    inbox = StagingSlotPool(slot_count=1, producer=False)
    slot = inbox.acquire(("transfer", 0))
    assert slot is not None
    inbox.transition(
        slot.slot_id, ConsumerSlotState.READING, ConsumerSlotState.SCATTERING
    )
    inbox.release(slot.slot_id)
    assert inbox.acquire(("next", 0)) is not None


def test_ready_queue_is_fair_and_deduplicates():
    ready = FairReadyQueue()
    assert ready.push("p0", _ready("large", 0))
    assert ready.push("p0", _ready("large", 1))
    assert ready.push("p0", _ready("small", 0))
    assert ready.push("p1", _ready("peer", 0))
    assert not ready.push("p0", _ready("large", 0))
    order = [ready.pop(), ready.pop(), ready.pop(), ready.pop()]
    assert [(p, m.request_id) for p, m in order if p and m] == [
        ("p0", "large"),
        ("p1", "peer"),
        ("p0", "small"),
        ("p0", "large"),
    ]
    assert ready.pop() is None


def test_typed_protocol_round_trip_and_rejects_wrong_version():
    message = StageReadComplete(
        protocol_version=STAGING_PROTOCOL_VERSION,
        producer_generation="generation-1",
        transfer_id="transfer-1",
        chunk_index=3,
        source_slot_id=2,
        consumer_engine_id="decode-1",
        consumer_rank=0,
        consumer_generation="consumer-generation-1",
        source_slot_epoch=1,
        producer_engine_id="producer",
        producer_rank=0,
    )
    assert decode_stage_message(encode_stage_message(message)) == message
    invalid = msgspec.structs.replace(
        message, protocol_version=STAGING_PROTOCOL_VERSION + 1
    )
    with pytest.raises(ValueError, match="Unsupported"):
        decode_stage_message(STAGE_NOTIF_PREFIX + msgspec.msgpack.encode(invalid))


def test_read_completion_is_explicit_retryable_and_follows_done_and_release():
    ledger = ChunkLedger()
    ready = _ready("request", 0, epoch=1)
    ledger.observe_ready(ready)
    ledger.transition(ready, RemoteChunkState.QUEUED, RemoteChunkState.POSTING)
    ledger.transition(ready, RemoteChunkState.POSTING, RemoteChunkState.INFLIGHT)
    outbox = ReadCompletionOutbox(ledger)
    events: list[str] = []

    def release_handle() -> None:
        assert ledger.state(ready) == RemoteChunkState.DONE
        events.append("release")

    assert outbox.observe_done(ready, release_handle)
    assert events == ["release"]

    attempts = 0

    def send_notification(payload: bytes) -> None:
        nonlocal attempts
        attempts += 1
        completion = decode_stage_message(payload)
        assert isinstance(completion, StageReadComplete)
        assert completion.transfer_id == ready.transfer_id
        events.append("send")
        if attempts == 1:
            raise RuntimeError("delivery result unknown")

    with pytest.raises(RuntimeError, match="delivery result unknown"):
        outbox.send_next(send_notification)
    assert len(outbox) == 1
    assert outbox.send_next(send_notification)
    assert len(outbox) == 0
    assert events == ["release", "send", "send"]
    assert not outbox.observe_done(ready, release_handle)
    assert events == ["release", "send", "send"]


def test_read_completion_waits_for_successful_handle_release():
    ledger = ChunkLedger()
    ready = _ready("request", 0, epoch=1)
    ledger.observe_ready(ready)
    ledger.transition(ready, RemoteChunkState.QUEUED, RemoteChunkState.POSTING)
    ledger.transition(ready, RemoteChunkState.POSTING, RemoteChunkState.INFLIGHT)
    outbox = ReadCompletionOutbox(ledger)
    release_attempts = 0

    def release_handle() -> None:
        nonlocal release_attempts
        release_attempts += 1
        if release_attempts == 1:
            raise RuntimeError("release failed")

    with pytest.raises(RuntimeError, match="release failed"):
        outbox.observe_done(ready, release_handle)
    assert ledger.state(ready) == RemoteChunkState.DONE
    assert not outbox.send_next(lambda _: pytest.fail("sent before release"))

    assert outbox.observe_done(ready, release_handle)
    assert outbox.send_next(lambda _: None)
    assert release_attempts == 2


def test_chunk_ledger_enforces_posting_boundary_and_watermark():
    ledger = ChunkLedger()
    first = _ready("request", 0, epoch=1)
    assert ledger.observe_ready(first) == RemoteChunkState.QUEUED
    assert ledger.observe_ready(first) == RemoteChunkState.QUEUED
    ledger.transition(first, RemoteChunkState.QUEUED, RemoteChunkState.POSTING)
    ledger.transition(first, RemoteChunkState.POSTING, RemoteChunkState.UNKNOWN)
    with pytest.raises(RuntimeError, match="safe terminal"):
        ledger.observe_ready(_ready("request", 0, epoch=2))
    ledger.transition(first, RemoteChunkState.UNKNOWN, RemoteChunkState.DONE)
    second = _ready("request", 0, epoch=2)
    assert ledger.observe_ready(second) == RemoteChunkState.QUEUED
    query = StageStatusQuery(
        protocol_version=STAGING_PROTOCOL_VERSION,
        producer_generation="generation-1",
        consumer_generation=first.consumer_generation,
        transfer_id=first.transfer_id,
        chunk_index=0,
        source_slot_id=0,
        source_slot_epoch=1,
        producer_engine_id="producer",
        producer_rank=0,
        consumer_engine_id="consumer",
        consumer_rank=0,
    )
    assert ledger.status(query) == "safe_retired"


def test_mode_decisions_are_idempotent_and_conflicts_fail_closed():
    ledger = ModeDecisionLedger()
    commit = StageModeCommit("transfer", 1, "staged", (("p", 0, "d", 0, 64),))
    assert ledger.accept(commit)
    assert not ledger.accept(commit)
    with pytest.raises(RuntimeError, match="Conflicting"):
        ledger.accept(StageModeAbort("transfer", 1, "timeout"))
    with pytest.raises(ValueError, match="edges"):
        ModeDecisionLedger().accept(StageModeCommit("transfer", 1, "staged", ()))


def test_agent_metadata_defaults_to_no_staging_for_legacy_payload():
    payload = msgspec.msgpack.encode(
        {
            "engine_id": "producer",
            "agent_metadata": b"agent",
            "kv_caches_base_addr": [1234],
            "device_id": 0,
            "num_blocks": 8,
            "block_lens": [128],
            "kv_cache_layout": "HND",
            "block_size": 16,
            "ssm_sizes": (0, 0),
            "attn_backend_name": "FLASH_ATTN",
            "physical_blocks_per_logical_kv_block": 1,
        }
    )
    metadata = msgspec.msgpack.decode(payload, type=NixlAgentMetadata)
    assert metadata.staging_generation is None
    assert metadata.staging_pool_bytes == 0
    assert metadata.staging_protocol_version == 0
    assert not metadata.supports_staging

    capable = replace(
        metadata,
        staging_protocol_version=STAGING_PROTOCOL_VERSION,
        staging_generation="generation-1",
        staging_pool_base_addr=4096,
        staging_pool_bytes=128,
        staging_slot_bytes=64,
        staging_slot_count=2,
    )
    assert capable.supports_staging


def test_nixl_staging_read_uses_one_descriptor_and_no_completion_notif():
    class Wrapper:
        def __init__(self):
            self.next_handle = 1
            self.make_args = None
            self.transferred = []
            self.released = []

        def get_xfer_descs(self, data, memory_type):
            assert memory_type == "VRAM"
            assert data.shape == (1, 3)
            return data

        def prep_xfer_dlist(self, agent, descs):
            handle = self.next_handle
            self.next_handle += 1
            return handle

        def release_dlist_handle(self, handle):
            self.released.append(("dlist", handle))

        def make_prepped_xfer(self, *args, **kwargs):
            self.make_args = (args, kwargs)
            return 99

        def transfer(self, handle):
            self.transferred.append(handle)

        def check_xfer_state(self, handle):
            return "DONE"

        def release_xfer_handle(self, handle):
            self.released.append(("xfer", handle))

    wrapper = Wrapper()
    pool = StagingSlotPool(2, 128, device="cpu", producer=False)
    remote = NixlAgentMetadata(
        engine_id="producer",
        agent_metadata=b"agent",
        kv_caches_base_addr=[1],
        device_id=0,
        num_blocks=1,
        block_lens=[128],
        kv_cache_layout="HND",
        block_size=16,
        ssm_sizes=(0, 0),
        attn_backend_name="FLASH_ATTN",
        physical_blocks_per_logical_kv_block=1,
        staging_protocol_version=STAGING_PROTOCOL_VERSION,
        staging_generation="generation-1",
        staging_pool_base_addr=4096,
        staging_pool_bytes=256,
        staging_slot_bytes=128,
        staging_slot_count=2,
    )
    worker = SimpleNamespace(
        nixl_wrapper=wrapper,
        staging_config=StagingConfig(enabled=True, buffer_bytes=256, slot_bytes=128),
        staging_pool=pool,
        device_id=0,
        nixl_memory_type="VRAM",
        _staging_remote_metadata={"producer": {0: remote}},
        _remote_agents={"producer": {(0, 0): "producer-agent"}},
    )
    backend = _NixlStagingReadBackend(worker)
    handle = backend.post_read(_ready("request", 0), 1)
    assert handle == 99
    assert wrapper.transferred == [99]
    assert wrapper.make_args is not None
    assert wrapper.make_args[1]["notif_msg"] is None
    backend.release_read(handle)
    backend.close()


def test_protocol_rejects_incomplete_identity_and_invalid_lengths():
    missing_generation = msgspec.structs.replace(
        _ready("request", 0), consumer_generation=""
    )
    with pytest.raises(ValueError, match="consumer_generation"):
        encode_stage_message(missing_generation)

    invalid_length = msgspec.structs.replace(_ready("request", 0), valid_bytes=0)
    with pytest.raises(ValueError, match="valid_bytes"):
        encode_stage_message(invalid_length)


def test_chunk_ledger_rejects_conflicting_identity_at_same_epoch():
    ledger = ChunkLedger()
    ready = _ready("request", 0)
    ledger.observe_ready(ready)
    conflicting = msgspec.structs.replace(ready, transfer_id="other-transfer")
    with pytest.raises(RuntimeError, match="Conflicting READY"):
        ledger.observe_ready(conflicting)

    query = StageStatusQuery(
        protocol_version=STAGING_PROTOCOL_VERSION,
        producer_generation=ready.producer_generation,
        consumer_generation=ready.consumer_generation,
        transfer_id="other-transfer",
        chunk_index=ready.chunk_index,
        source_slot_id=ready.source_slot_id,
        source_slot_epoch=ready.source_slot_epoch,
        producer_engine_id=ready.producer_engine_id,
        producer_rank=ready.producer_rank,
        consumer_engine_id=ready.consumer_engine_id,
        consumer_rank=ready.consumer_rank,
    )
    assert ledger.status(query) == "unknown"


def test_producer_exposes_before_send_and_ignores_stale_proof():
    pool = StagingSlotPool(slot_count=1, slot_bytes=64, producer=True)
    slot = pool.acquire(("transfer-request", 0))
    assert slot is not None
    pool.transition(0, ProducerSlotState.PACKING, ProducerSlotState.READY_LOCAL)
    ready = msgspec.structs.replace(_ready("request", 0), source_slot_id=0)
    progress = ProducerProgress(pool, "producer", 0, "generation-1", 0.5)
    progress.add_ready((ready,))

    with pytest.raises(RuntimeError, match="uncertain"):
        progress.send_ready(
            lambda _message, _payload: (_ for _ in ()).throw(
                RuntimeError("delivery uncertain")
            ),
            now=1.0,
        )
    assert pool.slots[0].state == ProducerSlotState.EXPOSED

    stale = StageReadComplete(
        protocol_version=STAGING_PROTOCOL_VERSION,
        producer_generation=ready.producer_generation,
        consumer_generation=ready.consumer_generation,
        transfer_id=ready.transfer_id,
        chunk_index=ready.chunk_index,
        source_slot_id=ready.source_slot_id,
        source_slot_epoch=ready.source_slot_epoch + 1,
        producer_engine_id=ready.producer_engine_id,
        producer_rank=ready.producer_rank,
        consumer_engine_id=ready.consumer_engine_id,
        consumer_rank=ready.consumer_rank,
    )
    assert not progress.accept_read_complete(stale)
    assert progress.accept_read_complete(
        msgspec.structs.replace(stale, source_slot_epoch=ready.source_slot_epoch)
    )
    assert pool.slots[0].state == ProducerSlotState.FREE


class _ReadBackend:
    def __init__(self) -> None:
        self.released = False

    def post_read(self, ready: StageReady, local_slot_id: int) -> str:
        assert ready.valid_bytes <= 64
        assert local_slot_id == 0
        return "handle"

    def check_read(self, handle: str) -> str:
        assert handle == "handle"
        return "DONE"

    def release_read(self, handle: str) -> None:
        assert handle == "handle"
        self.released = True


def test_consumer_releases_remote_before_local_scatter_finishes():
    config = StagingConfig(
        enabled=True,
        buffer_bytes=64,
        slot_bytes=64,
        max_inflight=1,
        max_inflight_per_peer=1,
        max_ready_per_request=1,
    )
    pool = StagingSlotPool(slot_count=1, slot_bytes=64, producer=False)
    progress = ConsumerProgress(pool, config, "consumer", 0, "consumer-generation-1")
    ready = msgspec.structs.replace(_ready("request", 0), source_slot_id=0)
    backend = _ReadBackend()
    scatter_done = False

    assert progress.receive_ready(ready)
    assert progress.post_available(backend) == 1
    assert progress.poll_reads(backend, lambda _ready, _slot: lambda: scatter_done) == 1
    assert backend.released
    assert pool.slots[0].state == ConsumerSlotState.SCATTERING

    sent: list[StageReadComplete] = []
    assert progress.completions.send_next(
        lambda payload: sent.append(decode_stage_message(payload))  # type: ignore[arg-type]
    )
    assert len(sent) == 1
    assert progress.poll_scatters() == ()
    scatter_done = True
    assert progress.poll_scatters() == (ready,)
    assert pool.slots[0].state == ConsumerSlotState.FREE


def test_status_safe_complete_can_release_producer_slot():
    pool = StagingSlotPool(slot_count=1, slot_bytes=64, producer=True)
    slot = pool.acquire(("transfer-request", 0))
    assert slot is not None
    pool.transition(0, ProducerSlotState.PACKING, ProducerSlotState.READY_LOCAL)
    ready = msgspec.structs.replace(_ready("request", 0), source_slot_id=0)
    progress = ProducerProgress(pool, "producer", 0, "generation-1", 0.5)
    progress.add_ready((ready,))
    assert progress.send_ready(lambda _ready, _payload: None) == 1
    reply = StageStatusReply(
        protocol_version=STAGING_PROTOCOL_VERSION,
        producer_generation=ready.producer_generation,
        consumer_generation=ready.consumer_generation,
        transfer_id=ready.transfer_id,
        chunk_index=ready.chunk_index,
        source_slot_id=ready.source_slot_id,
        source_slot_epoch=ready.source_slot_epoch,
        status="safe_complete",
        producer_engine_id=ready.producer_engine_id,
        producer_rank=ready.producer_rank,
        consumer_engine_id=ready.consumer_engine_id,
        consumer_rank=ready.consumer_rank,
    )
    assert progress.accept_status_reply(reply)


def test_backend_safety_matrix_and_descriptor_lru_fail_closed():
    registry = BackendSafetyRegistry()
    key = ("0.9.0", "UCX", "VRAM")
    with pytest.raises(RuntimeError, match="No staging safety evidence"):
        registry.require_staging(key)
    unsafe = BackendSafetyCapabilities(True, True, True, False, False)
    registry.register(key, unsafe)
    with pytest.raises(RuntimeError, match="does not safely support"):
        registry.require_staging(key)

    released: list[int] = []
    cache = DescriptorCache[int, int](1, released.append)
    assert cache.get_or_create(1, lambda: 10) == 10
    assert cache.get_or_create(1, lambda: 11) == 10
    assert cache.get_or_create(2, lambda: 20) == 20
    assert released == [10]
    assert (cache.hits, cache.misses) == (1, 2)
    cache.clear()
    assert released == [10, 20]


def test_scatter_can_skip_locally_cached_wire_blocks():
    source = (torch.arange(24, dtype=torch.uint8).reshape(3, 8),)
    destination = (torch.zeros_like(source[0]),)
    plan = StagingCopyPlan.build(
        "prefix",
        ((0, 1, 2),),
        (8,),
        24,
        scatter_block_ids=(frozenset({2}),),
    )
    slot = torch.empty(24, dtype=torch.uint8)
    gather_chunk(source, slot, plan.chunks[0])
    scatter_chunk(destination, slot, plan.chunks[0])
    assert torch.count_nonzero(destination[0][:2]) == 0
    assert torch.equal(destination[0][2], source[0][2])


def _intent() -> StagingTransferIntent:
    return StagingTransferIntent(
        protocol_version=STAGING_PROTOCOL_VERSION,
        producer_generation="generation-1",
        consumer_generation="consumer-generation-1",
        transfer_id="transfer-request",
        producer_request_id="producer-request",
        consumer_request_id="consumer-request",
        producer_engine_id="producer",
        producer_rank=0,
        producer_tp_size=1,
        consumer_engine_id="consumer",
        consumer_rank=0,
        consumer_tp_size=1,
        producer_host="producer-host",
        producer_port=1234,
        consumer_host="consumer-host",
        consumer_port=5678,
        plan_id="plan-request",
        mode_attempt=1,
    )


def test_mode_coordinator_freezes_minimum_slot_and_replays_decision():
    store = InMemoryModeStore()
    coordinator = ModeCoordinator(store)
    coordinator.record_prepared(
        StageModePrepared(
            "transfer-request",
            1,
            "producer",
            0,
            "generation-1",
            "consumer",
            0,
            128,
            True,
        )
    )
    coordinator.record_prepared(
        StageModePrepared(
            "transfer-request",
            1,
            "consumer",
            0,
            "consumer-generation-1",
            "producer",
            0,
            64,
            True,
        )
    )
    commit = coordinator.decide(
        "transfer-request", 1, (("producer", 0, "consumer", 0),), "direct"
    )
    assert commit.mode == "staged"
    assert committed_wire_chunk_bytes(_intent(), commit) == 64
    assert coordinator.query(StageModeQuery("transfer-request", 1)) == commit
    assert (
        coordinator.decide(
            "transfer-request", 1, (("producer", 0, "consumer", 0),), "fail"
        )
        == commit
    )
    with pytest.raises(RuntimeError, match="durable decision"):
        coordinator.abort(StageModeAbort("transfer-request", 1, "late"))


def test_preflight_rejects_pp_and_plan_supports_topology_slices():
    intent = _intent()
    config = StagingConfig(enabled=True, buffer_bytes=128, slot_bytes=64)
    prepared = validate_preflight(
        intent,
        config,
        engine_id="producer",
        rank=0,
        generation="generation-1",
        pipeline_parallel_size=2,
    )
    assert not prepared.supported
    assert "pipeline" in prepared.reason

    plan = StagingCopyPlan.from_wire_segments(
        "heterogeneous-tp",
        (
            (0, 2, 4, 6, True),
            (1, 3, 1, 5, False),
        ),
        slot_bytes=8,
    )
    assert [chunk.valid_bytes for chunk in plan.chunks] == [8, 3]
    assert plan.total_bytes == 11


class _PostFailureBackend(_ReadBackend):
    def __init__(self, definitely_not_submitted: bool) -> None:
        super().__init__()
        self.definitely_not_submitted = definitely_not_submitted

    def post_read(self, ready: StageReady, local_slot_id: int) -> str:
        if self.definitely_not_submitted:
            raise DefinitelyNotSubmittedError
        raise RuntimeError("submission result unknown")


@pytest.mark.parametrize("definitely_not_submitted", [True, False])
def test_post_failure_only_reuses_slot_with_backend_proof(
    definitely_not_submitted: bool,
):
    config = StagingConfig(
        enabled=True,
        buffer_bytes=64,
        slot_bytes=64,
        quarantine_max_bytes=64,
    )
    pool = StagingSlotPool(1, 64, producer=False)
    progress = ConsumerProgress(pool, config, "consumer", 0, "consumer-generation-1")
    ready = msgspec.structs.replace(_ready("request", 0), source_slot_id=0)
    assert progress.receive_ready(ready)
    assert progress.post_available(_PostFailureBackend(definitely_not_submitted)) == 0
    if definitely_not_submitted:
        assert pool.slots[0].state == ConsumerSlotState.FREE
        query = progress.cancel_transfer(ready.transfer_id)
        assert query == ()
    else:
        assert pool.slots[0].state == ConsumerSlotState.QUARANTINED
        assert "producer" in progress.unavailable_peers
        barriers: list[str] = []
        progress.cleanup_peer_generation(
            "producer", "generation-1", lambda: barriers.append("safe")
        )
        assert barriers == ["safe"]
        assert pool.slots[0].state == ConsumerSlotState.FREE


def _committed_session(total_bytes: int = 64) -> StagingTransferSession:
    session = StagingTransferSession(_intent())
    session.accept_decision(
        StageModeCommit(
            "transfer-request", 1, "staged", (("producer", 0, "consumer", 0, 64),)
        )
    )
    session.freeze_plan(
        StagingCopyPlan.build("plan-request", ((0,),), (total_bytes,), slot_bytes=64)
    )
    return session


def _session_ready(chunk_index: int, epoch: int = 1) -> StageReady:
    return StageReady(
        protocol_version=STAGING_PROTOCOL_VERSION,
        producer_generation="generation-1",
        consumer_generation="consumer-generation-1",
        transfer_id="transfer-request",
        request_id="producer-request",
        chunk_index=chunk_index,
        source_slot_id=chunk_index,
        source_slot_epoch=epoch,
        valid_bytes=64,
        plan_id="plan-request",
        producer_engine_id="producer",
        producer_rank=0,
        consumer_engine_id="consumer",
        consumer_rank=0,
    )


def test_session_requires_commit_and_freezes_ready_geometry():
    session = StagingTransferSession(_intent())
    plan = StagingCopyPlan.build("plan-request", ((0,),), (64,), 64)
    with pytest.raises(RuntimeError, match="MODE_COMMIT"):
        session.freeze_plan(plan)

    assert session.accept_decision(
        StageModeCommit(
            "transfer-request", 1, "staged", (("producer", 0, "consumer", 0, 64),)
        )
    )
    assert session.freeze_plan(plan)
    assert not session.freeze_plan(plan)
    assert session.validate_ready(_session_ready(0)) == plan.chunks[0]

    stale = msgspec.structs.replace(
        _session_ready(0), consumer_generation="old-generation"
    )
    with pytest.raises(RuntimeError, match="intent"):
        session.validate_ready(stale)


def test_session_registry_supports_edges_but_prevents_edge_rebinding():
    registry = StagingSessionRegistry()
    assert registry.register(_intent()) is registry.register(_intent())
    second_edge = msgspec.structs.replace(
        _intent(), consumer_engine_id="other-consumer"
    )
    assert registry.register(second_edge).intent == second_edge
    rebound = msgspec.structs.replace(_intent(), consumer_request_id="other-request")
    with pytest.raises(RuntimeError, match="rebound"):
        registry.register(rebound)
    registry.retire("transfer-request")
    with pytest.raises(RuntimeError, match="terminal"):
        registry.register(_intent())


def test_transfer_completion_waits_for_every_scatter_and_is_idempotent():
    tracker = TransferCompletionTracker()
    tracker.register("transfer-request", 2)
    assert not tracker.observe_scatter(_session_ready(1))
    assert tracker.observe_scatter(_session_ready(0))
    assert not tracker.observe_scatter(_session_ready(0))

    failed = TransferCompletionTracker()
    failed.register("transfer-request", 1)
    failed.fail("transfer-request")
    assert not failed.observe_scatter(_session_ready(0))


def test_registered_consumer_rejects_uncommitted_or_stale_ready():
    registry = StagingSessionRegistry()
    session = registry.register(_intent())
    config = StagingConfig(enabled=True, buffer_bytes=64, slot_bytes=64)
    progress = ConsumerProgress(
        StagingSlotPool(1, 64, producer=False),
        config,
        "consumer",
        0,
        "consumer-generation-1",
        sessions=registry,
    )
    with pytest.raises(RuntimeError, match="frozen"):
        progress.receive_ready(_session_ready(0))

    committed = _committed_session()
    decision = committed.decision
    assert isinstance(decision, StageModeCommit)
    session.accept_decision(decision)
    assert committed.plan is not None
    session.freeze_plan(committed.plan)
    assert progress.receive_ready(_session_ready(0))
    with pytest.raises(RuntimeError, match="intent"):
        progress.receive_ready(
            msgspec.structs.replace(
                _session_ready(0), producer_generation="stale-generation"
            )
        )


def test_producer_status_queries_only_after_diagnostic_timeout():
    pool = StagingSlotPool(1, 64, producer=True)
    slot = pool.acquire(("transfer-request", 0))
    assert slot is not None
    pool.transition(0, ProducerSlotState.PACKING, ProducerSlotState.READY_LOCAL)
    ready = msgspec.structs.replace(_ready("request", 0), source_slot_id=0)
    progress = ProducerProgress(pool, "producer", 0, "generation-1", 0.5)
    progress.add_ready((ready,))
    progress.send_ready(lambda _ready, _payload: None, now=10.0)
    assert progress.status_queries(now=39.9, timeout=30.0) == ()
    assert len(progress.status_queries(now=40.0, timeout=30.0)) == 1


def test_scatter_start_failure_quarantines_local_slot_but_releases_source():
    config = StagingConfig(enabled=True, buffer_bytes=64, slot_bytes=64)
    pool = StagingSlotPool(1, 64, producer=False)
    progress = ConsumerProgress(pool, config, "consumer", 0, "consumer-generation-1")
    ready = msgspec.structs.replace(_ready("request", 0), source_slot_id=0)
    backend = _ReadBackend()
    assert progress.receive_ready(ready)
    assert progress.post_available(backend) == 1

    def fail_scatter(_ready: StageReady, _slot: int):
        raise RuntimeError("scatter submission unknown")

    assert progress.poll_reads(backend, fail_scatter) == 0
    assert pool.slots[0].state == ConsumerSlotState.QUARANTINED
    assert ready.transfer_id in progress.failed_transfers
    sent: list[bytes] = []
    assert progress.completions.send_next(sent.append)
    assert isinstance(decode_stage_message(sent[0]), StageReadComplete)


def _session_for(transfer_id: str) -> StagingTransferSession:
    intent = msgspec.structs.replace(
        _intent(),
        transfer_id=transfer_id,
        producer_request_id=f"producer-{transfer_id}",
        consumer_request_id=f"consumer-{transfer_id}",
        plan_id=f"plan-{transfer_id}",
    )
    session = StagingTransferSession(intent)
    session.accept_decision(
        StageModeCommit(
            transfer_id,
            1,
            "staged",
            (("producer", 0, "consumer", 0, 64),),
        )
    )
    session.freeze_plan(
        StagingCopyPlan.build(intent.plan_id, ((0,),), (64,), slot_bytes=64)
    )
    return session


def test_producer_pipeline_schedules_requests_fairly_and_waits_for_gather():
    config = StagingConfig(
        enabled=True,
        buffer_bytes=128,
        slot_bytes=64,
        max_ready_per_request=1,
    )
    pool = StagingSlotPool(2, 64, producer=True)
    pipeline = ProducerPipeline(pool, config, "producer", 0, "generation-1")
    pipeline.register_transfer(_session_for("transfer-a"))
    pipeline.register_transfer(_session_for("transfer-b"))
    gather_done = {"transfer-a": False, "transfer-b": False}
    started: list[str] = []

    def start_gather(session, _chunk, _slot):
        transfer_id = session.intent.transfer_id
        started.append(transfer_id)
        return lambda: gather_done[transfer_id]

    assert pipeline.start_available(start_gather) == 2
    assert started == ["transfer-a", "transfer-b"]
    assert pipeline.poll_gathers() == ()
    gather_done["transfer-b"] = True
    ready = pipeline.poll_gathers()
    assert [message.transfer_id for message in ready] == ["transfer-b"]
    assert pipeline.pop_source_stable() == {"transfer-b"}


def test_producer_pipeline_keeps_same_transfer_edges_independent():
    config = StagingConfig(enabled=True, buffer_bytes=128, slot_bytes=64)
    pool = StagingSlotPool(2, 64, producer=True)
    pipeline = ProducerPipeline(pool, config, "producer", 0, "generation-1")
    first = _session_for("transfer-shared")
    second_intent = msgspec.structs.replace(
        first.intent,
        consumer_engine_id="consumer-2",
        consumer_rank=1,
        consumer_generation="consumer-generation-2",
    )
    second = StagingTransferSession(second_intent)
    second.accept_decision(
        StageModeCommit(
            "transfer-shared",
            1,
            "staged",
            (("producer", 0, "consumer-2", 1, 64),),
        )
    )
    second.freeze_plan(
        StagingCopyPlan.build(second_intent.plan_id, ((0,),), (64,), slot_bytes=64)
    )
    pipeline.register_transfer(first)
    pipeline.register_transfer(second)

    def complete(ready: StageReady) -> StageReadComplete:
        values = msgspec.to_builtins(ready)
        del values["request_id"]
        del values["valid_bytes"]
        del values["plan_id"]
        del values["type"]
        return msgspec.convert(values, type=StageReadComplete)

    assert pipeline.start_available(lambda *_args: lambda: True) == 1
    first_ready = pipeline.poll_gathers()[0]
    assert pipeline.pop_source_stable() == set()
    pipeline.send_ready(lambda _ready, _payload: None)
    assert pipeline.progress.accept_read_complete(complete(first_ready))
    assert pipeline.pop_completed_transfers() == set()

    assert pipeline.start_available(lambda *_args: lambda: True) == 1
    second_ready = pipeline.poll_gathers()[0]
    assert pipeline.pop_source_stable() == {"transfer-shared"}
    pipeline.send_ready(lambda _ready, _payload: None)
    assert pipeline.progress.accept_read_complete(complete(second_ready))
    assert pipeline.pop_completed_transfers() == {"transfer-shared"}


def test_producer_pipeline_cancel_drains_submitted_gather_without_ready():
    config = StagingConfig(enabled=True, buffer_bytes=64, slot_bytes=64)
    pool = StagingSlotPool(1, 64, producer=True)
    pipeline = ProducerPipeline(pool, config, "producer", 0, "generation-1")
    pipeline.register_transfer(_session_for("transfer-a"))
    gather_done = False

    def start_gather(_session, _chunk, _slot):
        return lambda: gather_done

    assert pipeline.start_available(start_gather) == 1
    assert pipeline.cancel_transfer("transfer-a") == ()
    assert pipeline.poll_gathers() == ()
    gather_done = True
    assert pipeline.poll_gathers() == ()
    assert pool.slots[0].state == ProducerSlotState.FREE
    assert pipeline.pop_source_stable() == {"transfer-a"}


def test_producer_source_slot_waits_for_every_consumer_proof():
    pool = StagingSlotPool(1, 64, producer=True)
    slot = pool.acquire(("transfer-request", 0))
    assert slot is not None
    pool.transition(0, ProducerSlotState.PACKING, ProducerSlotState.READY_LOCAL)
    first = msgspec.structs.replace(_ready("request", 0), source_slot_id=0)
    second = msgspec.structs.replace(
        first,
        consumer_engine_id="consumer-2",
        consumer_rank=1,
        consumer_generation="consumer-generation-2",
    )
    progress = ProducerProgress(pool, "producer", 0, "generation-1", 0.5)
    progress.add_ready((first, second))
    assert progress.send_ready(lambda _ready, _payload: None) == 2

    first_done = StageReadComplete(
        first.protocol_version,
        first.producer_generation,
        first.consumer_generation,
        first.transfer_id,
        first.chunk_index,
        first.source_slot_id,
        first.source_slot_epoch,
        first.producer_engine_id,
        first.producer_rank,
        first.consumer_engine_id,
        first.consumer_rank,
    )
    second_done = msgspec.structs.replace(
        first_done,
        consumer_engine_id=second.consumer_engine_id,
        consumer_rank=second.consumer_rank,
        consumer_generation=second.consumer_generation,
    )
    assert not progress.accept_read_complete(first_done)
    assert pool.slots[0].state == ProducerSlotState.EXPOSED
    assert progress.accept_read_complete(second_done)
    assert pool.slots[0].state == ProducerSlotState.FREE


def test_notification_handler_routes_cancel_and_drops_invalid_payloads():
    config = StagingConfig(enabled=True, buffer_bytes=64, slot_bytes=64)
    consumer = ConsumerProgress(
        StagingSlotPool(1, 64, producer=False),
        config,
        "consumer",
        0,
        "consumer-generation-1",
    )
    handler = StagingNotificationHandler(consumer=consumer)
    ready = msgspec.structs.replace(_ready("request", 0), source_slot_id=0)
    assert handler.receive(encode_stage_message(ready)) == ()

    cancel = StageCancel(
        protocol_version=STAGING_PROTOCOL_VERSION,
        producer_generation=ready.producer_generation,
        consumer_generation=ready.consumer_generation,
        transfer_id=ready.transfer_id,
        request_id=ready.request_id,
        producer_request_id="producer-request",
        consumer_request_id="consumer-request",
        consumer_engine_id=ready.consumer_engine_id,
        consumer_rank=ready.consumer_rank,
    )
    replies = handler.receive(encode_stage_message(cancel))
    assert len(replies) == 1
    assert isinstance(replies[0], StageStatusReply)
    assert replies[0].status == "safe_not_submitted"

    assert handler.receive(STAGE_NOTIF_PREFIX + b"invalid") == ()
    assert handler.invalid_notifications == 1
