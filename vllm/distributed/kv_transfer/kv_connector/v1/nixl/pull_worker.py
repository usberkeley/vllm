# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pull-specific (READ) worker-side logic for the NIXL connector."""

import queue
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Hashable
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.staging import (
    STAGE_NOTIF_PREFIX,
    ConsumerProgress,
    DefinitelyNotSubmittedError,
    DescriptorCache,
    PossiblySubmittedError,
    ProducerPipeline,
    StageCancel,
    StageModeCommit,
    StageReadComplete,
    StageReady,
    StageStatusQuery,
    StageStatusReply,
    StagingCopyPlan,
    StagingSessionRegistry,
    StagingTransferIntent,
    StagingTransferSession,
    committed_wire_chunk_bytes,
    decode_stage_message,
    encode_stage_message,
    gather_chunk,
    scatter_chunk,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    ReadSpec,
    _is_attention_spec,
    _is_ssm_spec,
)
from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)

# Slack (seconds) subtracted from D's exported block-expiry deadline on the turn-2
# readback, absorbing clock-offset error and read latency.
_KV_BLOCKS_EXPIRY_SAFETY_MARGIN = 5.0


class _NixlStagingReadBackend:
    """One-descriptor NIXL READ backend for consumer staging slots."""

    def __init__(self, worker: "NixlPullConnectorWorker") -> None:
        self.worker = worker
        release = worker.nixl_wrapper.release_dlist_handle
        capacity = max(32, worker.staging_config.slot_count * 8)
        self.local = DescriptorCache[tuple[int, int], int](capacity, release)
        self.remote = DescriptorCache[tuple[Any, ...], int](capacity, release)
        self.active_handles: set[Hashable] = set()

    def _local_handle(self, slot_id: int, valid_bytes: int) -> int:
        pool = self.worker.staging_pool
        assert pool is not None

        def create() -> int:
            base, _, device_id = pool.slot_transfer_regions(self.worker.device_id)[
                slot_id
            ]
            data = np.asarray([[base, valid_bytes, device_id]], dtype=np.uint64)
            descs = self.worker.nixl_wrapper.get_xfer_descs(
                data, self.worker.nixl_memory_type
            )
            return self.worker.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", descs)

        return self.local.get_or_create((slot_id, valid_bytes), create)

    def _remote_handle(self, ready: StageReady) -> int:
        key = (
            ready.producer_engine_id,
            ready.producer_rank,
            ready.producer_generation,
            ready.source_slot_id,
            ready.valid_bytes,
        )

        def create() -> int:
            metadata = self.worker._staging_remote_metadata[ready.producer_engine_id][
                ready.producer_rank
            ]
            if (
                not metadata.supports_staging
                or metadata.staging_generation != ready.producer_generation
                or ready.source_slot_id >= metadata.staging_slot_count
                or ready.valid_bytes > metadata.staging_slot_bytes
            ):
                raise DefinitelyNotSubmittedError(
                    "READY does not match the producer staging handshake"
                )
            address = (
                metadata.staging_pool_base_addr
                + ready.source_slot_id * metadata.staging_slot_bytes
            )
            data = np.asarray(
                [[address, ready.valid_bytes, metadata.device_id]], dtype=np.uint64
            )
            descs = self.worker.nixl_wrapper.get_xfer_descs(
                data, self.worker.nixl_memory_type
            )
            agent = self.worker._remote_agents[ready.producer_engine_id][
                (0, ready.producer_rank)
            ]
            return self.worker.nixl_wrapper.prep_xfer_dlist(agent, descs)

        return self.remote.get_or_create(key, create)

    def post_read(self, ready: StageReady, local_slot_id: int) -> Hashable:
        try:
            local_handle = self._local_handle(local_slot_id, ready.valid_bytes)
            remote_handle = self._remote_handle(ready)
            handle = self.worker.nixl_wrapper.make_prepped_xfer(
                "READ",
                local_handle,
                np.asarray([0], dtype=np.uint64),
                remote_handle,
                np.asarray([0], dtype=np.uint64),
                notif_msg=None,
            )
            self.active_handles.add(handle)
        except DefinitelyNotSubmittedError:
            raise
        except Exception as exc:
            raise DefinitelyNotSubmittedError(str(exc)) from exc
        try:
            self.worker.nixl_wrapper.transfer(handle)
        except Exception as exc:
            raise PossiblySubmittedError(handle, str(exc)) from exc
        return handle

    def check_read(self, handle: Hashable) -> str:
        return self.worker.nixl_wrapper.check_xfer_state(handle)

    def release_read(self, handle: Hashable) -> None:
        try:
            telemetry = self.worker.nixl_wrapper.get_xfer_telemetry(handle)
            self.worker.xfer_stats.record_transfer(telemetry)
        except Exception:
            logger.debug("NIXL staging telemetry was unavailable", exc_info=True)
        self.worker.nixl_wrapper.release_xfer_handle(handle)
        self.active_handles.discard(handle)

    def close(self) -> None:
        for handle in self.active_handles:
            self.worker.nixl_wrapper.release_xfer_handle(handle)
        self.active_handles.clear()
        self.local.clear()
        self.remote.clear()


class NixlPullConnectorWorker(NixlBaseConnectorWorker):
    """Pull-specific (READ) worker logic."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)
        self._staging_sessions = StagingSessionRegistry()
        self._staging_activation: queue.SimpleQueue[
            tuple[bool, str, ReqMeta, StagingTransferIntent, StageModeCommit]
        ] = queue.SimpleQueue()
        self._staging_early_messages: list[Any] = []
        self._staging_transfer_to_request: dict[str, str] = {}
        self._staging_request_transfers: dict[str, set[str]] = defaultdict(set)
        self._staging_stable_transfers: set[str] = set()
        self._staging_done_sending: set[str] = set()
        self._staging_done_sending_queue: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._staging_done_recving_queue: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._staging_legacy_notifs: queue.SimpleQueue[bytes] = queue.SimpleQueue()
        self._staging_control_outbox: queue.SimpleQueue[tuple[str, int, bytes]] = (
            queue.SimpleQueue()
        )
        self._staging_cancelled_requests: set[str] = set()
        self._staging_cancel_queue: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._staging_nixl_calls: queue.SimpleQueue[Callable[[], None]] = (
            queue.SimpleQueue()
        )
        self._staging_reported_failures: set[str] = set()
        self._staging_direct_transfers: defaultdict[str, list[int]] = defaultdict(list)
        self._staging_stop = threading.Event()
        self._staging_wake = threading.Event()
        self._staging_thread: threading.Thread | None = None
        self._staging_source_events: dict[str, Any] = {}
        self._staging_gather_stream: Any = None
        self._staging_scatter_stream: Any = None
        self._staging_producer: ProducerPipeline | None = None
        self._staging_consumer: ConsumerProgress | None = None
        self._staging_read_backend: _NixlStagingReadBackend | None = None

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        super().register_kv_caches(kv_caches)
        if not self.staging_config.enabled:
            return
        assert self.staging_pool is not None
        if self.staging_pool.producer:
            self._staging_gather_stream = current_platform.Stream()
            self._staging_producer = ProducerPipeline(
                self.staging_pool,
                self.staging_config,
                self.engine_id,
                self.tp_rank,
                self.staging_generation,
                self._staging_sessions,
            )
        else:
            self._staging_scatter_stream = current_platform.Stream()
            self._staging_consumer = ConsumerProgress(
                self.staging_pool,
                self.staging_config,
                self.engine_id,
                self.tp_rank,
                self.staging_generation,
                self._staging_sessions,
            )
            self._staging_read_backend = _NixlStagingReadBackend(self)
        self._staging_thread = threading.Thread(
            target=self._staging_progress_loop,
            daemon=True,
            name="vllm-nixl-staging-progress",
        )
        self._staging_thread.start()

    def _matching_staging_intents(
        self, meta: ReqMeta, producer: bool
    ) -> tuple[StagingTransferIntent, ...]:
        commit = meta.staging_mode_commit
        if not self.staging_config.enabled or commit is None:
            return ()
        if commit.mode == "fail":
            raise RuntimeError("Router committed staging transfer failure")
        if commit.mode != "staged":
            return ()
        identity = (self.engine_id, self.tp_rank, self.staging_generation)
        intents = tuple(
            intent
            for intent in meta.staging_intents
            if (
                (
                    intent.producer_engine_id,
                    intent.producer_rank,
                    intent.producer_generation,
                )
                if producer
                else (
                    intent.consumer_engine_id,
                    intent.consumer_rank,
                    intent.consumer_generation,
                )
            )
            == identity
        )
        return intents

    def _build_staging_plan(
        self,
        meta: ReqMeta,
        intent: StagingTransferIntent,
        commit: StageModeCommit,
    ) -> StagingCopyPlan:
        heterogeneous_tp = intent.producer_tp_size != intent.consumer_tp_size
        if heterogeneous_tp and self._has_mamba:
            raise RuntimeError("hybrid SSM staging requires homogeneous TP")
        groups = [tuple(blocks) for blocks in meta.local_physical_block_ids]
        if len(groups) != len(self._group_spec_types):
            raise RuntimeError("staging block groups do not match KV cache groups")
        producer_side = intent.producer_engine_id == self.engine_id
        if intent.source_ranges_by_group:
            if len(intent.source_ranges_by_group) != len(groups):
                raise RuntimeError("source range count does not match KV groups")
            logical_groups = [tuple(blocks) for blocks in meta.local_block_ids]
            selected_logical: list[list[int]] = []
            for group_index, ranges in enumerate(intent.source_ranges_by_group):
                selected_count = sum(count for _, count in ranges)
                if (
                    not producer_side
                    and len(logical_groups[group_index]) == selected_count
                ):
                    selected_logical.append(list(logical_groups[group_index]))
                    continue
                selected: list[int] = []
                for start, count in ranges:
                    if start + count > len(logical_groups[group_index]):
                        raise RuntimeError("staging source range exceeds block table")
                    selected.extend(logical_groups[group_index][start : start + count])
                selected_logical.append(selected)
            groups = [
                tuple(blocks)
                for blocks in self._logical_to_kernel_block_ids(
                    selected_logical,
                    self._physical_blocks_per_logical_kv_block,
                )
            ]
        wire_chunk_bytes = committed_wire_chunk_bytes(intent, commit)
        if wire_chunk_bytes > self.staging_config.slot_bytes:
            raise RuntimeError("committed chunk exceeds the local staging slot")
        wire_segments: list[tuple[int, int, int, int, bool]] = []
        region_count = len(self._staging_regions)
        peer_engine = (
            intent.consumer_engine_id if producer_side else intent.producer_engine_id
        )
        peer_rank = intent.consumer_rank if producer_side else intent.producer_rank
        peer_metadata = self._staging_remote_metadata[peer_engine][peer_rank]
        expected_generation = (
            intent.consumer_generation if producer_side else intent.producer_generation
        )
        if (
            not peer_metadata.supports_staging
            or peer_metadata.staging_generation != expected_generation
            or wire_chunk_bytes > peer_metadata.staging_slot_bytes
        ):
            raise RuntimeError("peer handshake cannot honor staged mode commit")
        if len(peer_metadata.block_lens) != region_count:
            raise RuntimeError("staging peer region geometry does not match")
        total_heads = self.model_config.get_total_num_kv_heads()
        producer_head_ranks = min(intent.producer_tp_size, total_heads)
        consumer_head_ranks = min(intent.consumer_tp_size, total_heads)
        producer_head = intent.producer_rank * total_heads // intent.producer_tp_size
        consumer_head = intent.consumer_rank * total_heads // intent.consumer_tp_size
        remote_groups: list[tuple[int, ...]] | None = None
        if (
            not producer_side
            and not intent.source_ranges_by_group
            and meta.remote is not None
        ):
            remote_groups = [
                tuple(blocks)
                for blocks in self._logical_to_kernel_block_ids(
                    meta.remote.block_ids,
                    peer_metadata.physical_blocks_per_logical_kv_block,
                )
            ]
        for group_index, blocks in enumerate(groups):
            spec_type = self._group_spec_types[group_index]
            if _is_attention_spec(spec_type):
                for region_index, block_bytes in enumerate(self.block_len_per_layer):
                    transfer_bytes = block_bytes
                    block_offset = 0
                    if not self._region_is_mla[region_index]:
                        if (
                            producer_side
                            and consumer_head_ranks > producer_head_ranks
                            and consumer_head_ranks % producer_head_ranks == 0
                        ):
                            ratio = consumer_head_ranks // producer_head_ranks
                            transfer_bytes = block_bytes // ratio
                            heads_per_consumer = total_heads // consumer_head_ranks
                            slot = (consumer_head - producer_head) // heads_per_consumer
                            block_offset = slot * transfer_bytes
                        elif (
                            not producer_side
                            and producer_head_ranks > consumer_head_ranks
                            and producer_head_ranks % consumer_head_ranks == 0
                        ):
                            ratio = producer_head_ranks // consumer_head_ranks
                            transfer_bytes = block_bytes // ratio
                            heads_per_producer = total_heads // producer_head_ranks
                            slot = (producer_head - consumer_head) // heads_per_producer
                            block_offset = slot * transfer_bytes
                    if remote_groups is not None:
                        source_bytes = peer_metadata.block_lens[region_index]
                        if (
                            consumer_head_ranks > producer_head_ranks
                            and consumer_head_ranks % producer_head_ranks == 0
                            and not self._region_is_mla[region_index]
                        ):
                            source_bytes //= consumer_head_ranks // producer_head_ranks
                        prefix_bytes = (
                            len(remote_groups[group_index]) * source_bytes
                            - len(blocks) * transfer_bytes
                        )
                        if prefix_bytes < 0:
                            raise RuntimeError(
                                "local staging destination exceeds source wire range"
                            )
                        if prefix_bytes:
                            wire_segments.append(
                                (region_index, 0, 0, prefix_bytes, False)
                            )
                    wire_segments.extend(
                        (
                            region_index,
                            block_id,
                            block_offset,
                            transfer_bytes,
                            True,
                        )
                        for block_id in blocks
                    )
                continue
            if not _is_ssm_spec(spec_type) or self._conv_decomp is None:
                raise RuntimeError("unsupported staging KV cache group")
            conv_size, ssm_size = self._mamba_ssm_size
            for region_index in range(region_count):
                mamba_region = region_count + region_index
                for offset, length in self._conv_decomp.local_conv_offsets:
                    wire_segments.extend(
                        (mamba_region, block_id, offset, length, True)
                        for block_id in blocks
                    )
                wire_segments.extend(
                    (mamba_region, block_id, conv_size, ssm_size, True)
                    for block_id in blocks
                )
        return StagingCopyPlan.from_wire_segments(
            intent.plan_id, tuple(wire_segments), wire_chunk_bytes
        )

    def _queue_staging_activation(
        self,
        producer: bool,
        request_id: str,
        meta: ReqMeta,
        intent: StagingTransferIntent,
    ) -> None:
        commit = meta.staging_mode_commit
        assert commit is not None
        peer_id = intent.consumer_engine_id if producer else intent.producer_engine_id
        peer_host = intent.consumer_host if producer else intent.producer_host
        peer_port = intent.consumer_port if producer else intent.producer_port
        peer_tp = intent.consumer_tp_size if producer else intent.producer_tp_size
        future = self._ensure_handshake(
            peer_id,
            peer_host,
            peer_port,
            peer_tp,
            notif_agents_only=producer,
        )
        entry = (producer, request_id, meta, intent, commit)
        if future is None:
            self._staging_activation.put(entry)
            self._staging_wake.set()
            return

        def activated(future: Any, activation=entry) -> None:
            try:
                future.result()
            except Exception as exc:
                self._log_failure(
                    "staging_handshake_failed",
                    req_id=request_id,
                    error=exc,
                )
                if not producer:
                    self._handle_failed_transfer(request_id, None)
                return
            self._staging_activation.put(activation)
            self._staging_wake.set()

        future.add_done_callback(activated)

    def _activate_staging_requests(self) -> None:
        while True:
            try:
                producer, request_id, meta, intent, commit = (
                    self._staging_activation.get_nowait()
                )
            except queue.Empty:
                break
            try:
                if request_id in self._staging_cancelled_requests:
                    continue
                session = StagingTransferSession(intent)
                session.accept_decision(commit)
                session.freeze_plan(self._build_staging_plan(meta, intent, commit))
                self._staging_transfer_to_request[intent.transfer_id] = request_id
                self._staging_request_transfers[request_id].add(intent.transfer_id)
                if producer:
                    assert self._staging_producer is not None
                    self._staging_producer.register_transfer(session)
                else:
                    assert self._staging_consumer is not None
                    self._staging_consumer.register_transfer(session)
            except Exception as exc:
                self._log_failure(
                    "staging_activation_failed",
                    req_id=request_id,
                    error=exc,
                )
                if producer:
                    self._staging_done_sending.add(request_id)
                else:
                    self._handle_failed_transfer(request_id, None)
        early, self._staging_early_messages = self._staging_early_messages, []
        for payload in early:
            self._handle_stage_message(payload)

    def _start_staging_gather(
        self, session: StagingTransferSession, chunk: Any, slot_id: int
    ) -> Any:
        assert self.staging_pool is not None
        event = torch.Event()
        with current_platform.stream(self._staging_gather_stream):
            source_event = self._staging_source_events.get(session.intent.transfer_id)
            if source_event is not None:
                self._staging_gather_stream.wait_event(source_event)
            gather_chunk(
                self._staging_regions + self._staging_mamba_regions,
                self.staging_pool.slot_view(slot_id, chunk.valid_bytes),
                chunk,
            )
            event.record()
        return event.query

    def _start_staging_scatter(self, ready: StageReady, slot_id: int) -> Any:
        assert self.staging_pool is not None
        session = self._staging_sessions.get_for_ready(ready)
        assert session is not None
        chunk = session.validate_ready(ready)
        event = torch.Event()
        with current_platform.stream(self._staging_scatter_stream):
            scatter_chunk(
                self._staging_regions + self._staging_mamba_regions,
                self.staging_pool.slot_view(slot_id, ready.valid_bytes),
                chunk,
            )
            event.record()
        return event.query

    def _send_stage_message(self, engine_id: str, rank: int, payload: bytes) -> None:
        agent = self._remote_agents[engine_id][(0, rank)]
        self.nixl_wrapper.send_notif(agent, notif_msg=payload)

    def _progress_staging(self) -> None:
        if not self.staging_config.enabled or self.staging_pool is None:
            return
        self._activate_staging_requests()
        while True:
            try:
                self._staging_nixl_calls.get_nowait()()
            except queue.Empty:
                break
        if self._staging_direct_transfers:
            for request_id in self._pop_done_transfers(self._staging_direct_transfers):
                self._staging_done_recving_queue.put(request_id)
        while True:
            try:
                self._apply_staging_cancel(self._staging_cancel_queue.get_nowait())
            except queue.Empty:
                break
        while True:
            try:
                engine_id, rank, payload = self._staging_control_outbox.get_nowait()
            except queue.Empty:
                break
            try:
                self._send_stage_message(engine_id, rank, payload)
            except Exception:
                self._staging_control_outbox.put((engine_id, rank, payload))
                break
        if self._staging_producer is not None:
            pipeline = self._staging_producer
            pipeline.start_available(self._start_staging_gather)
            pipeline.poll_gathers()
            try:
                pipeline.send_ready(
                    lambda ready, payload: self._send_stage_message(
                        ready.consumer_engine_id, ready.consumer_rank, payload
                    )
                )
            except Exception:
                logger.warning("Failed to send NIXL staging READY; it will retry")
            for query in pipeline.progress.status_queries(
                time.monotonic(), self.staging_config.transfer_timeout
            ):
                try:
                    self._send_stage_message(
                        query.consumer_engine_id,
                        query.consumer_rank,
                        encode_stage_message(query),
                    )
                except Exception:
                    logger.warning("Failed to send NIXL staging STATUS_QUERY")
            self._staging_stable_transfers.update(pipeline.pop_source_stable())
            for request_id, transfers in list(self._staging_request_transfers.items()):
                if transfers and transfers <= self._staging_stable_transfers:
                    self._staging_done_sending_queue.put(request_id)
                    del self._staging_request_transfers[request_id]
            for transfer_id in pipeline.pop_completed_transfers():
                self._staging_transfer_to_request.pop(transfer_id, None)
                self._staging_source_events.pop(transfer_id, None)
        if self._staging_consumer is not None:
            consumer = self._staging_consumer
            backend = self._staging_read_backend
            assert backend is not None
            consumer.post_available(backend)
            consumer.poll_unknown_reads(backend)
            consumer.poll_reads(backend, self._start_staging_scatter)
            while True:
                try:
                    sent = consumer.completions.send_next(
                        lambda payload: self._send_stage_message(
                            decode_stage_message(payload).producer_engine_id,
                            decode_stage_message(payload).producer_rank,
                            payload,
                        )
                    )
                except Exception:
                    break
                if not sent:
                    break
            consumer.poll_scatters()
            for transfer_id in consumer.pop_completed_transfers():
                request_id = self._staging_transfer_to_request[transfer_id]
                self._staging_stable_transfers.add(transfer_id)
                transfers = self._staging_request_transfers[request_id]
                if transfers <= self._staging_stable_transfers:
                    self._staging_done_recving_queue.put(request_id)
                    del self._staging_request_transfers[request_id]
            for transfer_id in (
                consumer.failed_transfers - self._staging_reported_failures
            ):
                failed_request_id = self._staging_transfer_to_request.get(transfer_id)
                if failed_request_id is not None:
                    self._handle_failed_transfer(failed_request_id, None)
                self._staging_reported_failures.add(transfer_id)

    def _staging_progress_loop(self) -> None:
        current_platform.set_device(self.device_id)
        while not self._staging_stop.is_set():
            try:
                for notifs in self.nixl_wrapper.get_new_notifs().values():
                    for notif in notifs:
                        if notif.startswith(STAGE_NOTIF_PREFIX):
                            self._handle_stage_message(notif)
                        else:
                            self._staging_legacy_notifs.put(notif)
                self._progress_staging()
            except Exception:
                logger.exception("NIXL staging progress iteration failed")
            self._staging_wake.wait(0.001)
            self._staging_wake.clear()

    def _handle_stage_message(self, payload: bytes) -> None:
        message = decode_stage_message(payload)
        try:
            if isinstance(message, StageReady):
                if self._staging_consumer is None:
                    return
                self._staging_consumer.receive_ready(message)
            elif isinstance(message, StageReadComplete):
                if self._staging_producer is not None:
                    self._staging_producer.progress.accept_read_complete(message)
            elif isinstance(message, StageStatusReply):
                if self._staging_producer is not None:
                    self._staging_producer.progress.accept_status_reply(message)
            elif isinstance(message, StageStatusQuery):
                if self._staging_consumer is not None:
                    reply = self._staging_consumer.reply_status(message)
                    self._send_stage_message(
                        message.producer_engine_id,
                        message.producer_rank,
                        encode_stage_message(reply),
                    )
            elif isinstance(message, StageCancel):
                if self._staging_producer is not None:
                    for query in self._staging_producer.cancel_transfer(
                        message.transfer_id
                    ):
                        self._staging_control_outbox.put(
                            (
                                query.consumer_engine_id,
                                query.consumer_rank,
                                encode_stage_message(query),
                            )
                        )
                if self._staging_consumer is not None:
                    for reply in self._staging_consumer.cancel_transfer(
                        message.transfer_id
                    ):
                        self._staging_control_outbox.put(
                            (
                                reply.producer_engine_id,
                                reply.producer_rank,
                                encode_stage_message(reply),
                            )
                        )
        except RuntimeError:
            self._staging_early_messages.append(payload)

    def get_finished(self) -> tuple[set[str], set[str]]:
        while True:
            try:
                req_id = self._staging_done_recving_queue.get_nowait()
            except queue.Empty:
                break
            self._recving_transfers[req_id] = []
        return super().get_finished()

    def _cancel_staging_request(self, request_id: str) -> None:
        self._staging_cancel_queue.put(request_id)
        self._staging_wake.set()

    def _apply_staging_cancel(self, request_id: str) -> None:
        self._staging_cancelled_requests.add(request_id)
        for transfer_id in self._staging_request_transfers.get(request_id, ()):
            sessions = self._staging_sessions.for_transfer(transfer_id)
            if self._staging_producer is not None:
                self._staging_producer.cancel_transfer(transfer_id)
            if self._staging_consumer is not None:
                for reply in self._staging_consumer.cancel_transfer(transfer_id):
                    self._staging_control_outbox.put(
                        (
                            reply.producer_engine_id,
                            reply.producer_rank,
                            encode_stage_message(reply),
                        )
                    )
            for session in sessions:
                intent = session.intent
                cancel = StageCancel(
                    protocol_version=intent.protocol_version,
                    producer_generation=intent.producer_generation,
                    consumer_generation=intent.consumer_generation,
                    transfer_id=transfer_id,
                    request_id=request_id,
                    producer_request_id=intent.producer_request_id,
                    consumer_request_id=intent.consumer_request_id,
                    consumer_engine_id=intent.consumer_engine_id,
                    consumer_rank=intent.consumer_rank,
                )
                if self._staging_producer is not None:
                    peer_id, peer_rank = (
                        intent.consumer_engine_id,
                        intent.consumer_rank,
                    )
                else:
                    peer_id, peer_rank = (
                        intent.producer_engine_id,
                        intent.producer_rank,
                    )
                self._staging_control_outbox.put(
                    (peer_id, peer_rank, encode_stage_message(cancel))
                )

    def shutdown(self) -> None:
        backend = getattr(self, "_staging_read_backend", None)
        stop = getattr(self, "_staging_stop", None)
        if stop is not None:
            stop.set()
            self._staging_wake.set()
        thread = getattr(self, "_staging_thread", None)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5)
            if thread.is_alive():
                logger.error(
                    "NIXL staging progress thread did not stop; retaining registered "
                    "memory to avoid unsafe reuse"
                )
                return
            self._staging_thread = None
        for engine_id in list(getattr(self, "_remote_agents", {})):
            self._cleanup_remote_engine(engine_id, log_eviction=False)
        for handles in getattr(self, "_staging_direct_transfers", {}).values():
            for handle in handles:
                self.nixl_wrapper.release_xfer_handle(handle)
        if hasattr(self, "_staging_direct_transfers"):
            self._staging_direct_transfers.clear()
        if backend is not None:
            backend.close()
            self._staging_read_backend = None
        super().shutdown()

    def start_load_kv(self, metadata: NixlConnectorMetadata):
        """
        Start loading by triggering non-blocking nixl_xfer.
        We check for these trnxs to complete in each step().
        """
        for req_id, meta in metadata.reqs_to_recv.items():
            meta.local_physical_block_ids = self._logical_to_kernel_block_ids(
                meta.local_block_ids, self._physical_blocks_per_logical_kv_block
            )
            try:
                staged_intents = self._matching_staging_intents(meta, producer=False)
            except RuntimeError as exc:
                self._recving_metadata[req_id] = meta
                self._log_failure("staging_mode_failed", req_id=req_id, error=exc)
                self._handle_failed_transfer(req_id, None)
                continue
            if (
                meta.staging_mode_commit is not None
                and meta.staging_mode_commit.mode == "staged"
                and not staged_intents
            ):
                self._recving_metadata[req_id] = meta
                self._log_failure(
                    "staging_intent_missing_for_rank",
                    req_id=req_id,
                )
                self._handle_failed_transfer(req_id, None)
                continue
            if staged_intents:
                self._recving_metadata[req_id] = meta
                for intent in staged_intents:
                    self._queue_staging_activation(False, req_id, meta, intent)
                continue
            assert meta.remote is not None
            # Remote block IDs are kept logical here; expanded in
            # _read_blocks_for_req using the remote engine's phys ratio.
            remote_engine_id = meta.remote.engine_id
            logger.debug(
                "start_load_kv for request %s from remote engine %s. "
                "Num local_block_ids: %s. Num remote_block_ids: %s. ",
                req_id,
                remote_engine_id,
                len(meta.local_physical_block_ids),
                len(meta.remote.block_ids),
            )
            # always store metadata for failure recovery
            self._recving_metadata[req_id] = meta
            if remote_engine_id not in self._remote_agents:
                # Initiate handshake with remote engine to exchange metadata.
                with self._handshake_lock:
                    if remote_engine_id not in self._remote_agents:
                        self._background_nixl_handshake(req_id, remote_engine_id, meta)
                        continue

            # Handshake already completed, start async read xfer.
            self._read_blocks_for_req(req_id, meta)

        # Start transfers for requests whose handshakes have now finished.
        while not self._ready_requests.empty():
            self._read_blocks_for_req(*self._ready_requests.get_nowait())

        for req_id, meta in metadata.staging_reqs_to_send.items():
            meta.local_physical_block_ids = self._logical_to_kernel_block_ids(
                meta.local_block_ids, self._physical_blocks_per_logical_kv_block
            )
            try:
                staged_intents = self._matching_staging_intents(meta, producer=True)
            except RuntimeError as exc:
                self._log_failure("staging_mode_failed", req_id=req_id, error=exc)
                self._staging_done_sending.add(req_id)
                continue
            if (
                meta.staging_mode_commit is not None
                and meta.staging_mode_commit.mode == "staged"
                and not staged_intents
            ):
                self._log_failure(
                    "staging_intent_missing_for_rank",
                    req_id=req_id,
                )
                self._staging_done_sending.add(req_id)
                continue
            for intent in staged_intents:
                source_event = torch.Event()
                source_event.record()
                self._staging_source_events[intent.transfer_id] = source_event
                self._queue_staging_activation(True, req_id, meta, intent)

        # Keep around the requests that have been part of a batch. This is
        # needed because async scheduling pushes the misalignment between the
        # moment in which requests expiration is set (P side) and the moment in
        # which blocks are read from D. As P can now more easily lag behind D
        # while processing the next batch, we make sure to only set an
        # expiration for requests that have not been read from D yet.
        for req_id in metadata.reqs_in_batch:
            self._reqs_to_process.add(req_id)

        # Remove all requests that are not to be processed (eg aborted).
        for req_id in metadata.reqs_not_processed:
            self._reqs_to_process.discard(req_id)
            if req_id in self._staging_request_transfers:
                self._cancel_staging_request(req_id)
                self._reqs_to_send.pop(req_id, None)
                continue
            # We should never get an abort after setting an expiry timer
            assert req_id not in self._reqs_to_send

        # Add to requests that are waiting to be read and track expiration.
        # Deadlines are stamped with the scheduler process's perf_counter,
        # which is not comparable to ours when the worker runs in another
        # process on another node (perf_counter epochs differ by boot time).
        # Rebase the remaining TTL onto our clock; broadcast latency only
        # lengthens the lease, which is the safe direction. A cross-node
        # epoch gap larger than the TTL otherwise expires the lease on
        # arrival and the blocks are freed before D reads them.
        now_local = time.perf_counter()
        for req_id, expiration_time in metadata.reqs_to_send.items():
            if req_id in self._reqs_to_process:
                if metadata.scheduler_clock:
                    expiration_time = now_local + (
                        expiration_time - metadata.scheduler_clock
                    )
                self._reqs_to_send[req_id] = expiration_time

        for req_id in metadata.staging_reqs_to_send:
            if req_id in self._reqs_to_send:
                self._reqs_to_send[req_id] = float("inf")

        # Send heartbeats to P-side engines to keep KV blocks alive while
        # requests sit in the D scheduler WAITING queue.
        self._send_heartbeats(metadata)
        self._staging_wake.set()

    def _is_turn2_read_expired(self, meta: ReqMeta) -> bool:
        """Whether D's cached blocks for this turn-2 readback have (nearly) expired."""
        assert meta.remote is not None
        blocks_expiry_time = meta.remote.blocks_expiry_time
        # Deadline may be absent (router may not forward it) -> read as usual.
        if blocks_expiry_time is None or not meta.local_physical_block_ids:
            return False
        clock_offset = self._engine_clock_offset[meta.remote.engine_id]
        deadline = blocks_expiry_time - clock_offset
        return time.perf_counter() + _KV_BLOCKS_EXPIRY_SAFETY_MARGIN >= deadline

    def _read_blocks_for_req(self, req_id: str, meta: ReqMeta):
        assert meta.remote is not None and self.transfer_topo is not None
        engine_id = meta.remote.engine_id
        # Update last activity from this remote. Mind that cleanup is done on main
        # thread (this one), so we don't race on this structure.
        self._engine_last_active[engine_id] = time.perf_counter()

        if self._bidirectional_kv_xfer_enabled and self._is_turn2_read_expired(meta):
            logger.warning(
                "Declining expired remote read for %s from engine %s.",
                req_id,
                engine_id,
            )
            self.xfer_stats.record_kv_expired_req()
            self._handle_failed_transfer(req_id, None)
            return

        plan = self.tp_mappings[engine_id]
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        tp_ratio = self.transfer_topo.tp_ratio(remote_info.remote_tp_size)

        meta.remote.block_ids = self._logical_to_kernel_block_ids(
            meta.remote.block_ids,
            remote_info.remote_physical_blocks_per_logical,
        )
        remote_block_ids = meta.remote.block_ids
        local_block_ids = meta.local_physical_block_ids
        num_groups = len(local_block_ids)
        read_specs = [
            ReadSpec(
                remote_rank=rank,
                local_block_ids=[
                    list(local_block_ids[g])
                    if rank in plan.source_ranks_per_group[g]
                    else []
                    for g in range(num_groups)
                ],
                remote_block_ids=[
                    list(remote_block_ids[g])
                    if rank in plan.source_ranks_per_group[g]
                    else []
                    for g in range(num_groups)
                ],
            )
            for rank in plan.all_source_ranks
        ]

        # D may have to perform multiple reads from different remote ranks.
        # Pure MLA reads once because its cache is replicated. Hybrid
        # MLA+SSM still needs one read per SSM source rank.
        if self.use_mla and tp_ratio < 0 and not self._has_mamba:
            assert len(read_specs) == 1

        for i, spec in enumerate(read_specs):
            remote_block_size = remote_info.remote_block_size
            logger.debug(
                "Remote agent %s available, calling _read_blocks"
                " on remote rank %s with remote block size %s for req %s",
                meta.remote.engine_id,
                spec.remote_rank,
                remote_block_size,
                req_id,
            )
            # Get side handles.
            if tp_ratio < 0 and (not self.use_mla or len(read_specs) > 1):
                # Remote tp_size > local tp_size: we must perform multiple
                # reads. Get the memory chunk onto which we will write to.
                split_key = (tp_ratio, remote_block_size)
                local_xfer_side_handle = self.src_xfer_handles_by_tp_ratio[split_key][i]
            else:
                # Single read from remote, we write to the whole memory region.
                # Also handle remote block size different from local block size.
                local_xfer_side_handle = self.src_xfer_handles_by_block_size[
                    remote_block_size
                ]

            # Destination handle: remote_engine_id -> remote_rank -> handle.
            remote_xfer_side_handle = self.dst_xfer_side_handles[meta.remote.engine_id][
                spec.remote_rank
            ]

            self._read_blocks(
                read_spec=spec,
                request_id=req_id,
                dst_engine_id=meta.remote.engine_id,
                remote_request_id=meta.remote.request_id,
                local_xfer_side_handle=local_xfer_side_handle,
                remote_xfer_side_handle=remote_xfer_side_handle,
            )

        if self.use_mla and tp_ratio < 0 and len(read_specs) == 1:
            # ..but we still need to notify the other remote ranks that we
            # have the blocks we need so they can update the request state.
            notif_id = f"{meta.remote.request_id}:{self.world_size}".encode()
            remote_agents = self._remote_agents[meta.remote.engine_id]
            for rank_to_notify, agent in remote_agents.items():
                if rank_to_notify != (0, read_specs[0].remote_rank):
                    self.nixl_wrapper.send_notif(agent, notif_msg=notif_id)

    def _read_blocks(
        self,
        read_spec: ReadSpec,
        dst_engine_id: str,
        request_id: str,
        remote_request_id: str,
        local_xfer_side_handle: int,
        remote_xfer_side_handle: int,
    ):
        """
        Post a READ point-to-point xfer request from a single local worker to
        a single remote worker.
        """
        if (
            self.staging_config.enabled
            and self._staging_thread is not None
            and threading.current_thread() is not self._staging_thread
        ):
            self._staging_nixl_calls.put(
                partial(
                    self._read_blocks,
                    read_spec,
                    dst_engine_id,
                    request_id,
                    remote_request_id,
                    local_xfer_side_handle,
                    remote_xfer_side_handle,
                )
            )
            self._staging_wake.set()
            return
        assert self.transfer_topo is not None
        remote_rank = read_spec.remote_rank
        local_block_ids = read_spec.local_block_ids
        remote_block_ids = read_spec.remote_block_ids

        remote_info = self.transfer_topo.get_engine_info(dst_engine_id)
        block_size_ratio = self.transfer_topo.block_size_ratio(
            remote_info.remote_block_size
        )
        if block_size_ratio > 1:
            local_block_ids, remote_block_ids = (
                self._map_block_ids_for_block_size_ratio(
                    local_block_ids, remote_block_ids, block_size_ratio
                )
            )
        # NOTE(rob): having the staging blocks be on the READER side is
        # not going to work well (since we will have to call rearrange tensors).
        # after we detect the txn is complete (which means we cannot make the
        # read trxn async easily). If we want to make "READ" happen cleanly,
        # then we will need to have the staging blocks on the remote side.

        # NOTE(rob): according to nvidia the staging blocks are used to
        # saturate IB with heterogeneous TP sizes.

        # Number of D TP workers that will read from dst P. Propagate info
        # on notification so that dst worker can wait before freeing blocks.
        notif_id = f"{remote_request_id}:{self.world_size}".encode()

        # Full prefix cache hit: do not need to read remote blocks,
        # just notify P worker that we have the blocks we need.
        if len(local_block_ids) == 0:
            # A full prefix cache hit is indicated with an empty list.
            agent_name = self._remote_agents[dst_engine_id][(0, remote_rank)]
            try:
                self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_id)
            except Exception as e:
                self._log_failure(
                    failure_type="notification_failed",
                    msg="P worker blocks will be freed after timeout. "
                    "This may indicate network issues.",
                    req_id=request_id,
                    error=e,
                    dst_engine_id=dst_engine_id,
                    remote_rank=remote_rank,
                    remote_agent_name=agent_name,
                )
                self.xfer_stats.record_failed_notification()
            transfers = (
                self._staging_direct_transfers
                if self.staging_config.enabled
                else self._recving_transfers
            )
            transfers.setdefault(request_id, [])
            return

        assert (
            len(remote_block_ids)
            == len(local_block_ids)
            == len(self.kv_cache_config.kv_cache_groups)
        )
        local_block_ids, remote_block_ids = self._apply_prefix_caching(
            decode_block_ids=local_block_ids,
            prefill_block_ids=remote_block_ids,
            decode_physical_per_logical=self._physical_blocks_per_logical_kv_block,
            prefill_physical_per_logical=remote_info.remote_physical_blocks_per_logical,
        )

        # NOTE (nicolo) With homogeneous TP, each TP worker loads KV from
        # corresponding rank. With heterogeneous TP, fixing D>P, the D tp
        # workers will issue xfers to parts of the P worker remote kv caches.

        # Get descs ids.
        remote_block_descs_ids = self._compute_desc_ids(
            block_ids=remote_block_ids,
            dst_num_blocks=self.dst_num_blocks[dst_engine_id],
            block_size_ratio=None,
            physical_blocks_per_logical=remote_info.remote_physical_blocks_per_logical,
        )
        local_block_descs_ids = self._compute_desc_ids(
            block_ids=local_block_ids,
            dst_num_blocks=self.dst_num_blocks[self.engine_id],
            block_size_ratio=block_size_ratio,
            physical_blocks_per_logical=self._physical_blocks_per_logical_kv_block,
        )

        assert len(local_block_descs_ids) == len(remote_block_descs_ids)

        # Prepare transfer with Nixl.
        handle = None
        try:
            handle = self.nixl_wrapper.make_prepped_xfer(
                "READ",
                local_xfer_side_handle,
                local_block_descs_ids,
                remote_xfer_side_handle,
                remote_block_descs_ids,
                notif_msg=notif_id,
            )

            # Begin async xfer.
            self.nixl_wrapper.transfer(handle)

            # Use handle to check completion in future step().
            if self.staging_config.enabled:
                self._staging_direct_transfers[request_id].append(handle)
            else:
                self._recving_transfers[request_id].append(handle)
        except Exception as e:
            # mark all (logical) blocks for this request as invalid
            self._log_failure(
                failure_type="transfer_setup_failed",
                req_id=request_id,
                msg="Marking blocks as invalid",
                error=e,
                dst_engine_id=dst_engine_id,
                remote_rank=remote_rank,
            )
            self._handle_failed_transfer(request_id, handle)

    def _get_new_notifs(self) -> set[str]:
        """
        Get req_ids which got a remote xfer message. When multiple consumers
        are reading from the same producer (heterogeneous TP scenario), wait
        for all consumers to be done pulling.

        Also handles heartbeat notifications ("HB:req1,req2,...") by
        extending the lease on the referenced requests.
        """
        assert self.transfer_topo is not None
        notified_req_ids = self._staging_done_sending
        self._staging_done_sending = set()
        while True:
            try:
                notified_req_ids.add(self._staging_done_sending_queue.get_nowait())
            except queue.Empty:
                break
        for req_id in notified_req_ids:
            self._reqs_to_process.discard(req_id)
            self._reqs_to_send.pop(req_id, None)
        if self.staging_config.enabled:
            pending_notifs: list[bytes] = []
            while True:
                try:
                    pending_notifs.append(self._staging_legacy_notifs.get_nowait())
                except queue.Empty:
                    break
            notification_batches = (pending_notifs,)
        else:
            notification_batches = self.nixl_wrapper.get_new_notifs().values()
        for notifs in notification_batches:
            for notif in notifs:
                msg = notif.decode("utf-8")

                # Handle heartbeat messages from D-side.
                if msg.startswith("HB:"):
                    self._handle_heartbeat(msg[3:])
                    continue

                req_id, tp_size = msg.rsplit(":", 1)
                if (
                    req_id not in self._reqs_to_send
                    and req_id not in self._reqs_to_process
                ):
                    logger.error(
                        "Potentially invalid KV blocks for "
                        "unrecognized request %s were retrieved by "
                        "a decode worker. They may have expired.",
                        req_id,
                    )
                    continue

                # NOTE: `tp_ratio` is the opposite when swapping local<>remote
                n_consumers = int(tp_size)
                tp_ratio = self.transfer_topo.tp_ratio(n_consumers)

                # Number of reads *per producer* to wait for.
                # When remote D TP > local P TP we expect `tp_ratio` reads.
                consumers_per_producer = (
                    -tp_ratio if n_consumers > self.world_size else 1
                )

                self.consumer_notification_counts_by_req[req_id] += 1
                # Wait all consumers (D) to be done reading before freeing.
                if (
                    self.consumer_notification_counts_by_req[req_id]
                    == consumers_per_producer
                ):
                    notified_req_ids.add(req_id)
                    del self.consumer_notification_counts_by_req[req_id]
                    self._reqs_to_process.remove(req_id)
                    self._reqs_to_send.pop(req_id, None)
        return notified_req_ids
