# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Push-specific (WRITE) worker-side logic for the NIXL connector.

A dedicated ``nixl-push-writer`` thread owns all push-related NIXL ops:
calls ``get_new_notifs`` (routing PUSH_REG internally; HB / completion
notifs are forwarded to the engine main thread), sends PUSH_REG via
``send_notif``, matches D registrations with P finished blocks, and
issues WRITE transfers via ``make_prepped_xfer`` / ``transfer``.

The engine main thread feeds the writer through queues:
``_reg_send_inbox`` (D-side regs to send), ``_finished_blocks_inbox``
(P-side blocks from metadata) and ``_pending_completion_notifs``
(non-PUSH_REG notifs forwarded back for HB / completion accounting).
The handshake-completion callback feeds ``_deferred_push_inbox`` with
matched pushes whose P→D handshake has finished so the writer can
(re-)issue the WRITE without ever blocking on the network.

Wake model: the writer self-polls every
``_PUSH_WRITER_POLL_INTERVAL_MS`` only while it has unmatched
``_push_finished_blocks`` (i.e. P-side blocks waiting for a D PUSH_REG
notif that has no other wake source). All other progress is
event-driven: the engine main thread sets ``_push_writer_wake`` from
``start_load_kv`` (when handing it new work) and from ``get_finished``
(so each engine step gives the writer a chance to drain NIXL notifs);
the handshake-completion callback sets the same event after a deferred
PUSH_REG send or a deferred push WRITE has been queued. When a request's
lease expires (the base worker reports it via ``done_sending``) or the WRITE completes,
``get_finished`` enqueues an eviction onto ``_evict_finished_inbox`` so
the writer drops any leftover ``_push_finished_blocks`` /
``_pending_d_registrations`` and stops self-polling.
"""

import queue
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import msgspec
import numpy as np
import torch

from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    PUSH_REG_NOTIF_PREFIX,
    NixlAgentMetadata,
    NixlConnectorMetadata,
    RemoteMeta,
    ReqId,
    ReqMeta,
    TransferHandle,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.staging import (
    STAGE_ACK_NOTIF_PREFIX,
    STAGE_DATA_NOTIF_PREFIX,
    STAGE_RELEASE_NOTIF_PREFIX,
    STAGING_PROTOCOL_VERSION,
    NixlStagingConfig,
    RemoteStagingRegion,
    StagingCredit,
    StagingSlotPool,
    gather_staging_blocks,
    scatter_staging_blocks,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    ReadSpec,
    _is_attention_spec,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import get_base_request_id
from vllm.logger import init_logger
from vllm.platforms import current_platform

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)

# Writer-thread poll cadence while there is in-flight push state. When
# fully idle, the writer blocks on a wake event signalled by the engine
# main thread (start_load_kv / get_finished). Smaller -> lower latency
# while active, slightly more CPU.
_PUSH_WRITER_POLL_INTERVAL_MS = 1.0


@dataclass
class _StagingSendState:
    request_id: str
    decode_request_id: str
    decode_engine_id: str
    remote_rank: int
    local_block_ids: list[int]
    blocks_per_chunk: int
    total_chunks: int
    credits: deque[StagingCredit]
    source_ready_event: torch.cuda.Event | None = None
    next_block: int = 0
    next_chunk: int = 0
    packing: int = 0
    nixl_inflight: int = 0
    acked_chunks: int = 0
    awaiting_acks: dict[tuple[int, int], int] = field(default_factory=dict)
    cancelled: bool = False


@dataclass
class _StagingPackTask:
    request_id: str
    local_slot: int
    remote_credit: StagingCredit
    chunk_index: int
    block_start: int
    block_count: int
    valid_bytes: int
    event: torch.cuda.Event
    indices: torch.Tensor


@dataclass
class _StagingNixlTask:
    request_id: str
    local_slot: int
    remote_credit: StagingCredit
    handle: int
    local_dlist: int
    remote_dlist: int


@dataclass
class _StagingScatterTask:
    decode_request_id: str
    producer_request_id: str
    producer_engine_id: str
    credit: StagingCredit
    total_chunks: int
    event: torch.cuda.Event
    indices: torch.Tensor


@dataclass
class _StagingRecvProgress:
    total_chunks: int
    seen_chunks: set[int] = field(default_factory=set)
    completed_chunks: int = 0


class NixlPushConnectorWorker(NixlBaseConnectorWorker):
    """Push-specific (WRITE) worker logic. See module docstring."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)

        # Heartbeat handshakes to a PP-sharded producer must be notif-only,
        # like the PUSH_REG path.
        self._hb_handshake_notif_only = True

        # Push-specific state.
        # P-side: outgoing WRITE handles awaiting completion, keyed by
        # request_id. Mutated by writer (submit) and main thread
        # (``_pop_done_transfers``); guarded by
        # ``_sending_transfers_lock``.
        self._sending_transfers = defaultdict[ReqId, list[TransferHandle]](list)
        self._sending_transfers_lock = threading.Lock()

        # Writer-thread owned matching state.
        # P-side: finished request blocks received from scheduler metadata
        # that have not yet been matched with an incoming D registration.
        self._push_finished_blocks: dict[ReqId, BlockIds] = {}
        # P-side: D registrations received via NIXL notification that have
        # not yet been matched with a finished P request.
        self._pending_d_registrations: dict[ReqId, dict[str, Any]] = {}

        # Cross-thread channels.
        self._reg_send_inbox: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()
        self._finished_blocks_inbox: queue.Queue[tuple[str, BlockIds]] = queue.Queue()
        self._pending_completion_notifs: queue.Queue[bytes] = queue.Queue()
        # Main thread → writer: req_ids whose lease has expired or whose
        # WRITE has completed. Writer drops them from
        # ``_push_finished_blocks`` so an unmatched entry doesn't keep the
        # writer busy-polling forever.
        self._evict_finished_inbox: queue.Queue[str] = queue.Queue()
        # Handshakes that have just completed and are ready for the WRITE on wthread
        self._deferred_push_inbox = queue.Queue[tuple[str, BlockIds, dict[str, Any]]]()

        # Wake signal from engine main thread (start_load_kv / get_finished).
        # Writer self-polls at _PUSH_WRITER_POLL_INTERVAL_MS while it has
        # active in-flight state; otherwise it blocks until signalled.
        self._push_writer_wake = threading.Event()

        self._push_writer_stop = threading.Event()
        self._push_writer_thread: threading.Thread | None = None

        total_memory = current_platform.get_device_total_memory()
        self._staging_config = NixlStagingConfig.from_vllm_config(
            vllm_config, total_memory
        )
        self._staging_enabled = self._staging_config.active
        self._staging_pool: StagingSlotPool | None = None
        self._staging_copy_stream: torch.cuda.Stream | None = None
        self._staging_caches: tuple[torch.Tensor, ...] = ()
        self._staging_page_bytes: tuple[int, ...] = ()
        self._staging_block_bytes = 0

        # Metadata learned during full P->D handshakes.
        self._remote_staging_regions: dict[tuple[str, int], RemoteStagingRegion] = {}
        self._remote_staging_block_lens: dict[tuple[str, int], tuple[int, ...]] = {}
        # Feature bit learned by D's notification-only handshake to P.
        self._remote_staging_capable: dict[str, bool] = {}

        # D-side receive-slot ownership. Epochs advance only after scatter.
        self._staging_recv_epochs: list[int] = []
        self._staging_recv_owners: dict[int, tuple[str, int]] = {}
        self._staging_recv_busy: set[int] = set()
        self._staging_registration_credits: dict[str, list[StagingCredit]] = {}
        self._staging_waiting_registrations: dict[str, dict[str, Any]] = {}
        self._staging_recv_progress: dict[str, _StagingRecvProgress] = {}

        # P-side bounded send pipeline.
        self._staging_sends: dict[str, _StagingSendState] = {}
        self._staging_send_order: deque[str] = deque()
        self._staging_pack_tasks: list[_StagingPackTask] = []
        self._staging_nixl_tasks: dict[int, _StagingNixlTask] = {}
        self._staging_scatter_tasks: list[_StagingScatterTask] = []
        self._staging_done_sending: queue.Queue[str] = queue.Queue()
        self._staging_done_recving: queue.Queue[str] = queue.Queue()
        self._staging_source_ready_events: dict[str, torch.cuda.Event] = {}

    # --- Lifecycle ----------------------------------------------------- #

    def register_kv_caches(self, kv_caches: dict[str, "torch.Tensor"]):
        super().register_kv_caches(kv_caches)
        if self._staging_enabled:
            self._initialize_staging(kv_caches)
        if self._push_writer_thread is None:
            self._push_writer_thread = threading.Thread(
                target=self._push_writer_loop,
                daemon=True,
                name="nixl-push-writer",
            )
            self._push_writer_thread.start()
            logger.info("nixl-push-writer thread started (rank=%d)", self.tp_rank)

    def _initialize_staging(self, kv_caches: dict[str, torch.Tensor]) -> None:
        reason = self._staging_unsupported_reason(kv_caches)
        if reason is not None:
            self._disable_or_raise_staging(reason)
            return

        assert self._staging_config.pool_bytes > 0
        first_cache = next(iter(kv_caches.values()))
        try:
            staging_tensor = torch.empty(
                self._staging_config.pool_bytes,
                dtype=torch.uint8,
                device=first_cache.device,
            )
        except torch.OutOfMemoryError:
            if self._staging_config.fallback == "fail":
                raise
            logger.exception(
                "Disabling NIXL GPU staging because its buffer allocation failed"
            )
            self._staging_enabled = False
            return

        self._staging_pool = StagingSlotPool(
            staging_tensor, self._staging_config.slot_bytes
        )
        self._staging_copy_stream = torch.cuda.Stream(device=first_cache.device)
        self._staging_recv_epochs = [0] * self._staging_pool.slot_count

        caches: list[torch.Tensor] = []
        seen_ptrs: set[int] = set()
        for cache in kv_caches.values():
            if cache.data_ptr() in seen_ptrs:
                continue
            seen_ptrs.add(cache.data_ptr())
            caches.append(cache)
        self._staging_caches = tuple(caches)
        self._staging_page_bytes = tuple(
            cache[0].numel() * cache.element_size() for cache in caches
        )
        self._staging_block_bytes = sum(self._staging_page_bytes)

        staging_data = [
            (
                staging_tensor.data_ptr(),
                staging_tensor.numel(),
                self.device_id,
                "",
            )
        ]
        descs = self.nixl_wrapper.get_reg_descs(staging_data, self.nixl_memory_type)
        self.nixl_wrapper.register_memory(descs, backends=self.nixl_backends)
        self._registered_descs.append(descs)
        self._advertise_staging_region()
        logger.info(
            "NIXL GPU staging enabled: pool=%d bytes, slot=%d bytes, slots=%d, "
            "KV bytes per physical block=%d",
            staging_tensor.numel(),
            self._staging_pool.slot_bytes,
            self._staging_pool.slot_count,
            self._staging_block_bytes,
        )

    def _staging_unsupported_reason(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> str | None:
        if self.device_type != "cuda" or self.kv_buffer_device != "cuda":
            return "the first staging implementation requires CUDA device buffers"
        if self.kv_transfer_config.kv_role == "kv_both":
            return "kv_both staging pools are not implemented"
        if self.pp_size != 1:
            return "pipeline-parallel staging transfers are not implemented"
        if self.use_mla or self._has_mamba or self._is_hma_required:
            return "MLA, Mamba, and HMA staging copy plans are not implemented"
        if self.kv_cache_layout != "HND":
            return "the first staging implementation requires HND KV layout"
        if len(self.kv_cache_config.kv_cache_groups) != 1:
            return "the first staging implementation requires one KV cache group"
        if self._physical_blocks_per_logical_kv_block != 1:
            return "heterogeneous physical/logical block staging is not implemented"

        seen_ptrs: set[int] = set()
        unique_caches: list[torch.Tensor] = []
        for cache in kv_caches.values():
            if cache.data_ptr() in seen_ptrs:
                continue
            seen_ptrs.add(cache.data_ptr())
            unique_caches.append(cache)
        if len(unique_caches) != self.num_regions:
            return "packed KV cache layouts are not supported by staging"
        for cache, block_len in zip(unique_caches, self.block_len_per_layer):
            if (
                cache.device.type != "cuda"
                or not cache.is_contiguous()
                or cache.ndim == 0
                or cache.shape[0] != self.num_blocks
            ):
                return "staging requires contiguous block-major CUDA KV tensors"
            page_bytes = cache[0].numel() * cache.element_size()
            if page_bytes != block_len:
                return "KV tensor page geometry does not match NIXL regions"
        return None

    def _disable_or_raise_staging(self, reason: str) -> None:
        if self._staging_config.fallback == "fail":
            raise NotImplementedError(f"NIXL GPU staging is unsupported: {reason}")
        logger.warning("NIXL GPU staging disabled: %s; using direct WRITE", reason)
        self._staging_enabled = False

    def _advertise_staging_region(self) -> None:
        assert self._staging_pool is not None
        assert self.xfer_handshake_metadata is not None
        decoder = msgspec.msgpack.Decoder(NixlAgentMetadata)
        metadata = decoder.decode(self.xfer_handshake_metadata.agent_metadata_bytes)
        metadata.staging_protocol_version = STAGING_PROTOCOL_VERSION
        metadata.staging_buffer_base_addr = self._staging_pool.tensor.data_ptr()
        metadata.staging_buffer_size = self._staging_pool.tensor.numel()
        metadata.staging_slot_size = self._staging_pool.slot_bytes
        metadata.staging_slot_count = self._staging_pool.slot_count
        self.xfer_handshake_metadata = type(self.xfer_handshake_metadata)(
            compatibility_hash=self.xfer_handshake_metadata.compatibility_hash,
            agent_metadata_bytes=msgspec.msgpack.encode(metadata),
        )

    def add_remote_agent(
        self,
        nixl_agent_meta: NixlAgentMetadata,
        remote_tp_rank: int = 0,
        remote_tp_size: int = 1,
    ) -> str:
        agent_name = super().add_remote_agent(
            nixl_agent_meta, remote_tp_rank, remote_tp_size
        )
        region = RemoteStagingRegion(
            base_addr=nixl_agent_meta.staging_buffer_base_addr,
            pool_bytes=nixl_agent_meta.staging_buffer_size,
            slot_bytes=nixl_agent_meta.staging_slot_size,
            slot_count=nixl_agent_meta.staging_slot_count,
            device_id=nixl_agent_meta.device_id,
            protocol_version=nixl_agent_meta.staging_protocol_version,
        )
        key = (nixl_agent_meta.engine_id, remote_tp_rank)
        self._remote_staging_regions[key] = region
        self._remote_staging_block_lens[key] = tuple(nixl_agent_meta.block_lens)
        return agent_name

    def _add_notif_only_remote_agent(
        self, metadata: NixlAgentMetadata, remote_tp_size: int
    ) -> str:
        agent_name = super()._add_notif_only_remote_agent(metadata, remote_tp_size)
        self._remote_staging_capable[metadata.engine_id] = (
            metadata.staging_protocol_version == STAGING_PROTOCOL_VERSION
            and metadata.staging_buffer_base_addr > 0
        )
        return agent_name

    def _cleanup_remote_engine(self, engine_id: str, log_eviction: bool = True) -> None:
        super()._cleanup_remote_engine(engine_id, log_eviction=log_eviction)
        self._remote_staging_capable.pop(engine_id, None)
        for key in [key for key in self._remote_staging_regions if key[0] == engine_id]:
            self._remote_staging_regions.pop(key, None)
            self._remote_staging_block_lens.pop(key, None)

    def shutdown(self):
        self._push_writer_stop.set()
        # Unblock the writer if it's waiting in the no-active-state branch.
        self._push_writer_wake.set()
        if self._push_writer_thread is not None:
            self._push_writer_thread.join(timeout=2)
            self._push_writer_thread = None
        with self._sending_transfers_lock:
            for handles in self._sending_transfers.values():
                for handle in handles:
                    self.nixl_wrapper.release_xfer_handle(handle)
            self._sending_transfers.clear()
        for task in getattr(self, "_staging_nixl_tasks", {}).values():
            self.nixl_wrapper.release_xfer_handle(task.handle)
            self.nixl_wrapper.release_dlist_handle(task.local_dlist)
            self.nixl_wrapper.release_dlist_handle(task.remote_dlist)
        if hasattr(self, "_staging_nixl_tasks"):
            self._staging_nixl_tasks.clear()
        super().shutdown()

    # --- Engine-main-thread entry point -------------------------------- #

    def start_load_kv(self, metadata: NixlConnectorMetadata):
        """Pre-process metadata; defer NIXL ops to the writer thread."""
        # D-side: track reqs waiting for P to push.
        for req_id, meta in metadata.reqs_to_recv.items():
            meta.local_physical_block_ids = self._logical_to_kernel_block_ids(
                meta.local_block_ids, self._physical_blocks_per_logical_kv_block
            )
            assert meta.remote is not None
            remote_engine_id = meta.remote.engine_id
            logger.debug(
                "start_load_kv (push) for request %s from remote engine %s. "
                "Num local_block_ids: %s. Num remote_block_ids: %s. ",
                req_id,
                remote_engine_id,
                len(meta.local_physical_block_ids),
                len(meta.remote.block_ids),
            )
            self._recving_metadata[req_id] = meta

        # --- D-side: registrations to send to P via NIXL ---
        if metadata.push_registrations:
            for req_id, reg_data in metadata.push_registrations.items():
                self._reg_send_inbox.put((req_id, reg_data))
            self._push_writer_wake.set()

        # --- P-side: newly finished blocks awaiting a D registration match ---
        if metadata.push_finished_blocks:
            for req_id, block_ids in metadata.push_finished_blocks.items():
                if getattr(self, "_staging_enabled", False):
                    ready_event = torch.cuda.Event()
                    ready_event.record(torch.cuda.current_stream(self.device_id))
                    self._staging_source_ready_events[req_id] = ready_event
                self._finished_blocks_inbox.put((req_id, block_ids))
            self._push_writer_wake.set()

        # Batch + lease tracking (same as pull).
        for req_id in metadata.reqs_in_batch:
            self._reqs_to_process.add(req_id)
        for req_id in metadata.reqs_not_processed:
            self._reqs_to_process.discard(req_id)
            assert req_id not in self._reqs_to_send
        # Rebase scheduler-clock deadlines onto this worker's clock — see the
        # equivalent block in pull_worker.start_load_kv for the rationale.
        now_local = time.perf_counter()
        for req_id, expiration_time in metadata.reqs_to_send.items():
            if req_id in self._reqs_to_process:
                if metadata.scheduler_clock:
                    expiration_time = now_local + (
                        expiration_time - metadata.scheduler_clock
                    )
                self._reqs_to_send[req_id] = expiration_time

        # Heartbeats still leave from the main thread (base worker behaviour).
        self._send_heartbeats(metadata)

    # --- Writer thread ------------------------------------------------- #

    def _push_writer_loop(self) -> None:
        sleep_s = _PUSH_WRITER_POLL_INTERVAL_MS / 1000.0
        if getattr(self, "_staging_enabled", False):
            current_platform.set_device(self.device_id)

        while not self._push_writer_stop.is_set():
            try:
                self._retry_waiting_staging_registration()

                # 1. D registrations to send.
                while True:
                    try:
                        rid, rd = self._reg_send_inbox.get_nowait()
                    except queue.Empty:
                        break
                    self._send_registration_to_p(rid, rd)

                # 2. Deferred P→D pushes whose handshake just completed; do xfer now
                while True:
                    try:
                        rid, blocks, rd = self._deferred_push_inbox.get_nowait()
                    except queue.Empty:
                        break
                    self._do_start_push_kv(rid, blocks, rd)

                # 3. P-side finished blocks; match against pending regs.
                while True:
                    try:
                        rid, blocks = self._finished_blocks_inbox.get_nowait()
                    except queue.Empty:
                        break
                    matched = self._pop_matching_registration(rid)
                    if matched is not None:
                        self._do_start_push_kv(rid, blocks, matched)
                    else:
                        self._push_finished_blocks[rid] = blocks

                # 3b. Evict finished blocks for requests that have either
                # completed (WRITE acknowledged) or whose lease expired
                # without a D registration.  Drop pending registrations
                # for the same reason so we don't leak state.
                while True:
                    try:
                        rid = self._evict_finished_inbox.get_nowait()
                    except queue.Empty:
                        break
                    self._push_finished_blocks.pop(rid, None)
                    self._pending_d_registrations.pop(rid, None)
                    getattr(self, "_staging_source_ready_events", {}).pop(rid, None)
                    if rid in getattr(self, "_staging_sends", {}):
                        self._fail_staging_send(rid)

                # 4. NIXL notifs: route push/staging control messages.
                for notifs in self.nixl_wrapper.get_new_notifs().values():
                    for notif in notifs:
                        if notif.startswith(PUSH_REG_NOTIF_PREFIX):
                            self._handle_push_reg_notif(notif)
                        elif notif.startswith(STAGE_DATA_NOTIF_PREFIX):
                            self._handle_staging_data_notif(notif)
                        elif notif.startswith(STAGE_ACK_NOTIF_PREFIX):
                            self._handle_staging_ack_notif(notif)
                        elif notif.startswith(STAGE_RELEASE_NOTIF_PREFIX):
                            self._handle_staging_release_notif(notif)
                        else:
                            self._pending_completion_notifs.put(notif)

                self._progress_staging()
            except Exception:
                logger.exception("nixl-push-writer error; continuing")

            # Self-poll only while there is no other wake source: P-side
            # finished blocks waiting for a D PUSH_REG match. All other
            # progress is event-driven (see module docstring).
            if self._push_finished_blocks or self._has_active_staging_work():
                self._push_writer_stop.wait(timeout=sleep_s)
            else:
                self._push_writer_wake.wait()
                self._push_writer_wake.clear()

    def _handle_push_reg_notif(self, notif: bytes) -> None:
        try:
            reg_data = msgspec.msgpack.decode(notif[len(PUSH_REG_NOTIF_PREFIX) :])
        except Exception:
            logger.exception("Failed to decode PUSH_REG notification payload")
            return
        rid = reg_data.get("request_id") if isinstance(reg_data, dict) else None
        if not isinstance(rid, str):
            logger.warning("PUSH_REG notif missing request_id; dropping")
            return

        match = self._pop_matching_finished_blocks(rid)
        if match is not None:
            fin_id, blocks = match
            self._do_start_push_kv(fin_id, blocks, reg_data)
        else:
            self._pending_d_registrations[rid] = reg_data

    # --- D-side registration send (writer thread) ---------------------- #

    def _send_registration_to_p(
        self,
        req_id: str,
        reg_data: dict[str, Any],
    ) -> None:
        """Handshake (if needed) then send PUSH_REG. ``send_notif`` always
        executes on the writer; the handshake runs on the background executor
        and the request is re-queued onto ``_reg_send_inbox`` once it
        completes (at which point ``_ensure_handshake`` returns ``None`` and we
        send directly)."""
        remote_pp_size = reg_data.get("remote_pp_size", 1)
        fut = self._ensure_handshake(
            reg_data["remote_engine_id"],
            reg_data["remote_host"],
            reg_data["remote_port"],
            reg_data["remote_tp_size"],
            pp_size=remote_pp_size,
            # D only ever sends PUSH_REG notifs to P and never reads or writes
            # P's memory in push mode, so it never needs the transfer
            # descriptors set up by the full add_remote_agent path.
            notif_agents_only=True,
        )
        if fut is None:
            self._do_send_reg_notif(req_id, reg_data)
            return

        def _on_handshake(
            f: Future[tuple[dict[tuple[int, int], str], float]],
            rid: str = req_id,
            rd: dict[str, Any] = reg_data,
        ) -> None:
            try:
                f.result()
            except Exception as e:
                self._log_failure(
                    failure_type="push_reg_handshake_failed", req_id=rid, error=e
                )
                self._handle_failed_transfer(rid, None)
                return
            # Re-queue for the writer to send now that the handshake is done.
            self._reg_send_inbox.put((rid, rd))
            # Wake the writer so it sends the PUSH_REG promptly even if
            # otherwise parked.
            self._push_writer_wake.set()

        fut.add_done_callback(_on_handshake)

    def _do_send_reg_notif(self, req_id: str, reg_data: dict[str, Any]) -> None:
        engine_id = reg_data["remote_engine_id"]
        agents = self._remote_agents.get(engine_id)
        if not agents:
            logger.error(
                "No remote agents for engine %s; cannot send registration for %s",
                engine_id,
                req_id,
            )
            self._handle_failed_transfer(req_id, None)
            return
        prepared_reg = self._prepare_staging_registration(req_id, reg_data)
        if prepared_reg is None:
            self._staging_waiting_registrations[req_id] = reg_data
            return
        notif_msg = PUSH_REG_NOTIF_PREFIX + msgspec.msgpack.encode(prepared_reg)
        for rank, agent_name in agents.items():
            try:
                self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_msg)
            except Exception as e:
                self._log_failure(
                    failure_type="push_reg_notif_failed",
                    req_id=req_id,
                    error=e,
                    remote_rank=rank,
                )
        logger.debug(
            "Sent PUSH_REG for %s to engine %s (%dB)", req_id, engine_id, len(notif_msg)
        )

    def _prepare_staging_registration(
        self, req_id: str, reg_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        if not getattr(
            self, "_staging_enabled", False
        ) or not self._remote_staging_capable.get(reg_data["remote_engine_id"], False):
            return reg_data
        assert self._staging_pool is not None

        credits = self._staging_registration_credits.get(req_id)
        if credits is None:
            credits = []
            max_credits = self._staging_config.max_inflight_per_request
            while len(credits) < max_credits:
                slot_id = self._staging_pool.acquire()
                if slot_id is None:
                    break
                epoch = self._staging_recv_epochs[slot_id]
                self._staging_recv_owners[slot_id] = (req_id, epoch)
                credits.append(StagingCredit(slot_id, epoch))
            if not credits:
                return None
            self._staging_registration_credits[req_id] = credits

        prepared = dict(reg_data)
        prepared["staging_protocol_version"] = STAGING_PROTOCOL_VERSION
        prepared["staging_credits"] = [
            {"slot_id": credit.slot_id, "epoch": credit.epoch} for credit in credits
        ]
        return prepared

    def _retry_waiting_staging_registration(self) -> None:
        if not getattr(self, "_staging_waiting_registrations", None):
            return
        assert self._staging_pool is not None
        if self._staging_pool.num_free == 0:
            return
        req_id = next(iter(self._staging_waiting_registrations))
        reg_data = self._staging_waiting_registrations.pop(req_id)
        self._do_send_reg_notif(req_id, reg_data)

    # --- Matching helpers --------------------------------------------- #

    def _pop_matching_registration(self, request_id: str) -> dict[str, Any] | None:
        """Pop the D-side registration matching *request_id*.

        Exact key first, then a match after stripping the random suffix from
        both sides. No match leaves the request unmatched (push not started).
        """
        data = self._pending_d_registrations.pop(request_id, None)
        if data is not None:
            return data
        base_id = get_base_request_id(request_id)
        for reg_id in list(self._pending_d_registrations):
            if get_base_request_id(reg_id) == base_id:
                return self._pending_d_registrations.pop(reg_id)
        return None

    def _pop_matching_finished_blocks(
        self, request_id: str
    ) -> tuple[str, BlockIds] | None:
        """Pop the P-side finished blocks matching *request_id*.

        Same lookup as ``_pop_matching_registration``: exact key, then a
        match after stripping the random suffix from both sides.
        """
        blocks = self._push_finished_blocks.pop(request_id, None)
        if blocks is not None:
            return request_id, blocks
        base_id = get_base_request_id(request_id)
        for fin_id in list(self._push_finished_blocks):
            if get_base_request_id(fin_id) == base_id:
                return fin_id, self._push_finished_blocks.pop(fin_id)
        return None

    # --- WRITE transfer logic (writer thread) ------------------------- #

    def _do_start_push_kv(
        self,
        request_id: str,
        local_block_ids: BlockIds,
        registration_data: dict[str, Any],
    ) -> None:
        """Start push-based KV transfer from P worker to D node.

        The P→D handshake runs on the base worker's background executor.
        If it isn't ready yet we register a completion callback, defer the
        WRITE, and re-drive this request via ``_deferred_push_inbox`` once
        the handshake resolves -- so the writer thread never blocks on the
        network (mirrors ``_send_registration_to_p``).
        """
        if not local_block_ids:
            logger.warning("No local blocks to push for request %s", request_id)
            getattr(self, "_staging_source_ready_events", {}).pop(request_id, None)
            return

        # ``local_block_ids`` are P's logical block IDs; ``remote_block_ids``
        # (D's, from the PUSH_REG notif) are also logical.
        decode_engine_id = registration_data["decode_engine_id"]
        remote_block_ids = registration_data["local_block_ids"]
        decode_request_id = registration_data["request_id"]

        # Runs on the background executor; defer the WRITE until it's ready.
        fut = self._ensure_handshake(
            decode_engine_id,
            registration_data["decode_host"],
            registration_data["decode_port"],
            registration_data["decode_tp_size"],
        )
        if fut is not None:

            def _on_handshake(
                f: Future[tuple[dict[tuple[int, int], str], float]],
                rid: str = request_id,
                blocks: BlockIds = local_block_ids,
                rd: dict[str, Any] = registration_data,
            ) -> None:
                if (e := f.exception()) is not None:
                    # The engine reclaims the blocks via the TTL so we dont free here
                    self._log_failure(
                        failure_type="push_handshake_failed", req_id=rid, error=e
                    )
                    getattr(self, "_staging_source_ready_events", {}).pop(rid, None)
                    return
                self._deferred_push_inbox.put((rid, blocks, rd))
                self._push_writer_wake.set()

            fut.add_done_callback(_on_handshake)
            return

        # Both sides stay logical here; ``_xfer_blocks_for_req`` converts each
        # to physical with its own physical-blocks-per-logical ratio -- P uses
        # ``self._physical_blocks_per_logical_kv_block``, D's is learned during
        # the NIXL handshake.
        logical_local = self._as_grouped_block_ids(local_block_ids)
        logical_remote = self._as_grouped_block_ids(remote_block_ids)
        physical_local = self._logical_to_kernel_block_ids(
            logical_local, self._physical_blocks_per_logical_kv_block
        )

        push_meta = ReqMeta(
            local_block_ids=logical_local,
            local_physical_block_ids=physical_local,
            tp_size=self.world_size,
            remote=RemoteMeta(
                block_ids=logical_remote,
                host="",
                port=0,
                engine_id=decode_engine_id,
                request_id=decode_request_id,
            ),
        )

        if self._try_start_staging_send(request_id, push_meta, registration_data):
            return

        getattr(self, "_staging_source_ready_events", {}).pop(request_id, None)
        t0 = time.perf_counter()
        self._xfer_blocks_for_req(req_id=request_id, meta=push_meta)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if elapsed_ms > 200.0:
            logger.warning(
                "_do_start_push_kv for %s took %.1fms (slow NIXL submission)",
                request_id,
                elapsed_ms,
            )

    @staticmethod
    def _as_grouped_block_ids(block_ids: BlockIds) -> BlockIds:
        """Normalise a sequence of block IDs to a tuple-of-groups shape.

        ``BlockIds`` is canonically a tuple of per-group lists, but some
        registration payloads collapse a single-group case to a flat
        list. Re-wrap that case so downstream group-aware helpers see a
        consistent shape."""
        if block_ids and not isinstance(block_ids[0], (list, tuple)):
            return (list(block_ids),)
        return block_ids

    def _try_start_staging_send(
        self,
        request_id: str,
        meta: ReqMeta,
        registration_data: dict[str, Any],
    ) -> bool:
        raw_credits = registration_data.get("staging_credits")
        if not raw_credits:
            return False

        try:
            credits = deque(
                StagingCredit(int(item["slot_id"]), int(item["epoch"]))
                for item in raw_credits
            )
        except (KeyError, TypeError, ValueError):
            logger.exception("Invalid NIXL staging credits for %s", request_id)
            return False
        if (
            not getattr(self, "_staging_enabled", False)
            or registration_data.get("staging_protocol_version")
            != STAGING_PROTOCOL_VERSION
        ):
            self._release_remote_staging_credits(
                registration_data["decode_engine_id"],
                registration_data["request_id"],
                credits,
            )
            return False

        assert meta.remote is not None and self.transfer_topo is not None
        remote_info = self.transfer_topo.get_engine_info(meta.remote.engine_id)
        plan = self.tp_mappings[meta.remote.engine_id]
        supported = (
            remote_info.remote_tp_size == self.world_size
            and remote_info.remote_block_size == self.block_size
            and remote_info.remote_physical_blocks_per_logical
            == self._physical_blocks_per_logical_kv_block
            and len(plan.all_source_ranks) == 1
            and len(meta.local_physical_block_ids) == 1
            and len(meta.remote.block_ids) == 1
        )
        remote_rank = plan.all_source_ranks[0] if supported else -1
        remote_region = self._remote_staging_regions.get(
            (meta.remote.engine_id, remote_rank)
        )
        remote_block_lens = self._remote_staging_block_lens.get(
            (meta.remote.engine_id, remote_rank)
        )
        supported = (
            supported
            and remote_region is not None
            and remote_region.enabled
            and remote_block_lens == self._staging_page_bytes
        )
        if supported:
            assert remote_region is not None
            credit_keys = {(credit.slot_id, credit.epoch) for credit in credits}
            supported = len(credit_keys) == len(credits) and all(
                0 <= credit.slot_id < remote_region.slot_count and credit.epoch >= 0
                for credit in credits
            )
        if not supported:
            self._release_remote_staging_credits(
                meta.remote.engine_id, meta.remote.request_id, credits
            )
            if self._staging_config.fallback == "fail":
                raise NotImplementedError(
                    "NIXL GPU staging requires homogeneous TP, block size, and "
                    "KV page geometry"
                )
            return False

        assert remote_region is not None
        payload_bytes = min(self._staging_config.slot_bytes, remote_region.slot_bytes)
        blocks_per_chunk = payload_bytes // self._staging_block_bytes
        if blocks_per_chunk == 0:
            self._release_remote_staging_credits(
                meta.remote.engine_id, meta.remote.request_id, credits
            )
            if self._staging_config.fallback == "fail":
                raise ValueError(
                    "NIXL staging slot is smaller than one complete KV block bundle"
                )
            return False

        local_blocks = list(meta.local_physical_block_ids[0])
        remote_blocks = list(meta.remote.block_ids[0])
        num_blocks = min(len(local_blocks), len(remote_blocks))
        local_blocks = local_blocks[:num_blocks]
        if not local_blocks:
            self._release_remote_staging_credits(
                meta.remote.engine_id, meta.remote.request_id, credits
            )
            return False

        total_chunks = (num_blocks + blocks_per_chunk - 1) // blocks_per_chunk
        self._staging_sends[request_id] = _StagingSendState(
            request_id=request_id,
            decode_request_id=meta.remote.request_id,
            decode_engine_id=meta.remote.engine_id,
            remote_rank=remote_rank,
            local_block_ids=local_blocks,
            blocks_per_chunk=blocks_per_chunk,
            total_chunks=total_chunks,
            credits=credits,
            source_ready_event=getattr(self, "_staging_source_ready_events", {}).pop(
                request_id, None
            ),
        )
        self._staging_send_order.append(request_id)
        logger.debug(
            "Queued staged NIXL push %s: blocks=%d, chunks=%d, credits=%d",
            request_id,
            num_blocks,
            total_chunks,
            len(credits),
        )
        return True

    def _release_remote_staging_credits(
        self,
        decode_engine_id: str,
        decode_request_id: str,
        credits: deque[StagingCredit] | list[StagingCredit],
    ) -> None:
        if not credits:
            return
        agents = self._remote_agents.get(decode_engine_id, {})
        if not agents:
            return
        payload = {
            "request_id": decode_request_id,
            "credits": [
                {"slot_id": credit.slot_id, "epoch": credit.epoch} for credit in credits
            ],
        }
        notif = STAGE_RELEASE_NOTIF_PREFIX + msgspec.msgpack.encode(payload)
        for agent_name in agents.values():
            self.nixl_wrapper.send_notif(agent_name, notif_msg=notif)

    def _has_active_staging_work(self) -> bool:
        return bool(
            getattr(self, "_staging_sends", None)
            or getattr(self, "_staging_pack_tasks", None)
            or getattr(self, "_staging_nixl_tasks", None)
            or getattr(self, "_staging_scatter_tasks", None)
            or getattr(self, "_staging_waiting_registrations", None)
        )

    def _progress_staging(self) -> None:
        if not getattr(self, "_staging_enabled", False):
            return
        self._poll_staging_scatter_tasks()
        self._poll_staging_nixl_tasks()
        self._poll_staging_pack_tasks()
        self._schedule_staging_packs()

    def _schedule_staging_packs(self) -> None:
        if not self._staging_sends or not self._staging_send_order:
            return
        assert self._staging_pool is not None

        attempts = len(self._staging_send_order)
        while (
            attempts > 0
            and self._staging_pool.num_free > 0
            and len(self._staging_nixl_tasks) + len(self._staging_pack_tasks)
            < self._staging_config.max_inflight
        ):
            attempts -= 1
            request_id = self._staging_send_order.popleft()
            state = self._staging_sends.get(request_id)
            if state is None:
                continue
            self._staging_send_order.append(request_id)
            request_inflight = state.packing + state.nixl_inflight
            if (
                state.next_block >= len(state.local_block_ids)
                or not state.credits
                or request_inflight >= self._staging_config.max_inflight_per_request
            ):
                continue
            self._start_staging_gather(state)
            attempts = len(self._staging_send_order)

    def _start_staging_gather(self, state: _StagingSendState) -> None:
        assert self._staging_pool is not None
        assert self._staging_copy_stream is not None
        local_slot = self._staging_pool.acquire()
        if local_slot is None:
            return
        remote_credit = state.credits.popleft()
        block_start = state.next_block
        block_count = min(
            state.blocks_per_chunk,
            len(state.local_block_ids) - block_start,
        )
        block_ids = state.local_block_ids[block_start : block_start + block_count]
        valid_bytes = block_count * self._staging_block_bytes

        try:
            with torch.cuda.stream(self._staging_copy_stream):
                if state.source_ready_event is not None:
                    self._staging_copy_stream.wait_event(state.source_ready_event)
                indices = torch.tensor(
                    block_ids,
                    device=self._staging_pool.tensor.device,
                    dtype=torch.long,
                )
                slot = self._staging_pool.view(local_slot, valid_bytes)
                copied_bytes = gather_staging_blocks(
                    self._staging_caches, slot, indices
                )
                assert copied_bytes == valid_bytes
                event = torch.cuda.Event()
                event.record(self._staging_copy_stream)
        except Exception:
            self._staging_pool.release(local_slot)
            state.credits.appendleft(remote_credit)
            raise

        task = _StagingPackTask(
            request_id=state.request_id,
            local_slot=local_slot,
            remote_credit=remote_credit,
            chunk_index=state.next_chunk,
            block_start=block_start,
            block_count=block_count,
            valid_bytes=valid_bytes,
            event=event,
            indices=indices,
        )
        self._staging_pack_tasks.append(task)
        state.next_block += block_count
        state.next_chunk += 1
        state.packing += 1

    def _poll_staging_pack_tasks(self) -> None:
        pending: list[_StagingPackTask] = []
        for task in self._staging_pack_tasks:
            if not task.event.query():
                pending.append(task)
                continue
            state = self._staging_sends.get(task.request_id)
            if state is None:
                assert self._staging_pool is not None
                self._staging_pool.release(task.local_slot)
                continue
            if state.cancelled:
                assert self._staging_pool is not None
                self._staging_pool.release(task.local_slot)
                state.packing -= 1
                self._release_remote_staging_credits(
                    state.decode_engine_id,
                    state.decode_request_id,
                    [task.remote_credit],
                )
                self._finish_cancelled_staging_send(state)
                continue
            try:
                self._submit_staging_pack(state, task)
            except Exception as e:
                self._log_failure(
                    failure_type="staging_transfer_setup_failed",
                    req_id=task.request_id,
                    error=e,
                )
                assert self._staging_pool is not None
                self._staging_pool.release(task.local_slot)
                state.packing -= 1
                self._release_remote_staging_credits(
                    state.decode_engine_id,
                    state.decode_request_id,
                    [task.remote_credit],
                )
                self._fail_staging_send(task.request_id)
                self._finish_cancelled_staging_send(state)
        self._staging_pack_tasks = [
            task for task in pending if task.request_id in self._staging_sends
        ]

    def _submit_staging_pack(
        self, state: _StagingSendState, task: _StagingPackTask
    ) -> None:
        assert self._staging_pool is not None
        remote_region = self._remote_staging_regions[
            (state.decode_engine_id, state.remote_rank)
        ]
        agent_name = self._remote_agents[state.decode_engine_id][(0, state.remote_rank)]
        local_data = np.asarray(
            [
                (
                    self._staging_pool.address(task.local_slot),
                    task.valid_bytes,
                    self.device_id,
                )
            ],
            dtype=np.uint64,
        )
        remote_data = np.asarray(
            [
                (
                    remote_region.base_addr
                    + task.remote_credit.slot_id * remote_region.slot_bytes,
                    task.valid_bytes,
                    remote_region.device_id,
                )
            ],
            dtype=np.uint64,
        )
        local_descs = self.nixl_wrapper.get_xfer_descs(
            local_data, self.nixl_memory_type
        )
        remote_descs = self.nixl_wrapper.get_xfer_descs(
            remote_data, self.nixl_memory_type
        )
        payload = {
            "request_id": state.decode_request_id,
            "producer_request_id": state.request_id,
            "producer_engine_id": self.engine_id,
            "slot_id": task.remote_credit.slot_id,
            "epoch": task.remote_credit.epoch,
            "chunk_index": task.chunk_index,
            "total_chunks": state.total_chunks,
            "block_start": task.block_start,
            "block_count": task.block_count,
            "valid_bytes": task.valid_bytes,
            "producer_tp_size": self.world_size,
        }
        notif = STAGE_DATA_NOTIF_PREFIX + msgspec.msgpack.encode(payload)
        indexes = np.asarray([0], dtype=np.int64)
        handle = None
        local_dlist = None
        remote_dlist = None
        try:
            local_dlist = self.nixl_wrapper.prep_xfer_dlist(
                "NIXL_INIT_AGENT", local_descs
            )
            remote_dlist = self.nixl_wrapper.prep_xfer_dlist(agent_name, remote_descs)
            handle = self.nixl_wrapper.make_prepped_xfer(
                "WRITE",
                local_dlist,
                indexes,
                remote_dlist,
                indexes,
                notif_msg=notif,
            )
            self.nixl_wrapper.transfer(handle)
        except Exception:
            if handle is not None:
                self.nixl_wrapper.release_xfer_handle(handle)
            if local_dlist is not None:
                self.nixl_wrapper.release_dlist_handle(local_dlist)
            if remote_dlist is not None:
                self.nixl_wrapper.release_dlist_handle(remote_dlist)
            raise

        assert local_dlist is not None and remote_dlist is not None
        state.packing -= 1
        state.nixl_inflight += 1
        state.awaiting_acks[(task.remote_credit.slot_id, task.remote_credit.epoch)] = (
            task.chunk_index
        )
        self._staging_nixl_tasks[handle] = _StagingNixlTask(
            request_id=state.request_id,
            local_slot=task.local_slot,
            remote_credit=task.remote_credit,
            handle=handle,
            local_dlist=local_dlist,
            remote_dlist=remote_dlist,
        )

    def _poll_staging_nixl_tasks(self) -> None:
        assert self._staging_pool is not None
        for handle, task in list(self._staging_nixl_tasks.items()):
            try:
                xfer_state = self.nixl_wrapper.check_xfer_state(handle)
                if xfer_state == "PROC":
                    continue
                if xfer_state != "DONE":
                    raise RuntimeError(f"staged NIXL transfer state is {xfer_state}")
                telemetry = self.nixl_wrapper.get_xfer_telemetry(handle)
                self.xfer_stats.record_transfer(telemetry)
            except Exception as e:
                self._log_failure(
                    failure_type="staging_transfer_failed",
                    req_id=task.request_id,
                    error=e,
                )
                self.xfer_stats.record_failed_transfer()
                state = self._staging_sends.get(task.request_id)
                self._fail_staging_send(task.request_id)
                self.nixl_wrapper.release_xfer_handle(handle)
                self.nixl_wrapper.release_dlist_handle(task.local_dlist)
                self.nixl_wrapper.release_dlist_handle(task.remote_dlist)
                self._staging_pool.release(task.local_slot)
                del self._staging_nixl_tasks[handle]
                if state is not None:
                    state.nixl_inflight -= 1
                    state.awaiting_acks.pop(
                        (task.remote_credit.slot_id, task.remote_credit.epoch), None
                    )
                    self._release_remote_staging_credits(
                        state.decode_engine_id,
                        state.decode_request_id,
                        [task.remote_credit],
                    )
                    self._finish_cancelled_staging_send(state)
                continue

            self.nixl_wrapper.release_xfer_handle(handle)
            self.nixl_wrapper.release_dlist_handle(task.local_dlist)
            self.nixl_wrapper.release_dlist_handle(task.remote_dlist)
            self._staging_pool.release(task.local_slot)
            del self._staging_nixl_tasks[handle]
            if state := self._staging_sends.get(task.request_id):
                state.nixl_inflight -= 1
                if state.cancelled:
                    self._finish_cancelled_staging_send(state)
                else:
                    self._finish_staging_send_if_ready(state)

    def _fail_staging_send(self, request_id: str) -> None:
        state = self._staging_sends.get(request_id)
        if state is None:
            return
        if state.cancelled:
            return
        state.cancelled = True
        self._staging_send_order = deque(
            item for item in self._staging_send_order if item != request_id
        )
        if state.credits:
            self._release_remote_staging_credits(
                state.decode_engine_id, state.decode_request_id, state.credits
            )
        state.credits.clear()
        self._finish_cancelled_staging_send(state)

    def _finish_cancelled_staging_send(self, state: _StagingSendState) -> None:
        if state.packing or state.nixl_inflight or state.awaiting_acks:
            return
        self._staging_sends.pop(state.request_id, None)

    def _finish_staging_send_if_ready(self, state: _StagingSendState) -> None:
        if (
            state.acked_chunks != state.total_chunks
            or state.packing
            or state.nixl_inflight
        ):
            return
        self._release_remote_staging_credits(
            state.decode_engine_id, state.decode_request_id, state.credits
        )
        self._staging_sends.pop(state.request_id, None)
        self._staging_send_order = deque(
            item for item in self._staging_send_order if item != state.request_id
        )
        self._staging_done_sending.put(state.request_id)

    def _handle_staging_ack_notif(self, notif: bytes) -> None:
        try:
            payload = msgspec.msgpack.decode(notif[len(STAGE_ACK_NOTIF_PREFIX) :])
            request_id = payload["producer_request_id"]
            decode_request_id = payload["request_id"]
            decode_engine_id = payload["decode_engine_id"]
            old_credit = StagingCredit(int(payload["slot_id"]), int(payload["epoch"]))
            new_credit = StagingCredit(old_credit.slot_id, int(payload["next_epoch"]))
        except (KeyError, TypeError, ValueError, msgspec.DecodeError):
            logger.exception("Invalid NIXL staging ACK notification")
            return

        state = self._staging_sends.get(request_id)
        if state is None:
            self._release_remote_staging_credits(
                decode_engine_id, decode_request_id, [new_credit]
            )
            return
        key = (old_credit.slot_id, old_credit.epoch)
        if key not in state.awaiting_acks:
            logger.warning("Ignoring duplicate NIXL staging ACK for %s", key)
            return
        del state.awaiting_acks[key]
        if state.cancelled:
            self._release_remote_staging_credits(
                state.decode_engine_id, decode_request_id, [new_credit]
            )
            self._finish_cancelled_staging_send(state)
            return
        state.credits.append(new_credit)
        state.acked_chunks += 1
        self._finish_staging_send_if_ready(state)

    def _handle_staging_data_notif(self, notif: bytes) -> None:
        try:
            payload = msgspec.msgpack.decode(notif[len(STAGE_DATA_NOTIF_PREFIX) :])
            request_id = str(payload["request_id"])
            producer_request_id = str(payload["producer_request_id"])
            producer_engine_id = str(payload["producer_engine_id"])
            credit = StagingCredit(int(payload["slot_id"]), int(payload["epoch"]))
            block_start = int(payload["block_start"])
            block_count = int(payload["block_count"])
            chunk_index = int(payload["chunk_index"])
            total_chunks = int(payload["total_chunks"])
            valid_bytes = int(payload["valid_bytes"])
        except (KeyError, TypeError, ValueError, msgspec.DecodeError):
            logger.exception("Invalid NIXL staging DATA notification")
            return

        owner = self._staging_recv_owners.get(credit.slot_id)
        meta = self._recving_metadata.get(request_id)
        if (
            owner != (request_id, credit.epoch)
            or credit.slot_id in self._staging_recv_busy
            or meta is None
            or len(meta.local_physical_block_ids) != 1
        ):
            logger.error(
                "Dropping stale or unowned NIXL staging DATA for request %s, "
                "slot=%d, epoch=%d",
                request_id,
                credit.slot_id,
                credit.epoch,
            )
            return
        assert self._staging_pool is not None
        local_blocks = list(meta.local_physical_block_ids[0])
        progress = self._staging_recv_progress.get(request_id)
        if (
            block_start < 0
            or block_count <= 0
            or total_chunks <= 0
            or not 0 <= chunk_index < total_chunks
            or block_start + block_count > len(local_blocks)
            or valid_bytes != block_count * self._staging_block_bytes
            or valid_bytes > self._staging_pool.slot_bytes
            or (
                progress is not None
                and (
                    progress.total_chunks != total_chunks
                    or chunk_index in progress.seen_chunks
                )
            )
        ):
            self._handle_failed_transfer(request_id, None)
            return

        assert self._staging_copy_stream is not None
        block_ids = local_blocks[block_start : block_start + block_count]
        with torch.cuda.stream(self._staging_copy_stream):
            indices = torch.tensor(
                block_ids,
                device=self._staging_pool.tensor.device,
                dtype=torch.long,
            )
            slot = self._staging_pool.view(credit.slot_id, valid_bytes)
            copied_bytes = scatter_staging_blocks(self._staging_caches, slot, indices)
            assert copied_bytes == valid_bytes
            event = torch.cuda.Event()
            event.record(self._staging_copy_stream)
        self._staging_recv_busy.add(credit.slot_id)
        if progress is None:
            progress = _StagingRecvProgress(total_chunks)
            self._staging_recv_progress[request_id] = progress
        progress.seen_chunks.add(chunk_index)
        self._staging_scatter_tasks.append(
            _StagingScatterTask(
                decode_request_id=request_id,
                producer_request_id=producer_request_id,
                producer_engine_id=producer_engine_id,
                credit=credit,
                total_chunks=total_chunks,
                event=event,
                indices=indices,
            )
        )

    def _poll_staging_scatter_tasks(self) -> None:
        pending: list[_StagingScatterTask] = []
        for task in self._staging_scatter_tasks:
            if not task.event.query():
                pending.append(task)
                continue
            self._staging_recv_busy.discard(task.credit.slot_id)
            owner = self._staging_recv_owners.get(task.credit.slot_id)
            if owner != (task.decode_request_id, task.credit.epoch):
                continue
            next_epoch = task.credit.epoch + 1
            self._staging_recv_epochs[task.credit.slot_id] = next_epoch
            self._staging_recv_owners[task.credit.slot_id] = (
                task.decode_request_id,
                next_epoch,
            )
            credits = self._staging_registration_credits.get(task.decode_request_id, [])
            for index, credit in enumerate(credits):
                if credit.slot_id == task.credit.slot_id:
                    credits[index] = StagingCredit(credit.slot_id, next_epoch)
                    break
            self._send_staging_ack(task, next_epoch)

            progress = self._staging_recv_progress[task.decode_request_id]
            progress.completed_chunks += 1
            if progress.completed_chunks == progress.total_chunks:
                self._staging_done_recving.put(task.decode_request_id)
        self._staging_scatter_tasks = pending

    def _send_staging_ack(self, task: _StagingScatterTask, next_epoch: int) -> None:
        agents = self._remote_agents.get(task.producer_engine_id, {})
        payload = {
            "request_id": task.decode_request_id,
            "producer_request_id": task.producer_request_id,
            "decode_engine_id": self.engine_id,
            "slot_id": task.credit.slot_id,
            "epoch": task.credit.epoch,
            "next_epoch": next_epoch,
        }
        notif = STAGE_ACK_NOTIF_PREFIX + msgspec.msgpack.encode(payload)
        for agent_name in agents.values():
            self.nixl_wrapper.send_notif(agent_name, notif_msg=notif)

    def _handle_staging_release_notif(self, notif: bytes) -> None:
        try:
            payload = msgspec.msgpack.decode(notif[len(STAGE_RELEASE_NOTIF_PREFIX) :])
            request_id = str(payload["request_id"])
            credits = [
                StagingCredit(int(item["slot_id"]), int(item["epoch"]))
                for item in payload["credits"]
            ]
        except (KeyError, TypeError, ValueError, msgspec.DecodeError):
            logger.exception("Invalid NIXL staging RELEASE notification")
            return
        assert self._staging_pool is not None
        for credit in credits:
            if credit.slot_id in self._staging_recv_busy:
                logger.error(
                    "Ignoring staging RELEASE for busy slot %d", credit.slot_id
                )
                continue
            if self._staging_recv_owners.get(credit.slot_id) != (
                request_id,
                credit.epoch,
            ):
                logger.warning(
                    "Ignoring stale staging RELEASE for slot=%d epoch=%d",
                    credit.slot_id,
                    credit.epoch,
                )
                continue
            del self._staging_recv_owners[credit.slot_id]
            self._staging_pool.release(credit.slot_id)
        if not any(
            owner[0] == request_id for owner in self._staging_recv_owners.values()
        ):
            self._staging_registration_credits.pop(request_id, None)
            self._staging_recv_progress.pop(request_id, None)

    def _xfer_blocks_for_req(self, req_id: str, meta: ReqMeta):
        """Issue WRITE transfers to one or more remote TP ranks."""
        assert meta.remote is not None and self.transfer_topo is not None
        engine_id = meta.remote.engine_id
        plan = self.tp_mappings[engine_id]
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        tp_ratio = self.transfer_topo.tp_ratio(remote_info.remote_tp_size)

        # Expand D's logical IDs using the ratio learned during the
        # NIXL handshake. ``meta`` is freshly built by
        # ``_do_start_push_kv`` so mutating it here is safe.
        meta.remote.block_ids = self._logical_to_kernel_block_ids(
            meta.remote.block_ids,
            remote_info.remote_physical_blocks_per_logical,
        )
        remote_block_ids = meta.remote.block_ids
        local_block_ids = meta.local_physical_block_ids
        num_groups = len(local_block_ids)

        # MLA latent is replicated across D's TP ranks: the tp-mapping
        # collapses it to one rank (fine for reads), but push must WRITE every
        # D rank or the rest decode stale KV. For hybrid MLA+SSM the sharded
        # SSM state already targets every covered D rank, so only the
        # attention groups need widening; pure MLA writes to all handshaked
        # ranks (only the dst differs per rank).
        replicate_attn = self.use_mla and tp_ratio < 0
        if replicate_attn and not self._has_mamba:
            assert len(plan.all_source_ranks) == 1
            write_ranks = sorted(self.dst_xfer_side_handles[engine_id])
        else:
            write_ranks = list(plan.all_source_ranks)

        def group_ids(block_ids: BlockIds, rank: int) -> BlockIds:
            return [
                list(block_ids[g])
                if (replicate_attn and _is_attention_spec(self._group_spec_types[g]))
                or rank in plan.source_ranks_per_group[g]
                else []
                for g in range(num_groups)
            ]

        read_specs = [
            ReadSpec(
                remote_rank=rank,
                local_block_ids=group_ids(local_block_ids, rank),
                remote_block_ids=group_ids(remote_block_ids, rank),
            )
            for rank in write_ranks
        ]

        handles: list[int] = []
        for i, spec in enumerate(read_specs):
            remote_block_size = remote_info.remote_block_size
            logger.debug(
                "Remote agent %s available, calling _xfer_blocks"
                " on remote rank %s with remote block size %s for req %s",
                meta.remote.engine_id,
                spec.remote_rank,
                remote_block_size,
                req_id,
            )
            if tp_ratio < 0 and (not self.use_mla or len(plan.all_source_ranks) > 1):
                # Multiple targets: write each rank its chunk of local memory.
                # Hybrid MLA+SSM also lands here: its split handles replicate
                # the attention descriptors and chunk only the SSM state.
                split_key = (tp_ratio, remote_block_size)
                local_xfer_side_handle = self.src_xfer_handles_by_tp_ratio[split_key][i]
            else:
                local_xfer_side_handle = self.src_xfer_handles_by_block_size[
                    remote_block_size
                ]

            remote_xfer_side_handle = self.dst_xfer_side_handles[meta.remote.engine_id][
                spec.remote_rank
            ]

            handle = self._xfer_blocks(
                read_spec=spec,
                request_id=req_id,
                dst_engine_id=meta.remote.engine_id,
                remote_request_id=meta.remote.request_id,
                local_xfer_side_handle=local_xfer_side_handle,
                remote_xfer_side_handle=remote_xfer_side_handle,
            )
            if handle is not None:
                handles.append(handle)

        # Publish all the request's WRITE handles in one locked update: a
        # partial set would let ``_pop_done_transfers`` finish the request
        # early, then double-report it as the remaining writes land.
        if handles:
            with self._sending_transfers_lock:
                self._sending_transfers[req_id].extend(handles)

    def _xfer_blocks(
        self,
        read_spec: ReadSpec,
        dst_engine_id: str,
        request_id: str,
        remote_request_id: str,
        local_xfer_side_handle: int,
        remote_xfer_side_handle: int,
    ) -> int | None:
        """Post a WRITE point-to-point xfer request.

        Returns the in-flight transfer handle (so the caller can track all of
        a request's handles atomically), or ``None`` if nothing was submitted.
        """
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

        notif_id = f"{remote_request_id}:{self.world_size}".encode()

        if len(local_block_ids) == 0:
            logger.warning("No blocks to push for request %s", request_id)
            return None

        # Align per-group block counts for push.
        local_block_ids = list(local_block_ids)
        remote_block_ids = list(remote_block_ids)
        for i in range(min(len(local_block_ids), len(remote_block_ids))):
            num_local = len(local_block_ids[i])
            num_remote = len(remote_block_ids[i])
            if num_local > num_remote:
                local_block_ids[i] = local_block_ids[i][:num_remote]
            elif num_local < num_remote:
                remote_block_ids[i] = remote_block_ids[i][:num_local]

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

        handle = None
        try:
            handle = self.nixl_wrapper.make_prepped_xfer(
                "WRITE",
                local_xfer_side_handle,
                local_block_descs_ids,
                remote_xfer_side_handle,
                remote_block_descs_ids,
                notif_msg=notif_id,
            )
            self.nixl_wrapper.transfer(handle)
            # Caller tracks the handle (atomically with the request's other
            # writes) so P can free blocks once all of them are done.
            return handle
        except Exception as e:
            self._log_failure(
                failure_type="transfer_setup_failed",
                req_id=request_id,
                msg="Push WRITE submission failed; releasing handle",
                error=e,
                dst_engine_id=dst_engine_id,
                remote_rank=remote_rank,
            )
            # On the P side this WRITE failure is purely outbound; we
            # don't have a ``_recving_metadata`` entry to invalidate, so
            # we just release the handle and let the engine reschedule
            # via the lease / watchdog.
            if handle is not None:
                self.nixl_wrapper.release_xfer_handle(handle)
            self.xfer_stats.record_failed_transfer()
            return None

    # --- Notification handling on engine main thread ------------------ #

    def _get_new_notifs(self) -> set[str]:
        """Drain HB / completion notifs forwarded by the writer thread.

        The writer owns ``nixl_wrapper.get_new_notifs`` for push; PUSH_REG
        notifs are handled there. Everything else is forwarded here for
        existing accounting.
        """
        assert self.transfer_topo is not None
        notified_req_ids: set[str] = set()
        while True:
            try:
                notif = self._pending_completion_notifs.get_nowait()
            except queue.Empty:
                break

            msg = notif.decode("utf-8")
            if msg.startswith("HB:"):
                self._handle_heartbeat(msg[3:])
                continue

            req_id, tp_size = msg.rsplit(":", 1)

            # Not tracked as a P-side send/process for this notif.
            if req_id not in self._reqs_to_send and req_id not in self._reqs_to_process:
                if (meta := self._recving_metadata.get(req_id)) is not None:
                    # Consumer waits for one notif per producer rank writing
                    # here: pp_size stages * producers-per-consumer (>1 when
                    # producer TP > consumer TP; tp_size is the producer TP).
                    producers_per_consumer = max(1, int(tp_size) // self.world_size)
                    expected_notifs = meta.pp_size * producers_per_consumer
                    self.consumer_notification_counts_by_req[req_id] += 1
                    notifs = self.consumer_notification_counts_by_req[req_id]
                    if notifs < expected_notifs:
                        continue
                    del self.consumer_notification_counts_by_req[req_id]
                    # P drove the transfer (we own no NIXL handle), so
                    # materialise an empty ``_recving_transfers`` entry for
                    # ``_pop_done_transfers`` to report done.
                    self._recving_transfers.setdefault(req_id, [])
                else:
                    # Not tracked on either side (lease may have expired
                    # before the notif arrived). Log and skip.
                    logger.error(
                        "Unrecognized request %s notif (may have expired).",
                        req_id,
                    )
                continue

            n_consumers = int(tp_size)
            tp_ratio = self.transfer_topo.tp_ratio(n_consumers)
            consumers_per_producer = -tp_ratio if n_consumers > self.world_size else 1
            self.consumer_notification_counts_by_req[req_id] += 1
            if (
                self.consumer_notification_counts_by_req[req_id]
                == consumers_per_producer
            ):
                notified_req_ids.add(req_id)
                del self.consumer_notification_counts_by_req[req_id]
                self._reqs_to_process.remove(req_id)
                self._reqs_to_send.pop(req_id, None)
        return notified_req_ids

    def get_finished(self) -> tuple[set[str], set[str]]:
        # Engine main thread asking for completions: also wake the writer
        # so it gets a chance to drain NIXL notifs (heartbeats, completion
        # notifs, late PUSH_REGs) even if it had been parked.
        self._push_writer_wake.set()

        staging_done_recving = getattr(self, "_staging_done_recving", None)
        while staging_done_recving is not None:
            try:
                req_id = staging_done_recving.get_nowait()
            except queue.Empty:
                break
            # The staging thread already scattered the bytes. Let the base
            # path perform metadata cleanup and scheduler accounting.
            self._recving_transfers.setdefault(req_id, [])

        done_sending, done_recving = super().get_finished()

        # ``_pop_done_transfers`` mutates ``_sending_transfers``; the
        # writer thread also appends to it, so guard the pop.
        with self._sending_transfers_lock:
            done_pushing = self._pop_done_transfers(self._sending_transfers)
        for req_id in done_pushing:
            self._reqs_to_send.pop(req_id, None)
            self._reqs_to_process.discard(req_id)
            self.consumer_notification_counts_by_req.pop(req_id, None)
            done_sending.add(req_id)

        staging_done_sending = getattr(self, "_staging_done_sending", None)
        while staging_done_sending is not None:
            try:
                req_id = staging_done_sending.get_nowait()
            except queue.Empty:
                break
            self._reqs_to_send.pop(req_id, None)
            self._reqs_to_process.discard(req_id)
            self.consumer_notification_counts_by_req.pop(req_id, None)
            done_sending.add(req_id)

        # Tell the writer to drop any state it still holds for any
        # request that just finished (push completed) or expired
        # (lease ran out without a D registration ever arriving).
        for req_id in done_sending:
            self._evict_finished_inbox.put(req_id)
        if done_sending:
            self._push_writer_wake.set()

        return done_sending, done_recving
