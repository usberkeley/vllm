# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse selected-page KV transfer connector facade."""

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import msgspec
import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import NixlPullConnector
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlConnectorMetadata,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    NixlPullConnectorWorker,
)
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.page_offload.config import (
    SparsePageParallelTopology,
)
from vllm.v1.attention.backends.mla.page_offload.coordinator import (
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
from vllm.v1.attention.backends.mla.page_offload.selection import LogicalPage
from vllm.v1.kv_cache_interface import MLAAttentionSpec, UniformTypeKVCacheSpecs

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request

logger = init_logger(__name__)

SPARSE_PAGE_KV_TRANSFER_KEY = "sparse_page_offload"
_SPARSE_PAGE_DONE_PREFIX = b"SPARSE_PAGE_DONE:"


@dataclass
class _SparseSourceRegistration:
    request_id: str
    generation: int
    tensors: tuple[torch.Tensor, ...]
    registered_descs: Any


@dataclass
class _SparseReceiveJob:
    route: SparsePageRoute
    meta: ReqMeta
    pages: tuple[LogicalPage, ...]
    page_tensors: dict[LogicalPage, torch.Tensor]
    tail_pages: frozenset[tuple[str, int]]
    registered_descs: Any
    local_dlist: Any
    remote_dlist: Any
    source_rank: int


class SparsePageNixlPullConnectorWorker(NixlPullConnectorWorker):
    """NIXL pull worker with a separate page-granular DRAM data plane."""

    def __init__(self, vllm_config, engine_id, kv_cache_config):
        super().__init__(vllm_config, engine_id, kv_cache_config)
        self._coordinator = get_sparse_page_offload_coordinator(vllm_config)
        self._sparse_source_registrations: dict[str, _SparseSourceRegistration] = {}
        self._pending_sparse_receives: dict[
            str, tuple[dict[str, Any], ReqMeta]
        ] = {}
        self._sparse_receive_jobs: dict[str, _SparseReceiveJob] = {}
        self._sparse_done_counts: defaultdict[str, int] = defaultdict(int)
        self._failed_sparse_requests: set[str] = set()

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        sparse_layers = self._get_sparse_c4a_layers()
        if not sparse_layers:
            result = super().register_kv_caches(kv_caches)
            self._regular_device_kv_caches = kv_caches
            return result
        regular_kv_caches = {
            layer_name: kv_cache
            for layer_name, kv_cache in kv_caches.items()
            if layer_name not in sparse_layers
        }
        if not regular_kv_caches:
            raise RuntimeError(
                "SparsePageConnector requires at least one non-c4a KV cache "
                "region for its regular NIXL data plane."
            )
        super().register_kv_caches(regular_kv_caches)
        self._regular_device_kv_caches = regular_kv_caches
        self.device_kv_caches = kv_caches

    def post_process_device_kv_on_receive(
        self,
        block_size_ratio: int,
        block_ids_list: list[list[int]],
    ):
        all_kv_caches = self.device_kv_caches
        self.device_kv_caches = self._regular_device_kv_caches
        try:
            return super().post_process_device_kv_on_receive(
                block_size_ratio,
                block_ids_list,
            )
        finally:
            self.device_kv_caches = all_kv_caches

    def sync_recved_kv_to_device(self, req_id: str, meta: ReqMeta):
        all_kv_caches = self.device_kv_caches
        self.device_kv_caches = self._regular_device_kv_caches
        try:
            return super().sync_recved_kv_to_device(req_id, meta)
        finally:
            self.device_kv_caches = all_kv_caches

    def save_kv_to_host(self, metadata: NixlConnectorMetadata):
        all_kv_caches = self.device_kv_caches
        self.device_kv_caches = self._regular_device_kv_caches
        try:
            return super().save_kv_to_host(metadata)
        finally:
            self.device_kv_caches = all_kv_caches

    def post_process_device_kv_on_receive_heterogeneous_attn(
        self,
        block_ids: list[int],
    ):
        all_kv_caches = self.device_kv_caches
        self.device_kv_caches = self._regular_device_kv_caches
        try:
            return super().post_process_device_kv_on_receive_heterogeneous_attn(
                block_ids
            )
        finally:
            self.device_kv_caches = all_kv_caches

    def _get_sparse_c4a_layers(self) -> set[str]:
        layers: set[str] = set()
        for group in self.kv_cache_config.kv_cache_groups:
            group_spec = group.kv_cache_spec
            if not isinstance(group_spec, UniformTypeKVCacheSpecs):
                continue
            for layer_name, layer_spec in group_spec.kv_cache_specs.items():
                if (
                    isinstance(layer_spec, MLAAttentionSpec)
                    and layer_spec.model_version == "deepseek_v4"
                    and layer_spec.compress_ratio == 4
                    and layer_spec.cache_dtype_str == "fp8_ds_mla"
                    and self._coordinator is not None
                    and self._coordinator.config.includes_layer(layer_name)
                ):
                    layers.add(layer_name)
        return layers

    def register_sparse_source_pages(
        self,
        *,
        request_id: str,
        generation: int,
        pages: tuple[tuple[LogicalPage, torch.Tensor], ...],
    ) -> SparsePageRankTransfer:
        if request_id in self._sparse_source_registrations:
            raise ValueError(
                f"Sparse source pages are already registered for {request_id!r}."
            )
        ordered_pages = tuple(sorted(pages, key=lambda item: item[0]))
        tensors = tuple(tensor for _, tensor in ordered_pages)
        memory_data = [
            (tensor.data_ptr(), tensor.nbytes, 0, "") for tensor in tensors
        ]
        registered_descs = self.nixl_wrapper.get_reg_descs(memory_data, "DRAM")
        self.nixl_wrapper.register_memory(
            registered_descs, backends=self.nixl_backends
        )
        self._sparse_source_registrations[request_id] = _SparseSourceRegistration(
            request_id=request_id,
            generation=generation,
            tensors=tensors,
            registered_descs=registered_descs,
        )
        return SparsePageRankTransfer(
            tp_rank=self.tp_rank,
            pages=tuple(
                SparsePageTransferPage(
                    layer_name=page.layer_name,
                    page_idx=page.page_idx,
                    source_address=tensor.data_ptr(),
                    page_size_bytes=tensor.nbytes,
                )
                for page, tensor in ordered_pages
            ),
        )

    def start_load_kv(self, metadata: NixlConnectorMetadata):
        super().start_load_kv(metadata)
        sparse_params = getattr(metadata, "sparse_page_request_params", {})
        for request_id, params in sparse_params.items():
            if request_id in self._sparse_receive_jobs:
                continue
            meta = metadata.reqs_to_recv.get(request_id)
            if meta is not None:
                self._pending_sparse_receives[request_id] = (params, meta)
        for request_id in tuple(self._pending_sparse_receives):
            params, meta = self._pending_sparse_receives[request_id]
            if self._try_start_sparse_receive(request_id, params, meta):
                del self._pending_sparse_receives[request_id]

    def _try_start_sparse_receive(
        self,
        request_id: str,
        params: dict[str, Any],
        meta: ReqMeta,
    ) -> bool:
        if self._coordinator is None:
            raise RuntimeError("Sparse page receive requires the MLA coordinator.")
        route_value = params.get("route")
        if not isinstance(route_value, dict):
            raise ValueError("Sparse page receive is missing route metadata.")
        route = SparsePageRoute.from_dict(route_value)
        engine_id = route.producer_engine_id
        if engine_id not in self._remote_agents or engine_id not in self.tp_mappings:
            return False

        source_ranks = self.tp_mappings[engine_id].all_source_ranks
        if len(source_ranks) != 1:
            raise ValueError(
                "Sparse MLA page transfer requires exactly one replicated source "
                f"rank, got {source_ranks}."
            )
        source_rank = source_ranks[0]
        sideband = SparsePagePrefillSideband.from_kv_transfer_params(params)
        rank_transfer = next(
            (
                transfer
                for transfer in sideband.rank_transfers
                if transfer.tp_rank == source_rank
            ),
            None,
        )
        if rank_transfer is None or not rank_transfer.pages:
            raise ValueError(
                "Sparse page sideband has no source pages for producer "
                f"tp_rank={source_rank}."
            )

        page_refs = tuple(
            SparsePageReference(page.layer_name, page.page_idx)
            for page in rank_transfer.pages
        )
        page_tensors = self._coordinator.reserve_received_cpu_pages(
            request_id=request_id,
            generation=route.generation,
            pages=page_refs,
            kv_caches=self.device_kv_caches,
        )
        ordered_logical_pages = tuple(page_tensors)
        ordered_tensors = tuple(page_tensors.values())
        for source_page, tensor in zip(
            rank_transfer.pages, ordered_tensors, strict=True
        ):
            if tensor.nbytes != source_page.page_size_bytes:
                raise ValueError(
                    "Sparse page size mismatch for "
                    f"layer={source_page.layer_name!r} page={source_page.page_idx}: "
                    f"local={tensor.nbytes} remote={source_page.page_size_bytes}."
                )

        handle = None
        registered_descs = None
        local_dlist = None
        remote_dlist = None
        try:
            local_memory = [
                (tensor.data_ptr(), tensor.nbytes, 0, "")
                for tensor in ordered_tensors
            ]
            registered_descs = self.nixl_wrapper.get_reg_descs(
                local_memory, "DRAM"
            )
            self.nixl_wrapper.register_memory(
                registered_descs, backends=self.nixl_backends
            )
            local_xfer_descs = self.nixl_wrapper.get_xfer_descs(
                [
                    (address, length, device_id)
                    for address, length, device_id, _ in local_memory
                ],
                "DRAM",
            )
            local_dlist = self.nixl_wrapper.prep_xfer_dlist(
                "NIXL_INIT_AGENT", local_xfer_descs
            )
            remote_xfer_descs = self.nixl_wrapper.get_xfer_descs(
                [
                    (page.source_address, page.page_size_bytes, 0)
                    for page in rank_transfer.pages
                ],
                "DRAM",
            )
            remote_agent = self._remote_agents[engine_id][(0, source_rank)]
            remote_dlist = self.nixl_wrapper.prep_xfer_dlist(
                remote_agent, remote_xfer_descs
            )
            indices = list(range(len(rank_transfer.pages)))
            handle = self.nixl_wrapper.make_prepped_xfer(
                "READ",
                local_dlist,
                indices,
                remote_dlist,
                indices,
            )
            self.nixl_wrapper.transfer(handle)
        except Exception as exc:
            logger.error(
                "Sparse page transfer setup failed for request %s: %s",
                request_id,
                exc,
            )
            self._handle_failed_transfer(request_id, handle)
            if local_dlist is not None:
                self.nixl_wrapper.release_dlist_handle(local_dlist)
            if remote_dlist is not None:
                self.nixl_wrapper.release_dlist_handle(remote_dlist)
            if registered_descs is not None:
                self.nixl_wrapper.deregister_memory(registered_descs)
            self._coordinator.cleanup_request(request_id)
            return True
        self._recving_transfers[request_id].append(handle)
        self._sparse_receive_jobs[request_id] = _SparseReceiveJob(
            route=route,
            meta=meta,
            pages=ordered_logical_pages,
            page_tensors=page_tensors,
            tail_pages=frozenset(
                (page.layer_name, page.page_idx) for page in sideband.tail_pages
            ),
            registered_descs=registered_descs,
            local_dlist=local_dlist,
            remote_dlist=remote_dlist,
            source_rank=source_rank,
        )
        return True

    def get_finished(self) -> tuple[set[str], set[str]]:
        done_sending, done_recving = super().get_finished()
        for request_id in done_recving:
            job = self._sparse_receive_jobs.pop(request_id, None)
            if request_id in self._failed_sparse_requests:
                if job is not None:
                    self._abort_sparse_receive(request_id, job)
                elif self._coordinator is not None:
                    self._coordinator.cleanup_request(request_id)
                self._pending_sparse_receives.pop(request_id, None)
                self._failed_sparse_requests.discard(request_id)
            elif job is None:
                continue
            else:
                self._finish_sparse_receive(request_id, job)
        return done_sending, done_recving

    def _handle_failed_transfer(self, req_id: str, handle: int | None):
        self._failed_sparse_requests.add(req_id)
        super()._handle_failed_transfer(req_id, handle)

    def _finish_sparse_receive(
        self,
        request_id: str,
        job: _SparseReceiveJob,
    ) -> None:
        assert self._coordinator is not None
        try:
            self._restore_tail_pages(job)
            self._coordinator.mark_received_cpu_pages_ready(job.pages)
        finally:
            self._release_sparse_receive_resources(job)

        payload = _SPARSE_PAGE_DONE_PREFIX + msgspec.msgpack.encode(
            {
                "request_id": request_id,
                "generation": job.route.generation,
                "consumer_tp_size": self.world_size,
            }
        )
        remote_agents = self._remote_agents[job.route.producer_engine_id]
        if job.route.producer_tp_size > self.world_size:
            agents_to_notify = remote_agents.values()
        else:
            agents_to_notify = (remote_agents[(0, job.source_rank)],)
        for agent in agents_to_notify:
            self.nixl_wrapper.send_notif(agent, notif_msg=payload)

    def _abort_sparse_receive(
        self,
        request_id: str,
        job: _SparseReceiveJob,
    ) -> None:
        self._release_sparse_receive_resources(job)
        if self._coordinator is not None:
            self._coordinator.cleanup_request(request_id)

    def _release_sparse_receive_resources(self, job: _SparseReceiveJob) -> None:
        self.nixl_wrapper.release_dlist_handle(job.local_dlist)
        self.nixl_wrapper.release_dlist_handle(job.remote_dlist)
        self.nixl_wrapper.deregister_memory(job.registered_descs)

    def _restore_tail_pages(self, job: _SparseReceiveJob) -> None:
        if not job.tail_pages:
            return
        if self._coordinator is not None and self._coordinator.config.allocate_partial:
            for page, cpu_tensor in job.page_tensors.items():
                if (page.layer_name, page.page_idx) not in job.tail_pages:
                    continue
                self._coordinator.restore_received_tail(
                    page,
                    cpu_tensor,
                    self.device_kv_caches[page.layer_name],
                )
            return
        group_by_layer = {
            layer_name: group_idx
            for group_idx, group in enumerate(self.kv_cache_config.kv_cache_groups)
            for layer_name in group.layer_names
        }
        for page, cpu_tensor in job.page_tensors.items():
            if (page.layer_name, page.page_idx) not in job.tail_pages:
                continue
            group_idx = group_by_layer.get(page.layer_name)
            if group_idx is None:
                raise ValueError(
                    f"Sparse tail layer has no KV cache group: {page.layer_name!r}."
                )
            block_ids = job.meta.local_physical_block_ids[group_idx]
            if page.page_idx >= len(block_ids):
                raise ValueError(
                    "Sparse tail page is outside the local request block table: "
                    f"{page!r}."
                )
            dst_block_id = block_ids[page.page_idx]
            self.device_kv_caches[page.layer_name][dst_block_id].copy_(cpu_tensor)

    def _handle_custom_notif(self, notif: bytes) -> bool:
        if not notif.startswith(_SPARSE_PAGE_DONE_PREFIX):
            return False
        try:
            payload = msgspec.msgpack.decode(
                notif[len(_SPARSE_PAGE_DONE_PREFIX) :]
            )
            request_id = str(payload["request_id"])
            generation = int(payload["generation"])
            consumer_tp_size = int(payload["consumer_tp_size"])
        except (KeyError, TypeError, ValueError, msgspec.DecodeError) as exc:
            logger.warning("Ignoring malformed sparse page completion: %s", exc)
            return True
        registration = self._sparse_source_registrations.get(request_id)
        if registration is None:
            logger.warning(
                "Ignoring sparse page completion for unknown request %s.", request_id
            )
            return True
        if generation != registration.generation:
            logger.warning(
                "Ignoring stale sparse page completion for request %s: "
                "got generation %d, expected %d.",
                request_id,
                generation,
                registration.generation,
            )
            return True
        expected = max(1, consumer_tp_size // self.world_size)
        self._sparse_done_counts[request_id] += 1
        if self._sparse_done_counts[request_id] >= expected:
            self.nixl_wrapper.deregister_memory(registration.registered_descs)
            del self._sparse_source_registrations[request_id]
            del self._sparse_done_counts[request_id]
            if self._coordinator is not None:
                self._coordinator.staging_manager.cleanup_request(
                    registration.request_id
                )
        return True

    def shutdown(self):
        source_registrations = getattr(self, "_sparse_source_registrations", {})
        receive_jobs = getattr(self, "_sparse_receive_jobs", {})
        for registration in source_registrations.values():
            self.nixl_wrapper.deregister_memory(registration.registered_descs)
        for job in receive_jobs.values():
            self.nixl_wrapper.release_dlist_handle(job.local_dlist)
            self.nixl_wrapper.release_dlist_handle(job.remote_dlist)
            self.nixl_wrapper.deregister_memory(job.registered_descs)
        source_registrations.clear()
        receive_jobs.clear()
        if hasattr(self, "_background_executor"):
            super().shutdown()


class SparsePageConnector(NixlPullConnector):
    """NIXL-backed facade for DeepSeek V4 selected-page offload.

    The connector name is the public switch for the MLA page-offload control
    plane. Request transfer still reuses the NIXL pull connector primitives;
    c4a selected-page residency is owned by the DeepSeek V4 MLA coordinator.
    """

    worker_cls = SparsePageNixlPullConnectorWorker

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        kv_role = getattr(vllm_config.kv_transfer_config, "kv_role", None)
        if kv_role not in ("kv_producer", "kv_consumer"):
            raise ValueError(
                "SparsePageConnector requires kv_role='kv_producer' or "
                f"'kv_consumer', got {kv_role!r}."
            )
        self._topology = SparsePageParallelTopology.from_vllm_config(vllm_config)
        self._topology.validate()
        super().__init__(vllm_config, role, kv_cache_config)
        transfer_backend = vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "sparse_page_transfer_backend",
            "nixl",
        )
        if transfer_backend != "nixl":
            raise NotImplementedError(
                "SparsePageConnector currently supports only "
                "sparse_page_transfer_backend='nixl'."
            )
        self._pending_sideband_by_request: dict[str, SparsePagePrefillSideband] = {}
        self._active_sparse_params_by_request: dict[str, dict[str, Any]] = {}
        self._route_tracker = SparsePageRouteTracker()
        logger.info_once(
            "SparsePageConnector enabled with NIXL transfer primitives. "
            "DeepSeek V4 c4a selected-page staging is handled by the MLA "
            "page-offload coordinator. engine_id=%s dp_rank=%d dp_size=%d "
            "tp_size=%d expert_parallel=%r",
            self._topology.engine_id,
            self._topology.dp_rank,
            self._topology.dp_size,
            self._topology.tp_size,
            self._topology.expert_parallel,
        )

    @property
    def requires_connector_output_before_request_finished(self) -> bool:
        return True

    def build_connector_meta(self, scheduler_output: "SchedulerOutput"):
        request_ids_by_row = _ordered_request_ids_for_sparse_rows(
            scheduler_output.num_scheduled_tokens
        )
        connector_metadata = super().build_connector_meta(scheduler_output)
        # DeepSeek sparse metadata rows follow the model runner's request order.
        # For pure prefill batches (the only case that seals c4a pages), this is
        # the scheduler request order sorted by query length.
        connector_metadata.sparse_page_request_ids_by_row = request_ids_by_row
        if self.kv_transfer_config.kv_role == "kv_producer":
            connector_metadata.sparse_page_generation_by_request = {
                request_id: self._route_tracker.get_or_create_producer_generation(
                    request_id
                )
                for request_id in request_ids_by_row
            }
        connector_metadata.sparse_page_request_params = {
            request_id: dict(self._active_sparse_params_by_request[request_id])
            for request_id in request_ids_by_row
            if request_id in self._active_sparse_params_by_request
        }
        return connector_metadata

    def on_new_request(self, request: "Request") -> None:
        is_producer = self.kv_transfer_config.kv_role == "kv_producer"
        if is_producer:
            self._route_tracker.validate_new_producer_request(request.request_id)
        else:
            self._validate_and_bind_consumer_route(request)
            params = request.kv_transfer_params
            if isinstance(params, dict):
                sparse_params = params.get(SPARSE_PAGE_KV_TRANSFER_KEY)
                if isinstance(sparse_params, dict):
                    self._active_sparse_params_by_request[request.request_id] = dict(
                        sparse_params
                    )
        try:
            super().on_new_request(request)
        except Exception:
            self._active_sparse_params_by_request.pop(request.request_id, None)
            self._route_tracker.discard_consumer_request(request.request_id)
            raise
        if is_producer:
            self._route_tracker.begin_producer_request(request.request_id)

    def update_connector_output(self, connector_output: "KVConnectorOutput"):
        super().update_connector_output(connector_output)
        worker_metadata = connector_output.kv_connector_worker_meta
        if not isinstance(worker_metadata, SparsePagePrefillWorkerMetadata):
            return
        for request_id, sideband in worker_metadata.request_sidebands.items():
            current = self._pending_sideband_by_request.get(request_id)
            self._pending_sideband_by_request[request_id] = (
                sideband if current is None else current.merge(sideband)
            )

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        delay_free, kv_transfer_params = super().request_finished(request, block_ids)
        merged_params = self._finalize_request(
            request.request_id,
            kv_transfer_params,
        )
        return delay_free, merged_params

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: BlockIds,
    ) -> tuple[bool, dict[str, Any] | None]:
        delay_free, kv_transfer_params = super().request_finished_all_groups(
            request, block_ids
        )
        merged_params = self._finalize_request(
            request.request_id,
            kv_transfer_params,
        )
        return delay_free, merged_params

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        result = super().get_finished(finished_req_ids)
        if self.kv_transfer_config.kv_role == "kv_consumer":
            coordinator = get_sparse_page_offload_coordinator(self._vllm_config)
            if coordinator is not None:
                for request_id in finished_req_ids:
                    coordinator.cleanup_request(request_id)
        return result

    def build_connector_worker_meta(self) -> SparsePagePrefillWorkerMetadata | None:
        if self.connector_worker is None:
            return None
        coordinator = get_sparse_page_offload_coordinator(self._vllm_config)
        if coordinator is None:
            return None
        request_ids = getattr(
            self._connector_metadata,
            "sparse_page_request_ids_by_row",
            (),
        )
        generations = getattr(
            self._connector_metadata,
            "sparse_page_generation_by_request",
            {},
        )
        metadata = coordinator.pop_prefill_worker_metadata(request_ids)
        if metadata is None:
            return None
        if not isinstance(self.connector_worker, SparsePageNixlPullConnectorWorker):
            return metadata
        request_sidebands = dict(metadata.request_sidebands)
        for request_row, request_id in enumerate(request_ids):
            sideband = request_sidebands.get(request_id)
            if sideband is None:
                continue
            pages = coordinator.get_prefill_cpu_pages(request_row, request_id)
            if not pages:
                raise RuntimeError(
                    f"Sparse prefill produced no CPU pages for {request_id!r}."
                )
            rank_transfer = self.connector_worker.register_sparse_source_pages(
                request_id=request_id,
                generation=int(generations[request_id]),
                pages=pages,
            )
            request_sidebands[request_id] = sideband.merge(
                SparsePagePrefillSideband(rank_transfers=(rank_transfer,))
            )
        return SparsePagePrefillWorkerMetadata(
            request_sidebands=request_sidebands
        )

    def start_load_kv(self, forward_context, **kwargs) -> None:
        coordinator = get_sparse_page_offload_coordinator(self._vllm_config)
        metadata = self._connector_metadata
        if coordinator is not None and metadata is not None:
            request_ids = getattr(metadata, "sparse_page_request_ids_by_row", ())
            request_params = getattr(metadata, "sparse_page_request_params", {})
            identities: list[tuple[str, int]] = []
            for request_id in request_ids:
                sparse_params = request_params.get(request_id, {})
                route_value = sparse_params.get("route")
                generation = (
                    SparsePageRoute.from_dict(route_value).generation
                    if isinstance(route_value, dict)
                    else 0
                )
                identities.append((request_id, generation))
            coordinator.bind_request_rows(identities)
        super().start_load_kv(forward_context, **kwargs)

    def _merge_prefill_sideband(
        self,
        request_id: str,
        kv_transfer_params: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        sideband = self._pending_sideband_by_request.pop(request_id, None)
        if sideband is None:
            return kv_transfer_params
        merged_params = dict(kv_transfer_params or {})
        sideband_params = merged_params.get(SPARSE_PAGE_KV_TRANSFER_KEY)
        if isinstance(sideband_params, dict):
            sideband_params = _merge_sideband_params(
                sideband_params,
                sideband.to_kv_transfer_params(),
            )
        else:
            sideband_params = sideband.to_kv_transfer_params()
        generation = self._route_tracker.get_or_create_producer_generation(request_id)
        route = SparsePageRoute.from_producer(
            self._topology,
            generation,
        )
        existing_route = sideband_params.get("route")
        if existing_route is not None and existing_route != route.to_dict():
            raise ValueError(
                f"Conflicting sparse page route metadata for request {request_id!r}."
            )
        sideband_params["version"] = SPARSE_PAGE_SIDEBAND_VERSION
        sideband_params["route"] = route.to_dict()
        merged_params[SPARSE_PAGE_KV_TRANSFER_KEY] = sideband_params
        return merged_params

    def _finalize_request(
        self,
        request_id: str,
        kv_transfer_params: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        try:
            return self._merge_prefill_sideband(request_id, kv_transfer_params)
        finally:
            self._active_sparse_params_by_request.pop(request_id, None)
            self._route_tracker.finish_request(request_id)

    def _validate_and_bind_consumer_route(self, request: "Request") -> None:
        self._route_tracker.validate_new_consumer_request(request.request_id)
        kv_transfer_params = request.kv_transfer_params
        if not isinstance(kv_transfer_params, dict):
            return
        sideband_params = kv_transfer_params.get(SPARSE_PAGE_KV_TRANSFER_KEY)
        if not isinstance(sideband_params, dict):
            return
        version = int(sideband_params.get("version", 0))
        if version != SPARSE_PAGE_SIDEBAND_VERSION:
            raise ValueError(
                "Unsupported sparse page sideband version: "
                f"{version}; expected {SPARSE_PAGE_SIDEBAND_VERSION}."
            )
        route_value = sideband_params.get("route")
        if not isinstance(route_value, dict):
            raise ValueError("Sparse page sideband is missing route metadata.")
        route = SparsePageRoute.from_dict(route_value)
        nixl_tp_size = kv_transfer_params.get("tp_size")
        if nixl_tp_size is None or int(nixl_tp_size) != route.producer_tp_size:
            raise ValueError(
                "Sparse page producer TP size does not match NIXL route: "
                f"sparse={route.producer_tp_size}, nixl={nixl_tp_size!r}."
            )
        route = route.bind_consumer(
            self._topology,
            remote_engine_id=kv_transfer_params.get("remote_engine_id"),
        )
        producer_request_id = str(
            kv_transfer_params.get("remote_request_id", request.request_id)
        )
        producer_request = (route.producer_engine_id, producer_request_id)
        self._route_tracker.bind_consumer_request(
            request.request_id,
            producer_request,
            route.generation,
        )
        bound_sideband_params = dict(sideband_params)
        bound_sideband_params["route"] = route.to_dict()
        kv_transfer_params[SPARSE_PAGE_KV_TRANSFER_KEY] = bound_sideband_params


def _ordered_request_ids_for_sparse_rows(
    scheduled_token_count_by_request: dict[str, int],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            scheduled_token_count_by_request,
            key=scheduled_token_count_by_request.__getitem__,
        )
    )


def _merge_sideband_params(
    existing_params: dict[str, Any],
    new_params: dict[str, Any],
) -> dict[str, Any]:
    merged_params = dict(existing_params)
    merged_params["version"] = new_params["version"]
    merged_params["tail_pages"] = _merge_serialized_pages(
        existing_params.get("tail_pages", ()),
        new_params.get("tail_pages", ()),
    )
    if "rank_transfers" in existing_params or "rank_transfers" in new_params:
        merged_params["rank_transfers"] = _merge_rank_transfers(
            existing_params.get("rank_transfers", ()),
            new_params.get("rank_transfers", ()),
        )
    return merged_params


def _merge_serialized_pages(
    existing_pages: Any,
    new_pages: Any,
) -> list[dict[str, Any]]:
    pages = {
        (str(page["layer_name"]), int(page["page_idx"]))
        for page in (*tuple(existing_pages), *tuple(new_pages))
    }
    return [
        {
            "layer_name": layer_name,
            "page_idx": page_idx,
        }
        for layer_name, page_idx in sorted(pages)
    ]


def _merge_rank_transfers(
    existing_transfers: Any,
    new_transfers: Any,
) -> list[dict[str, Any]]:
    sideband = SparsePagePrefillSideband.from_kv_transfer_params(
        {"rank_transfers": existing_transfers}
    ).merge(
        SparsePagePrefillSideband.from_kv_transfer_params(
            {"rank_transfers": new_transfers}
        )
    )
    return [transfer.to_dict() for transfer in sideband.rank_transfers]


__all__ = ["SparsePageConnector"]
