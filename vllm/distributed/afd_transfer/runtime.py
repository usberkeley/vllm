# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-closed segmented model adapter for the AFD event loops."""

from __future__ import annotations

import inspect
import operator
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from vllm.compilation.cuda_graph import CUDAGraphOptions, CUDAGraphWrapper
from vllm.config import CUDAGraphMode
from vllm.distributed.afd_transfer.graph_cut import (
    AFDFXAttentionContext,
    AFDFXAttentionExecutor,
    partition_afd_graph,
)
from vllm.distributed.afd_transfer.scheduler import pick_bucket
from vllm.forward_context import (
    BatchDescriptor,
    get_forward_context,
    set_forward_context,
)
from vllm.logger import init_logger
from vllm.model_executor.models.afd import (
    AFDAttnBoundary,
    AFDFfnBoundary,
)

logger = init_logger(__name__)


class _AFDCUDAGraphSegmentRunner:
    """Run one FX segment with stable per-graph input addresses."""

    def __init__(
        self,
        runnable: Callable[..., Any],
        vllm_config,
        wrapper: Callable[..., Any] | None = None,
    ) -> None:
        self.runnable = runnable
        self.wrapper = (
            wrapper
            if wrapper is not None
            else CUDAGraphWrapper(
                runnable,
                vllm_config,
                runtime_mode=CUDAGraphMode.FULL,
                cudagraph_options=CUDAGraphOptions(
                    gc_disable=True,
                    weak_ref_output=False,
                ),
            )
        )
        self._static_args: dict[BatchDescriptor, tuple[Any, ...]] = {}

    @staticmethod
    def _new_static_arg(arg: Any) -> Any:
        if not isinstance(arg, torch.Tensor):
            return arg
        static = torch.empty_strided(
            arg.size(),
            arg.stride(),
            dtype=arg.dtype,
            device=arg.device,
        )
        static.copy_(arg)
        return static

    @staticmethod
    def _copy_arg(static: Any, current: Any) -> None:
        if isinstance(static, torch.Tensor):
            if not isinstance(current, torch.Tensor):
                raise TypeError("AFD CUDA Graph tensor input changed type.")
            if (
                static.shape != current.shape
                or static.stride() != current.stride()
                or static.dtype != current.dtype
                or static.device != current.device
            ):
                raise ValueError(
                    "AFD CUDA Graph input shape, stride, dtype, and device "
                    "must remain stable for one batch descriptor."
                )
            static.copy_(current)
        elif static != current:
            raise ValueError(
                "AFD CUDA Graph non-tensor input changed for one batch descriptor."
            )

    def __call__(self, *args: Any) -> Any:
        forward_context = get_forward_context()
        if forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL:
            return self.runnable(*args)
        descriptor = forward_context.batch_descriptor
        if descriptor is None:
            raise RuntimeError("AFD CUDA Graph execution requires a batch descriptor.")

        static_args = self._static_args.get(descriptor)
        if static_args is None:
            static_args = tuple(self._new_static_arg(arg) for arg in args)
            self._static_args[descriptor] = static_args
        else:
            if len(static_args) != len(args):
                raise ValueError(
                    "AFD CUDA Graph input count changed for one batch descriptor."
                )
            for static, current in zip(static_args, args):
                self._copy_arg(static, current)
        return self.wrapper(*static_args)

    def clear(self) -> None:
        clear_graphs = getattr(self.wrapper, "clear_graphs", None)
        if clear_graphs is not None:
            clear_graphs()
        self._static_args.clear()


@dataclass
class AFDAttentionContext:
    """Local state retained while one scheduler batch is at the FFN peer."""

    positions: torch.Tensor
    attn_metadata: Any
    slot_mapping: Any
    residual: torch.Tensor | None = None
    next_layer_index: int = 0
    order: int = 0
    fx_context: AFDFXAttentionContext | None = None


class _AFDDecoderSegment(nn.Module):
    """Pure local decoder layers ending at one AFD boundary."""

    def __init__(self, layers: list[nn.Module]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        return hidden_states, residual


class _AFDDecoderTail(_AFDDecoderSegment):
    """Decoder layers after the final boundary plus the final norm."""

    def __init__(self, layers: list[nn.Module], norm: nn.Module) -> None:
        super().__init__(layers)
        self.norm = norm

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden_states, residual = super().forward(
            positions,
            hidden_states,
            residual,
        )
        normalized = self.norm(hidden_states, residual)
        return normalized[0] if isinstance(normalized, tuple) else normalized


class AFDSegmentedModel:
    """Executes a standard residual decoder one MoE boundary at a time.

    The adapter deliberately accepts only the common ``model.layers`` contract
    whose layer forward is ``(positions, hidden_states, residual, ...)`` and
    whose optional trailing arguments have defaults. Models outside this
    contract fail during startup instead of silently running with wrong KV or
    residual state.
    """

    def __init__(self, model: nn.Module, role: str) -> None:
        if role not in ("attention", "ffn"):
            raise ValueError(f"unsupported segmented AFD role {role}.")
        self.model = model
        self.role = role
        self.backbone = self._find_backbone(model)
        self.layers = self.backbone.layers
        self.start_layer = int(getattr(self.backbone, "start_layer", 0))
        self.end_layer = int(getattr(self.backbone, "end_layer", len(self.layers)))
        if self.start_layer != 0 or self.end_layer != len(self.layers):
            raise NotImplementedError(
                "AFD segmented runtime does not yet support pipeline-parallel "
                "partial layer ranges."
            )

        boundary_type = AFDAttnBoundary if role == "attention" else AFDFfnBoundary
        self.boundaries: dict[int, tuple[int, nn.Module]] = {}
        for index, layer in enumerate(self.layers):
            for module in layer.modules():
                if isinstance(module, boundary_type):
                    if module.layer_id in self.boundaries:
                        raise RuntimeError(
                            f"duplicate AFD boundary for layer {module.layer_id}."
                        )
                    self.boundaries[module.layer_id] = (index, module)

        if not self.boundaries:
            raise NotImplementedError(
                f"AFD found no {boundary_type.__name__} modules in model.layers."
            )
        self.layer_ids = sorted(self.boundaries)
        self.fx_executor: AFDFXAttentionExecutor | None = None
        self._cudagraph_runners: list[_AFDCUDAGraphSegmentRunner] = []
        if role == "attention":
            self._validate_attention_layers()
            config = getattr(self.backbone, "config", None)
            if getattr(config, "llama_4_scaling", None) is not None:
                raise NotImplementedError(
                    "AFD segmented runtime does not yet propagate "
                    "llama_4_scaling to decoder layers."
                )
            if not hasattr(self.backbone, "embed_input_ids"):
                raise NotImplementedError(
                    "AFD Attention runtime requires backbone.embed_input_ids."
                )
            if not hasattr(self.backbone, "norm"):
                raise NotImplementedError(
                    "AFD Attention runtime requires a final backbone.norm."
                )
            try:
                inspect.signature(self.backbone.norm.forward).bind(object(), None)
            except TypeError as error:
                raise NotImplementedError(
                    "AFD Attention runtime requires norm(hidden, residual)."
                ) from error
            self.fx_executor = self._build_attention_fx_executor()
        else:
            self._validate_ffn_boundaries()

    def enable_cudagraph(self, vllm_config) -> None:
        """Wrap every Attention FX segment with persistent-input CUDA Graphs."""
        if self.fx_executor is None:
            raise RuntimeError("Only AFD Attention segments support CUDA Graphs.")
        if self._cudagraph_runners:
            return
        self._cudagraph_runners = [
            _AFDCUDAGraphSegmentRunner(segment.graph_module, vllm_config)
            for segment in self.fx_executor.program.segments
        ]
        self.fx_executor.program.set_segment_runners(list(self._cudagraph_runners))

    def clear_cudagraphs(self) -> None:
        for runner in self._cudagraph_runners:
            runner.clear()

    def _build_attention_fx_executor(self) -> AFDFXAttentionExecutor:
        ordered = sorted(
            (index, layer_id) for layer_id, (index, _) in self.boundaries.items()
        )
        ordered_layer_ids = [layer_id for _, layer_id in ordered]
        if ordered_layer_ids != self.layer_ids:
            raise RuntimeError(
                "AFD layer ids must increase in decoder execution order."
            )

        root = nn.Module()
        graph = torch.fx.Graph()
        positions = graph.placeholder("positions")
        hidden_states = graph.placeholder("hidden_states")
        residual = graph.placeholder("residual")

        next_index = self.start_layer
        for segment_id, (target_index, layer_id) in enumerate(ordered):
            segment = _AFDDecoderSegment(
                list(self.layers[next_index : target_index + 1])
            )
            name = f"attention_segment_{segment_id}"
            root.add_module(name, segment)
            output = graph.call_module(
                name,
                (positions, hidden_states, residual),
            )
            hidden_states = graph.call_function(operator.getitem, (output, 0))
            residual = graph.call_function(operator.getitem, (output, 1))
            hidden_states = graph.call_function(
                torch.ops.vllm.afd_cut.default,
                (hidden_states, layer_id),
            )
            next_index = target_index + 1

        tail = _AFDDecoderTail(
            list(self.layers[next_index : self.end_layer]),
            self.backbone.norm,
        )
        root.add_module("attention_tail", tail)
        output = graph.call_module(
            "attention_tail",
            (positions, hidden_states, residual),
        )
        graph.output(output)
        graph_module = torch.fx.GraphModule(root, graph)
        graph_module.graph.lint()
        graph_module.recompile()
        executor = AFDFXAttentionExecutor(partition_afd_graph(graph_module))
        for _, boundary in self.boundaries.values():
            assert isinstance(boundary, AFDAttnBoundary)
            boundary.emit_cut = False
        return executor

    @staticmethod
    def _find_backbone(model: nn.Module) -> nn.Module:
        candidates = [model]
        seen: set[int] = set()
        while candidates:
            candidate = candidates.pop(0)
            if id(candidate) in seen:
                continue
            seen.add(id(candidate))
            if hasattr(candidate, "layers") and isinstance(
                candidate.layers, nn.ModuleList
            ):
                return candidate
            for attr in ("model", "language_model", "transformer"):
                child = getattr(candidate, attr, None)
                if isinstance(child, nn.Module):
                    candidates.append(child)
        raise NotImplementedError(
            "AFD segmented runtime requires a backbone exposing nn.ModuleList layers."
        )

    def _validate_attention_layers(self) -> None:
        for index, layer in enumerate(self.layers):
            signature = inspect.signature(layer.forward)
            parameters = list(signature.parameters.values())
            if len(parameters) < 3:
                raise NotImplementedError(
                    f"AFD layer {index} must accept positions, hidden_states, residual."
                )
            names = [parameter.name for parameter in parameters[:3]]
            if (
                "position" not in names[0]
                or "hidden" not in names[1]
                or "residual" not in names[2]
            ):
                raise NotImplementedError(
                    f"AFD layer {index} has unsupported forward prefix {names}."
                )
            required_trailing = [
                parameter.name
                for parameter in parameters[3:]
                if parameter.default is inspect.Parameter.empty
                and parameter.kind
                not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
            ]
            if required_trailing:
                raise NotImplementedError(
                    f"AFD layer {index} requires extra arguments "
                    f"{required_trailing}; add a model adapter."
                )

    def _validate_ffn_boundaries(self) -> None:
        for layer_id, (_, boundary) in self.boundaries.items():
            assert isinstance(boundary, AFDFfnBoundary)
            signature = inspect.signature(boundary.mlp.forward)
            required = [
                parameter.name
                for parameter in list(signature.parameters.values())[1:]
                if parameter.default is inspect.Parameter.empty
                and parameter.kind
                not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                )
            ]
            if required:
                raise NotImplementedError(
                    f"AFD FFN layer {layer_id} requires extra arguments "
                    f"{required}; activation-only transport is insufficient."
                )

    def embed(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            return inputs_embeds
        if input_ids is None:
            raise ValueError("AFD Attention runtime requires model inputs.")
        return self.backbone.embed_input_ids(input_ids)

    def run_attention_segment(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        context: AFDAttentionContext,
    ) -> torch.Tensor:
        assert self.fx_executor is not None
        if context.fx_context is None:
            context.fx_context = AFDFXAttentionContext(
                (context.positions, hidden_states, context.residual)
            )
        return self.fx_executor.run_attention_segment(
            layer_id,
            hidden_states,
            context.fx_context,
        )

    def finalize_attention(
        self,
        hidden_states: torch.Tensor,
        context: AFDAttentionContext,
    ) -> torch.Tensor:
        assert self.fx_executor is not None
        if context.fx_context is None:
            raise RuntimeError("AFD Attention graph has not started.")
        return self.fx_executor.finalize_attention(
            hidden_states,
            context.fx_context,
        )

    def run_ffn(self, layer_id: int, hidden_states: torch.Tensor) -> torch.Tensor:
        _, boundary = self.boundaries[layer_id]
        assert isinstance(boundary, AFDFfnBoundary)
        return boundary.mlp(hidden_states)


class AFDFfnService:
    """Background FFN event-loop service for a request-less FFN engine."""

    def __init__(
        self,
        segmented: AFDSegmentedModel,
        connector,
        scheduler,
        vllm_config,
    ) -> None:
        from vllm.distributed.afd_transfer.connector import STAGE_A2F
        from vllm.distributed.afd_transfer.worker_loop import AFDFfnEventLoop

        self.vllm_config = vllm_config
        self.segmented = segmented
        connector.start_listening(STAGE_A2F)
        self.loop = AFDFfnEventLoop(
            scheduler=scheduler,
            source=connector,
            sink=connector,
            replay_fn=self._replay,
            capture_sizes=None,
        )
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="vllm-afd-ffn",
            daemon=True,
        )

    def _replay(self, layer_id: int, hidden: torch.Tensor) -> torch.Tensor:
        with set_forward_context(
            None,
            self.vllm_config,
            num_tokens=hidden.shape[0],
        ):
            return self.segmented.run_ffn(layer_id, hidden)

    @torch.inference_mode()
    def _run(self) -> None:
        try:
            device = self.vllm_config.device_config.device
            if device.type == "cuda":
                torch.cuda.set_device(device)
            while not self._stop.is_set():
                if not self.loop.tick(time.monotonic()):
                    self._stop.wait(0.0005)
        except BaseException as error:
            self._error = error
            logger.exception("AFD FFN service stopped after an error.")

    def start(self) -> None:
        self._thread.start()

    def check_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("AFD FFN background service failed.") from self._error

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


class AFDRuntime:
    """Own AFD role setup, execution, transport, and service lifecycle."""

    def __init__(
        self,
        vllm_config,
        segmented: AFDSegmentedModel,
        connector,
        scheduler,
        transfer_ids,
    ) -> None:
        self.vllm_config = vllm_config
        self.config = vllm_config.afd_config
        self.segmented = segmented
        self.connector = connector
        self.scheduler = scheduler
        self.transfer_ids = transfer_ids
        self.ffn_service: AFDFfnService | None = None

    @classmethod
    def create(
        cls,
        vllm_config,
        model: nn.Module,
        *,
        supports_mm_inputs: bool,
        is_pooling_model: bool,
    ) -> AFDRuntime | None:
        """Create an AFD runtime and rewrite the raw model for its role."""
        config = vllm_config.afd_config
        if config is None:
            return None

        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise NotImplementedError(
                "AFD runtime does not support pipeline parallelism."
            )
        if vllm_config.speculative_config is not None:
            raise NotImplementedError(
                "AFD runtime does not yet support speculative decoding."
            )
        if supports_mm_inputs or is_pooling_model:
            raise NotImplementedError(
                "AFD runtime currently supports text generation models only."
            )
        if vllm_config.parallel_config.use_ubatching:
            raise ValueError(
                "AFD Phase 0 Attention scheduling requires an intact scheduler "
                "batch; disable --enable-dbo and --ubatch-size."
            )
        if vllm_config.lora_config is not None:
            raise NotImplementedError("AFD runtime does not yet support LoRA.")

        from vllm.distributed.afd_transfer.connector import (
            STAGE_F2A,
            AFDConnectorFactory,
            AFDConnectorRole,
            AFDTransferIdAllocator,
        )
        from vllm.distributed.afd_transfer.scheduler import AFDDynamicBatchScheduler
        from vllm.model_executor.models.afd import apply_afd_roles

        moe_layer_ids = apply_afd_roles(model, vllm_config)
        role = (
            AFDConnectorRole.ATTENTION if config.is_attention else AFDConnectorRole.FFN
        )
        connector = AFDConnectorFactory.create_connector(vllm_config, role)
        scheduler = AFDDynamicBatchScheduler(
            num_layers=max(moe_layer_ids) + 1,
            age_limit_s=config.age_limit_ms / 1000,
        )
        transfer_ids = AFDTransferIdAllocator() if config.is_attention else None
        segmented = AFDSegmentedModel(model, config.role)
        cudagraph_mode = getattr(
            vllm_config.compilation_config,
            "cudagraph_mode",
            CUDAGraphMode.NONE,
        )
        if (
            config.is_attention
            and cudagraph_mode is not None
            and cudagraph_mode.has_full_cudagraphs()
        ):
            segmented.enable_cudagraph(vllm_config)
        runtime = cls(
            vllm_config,
            segmented,
            connector,
            scheduler,
            transfer_ids,
        )
        if config.is_attention:
            connector.start_listening(STAGE_F2A)
        return runtime

    @property
    def is_attention(self) -> bool:
        return self.config.is_attention

    def configure_batch_execution(
        self,
        num_tokens: int,
        num_reqs: int,
        cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor: BatchDescriptor | None = None,
    ) -> tuple[CUDAGraphMode, BatchDescriptor, bool, None]:
        """Select full CUDA Graph execution or an unpadded eager fallback."""
        if not self.config.is_attention:
            raise RuntimeError("Only an AFD Attention runtime executes batches.")
        if cudagraph_mode == CUDAGraphMode.FULL:
            if batch_descriptor is None:
                capture_sizes = (
                    self.vllm_config.compilation_config.cudagraph_capture_sizes
                )
                if not capture_sizes or num_tokens > max(capture_sizes):
                    cudagraph_mode = CUDAGraphMode.NONE
                else:
                    batch_descriptor = BatchDescriptor(
                        num_tokens=pick_bucket(num_tokens, capture_sizes),
                        num_reqs=num_reqs,
                    )
            if batch_descriptor is not None:
                if batch_descriptor.num_tokens < num_tokens:
                    raise ValueError(
                        f"AFD CUDA Graph bucket {batch_descriptor.num_tokens} "
                        f"cannot hold {num_tokens} real tokens."
                    )
                return cudagraph_mode, batch_descriptor, False, None
        return (
            CUDAGraphMode.NONE,
            BatchDescriptor(num_tokens=num_tokens, num_reqs=num_reqs),
            False,
            None,
        )

    def start(self) -> None:
        """Start the activation-driven FFN service when this is an FFN role."""
        if not self.config.is_ffn or self.ffn_service is not None:
            return
        self.ffn_service = AFDFfnService(
            self.segmented,
            self.connector,
            self.scheduler,
            self.vllm_config,
        )
        self.ffn_service.start()

    def reject_scheduler_execution_for_ffn(self) -> None:
        """Reject scheduler-driven execution on an activation-driven FFN role."""
        if not self.config.is_ffn:
            return
        if self.ffn_service is not None:
            self.ffn_service.check_error()
        raise RuntimeError(
            "An AFD FFN worker is activation-driven and cannot execute "
            "scheduler requests."
        )

    def execute_attention(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        positions: torch.Tensor,
        attn_metadata,
        slot_mappings,
        num_tokens_unpadded: int,
        cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        batch_descriptor: BatchDescriptor | None = None,
    ) -> torch.Tensor:
        """Run one intact scheduler batch synchronously through AFD layers."""
        if not self.config.is_attention:
            raise RuntimeError("Only an AFD Attention runtime can execute requests.")

        from vllm.distributed.afd_transfer.worker_loop import AFDAttentionEventLoop

        padded_num_tokens = positions.shape[-1]
        if num_tokens_unpadded > padded_num_tokens:
            raise ValueError(
                f"AFD received {num_tokens_unpadded} real tokens in a "
                f"{padded_num_tokens}-token input."
            )
        real_input_ids = (
            input_ids[:num_tokens_unpadded] if input_ids is not None else None
        )
        real_inputs_embeds = (
            inputs_embeds[:num_tokens_unpadded] if inputs_embeds is not None else None
        )
        hidden_states = self.segmented.embed(real_input_ids, real_inputs_embeds)
        capture_sizes = (
            [padded_num_tokens] if cudagraph_mode == CUDAGraphMode.FULL else None
        )
        if cudagraph_mode == CUDAGraphMode.FULL and batch_descriptor is None:
            raise ValueError("AFD CUDA Graph execution requires a batch descriptor.")
        completed: list[torch.Tensor] = []

        def replay(layer_id, hidden, items):
            if len(items) != 1:
                raise RuntimeError(
                    "AFD Attention metadata adapter requires one scheduler batch "
                    "per replay."
                )
            context = items[0].context
            assert isinstance(context, AFDAttentionContext)
            with set_forward_context(
                context.attn_metadata,
                self.vllm_config,
                num_tokens=hidden.shape[0],
                cudagraph_runtime_mode=cudagraph_mode,
                batch_descriptor=batch_descriptor,
                slot_mapping=context.slot_mapping,
                skip_compiled=True,
            ):
                return self.segmented.run_attention_segment(
                    layer_id,
                    hidden,
                    context,
                )

        def finish(hidden, _meta, context):
            assert isinstance(context, AFDAttentionContext)
            with set_forward_context(
                context.attn_metadata,
                self.vllm_config,
                num_tokens=hidden.shape[0],
                cudagraph_runtime_mode=cudagraph_mode,
                batch_descriptor=batch_descriptor,
                slot_mapping=context.slot_mapping,
                skip_compiled=True,
            ):
                output = self.segmented.finalize_attention(hidden, context)
                completed.append(output[:num_tokens_unpadded])

        loop = AFDAttentionEventLoop(
            scheduler=self.scheduler,
            source=self.connector,
            sink=self.connector,
            replay_fn=replay,
            completion_fn=finish,
            capture_sizes=capture_sizes,
            layer_ids=self.segmented.layer_ids,
            transfer_ids=self.transfer_ids,
            credit_capacity=self.config.max_inflight_batches,
        )
        context = AFDAttentionContext(
            positions=positions,
            attn_metadata=attn_metadata,
            slot_mapping=slot_mappings,
        )
        loop.submit(hidden_states, now=time.monotonic(), context=context)

        timeout = self.config.transfer_timeout_s * (len(self.segmented.layer_ids) + 1)
        deadline = time.monotonic() + timeout
        while not completed:
            loop.tick(time.monotonic())
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for AFD Attention layer-segment completion."
                )
            time.sleep(0)
        return completed[0]

    def close(self) -> None:
        """Stop runtime-owned background work."""
        if self.ffn_service is not None:
            self.ffn_service.stop()
        self.segmented.clear_cudagraphs()
