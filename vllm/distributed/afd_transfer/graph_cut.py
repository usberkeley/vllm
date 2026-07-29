# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FX continuation programs for pure AFD graph cuts.

The partitioner turns a graph containing ``vllm::afd_cut`` markers into a
sequence of marker-free ``GraphModule`` compute fragments. Executing a fragment
ending at a cut yields the MoE input and a local continuation. Resuming injects
the remote MoE output as the value of the removed marker and runs the next
fragment.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.fx as fx
from torch.fx.node import map_arg

from vllm.model_executor.layers.afd import ops as afd_ops  # noqa: F401


def is_afd_cut_node(node: fx.Node) -> bool:
    """Return whether ``node`` is the pure AFD cut marker."""
    if node.op != "call_function":
        return False
    target = node.target
    if isinstance(target, torch._ops.OpOverloadPacket):
        return target._qualified_op_name == "vllm::afd_cut"
    if isinstance(target, torch._ops.OpOverload):
        return target.name() == "vllm::afd_cut"
    return False


@dataclass(frozen=True)
class AFDGraphSegment:
    """One marker-free local compute fragment."""

    graph_module: fx.GraphModule
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    cut_layer_id: int | None
    cut_node_name: str | None


@dataclass
class AFDGraphCutState:
    """Local values retained while an FX program waits for remote MoE output."""

    program_id: int
    values: dict[str, Any]
    next_segment: int = 0
    awaiting_cut: str | None = None
    expected_hidden: torch.Tensor | None = None
    done: bool = False


@dataclass(frozen=True)
class AFDGraphBoundary:
    """A suspended FX program and the activation to send to the FFN peer."""

    layer_id: int
    hidden_states: torch.Tensor
    state: AFDGraphCutState
    cut_node_name: str


@dataclass
class AFDFXAttentionContext:
    """Per-item state used while an event loop advances an FX continuation."""

    initial_inputs: tuple[Any, ...]
    boundary: AFDGraphBoundary | None = None


class AFDGraphCutProgram:
    """Execute FX compute fragments one AFD boundary at a time."""

    def __init__(
        self,
        graph_module: fx.GraphModule,
        segments: list[AFDGraphSegment],
        placeholder_names: tuple[str, ...],
    ) -> None:
        self.graph_module = graph_module
        self.segments = tuple(segments)
        self.segment_runners: list[Callable[..., Any]] = [
            segment.graph_module for segment in segments
        ]
        self.placeholder_names = placeholder_names
        self.layer_ids = tuple(
            segment.cut_layer_id
            for segment in self.segments
            if segment.cut_layer_id is not None
        )

    def set_segment_runners(
        self,
        runners: list[Callable[..., Any]],
    ) -> None:
        """Replace segment callables without changing the partitioned graph."""
        if len(runners) != len(self.segments):
            raise ValueError(
                f"AFD graph has {len(self.segments)} segments, "
                f"got {len(runners)} runners."
            )
        self.segment_runners = runners

    def start(self, *inputs: Any) -> AFDGraphBoundary | Any:
        """Start a new graph execution and run until its first cut."""
        if len(inputs) != len(self.placeholder_names):
            raise ValueError(
                f"AFD graph expects {len(self.placeholder_names)} inputs, "
                f"got {len(inputs)}."
            )
        state = AFDGraphCutState(
            program_id=id(self),
            values=dict(zip(self.placeholder_names, inputs)),
        )
        return self._advance(state)

    def resume(
        self,
        boundary: AFDGraphBoundary,
        moe_output: torch.Tensor,
    ) -> AFDGraphBoundary | Any:
        """Inject one remote MoE result and run to the next cut or graph output."""
        state = boundary.state
        if state.program_id != id(self):
            raise ValueError("AFD boundary belongs to another graph-cut program.")
        if state.done or state.awaiting_cut is None:
            raise RuntimeError("AFD graph is not waiting at a cut.")
        if state.awaiting_cut != boundary.cut_node_name:
            raise RuntimeError("AFD boundary is stale or has already been resumed.")
        expected = state.expected_hidden
        assert expected is not None
        if (
            moe_output.shape != expected.shape
            or moe_output.dtype != expected.dtype
            or moe_output.device != expected.device
        ):
            raise ValueError(
                "AFD MoE output must preserve the cut tensor's "
                "shape, dtype, and device."
            )
        state.values[state.awaiting_cut] = moe_output
        state.awaiting_cut = None
        state.expected_hidden = None
        return self._advance(state)

    def _advance(self, state: AFDGraphCutState) -> AFDGraphBoundary | Any:
        if state.awaiting_cut is not None:
            raise RuntimeError("AFD graph must receive its MoE output before resuming.")
        if state.next_segment >= len(self.segments):
            raise RuntimeError("AFD graph has no remaining segment.")

        segment = self.segments[state.next_segment]
        args = [state.values[name] for name in segment.input_names]
        result = self.segment_runners[state.next_segment](*args)
        state.next_segment += 1

        if segment.cut_layer_id is None:
            state.done = True
            state.values.clear()
            return result

        if not isinstance(result, tuple):
            raise RuntimeError("An AFD cut segment must return a tuple.")
        expected_outputs = 1 + len(segment.output_names)
        if len(result) != expected_outputs:
            raise RuntimeError(
                f"AFD cut segment returned {len(result)} values, "
                f"expected {expected_outputs}."
            )
        hidden_states = result[0]
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("AFD cut input must be a tensor.")
        for name, value in zip(segment.output_names, result[1:]):
            state.values[name] = value

        assert segment.cut_node_name is not None
        state.awaiting_cut = segment.cut_node_name
        state.expected_hidden = hidden_states
        return AFDGraphBoundary(
            segment.cut_layer_id,
            hidden_states,
            state,
            segment.cut_node_name,
        )


def partition_afd_graph(graph_module: fx.GraphModule) -> AFDGraphCutProgram:
    """Partition an FX graph into pure compute fragments around ``afd_cut``.

    Values used after a cut are returned by their producer fragment and retained
    in the local continuation state. The cut marker itself is removed; its value
    becomes a placeholder in the next fragment and is supplied by
    :meth:`AFDGraphCutProgram.resume`.
    """
    graph_nodes = list(graph_module.graph.nodes)
    placeholders = [node for node in graph_nodes if node.op == "placeholder"]
    output_nodes = [node for node in graph_nodes if node.op == "output"]
    if len(output_nodes) != 1:
        raise ValueError("AFD graph must have exactly one output node.")
    output_node = output_nodes[0]

    cuts = [node for node in graph_nodes if is_afd_cut_node(node)]
    if not cuts:
        raise ValueError("AFD graph contains no vllm::afd_cut marker.")

    groups: list[tuple[list[fx.Node], fx.Node | None]] = []
    current: list[fx.Node] = []
    for node in graph_nodes:
        if node.op in ("placeholder", "get_attr", "output"):
            continue
        if is_afd_cut_node(node):
            groups.append((current, node))
            current = []
        else:
            current.append(node)
    groups.append((current, None))

    node_group: dict[fx.Node, int] = {}
    for group_id, (nodes, cut) in enumerate(groups):
        for node in nodes:
            node_group[node] = group_id
        if cut is not None:
            node_group[cut] = group_id

    segments = [
        _build_segment(
            graph_module,
            group_id,
            nodes,
            cut,
            output_node,
            node_group,
        )
        for group_id, (nodes, cut) in enumerate(groups)
    ]
    return AFDGraphCutProgram(
        graph_module,
        segments,
        tuple(node.name for node in placeholders),
    )


class AFDFXAttentionExecutor:
    """Adapt an :class:`AFDGraphCutProgram` to Attention event-loop callbacks."""

    def __init__(self, program: AFDGraphCutProgram) -> None:
        self.program = program
        self.layer_ids = list(program.layer_ids)

    def run_attention_segment(
        self,
        layer_id: int,
        hidden_states: torch.Tensor,
        context: AFDFXAttentionContext,
    ) -> torch.Tensor:
        """Run the initial or resumed FX fragment ending at ``layer_id``."""
        if context.boundary is None:
            result = self.program.start(*context.initial_inputs)
        else:
            result = self.program.resume(context.boundary, hidden_states)
        if not isinstance(result, AFDGraphBoundary):
            raise RuntimeError(
                f"AFD FX graph completed before expected layer {layer_id}."
            )
        if result.layer_id != layer_id:
            raise RuntimeError(
                f"AFD FX graph reached layer {result.layer_id}, expected {layer_id}."
            )
        context.boundary = result
        return result.hidden_states

    def finalize_attention(
        self,
        hidden_states: torch.Tensor,
        context: AFDFXAttentionContext,
    ) -> Any:
        """Inject the final MoE output and return the graph's real output."""
        if context.boundary is None:
            raise RuntimeError("AFD FX graph has not reached a boundary.")
        result = self.program.resume(context.boundary, hidden_states)
        if isinstance(result, AFDGraphBoundary):
            raise RuntimeError(
                f"AFD FX graph still has pending layer {result.layer_id}."
            )
        context.boundary = None
        return result


def _build_segment(
    root: fx.GraphModule,
    group_id: int,
    nodes: list[fx.Node],
    cut: fx.Node | None,
    output_node: fx.Node,
    node_group: dict[fx.Node, int],
) -> AFDGraphSegment:
    graph = fx.Graph()
    internal = set(nodes)
    copied: dict[fx.Node, fx.Node] = {}
    input_nodes: list[fx.Node] = []

    def resolve(node: fx.Node) -> fx.Node:
        if node in copied:
            return copied[node]
        if node in internal:
            raise RuntimeError(f"AFD graph is not topologically ordered at {node}.")
        if node.op == "get_attr":
            copied[node] = graph.get_attr(str(node.target))
            return copied[node]
        placeholder = graph.placeholder(node.name)
        placeholder.meta = node.meta.copy()
        copied[node] = placeholder
        input_nodes.append(node)
        return placeholder

    for node in nodes:
        copied_node = graph.node_copy(node, resolve)
        copied_node.meta = node.meta.copy()
        copied[node] = copied_node

    if cut is None:
        final_output = map_arg(output_node.args[0], resolve)
        graph.output(final_output)
        segment = AFDGraphSegment(
            fx.GraphModule(root, graph),
            tuple(node.name for node in input_nodes),
            (),
            None,
            None,
        )
        segment.graph_module.graph.lint()
        segment.graph_module.recompile()
        return segment

    if len(cut.args) != 2 or not isinstance(cut.args[1], int):
        raise ValueError("afd_cut must receive (Tensor, constant int layer_id).")
    hidden_node = cut.args[0]
    if not isinstance(hidden_node, fx.Node):
        raise TypeError("afd_cut hidden input must be an FX node.")

    live_nodes = [
        node
        for node in nodes
        if any(
            user is not cut
            and (user is output_node or node_group.get(user, group_id) > group_id)
            for user in node.users
        )
    ]
    cut_output = (resolve(hidden_node), *(resolve(node) for node in live_nodes))
    graph.output(cut_output)
    segment = AFDGraphSegment(
        fx.GraphModule(root, graph),
        tuple(node.name for node in input_nodes),
        tuple(node.name for node in live_nodes),
        cut.args[1],
        cut.name,
    )
    segment.graph_module.graph.lint()
    segment.graph_module.recompile()
    return segment
