"""Truthful execution evidence for the Pydantic Graph runtime.

CONCEPT:AU-OS.observability.telemetry-observability

Pydantic Graph's convenience ``Graph.run`` consumes its public ``GraphRun``
iterator internally.  This module observes those same public iterator events so
the blocking and explicit-iterator paths produce one evidence contract without
patching the graph object or claiming that write-only state snapshots can resume
execution.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version
from typing import Any, cast

from agent_utilities.models import (
    GraphExecutionEvidence,
    GraphTaskEvidence,
    GraphTransitionEvidence,
)

_EVIDENCE_SCHEMA_VERSION = "graph-execution-evidence-v1"


def _content_digest(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _runtime_version() -> str:
    try:
        return version("pydantic-graph")
    except PackageNotFoundError:
        return ""


def _render_topology(graph: Any) -> str:
    """Return a deterministic structural rendering when the graph exposes one."""

    render = getattr(graph, "render", None)
    if not callable(render):
        return ""
    try:
        rendered = render()
    except Exception:  # noqa: BLE001 - evidence must not fail the graph run
        return ""
    if not isinstance(rendered, str) or not rendered.strip():
        return ""
    return "\n".join(line.rstrip() for line in rendered.strip().splitlines())


class GraphExecutionEvidenceCollector:
    """Collect ordered scheduler events and actual persisted checkpoint IDs."""

    def __init__(self, graph: Any, *, topology: str) -> None:
        rendered = _render_topology(graph)
        runtime_version = _runtime_version()
        topology_digest = _content_digest(rendered) if rendered else ""
        version_payload = json.dumps(
            {
                "evidence_schema": _EVIDENCE_SCHEMA_VERSION,
                "runtime_version": runtime_version,
                "topology_digest": topology_digest,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        self.topology = str(topology or "")
        self.topology_digest = topology_digest
        self.version_digest = _content_digest(version_payload)
        self.runtime_version = runtime_version
        self.node_sequence: list[str] = []
        self.transitions: list[GraphTransitionEvidence] = []
        self.checkpoint_ids: list[str] = []
        self._span: Any | None = None

    def bind_state(self, state: Any) -> None:
        """Seed the mutable graph state with its immutable execution identity."""

        state.graph_topology_digest = self.topology_digest
        state.graph_version_digest = self.version_digest
        state.graph_runtime_version = self.runtime_version

    def attach_span(self, span: Any | None) -> None:
        self._span = span

    def record_event(self, event: Any, *, state: Any | None = None) -> None:
        """Record one public ``GraphRun`` task-batch event, if present."""

        if not isinstance(event, Sequence) or isinstance(event, str | bytes):
            return
        tasks: list[GraphTaskEvidence] = []
        for item in event:
            node_id = getattr(item, "node_id", None)
            task_id = getattr(item, "task_id", None)
            if node_id is None or task_id is None:
                continue
            tasks.append(GraphTaskEvidence(node_id=str(node_id), task_id=str(task_id)))
        if not tasks:
            return

        transition = GraphTransitionEvidence(
            sequence=len(self.transitions) + 1,
            scheduled_tasks=tasks,
        )
        self.transitions.append(transition)
        self.node_sequence.extend(task.node_id for task in tasks)

        if state is not None:
            state.graph_node_sequence = list(self.node_sequence)
            state.graph_transition_sequence = [
                item.model_dump() for item in self.transitions
            ]

        span = self._span
        if span is not None:
            try:
                span.add_event(
                    "pydantic_graph.transition",
                    attributes={
                        "sequence": transition.sequence,
                        "node_ids": tuple(task.node_id for task in tasks),
                        "task_ids": tuple(task.task_id for task in tasks),
                    },
                )
            except Exception:  # noqa: BLE001 - tracing must never fail execution
                pass

    def refresh_checkpoints(self, state: Any) -> None:
        """Read only checkpoint identifiers confirmed by the checkpoint backend."""

        raw = getattr(state, "checkpoint_ids", None)
        if not isinstance(raw, list):
            self.checkpoint_ids = []
            return
        self.checkpoint_ids = [
            str(checkpoint_id)
            for checkpoint_id in raw
            if isinstance(checkpoint_id, str) and checkpoint_id
        ]

    def evidence(self, *, state: Any | None = None) -> GraphExecutionEvidence:
        if state is not None:
            self.refresh_checkpoints(state)
        return GraphExecutionEvidence(
            topology=self.topology,
            topology_digest=self.topology_digest,
            version_digest=self.version_digest,
            runtime_version=self.runtime_version,
            node_sequence=list(self.node_sequence),
            transitions=list(self.transitions),
            checkpoint_ids=list(self.checkpoint_ids),
        )

    def finish_span(self, *, state: Any | None = None) -> None:
        evidence = self.evidence(state=state)
        span = self._span
        if span is None:
            return
        try:
            span.set_attribute(
                "agent_utilities.graph.topology_digest", evidence.topology_digest
            )
            span.set_attribute(
                "agent_utilities.graph.version_digest", evidence.version_digest
            )
            span.set_attribute(
                "agent_utilities.graph.runtime_version", evidence.runtime_version
            )
            span.set_attribute(
                "agent_utilities.graph.node_sequence", tuple(evidence.node_sequence)
            )
            span.set_attribute(
                "agent_utilities.graph.transition_count", len(evidence.transitions)
            )
            span.set_attribute(
                "agent_utilities.graph.checkpoint_ids",
                tuple(evidence.checkpoint_ids),
            )
            span.set_attribute(
                "agent_utilities.graph.resume_supported",
                evidence.resume_supported,
            )
            for checkpoint_id in evidence.checkpoint_ids:
                span.add_event(
                    "pydantic_graph.checkpoint",
                    attributes={"checkpoint_id": checkpoint_id},
                )
        except Exception:  # noqa: BLE001 - tracing must never fail execution
            pass


class _ObservedGraphRun:
    """Public-API wrapper used by ``Graph.run`` to expose every task batch."""

    def __init__(
        self,
        graph_run: Any,
        collector: GraphExecutionEvidenceCollector,
        state: Any,
    ) -> None:
        self._graph_run = graph_run
        self._collector = collector
        self._state = state
        self._started = False

    async def next(self, value: Any = None) -> Any:
        # Pydantic Graph 2.21's convenience ``GraphRun.next`` intentionally
        # consumes the first iterator event internally.  Drive the same public
        # ``__anext__``/``override_next`` operations here so that event is
        # observed too.  Older 2.x implementations without ``override_next``
        # fall back to their own ``next`` method (with partial evidence rather
        # than altered execution semantics).
        if not hasattr(self._graph_run, "override_next"):
            result = await self._graph_run.next(value)
            self._collector.record_event(result, state=self._state)
            return result

        if not self._started:
            first = await anext(self._graph_run)
            self._collector.record_event(first, state=self._state)
            self._started = True
        if value is not None:
            self._graph_run.override_next(value)
        result = await anext(self._graph_run)
        self._collector.record_event(result, state=self._state)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._graph_run, name)


class _ObservedGraph:
    """Non-mutating proxy that lets the real ``Graph.run`` observe ``iter``."""

    def __init__(
        self,
        graph: Any,
        collector: GraphExecutionEvidenceCollector,
        state: Any,
    ) -> None:
        self._graph = graph
        self._collector = collector
        self._state = state

    @property
    def name(self) -> str | None:
        return self._graph.name

    @name.setter
    def name(self, value: str | None) -> None:
        self._graph.name = value

    @asynccontextmanager
    async def iter(self, **kwargs: Any):
        async with self._graph.iter(**kwargs) as graph_run:
            yield _ObservedGraphRun(graph_run, self._collector, self._state)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._graph, name)


async def run_with_execution_evidence(
    graph: Any,
    *,
    state: Any,
    deps: Any,
    collector: GraphExecutionEvidenceCollector,
) -> Any:
    """Run a graph while preserving a direct call to Pydantic Graph ``Graph.run``."""

    try:
        # The top-level ``pydantic_graph.Graph`` export is the deprecated
        # BaseNode runner in current releases.  Agent Utilities builds graphs
        # with GraphBuilder, whose concrete runtime lives in graph_builder.
        from pydantic_graph.graph_builder import Graph as BuilderGraph
    except ImportError:
        BuilderGraph = None  # type: ignore[assignment,misc]

    if BuilderGraph is not None and isinstance(graph, BuilderGraph):
        observed = _ObservedGraph(graph, collector, state)
        # Invoke the installed Pydantic Graph implementation itself.  The proxy
        # changes only which public ``iter`` context it receives; the topology
        # object remains untouched and safe for concurrent cached reuse.
        return await BuilderGraph.run(cast(Any, observed), state=state, deps=deps)

    # Test doubles and third-party graph-shaped objects keep their own run
    # implementation.  We do not invent a transition sequence they never emit.
    return await graph.run(state=state, deps=deps)
