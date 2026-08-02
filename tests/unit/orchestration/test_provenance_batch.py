"""Native RunTrace + ToolCall provenance batching (D-CDX-33).

The direct delegation path used to commit each trace, outcome, provenance edge,
and tool call independently.  These focused tests keep the native one-RPC path
truthful: it must preserve the portable graph shape, omit only unavailable
auxiliary links, and never turn an atomic authority failure into partial writes.
"""

from __future__ import annotations

import copy
import json
from typing import Any, cast

import pytest

from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
    EpistemicGraphBackend,
)
from agent_utilities.knowledge_graph.backends.fanout_backend import (
    AuthorityCommittedMirrorHandoffError,
    FanOutBackend,
)
from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine
from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.models.company_brain import ActorType
from agent_utilities.observability.trace_ontology import outcome_id, trace_id
from agent_utilities.orchestration import agent_runner
from agent_utilities.security.brain_context import ActorContext


class _ExistingGraph:
    def __init__(self, existing_ids: set[str] | None = None) -> None:
        self.existing_ids = existing_ids or set()
        self.batch_reads: list[list[str]] = []

    def has_batch(self, node_ids: list[str]) -> dict[str, bool]:
        self.batch_reads.append(list(node_ids))
        return {node_id: node_id in self.existing_ids for node_id in node_ids}

    def has_node(self, node_id: str) -> bool:
        return node_id in self.existing_ids


class _NativeTraceEngine:
    backend = object()

    def __init__(self, existing_ids: set[str] | None = None) -> None:
        self.graph = _ExistingGraph(existing_ids)
        self.batches: list[list[dict[str, object]]] = []

    def batch_typed_mutations(self, mutations: list[dict[str, object]]) -> bool:
        self.batches.append(copy.deepcopy(mutations))
        return True


class _PortableTraceEngine:
    backend = object()

    def __init__(self, existing_ids: set[str] | None = None) -> None:
        self.graph = _ExistingGraph(existing_ids)
        self.nodes: list[tuple[str, str, dict[str, object]]] = []
        self.edges: list[tuple[str, str, str]] = []

    def add_node(
        self,
        node_id: str,
        node_type: str,
        properties: dict[str, object] | None = None,
    ) -> None:
        self.nodes.append((node_id, node_type, dict(properties or {})))

    def link_nodes(
        self,
        source: str,
        target: str,
        rel_type: str,
        properties: dict[str, object] | None = None,
    ) -> None:
        self.edges.append((source, target, rel_type))


def _tool_call(*, target: str = "", error: str = "") -> dict[str, str]:
    args = {"incident_id": target} if target else {}
    return {
        "tool_name": "servicenow_get_incident",
        "args": json.dumps(args),
        "result": "incident details" if not error else "error: unavailable",
        "error": error,
    }


def _record(
    engine: object,
    *,
    run_id: str = "run:provenance-batch",
    tool_calls: list[dict[str, str]],
) -> bool:
    return agent_runner._record_execution_trace(
        engine,  # type: ignore[arg-type]
        run_id,
        "servicenow-mcp",
        "look up incident",
        status="completed",
        skill_used="servicenow-incident-resolution",
        bound_server="servicenow-mcp",
        skill_id="resource:skill:servicenow-incident-resolution",
        tool_calls=tool_calls,
        tool_call_server="servicenow-mcp",
    )


def _batch_graph_shape(
    mutations: list[dict[str, object]],
) -> tuple[set[tuple[str, str]], set[tuple[str, str, str]]]:
    nodes = {
        (str(mutation["id"]), str(mutation["node_type"]))
        for mutation in mutations
        if mutation["kind"] == "node"
    }
    edges = {
        (
            str(mutation["source"]),
            str(mutation["target"]),
            str(mutation["rel_type"]),
        )
        for mutation in mutations
        if mutation["kind"] == "edge"
    }
    return nodes, edges


def _portable_graph_shape(
    engine: _PortableTraceEngine,
) -> tuple[set[tuple[str, str]], set[tuple[str, str, str]]]:
    return (
        {(node_id, node_type) for node_id, node_type, _ in engine.nodes},
        set(engine.edges),
    )


def _tool_call_id(run_id: str, index: int = 0) -> str:
    return f"toolcall:{trace_id(run_id).removeprefix('trace:')}:{index}"


def test_native_provenance_uses_one_batch_for_the_previously_seven_writes():
    """One ServiceNow call has exactly one native write RPC, not seven serial ones."""
    engine = _NativeTraceEngine(
        {
            "srv:servicenow-mcp",
            "resource:skill:servicenow-incident-resolution",
        }
    )

    assert _record(engine, tool_calls=[_tool_call()])

    assert len(engine.graph.batch_reads) == 1
    assert len(engine.batches) == 1
    mutations = engine.batches[0]
    run_id = "run:provenance-batch"
    trace = trace_id(run_id)
    outcome = outcome_id(run_id)
    tool = _tool_call_id(run_id)
    # RunTrace + Outcome + PRODUCED_OUTCOME + EXECUTED_ON + USES_SKILL +
    # ToolCall + USED_TOOL = the seven authority operations from the profile.
    assert len(mutations) == 7
    assert _batch_graph_shape(mutations) == (
        {
            (trace, "RunTrace"),
            (outcome, "OutcomeEvaluation"),
            (tool, "ToolCall"),
        },
        {
            (
                trace,
                outcome,
                "PRODUCED_OUTCOME",
            ),
            (
                trace,
                "srv:servicenow-mcp",
                "EXECUTED_ON",
            ),
            (
                trace,
                "resource:skill:servicenow-incident-resolution",
                "USES_SKILL",
            ),
            (
                trace,
                tool,
                "USED_TOOL",
            ),
        },
    )


def test_native_batch_preserves_portable_provenance_shape():
    """The atomic path writes the same nodes and edges as the fallback path."""
    existing = {
        "srv:servicenow-mcp",
        "resource:skill:servicenow-incident-resolution",
        "incident:INC42",
    }
    native = _NativeTraceEngine(existing)
    portable = _PortableTraceEngine(existing)
    tool_calls = [_tool_call(target="incident:INC42"), _tool_call(error="denied")]

    assert _record(native, run_id="run:equivalent", tool_calls=tool_calls)
    assert _record(portable, run_id="run:equivalent", tool_calls=tool_calls)

    assert _batch_graph_shape(native.batches[0]) == _portable_graph_shape(portable)


def test_native_batch_omits_missing_auxiliary_endpoints_without_false_claims():
    engine = _NativeTraceEngine()

    assert _record(engine, tool_calls=[_tool_call(target="incident:missing")])

    assert len(engine.batches) == 1
    _, edges = _batch_graph_shape(engine.batches[0])
    relationships = {relationship for _, _, relationship in edges}
    assert {"PRODUCED_OUTCOME", "USED_TOOL"} <= relationships
    assert not {"EXECUTED_ON", "USES_SKILL", "ACTED_ON"} & relationships


def test_native_batch_failure_is_not_reinterpreted_as_serial_partial_success():
    class _FailingNative(_NativeTraceEngine):
        def __init__(self) -> None:
            super().__init__({"srv:servicenow-mcp"})
            self.serial_writes = 0

        def batch_typed_mutations(self, mutations: list[dict[str, object]]) -> bool:
            self.batches.append(copy.deepcopy(mutations))
            raise RuntimeError("authority rejected batch")

        def add_node(self, *args: object, **kwargs: object) -> None:
            self.serial_writes += 1

        def link_nodes(self, *args: object, **kwargs: object) -> None:
            self.serial_writes += 1

    engine = _FailingNative()

    assert not _record(engine, tool_calls=[_tool_call()])
    assert len(engine.batches) == 1
    assert engine.serial_writes == 0


def test_committed_authority_batch_is_not_reported_as_missing_after_mirror_error():
    class _CommittedButUnmirrored(_NativeTraceEngine):
        def __init__(self) -> None:
            super().__init__({"srv:servicenow-mcp"})
            self.serial_writes = 0

        def batch_typed_mutations(self, mutations: list[dict[str, object]]) -> bool:
            self.batches.append(copy.deepcopy(mutations))
            error = RuntimeError("mirror handoff unavailable")
            error.authority_committed = True  # type: ignore[attr-defined]
            raise error

        def add_node(self, *args: object, **kwargs: object) -> None:
            self.serial_writes += 1

        def link_nodes(self, *args: object, **kwargs: object) -> None:
            self.serial_writes += 1

    engine = _CommittedButUnmirrored()

    assert _record(engine, tool_calls=[_tool_call()])
    assert len(engine.batches) == 1
    assert engine.serial_writes == 0


def test_unavailable_native_batch_falls_back_to_portable_trace_and_tool_writes():
    engine = _PortableTraceEngine(
        {
            "srv:servicenow-mcp",
            "resource:skill:servicenow-incident-resolution",
        }
    )

    assert _record(engine, tool_calls=[_tool_call()])

    nodes, edges = _portable_graph_shape(engine)
    assert len(nodes) == 3
    assert len(edges) == 4
    assert (
        trace_id("run:provenance-batch"),
        _tool_call_id("run:provenance-batch"),
        "USED_TOOL",
    ) in edges


def test_failed_auxiliary_preflight_uses_portable_fallback():
    class _BrokenPreflightGraph(_ExistingGraph):
        def has_batch(self, node_ids: list[str]) -> dict[str, bool]:
            raise RuntimeError("availability probe failed")

    class _PreflightFallback(_PortableTraceEngine):
        def __init__(self) -> None:
            super().__init__()
            self.graph = _BrokenPreflightGraph()

        def batch_typed_mutations(self, mutations: list[dict[str, object]]) -> bool:
            raise AssertionError("preflight failure must not attempt the native batch")

    engine = _PreflightFallback()

    assert _record(engine, tool_calls=[_tool_call()])
    assert len(engine.nodes) == 3


class _CoreBatchBackend:
    def __init__(self) -> None:
        self.batches: list[list[dict[str, object]]] = []

    def apply_typed_batch(self, operations: list[dict[str, object]]) -> dict[str, int]:
        self.batches.append(copy.deepcopy(operations))
        return {"applied": len(operations)}


def _write_session() -> GraphSession:
    actor = ActorContext(
        actor_id="agent:provenance-test",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("kg:write",),
        tenant_id="test-tenant",
        authenticated=True,
    )
    return GraphSession(
        actor=actor,
        tenant="test-tenant",
        scopes=frozenset({"kg:write"}),
    )


def test_engine_prepares_governed_mutations_then_calls_native_backend_once():
    backend = _CoreBatchBackend()
    engine = cast(Any, object.__new__(IntelligenceGraphEngine))
    engine.backend = backend
    engine._compute_is_authority = True
    engine.active_schema_pack = None

    with use_session(_write_session()):
        assert engine.batch_typed_mutations(
            [
                {
                    "kind": "node",
                    "id": "trace:batch-core",
                    "node_type": "RunTrace",
                    "properties": {"run_id": "run:batch-core"},
                },
                {
                    "kind": "edge",
                    "source": "trace:batch-core",
                    "target": "outcome:batch-core",
                    "rel_type": "PRODUCED_OUTCOME",
                    "properties": {},
                },
            ]
        )

    assert len(backend.batches) == 1
    node, edge = backend.batches[0]
    assert node["op"] == "upsert_node"
    assert node["properties"]["tenant_id"] == "test-tenant"  # type: ignore[index]
    assert node["properties"]["classification"] == "confidential"  # type: ignore[index]
    assert edge["op"] == "upsert_edge"
    assert edge["properties"]["relationship"] == "PRODUCED_OUTCOME"  # type: ignore[index]
    assert edge["properties"]["confidence"] == 1.0  # type: ignore[index]


def test_engine_keeps_portable_path_when_compute_is_not_the_authority():
    backend = _CoreBatchBackend()
    engine = cast(Any, object.__new__(IntelligenceGraphEngine))
    engine.backend = backend
    engine._compute_is_authority = False

    assert not engine.batch_typed_mutations(
        [
            {
                "kind": "node",
                "id": "trace:separate-scratchpad",
                "node_type": "RunTrace",
                "properties": {},
            }
        ]
    )
    assert backend.batches == []


def test_epistemic_backend_forwards_one_ordered_native_batch():
    class _Graph:
        def __init__(self) -> None:
            self.batches: list[list[dict[str, object]]] = []

        def batch_update(self, operations: list[dict[str, object]]) -> dict[str, int]:
            self.batches.append(copy.deepcopy(operations))
            return {"applied": len(operations)}

    backend = cast(Any, object.__new__(EpistemicGraphBackend))
    backend._graph = _Graph()
    operations = [
        {"op": "upsert_node", "id": "x", "properties": {"node_type": "RunTrace"}}
    ]

    assert backend.apply_typed_batch(operations) == {"applied": 1}
    assert backend._graph.batches == [operations]


def test_fanout_batch_commits_once_then_enqueues_ordered_mirror_mutations():
    authority = _CoreBatchBackend()
    backend = cast(Any, object.__new__(FanOutBackend))
    backend._authority = authority
    backend._authority_writes = 0
    enqueued: list[tuple[str, dict[str, object]]] = []
    backend._enqueue = lambda op, payload: enqueued.append((op, payload))  # type: ignore[method-assign]
    operations = [
        {
            "op": "upsert_node",
            "id": "trace:fanout",
            "properties": {"id": "trace:fanout", "node_type": "RunTrace"},
        },
        {
            "op": "upsert_edge",
            "source": "trace:fanout",
            "target": "outcome:fanout",
            "properties": {"relationship": "PRODUCED_OUTCOME"},
        },
    ]

    backend.apply_typed_batch(operations)

    assert len(authority.batches) == 1
    assert backend._authority_writes == 1
    assert [op for op, _ in enqueued] == ["upsert_node", "upsert_edge"]


def test_fanout_handoff_error_marks_the_authority_as_already_committed():
    authority = _CoreBatchBackend()
    backend = cast(Any, object.__new__(FanOutBackend))
    backend._authority = authority
    backend._authority_writes = 0

    def _raise_after_authority(*_args: object, **_kwargs: object) -> None:
        raise OSError("outbox unavailable")

    backend._enqueue = _raise_after_authority
    with pytest.raises(AuthorityCommittedMirrorHandoffError) as raised:
        backend.apply_typed_batch(
            [
                {
                    "op": "upsert_node",
                    "id": "trace:handoff-failure",
                    "properties": {
                        "id": "trace:handoff-failure",
                        "node_type": "RunTrace",
                    },
                }
            ]
        )

    assert raised.value.authority_committed is True
    assert len(authority.batches) == 1
    assert backend._authority_writes == 1
