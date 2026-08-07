"""Native RunTrace + ToolCall provenance batching (D-CDX-33).

The direct delegation path used to commit each trace, outcome, provenance edge,
and tool call independently. These focused tests keep the native core-batch path
truthful: it must preserve the portable graph shape, keep the durable core
independent of auxiliary endpoints, and never turn an atomic authority failure
into partial writes.
"""

from __future__ import annotations

import copy
import json
import threading
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


def _all_batch_graph_shape(
    batches: list[list[dict[str, object]]],
) -> tuple[set[tuple[str, str]], set[tuple[str, str, str]]]:
    nodes: set[tuple[str, str]] = set()
    edges: set[tuple[str, str, str]] = set()
    for batch in batches:
        batch_nodes, batch_edges = _batch_graph_shape(batch)
        nodes.update(batch_nodes)
        edges.update(batch_edges)
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


def test_native_provenance_uses_core_and_optional_batches_instead_of_serial_writes():
    """One ServiceNow call writes a self-contained core, then optional links once."""
    engine = _NativeTraceEngine(
        {
            "srv:servicenow-mcp",
            "resource:skill:servicenow-incident-resolution",
        }
    )

    assert _record(engine, tool_calls=[_tool_call()])

    assert len(engine.graph.batch_reads) == 1
    assert len(engine.batches) == 2
    core_mutations, optional_mutations = engine.batches
    run_id = "run:provenance-batch"
    trace = trace_id(run_id)
    outcome = outcome_id(run_id)
    tool = _tool_call_id(run_id)
    assert len(core_mutations) == 5
    assert _batch_graph_shape(core_mutations) == (
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
                tool,
                "USED_TOOL",
            ),
        },
    )
    assert _batch_graph_shape(optional_mutations) == (
        set(),
        {
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

    assert _all_batch_graph_shape(native.batches) == _portable_graph_shape(portable)


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
    assert engine.graph.batch_reads == []
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
    assert len(engine.batches) == 2
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


def test_failed_auxiliary_preflight_keeps_the_native_core_without_serial_fallback():
    class _BrokenPreflightGraph(_ExistingGraph):
        def has_batch(self, node_ids: list[str]) -> dict[str, bool]:
            raise RuntimeError("availability probe failed")

    class _PreflightFailure(_NativeTraceEngine):
        def __init__(self) -> None:
            super().__init__()
            self.graph = _BrokenPreflightGraph()
            self.serial_writes = 0

        def add_node(self, *args: object, **kwargs: object) -> None:
            self.serial_writes += 1

        def link_nodes(self, *args: object, **kwargs: object) -> None:
            self.serial_writes += 1

    engine = _PreflightFailure()

    assert _record(engine, tool_calls=[_tool_call()])
    assert len(engine.batches) == 1
    assert engine.serial_writes == 0
    _, core_edges = _batch_graph_shape(engine.batches[0])
    assert {relationship for _, _, relationship in core_edges} == {
        "PRODUCED_OUTCOME",
        "USED_TOOL",
    }


def test_optional_endpoint_race_keeps_core_durable_without_serial_retry():
    class _EndpointDisappearsGraph(_ExistingGraph):
        def __init__(self, engine: Any) -> None:
            super().__init__()
            self.engine = engine

        def has_batch(self, node_ids: list[str]) -> dict[str, bool]:
            assert self.engine.core_durable
            self.batch_reads.append(list(node_ids))
            return {node_id: True for node_id in node_ids}

    class _EndpointRace(_NativeTraceEngine):
        def __init__(self) -> None:
            super().__init__()
            self.core_durable = False
            self.durable_batches: list[list[dict[str, object]]] = []
            self.serial_writes = 0
            self.graph = _EndpointDisappearsGraph(self)

        def batch_typed_mutations(self, mutations: list[dict[str, object]]) -> bool:
            self.batches.append(copy.deepcopy(mutations))
            if len(self.batches) == 1:
                self.durable_batches.append(copy.deepcopy(mutations))
                self.core_durable = True
                return True
            raise RuntimeError("Target node 'srv:servicenow-mcp' not found")

        def add_node(self, *args: object, **kwargs: object) -> None:
            self.serial_writes += 1

        def link_nodes(self, *args: object, **kwargs: object) -> None:
            self.serial_writes += 1

    engine = _EndpointRace()

    assert _record(engine, tool_calls=[_tool_call(target="incident:INC42")])
    assert len(engine.batches) == 2
    assert engine.durable_batches == [engine.batches[0]]
    assert engine.serial_writes == 0
    assert engine.graph.batch_reads == [
        [
            "srv:servicenow-mcp",
            "resource:skill:servicenow-incident-resolution",
            "incident:INC42",
        ]
    ]
    _, core_edges = _batch_graph_shape(engine.batches[0])
    assert {relationship for _, _, relationship in core_edges} == {
        "PRODUCED_OUTCOME",
        "USED_TOOL",
    }
    _, optional_edges = _batch_graph_shape(engine.batches[1])
    assert {relationship for _, _, relationship in optional_edges} == {
        "EXECUTED_ON",
        "USES_SKILL",
        "ACTED_ON",
    }


def test_contended_optional_batch_does_not_block_the_run_past_its_grace_period():
    """D-CDX-33: profiling showed the optional EXECUTED_ON/USES_SKILL batch
    costing roughly the SAME ~12s as the mandatory core batch against a
    contended shared engine, so provenance recording consumed 40-45% of a
    run's total wall clock -- well above the <10% target -- even though this
    enrichment is documented best-effort and the core RunTrace/Outcome/
    ToolCall write is ALREADY durable and readable by the time it starts.

    A contended ``has_batch`` preflight (simulated here with a real sleep
    well past the grace period) must not make ``_record_execution_trace``
    wait for it: the call must return promptly, the CORE batch must already
    be committed, and the optional batch completes in the background shortly
    after.
    """
    import threading
    import time

    class _ContendedGraph(_ExistingGraph):
        def __init__(self) -> None:
            super().__init__(
                {
                    "srv:servicenow-mcp",
                    "resource:skill:servicenow-incident-resolution",
                }
            )
            self.preflight_started = threading.Event()

        def has_batch(self, node_ids: list[str]) -> dict[str, bool]:
            self.preflight_started.set()
            # Well past agent_runner._OPTIONAL_PROVENANCE_GRACE_S (2.0s) --
            # simulates the "engine likely contended" ~12s RPC the profiled
            # runs actually measured, scaled down for a fast test.
            time.sleep(0.5)
            return super().has_batch(node_ids)

    engine = _NativeTraceEngine()
    engine.graph = _ContendedGraph()

    grace = agent_runner._OPTIONAL_PROVENANCE_GRACE_S
    try:
        agent_runner._OPTIONAL_PROVENANCE_GRACE_S = 0.05

        start = time.monotonic()
        result = _record(engine, tool_calls=[_tool_call()])
        elapsed = time.monotonic() - start
    finally:
        agent_runner._OPTIONAL_PROVENANCE_GRACE_S = grace

    assert result is True
    # The call returned well before the simulated 0.5s contended RPC
    # finished -- proof the caller was not blocked on it. A generous margin
    # (0.3s) keeps this robust against scheduler jitter while still being
    # far short of the 0.5s the old, fully-blocking call would have taken.
    assert elapsed < 0.3, f"run_agent's caller waited {elapsed:.3f}s -- the optional batch was NOT deferred"

    # The mandatory core batch is unaffected: durable immediately.
    assert len(engine.batches) == 1
    _, core_edges = _batch_graph_shape(engine.batches[0])
    assert {relationship for _, _, relationship in core_edges} == {
        "PRODUCED_OUTCOME",
        "USED_TOOL",
    }

    # The background thread is still doing its (now-started) work.
    assert engine.graph.preflight_started.wait(timeout=1.0)

    # Eventually the optional batch completes in the background.
    for _ in range(50):
        if len(engine.batches) == 2:
            break
        time.sleep(0.05)
    assert len(engine.batches) == 2, "optional batch never completed in the background"
    _, optional_edges = _batch_graph_shape(engine.batches[1])
    assert {relationship for _, _, relationship in optional_edges} == {
        "EXECUTED_ON",
        "USES_SKILL",
    }


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


def test_fanout_batch_commits_once_then_enqueues_ordered_mirror_mutations(
    tmp_path, monkeypatch
):
    authority = _CoreBatchBackend()
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.backends.fanout_backend._new_epistemic_authority",
        lambda: authority,
    )
    backend = FanOutBackend({}, outbox_path=str(tmp_path / "fanout.sqlite"))
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


def test_fanout_handoff_error_marks_the_authority_as_already_committed(
    tmp_path, monkeypatch
):
    authority = _CoreBatchBackend()
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.backends.fanout_backend._new_epistemic_authority",
        lambda: authority,
    )
    backend = FanOutBackend({}, outbox_path=str(tmp_path / "fanout.sqlite"))

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


def test_fanout_batch_holds_lifecycle_and_mutation_fences(tmp_path, monkeypatch):
    apply_started = threading.Event()
    release_apply = threading.Event()

    class _BlockingAuthority(_CoreBatchBackend):
        def apply_typed_batch(
            self, operations: list[dict[str, object]]
        ) -> dict[str, int]:
            apply_started.set()
            assert release_apply.wait(timeout=2)
            return super().apply_typed_batch(operations)

    authority = _BlockingAuthority()
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.backends.fanout_backend._new_epistemic_authority",
        lambda: authority,
    )
    backend = FanOutBackend({}, outbox_path=str(tmp_path / "fanout.sqlite"))
    node_id = "trace:fenced-batch"
    stripe = backend._mutation_lock(node_id)
    batch_done = threading.Event()
    stripe_acquired = threading.Event()

    def _write_batch() -> None:
        backend.apply_typed_batch(
            [
                {
                    "op": "upsert_node",
                    "id": node_id,
                    "properties": {"id": node_id, "node_type": "RunTrace"},
                }
            ]
        )
        batch_done.set()

    def _competing_writer() -> None:
        with stripe:
            stripe_acquired.set()

    writer = threading.Thread(target=_write_batch)
    competitor = threading.Thread(target=_competing_writer)
    writer.start()
    assert apply_started.wait(timeout=2)
    assert backend._active_producers == 1
    competitor.start()
    assert not stripe_acquired.wait(timeout=0.05)

    release_apply.set()
    writer.join(timeout=2)
    competitor.join(timeout=2)

    assert batch_done.is_set()
    assert stripe_acquired.is_set()
    assert backend._active_producers == 0
