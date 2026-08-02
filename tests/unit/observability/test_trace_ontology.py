"""Canonical execution ontology, cursor, and privacy boundary contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest

from agent_utilities.knowledge_graph.core.company_brain_runtime import (
    get_company_brain,
    reset_company_brain,
)
from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.knowledge_graph.orchestration.engine_query import QueryMixin
from agent_utilities.models.company_brain import ActorType, DataClassification, NodeACL
from agent_utilities.models.graph import (
    GraphExecutionEvidence,
    GraphTaskEvidence,
    GraphTransitionEvidence,
)
from agent_utilities.models.schema_definition import SCHEMA
from agent_utilities.observability.trace_ontology import (
    TRACE_PRODUCED_OUTCOME_EDGE,
    TRACE_USED_TOOL_EDGE,
    TraceCursor,
    load_trace_cursor,
    outcome_properties,
    save_trace_cursor,
    tool_call_properties,
    trace_properties,
)
from agent_utilities.security.brain_context import ActorContext, use_actor

_GRAPH_EVIDENCE = GraphExecutionEvidence(
    topology="multi_agent",
    topology_digest="sha256:topology",
    version_digest="sha256:version",
    runtime_version="2.21.0",
    node_sequence=["router", "dispatcher", "__end__"],
    transitions=[
        GraphTransitionEvidence(
            sequence=1,
            scheduled_tasks=[
                GraphTaskEvidence(node_id="router", task_id="task:router")
            ],
        ),
        GraphTransitionEvidence(
            sequence=2,
            scheduled_tasks=[
                GraphTaskEvidence(node_id="dispatcher", task_id="task:dispatcher")
            ],
        ),
        GraphTransitionEvidence(
            sequence=3,
            scheduled_tasks=[GraphTaskEvidence(node_id="__end__", task_id="task:end")],
        ),
    ],
    checkpoint_ids=["ckpt:fixture:1"],
)


@dataclass
class _TraceBackend:
    """Capture the governed temporal query without replacing its read seam."""

    rows: list[dict[str, Any]]
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def execute_read(self, query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        self.calls.append((query, params))
        return list(self.rows)


class _TraceQueryHarness:
    """Minimal host that invokes the production ``QueryMixin`` read path."""

    def __init__(self, backend: _TraceBackend) -> None:
        # The test backend deliberately implements only the governed read seam;
        # the production mixin's broader engine protocol is not exercised here.
        self.backend: Any = backend
        self.control_backend = None

    def query_cypher(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        clearance_level: int = 999,
        as_of: str | None = None,
        *,
        session: GraphSession | None = None,
        include_epistemic: bool = False,
    ) -> list[dict[str, Any]]:
        return QueryMixin.query_cypher(
            cast(QueryMixin, self),
            query,
            params,
            clearance_level,
            as_of,
            session=session,
            include_epistemic=include_epistemic,
        )

    def retrieve_orthogonal_context(
        self,
        query: str,
        views: list[str] | None = None,
    ) -> dict[str, Any]:
        return QueryMixin.retrieve_orthogonal_context(
            cast(QueryMixin, self), query, views
        )


@pytest.fixture
def trace_brain():
    reset_company_brain()
    yield get_company_brain()
    reset_company_brain()


def test_canonical_trace_edges_are_single_authority() -> None:
    assert TRACE_USED_TOOL_EDGE == "USED_TOOL"
    assert TRACE_PRODUCED_OUTCOME_EDGE == "PRODUCED_OUTCOME"
    used_tool = next(edge for edge in SCHEMA.edges if edge.type == TRACE_USED_TOOL_EDGE)
    assert used_tool.connections == [{"from": "RunTrace", "to": "ToolCall"}]


def test_temporal_view_returns_ordered_canonical_traces_under_tenant_scope(
    trace_brain,
) -> None:
    """The active consumer keeps its result rows and governed query boundary."""
    trace_ids = ("trace:latest", "trace:middle", "trace:earliest")
    rows: list[dict[str, dict[str, Any]]] = [
        {
            "r": {
                "id": "trace:latest",
                "event_sequence": 30,
                "tenant_id": "tenant-a",
            }
        },
        {
            "r": {
                "id": "trace:middle",
                "event_sequence": 20,
                "tenant_id": "tenant-a",
            }
        },
        {
            "r": {
                "id": "trace:earliest",
                "event_sequence": 10,
                "tenant_id": "tenant-a",
            }
        },
    ]
    for node_id in trace_ids:
        trace_brain.permissions.set_acl(
            NodeACL(
                node_id=node_id,
                classification=DataClassification.PUBLIC,
            )
        )
    backend = _TraceBackend(rows=rows)
    engine = _TraceQueryHarness(backend)
    actor = ActorContext(
        actor_id="agent:temporal-reader",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("kg:read",),
        tenant_id="tenant-a",
        authenticated=True,
    )
    session = GraphSession(
        actor=actor,
        tenant=actor.tenant_id,
        scopes=frozenset({"kg:read"}),
        policy_version="trace-test",
        audience="agent-services",
    )

    with use_actor(actor), use_session(session):
        context = engine.retrieve_orthogonal_context(
            "recent activity", views=["temporal"]
        )

    assert context == {"query": "recent activity", "views": {"temporal": rows}}
    assert len(backend.calls) == 1
    cypher, params = backend.calls[0]
    assert "MATCH (r:RunTrace)" in cypher
    assert "ORDER BY r.event_sequence DESC LIMIT 5" in cypher
    assert "Episode" not in cypher
    assert "r.tenant_id = $_tenant_scope_id" in cypher
    assert params["_tenant_scope_id"] == "tenant-a"


def test_trace_cursor_advances_numerically_from_rows() -> None:
    cursor = TraceCursor.from_rows(
        [{"event_sequence": 9}, {"event_sequence": "11"}, {"event_sequence": 10}]
    )
    assert cursor == TraceCursor(11)


def test_trace_consumer_cursor_is_graph_resident_monotonic_and_opaque() -> None:
    class _Engine:
        def __init__(self) -> None:
            self.nodes: dict[str, dict] = {}

        def query_cypher(self, query: str, params: dict) -> list[dict]:
            matching = [
                node
                for node in self.nodes.values()
                if node.get("consumer_ref") == params.get("consumer_ref")
            ]
            return sorted(
                ({"event_sequence": node["event_sequence"]} for node in matching),
                key=lambda row: row["event_sequence"],
                reverse=True,
            )[:1]

        def add_node(self, node_id: str, node_type: str, properties: dict) -> None:
            self.nodes[node_id] = {"type": node_type, **properties}

    engine = _Engine()
    consumer = "fixture-incremental-consumer"
    assert load_trace_cursor(engine, consumer) == TraceCursor()
    assert save_trace_cursor(engine, consumer, 12) == TraceCursor(12)
    assert save_trace_cursor(engine, consumer, 7) == TraceCursor(12)
    assert load_trace_cursor(engine, consumer) == TraceCursor(12)
    assert consumer not in str(engine.nodes)
    assert any(
        node.get("cursor_kind") == "checkpoint" for node in engine.nodes.values()
    )
    assert all(node.get("cursor_kind") != "head" for node in engine.nodes.values())
    assert load_trace_cursor(engine, consumer) == TraceCursor(12)


def test_trace_cursor_authority_failures_are_explicit_and_sanitized() -> None:
    class _FailedEngine:
        def query_cypher(self, *_args, **_kwargs):
            raise ConnectionError("private endpoint")

    with pytest.raises(RuntimeError, match="authority read failed") as exc_info:
        load_trace_cursor(_FailedEngine(), "consumer")
    assert "private endpoint" not in str(exc_info.value)
    with pytest.raises(RuntimeError, match="requires graph authority"):
        load_trace_cursor(None, "consumer")


def test_runtime_properties_sanitize_machine_locations_and_identity() -> None:
    trace = trace_properties(
        run_id="fixture-run",
        agent_name="fixture-agent",
        task="inspect /home/example-user/private/input.txt",
        status="failed",
        timestamp="2026-01-01T00:00:00Z",
        error="failed at C:\\Users\\agent-user\\private.txt",
        event_sequence=7,
        execution_mode="single_server_agent",
        graph_execution_evidence=_GRAPH_EVIDENCE.model_dump(),
    )
    call = tool_call_properties(
        run_id="fixture-run",
        tool_name="fixture_tool",
        args={"path": "/home/example-user/private/input.txt"},
        result="read /home/example-user/private/input.txt",
        error="failed at C:\\Users\\agent-user\\private.txt",
        status="error",
        sequence=0,
        timestamp="2026-01-01T00:00:00Z",
        event_sequence=8,
    )
    outcome = outcome_properties(
        run_id="fixture-run",
        status="failed",
        timestamp="2026-01-01T00:00:00Z",
        event_sequence=7,
        feedback="failed at /home/example-user/private/input.txt",
    )
    persisted = str({"trace": trace, "call": call, "outcome": outcome})
    assert "/home/example-user" not in persisted
    assert "C:\\Users\\agent-user" not in persisted
    assert "fixture-agent" not in persisted
    assert "trace:fixture-run" not in persisted
    assert trace["task"] == ""
    assert trace["error"] == ""
    assert trace["execution_mode"] == "single_server_agent"
    assert trace["graph_evidence_schema_version"] == "graph-execution-evidence-v1"
    assert trace["graph_topology"] == "multi_agent"
    assert trace["graph_node_sequence"] == ["router", "dispatcher", "__end__"]
    assert trace["graph_transition_count"] == 3
    assert trace["graph_checkpoint_ids"] == ["ckpt:fixture:1"]
    assert trace["graph_resume_supported"] is False
    assert call["args"] == ""
    assert call["result"] == ""
    assert call["error"] == ""
    assert outcome["feedback_text"] == ""
    assert trace["task_digest"].startswith("pref_trace_content_")

    columns = {table.name: set(table.columns) for table in SCHEMA.nodes}
    assert set(trace) <= columns["RunTrace"]
    assert set(call) <= columns["ToolCall"]
    assert set(outcome) <= columns["OutcomeEvaluation"]


def test_active_trace_consumers_do_not_reintroduce_legacy_edge_or_episode_query() -> (
    None
):
    root = Path(__file__).resolve().parents[3] / "agent_utilities"
    targets = (
        root / "orchestration" / "agent_runner.py",
        root / "capabilities" / "hooks.py",
        root / "knowledge_graph" / "research" / "trace_pattern_miner.py",
        root / "knowledge_graph" / "research" / "placement_mining.py",
        root / "harness" / "trace_examples.py",
        root / "knowledge_graph" / "retrieval" / "context_compiler.py",
        root / "knowledge_graph" / "orchestration" / "engine_ahe.py",
        root / "knowledge_graph" / "orchestration" / "engine_query.py",
        root / "runtime" / "provenance.py",
        root / "workflows" / "runner.py",
    )
    for target in targets:
        source = target.read_text(encoding="utf-8")
        assert "MADE_TOOL_CALL" not in source
        assert "MATCH (e:Episode)" not in source


def test_lifecycle_hooks_cannot_write_a_parallel_tool_trace_shape() -> None:
    root = Path(__file__).resolve().parents[3] / "agent_utilities"
    hooks_source = (root / "capabilities" / "hooks.py").read_text(encoding="utf-8")
    factory_source = (root / "agent" / "factory.py").read_text(encoding="utf-8")
    model_source = (root / "models" / "knowledge_graph.py").read_text(encoding="utf-8")

    for forbidden in (
        "ToolCallNode",
        "auto_graph_trace",
        "USED_TOOL",
        "graph.add_node",
    ):
        assert forbidden not in hooks_source
    assert "auto_graph_trace" not in factory_source
    assert "class ToolCallNode" not in model_source
