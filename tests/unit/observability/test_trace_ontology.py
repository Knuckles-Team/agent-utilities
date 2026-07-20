"""Canonical execution ontology, cursor, and privacy boundary contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

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


def test_canonical_trace_edges_are_single_authority() -> None:
    assert TRACE_USED_TOOL_EDGE == "USED_TOOL"
    assert TRACE_PRODUCED_OUTCOME_EDGE == "PRODUCED_OUTCOME"
    used_tool = next(edge for edge in SCHEMA.edges if edge.type == TRACE_USED_TOOL_EDGE)
    assert used_tool.connections == [{"from": "RunTrace", "to": "ToolCall"}]


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
