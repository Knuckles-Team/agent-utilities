"""ToolCall -[:ACTED_ON]-> target provenance edge (G23, audit-trail closure).

CONCEPT:AU-KG.audit.tool-call-acted-on-reverse-index

``agent_runner._persist_tool_calls`` best-effort links a persisted ``:ToolCall``
to a recognizable target entity id in its args, so
``Orchestrator.get_tool_calls_for_target`` can reconstruct "what happened to
entity X" without a new engine primitive. These tests exercise the extraction
helper and the write path against a minimal fake engine — no real KG backend.
"""

from __future__ import annotations

import json

from agent_utilities.orchestration.agent_runner import (
    _extract_tool_call_target,
    _persist_tool_calls,
)


class _FakeGraph:
    def __init__(self, existing_ids=()):
        self.existing = set(existing_ids)

    def has_node(self, node_id):
        return node_id in self.existing


class _FakeEngine:
    def __init__(self, existing_ids=()):
        self.node_calls: list[tuple[str, str, dict]] = []
        self.edge_calls: list[tuple[str, str, str]] = []
        self.graph = _FakeGraph(existing_ids)

    def add_node(self, node_id, node_type, properties=None):
        self.node_calls.append((node_id, node_type, dict(properties or {})))

    def link_nodes(self, source, target, rel_type, properties=None):
        self.edge_calls.append((source, target, rel_type))


def _tc(tool_name="engine_nodes", args=None, result="ok", error=""):
    return {
        "tool_name": tool_name,
        "args": json.dumps(args or {}),
        "result": result,
        "error": error,
    }


# ── _extract_tool_call_target ───────────────────────────────────────────────


def test_extract_target_finds_recognized_key():
    args = json.dumps({"incident_id": "incident:INC1", "note": "x"})
    assert _extract_tool_call_target(args) == "incident:INC1"


def test_extract_target_prefers_first_key_in_priority_order():
    # node_id is checked before id in _TOOL_ARG_TARGET_KEYS.
    args = json.dumps({"id": "id:2", "node_id": "node:1"})
    assert _extract_tool_call_target(args) == "node:1"


def test_extract_target_missing_key_returns_empty():
    assert _extract_tool_call_target(json.dumps({"unrelated": "x"})) == ""


def test_extract_target_non_json_string_returns_empty():
    assert _extract_tool_call_target("not json") == ""


def test_extract_target_non_dict_json_returns_empty():
    assert _extract_tool_call_target(json.dumps([1, 2, 3])) == ""


def test_extract_target_empty_or_non_string_returns_empty():
    assert _extract_tool_call_target("") == ""
    assert _extract_tool_call_target(None) == ""


# ── _persist_tool_calls ACTED_ON wiring ─────────────────────────────────────


def test_persist_tool_calls_links_acted_on_when_target_exists():
    engine = _FakeEngine(existing_ids={"incident:INC1"})
    tcs = [_tc(args={"incident_id": "incident:INC1"})]
    written = _persist_tool_calls(engine, "run:1", "agent-x", "server-y", tcs)
    assert written == 1
    tc_id = "toolcall:1:0"
    assert (tc_id, "incident:INC1", "ACTED_ON") in engine.edge_calls


def test_persist_tool_calls_skips_acted_on_when_target_missing():
    """A candidate id that resolves to no real node never vivifies a phantom."""
    engine = _FakeEngine(existing_ids=set())
    tcs = [_tc(args={"incident_id": "incident:DOES-NOT-EXIST"})]
    _persist_tool_calls(engine, "run:1", "agent-x", "server-y", tcs)
    assert not any(e[2] == "ACTED_ON" for e in engine.edge_calls)


def test_persist_tool_calls_no_candidate_writes_no_acted_on_edge():
    engine = _FakeEngine(existing_ids={"whatever"})
    tcs = [_tc(args={"unrelated_field": "x"})]
    _persist_tool_calls(engine, "run:1", "agent-x", "server-y", tcs)
    assert not any(e[2] == "ACTED_ON" for e in engine.edge_calls)
    # The MADE_TOOL_CALL provenance edge still lands regardless.
    assert any(e[2] == "MADE_TOOL_CALL" for e in engine.edge_calls)


def test_persist_tool_calls_best_effort_on_has_node_error():
    class _RaisingGraph:
        def has_node(self, node_id):
            raise RuntimeError("engine unreachable")

    engine = _FakeEngine()
    engine.graph = _RaisingGraph()
    tcs = [_tc(args={"incident_id": "incident:INC1"})]
    # Must not raise — the ToolCall + MADE_TOOL_CALL write still succeeds.
    written = _persist_tool_calls(engine, "run:1", "agent-x", "server-y", tcs)
    assert written == 1
