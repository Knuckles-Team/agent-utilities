"""Goal state collapsed onto the KG Loop node — one durable source of truth (KG-2.78)."""

from __future__ import annotations

import json

import agent_utilities.core.sessions as sessions
from agent_utilities.knowledge_graph.research.loops import active_loops


class _Engine:
    """Fake KG engine emulating the goal-node read/write surface."""

    def __init__(self):
        self.nodes: dict[str, dict] = {}

    def add_node(self, nid, ntype, properties=None):
        cur = self.nodes.get(nid, {})
        cur.update({"type": ntype, **(properties or {})})
        self.nodes[nid] = cur

    def _row(self, nid, n):
        return {
            "goal_id": nid,
            "session_id": n.get("session_id", ""),
            "status": n.get("status", "pending"),
            "objective": n.get("objective", ""),
            "owner_host": n.get("owner_host", ""),
            "summary": n.get("summary", ""),
            "error": n.get("error", ""),
            "total_iterations": n.get("total_iterations", 0),
            "total_duration_ms": n.get("total_duration_ms", 0),
            "total_tool_calls": n.get("total_tool_calls", 0),
            "iterations_json": n.get("iterations_json", "[]"),
            "validation_cmd": n.get("validation_cmd", ""),
            "max_iterations": n.get("max_iterations", 20),
            "updated_at": n.get("updated_at", 0),
            # active_loops fields
            "id": nid,
            "name": n.get("name", ""),
            "loop_kind": n.get("loop_kind", ""),
            "skill_ref": n.get("skill_ref", ""),
            "end_state": n.get("end_state", ""),
        }

    def query_cypher(self, q, params=None):
        params = params or {}
        if "c.id = $id" in q:
            nid = params.get("id")
            n = self.nodes.get(nid)
            return [self._row(nid, n)] if n else []
        if "c.loop_kind = 'develop'" in q:
            return [
                self._row(i, n)
                for i, n in self.nodes.items()
                if n.get("loop_kind") == "develop"
            ]
        if "ADDRESSED_BY" in q:
            return []
        # generic concept scan (active_loops)
        return [self._row(i, n) for i, n in self.nodes.items()]


def test_persist_goal_writes_to_kg_loop_node(monkeypatch):
    eng = _Engine()
    monkeypatch.setattr(sessions, "_goal_engine", lambda: eng)
    sessions.active_goals["loop:develop:g1"] = {
        "goal_id": "loop:develop:g1",
        "session_id": "sess-1",
        "status": "running",
        "objective": "fix the tests",
        "iterations": [{"iteration": 1, "is_complete": False}],
        "total_iterations": 1,
    }
    try:
        sessions._persist_goal("loop:develop:g1")
    finally:
        sessions.active_goals.pop("loop:develop:g1", None)
    node = eng.nodes["loop:develop:g1"]
    assert node["loop_kind"] == "develop" and node["status"] == "running"
    assert node["session_id"] == "sess-1"
    assert json.loads(node["iterations_json"])[0]["iteration"] == 1


def test_list_goal_entries_filters_to_goal_loops(monkeypatch):
    eng = _Engine()
    # a goal-originated develop loop (has session_id) and a bare develop loop (none)
    eng.add_node(
        "loop:develop:goal",
        "Concept",
        properties={"loop_kind": "develop", "session_id": "s", "objective": "g"},
    )
    eng.add_node(
        "loop:develop:bare",
        "Concept",
        properties={"loop_kind": "develop", "objective": "b"},
    )
    entries = sessions._list_goal_entries(eng)
    ids = {e["goal_id"] for e in entries}
    assert "loop:develop:goal" in ids
    assert "loop:develop:bare" not in ids  # no session_id → not a goal


def test_load_goal_entry_roundtrips_iterations(monkeypatch):
    eng = _Engine()
    eng.add_node(
        "loop:develop:g",
        "Concept",
        properties={
            "loop_kind": "develop",
            "session_id": "s",
            "status": "completed",
            "iterations_json": json.dumps([{"iteration": 1}, {"iteration": 2}]),
        },
    )
    entry = sessions._load_goal_entry(eng, "loop:develop:g")
    assert entry is not None
    assert entry["status"] == "completed"
    assert len(entry["iterations"]) == 2
    assert sessions._load_goal_entry(eng, "missing") is None


def test_active_loops_excludes_in_flight_running_develop_loops():
    eng = _Engine()
    eng.add_node(
        "loop:develop:run",
        "Concept",
        properties={"loop_kind": "develop", "status": "running", "name": "r"},
    )
    eng.add_node(
        "loop:develop:wait",
        "Concept",
        properties={"loop_kind": "develop", "status": "pending", "name": "w"},
    )
    ids = {loop["id"] for loop in active_loops(eng, limit=10)}
    assert "loop:develop:wait" in ids  # claimable
    assert "loop:develop:run" not in ids  # in-flight → skipped
