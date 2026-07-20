"""Durable goal registry + restart rehydration (CONCEPT:AU-KG.research.these-properties-carry / OS-5.18).

``active_goals``/``background_goal_runs`` are process memory; the durable source of
truth is now the **KG Loop node** (a develop ``Concept``) — the ``goals`` SQLite table
was collapsed onto the one Loop model. On restart, non-terminal goal Loops retain
their exact WorkItem status, and the supervisory plane's pause/kill desired-state
requests are honored by the owning goal loop (driven by ``LoopController.run_loop``).
"""

from __future__ import annotations

import json
import sqlite3
import time

import pytest

from agent_utilities.core import sessions as _sessions
from agent_utilities.orchestration.work_item import WorkItemStatus, loop_work_item_id


class _GoalEngine:
    """Fake KG engine: the goal Loop-node store (add_node + goal queries)."""

    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.work_items: dict[str, dict] = {}

    def add_node(self, nid, ntype, properties=None):
        cur = self.nodes.get(nid, {})
        cur.update({"type": ntype, **(properties or {})})
        self.nodes[nid] = cur

    def _row(self, nid, n):
        return {
            "goal_id": nid,
            "session_id": n.get("session_id", ""),
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
        }

    def query_cypher(self, q, params=None):
        params = params or {}
        if "MATCH (w:WorkItem" in q:
            item = self.work_items.get(params.get("id"))
            return [dict(item)] if item else []
        if "c.id = $id" in q:
            n = self.nodes.get(params.get("id"))
            return [self._row(params.get("id"), n)] if n else []
        if "c.loop_kind = 'develop'" in q:
            return [
                self._row(i, n)
                for i, n in self.nodes.items()
                if n.get("loop_kind") == "develop"
            ]
        return []


@pytest.fixture
def goal_db(tmp_path, monkeypatch):
    db = tmp_path / "sessions.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(_sessions._SQLITE_DDL)
    conn.commit()
    conn.close()
    eng = _GoalEngine()
    monkeypatch.setattr(_sessions, "_get_db_path", lambda: db)
    monkeypatch.setattr(_sessions, "_goal_engine", lambda: eng)
    monkeypatch.setattr(_sessions, "_rehydrated", False)
    monkeypatch.setattr(_sessions, "active_goals", {})
    monkeypatch.setattr(_sessions, "background_goal_runs", {})
    return eng  # the KG node store is the source of truth now


def test_persist_goal_upserts(goal_db):
    _sessions.active_goals["loop:develop:g1"] = {
        "goal_id": "loop:develop:g1",
        "session_id": "s1",
        "status": WorkItemStatus.RUNNING.value,
        "objective": "do the thing",
        "iterations": [],
        "total_iterations": 0,
    }
    _sessions._persist_goal("loop:develop:g1")
    node = goal_db.nodes["loop:develop:g1"]
    assert node["objective"] == "do the thing"
    assert node["owner_host"] == _sessions._owner_token()
    assert node["loop_kind"] == "develop"

    # Update path
    _sessions.active_goals["loop:develop:g1"]["status"] = (
        WorkItemStatus.SUCCEEDED.value
    )
    _sessions.active_goals["loop:develop:g1"]["total_iterations"] = 3
    _sessions._persist_goal("loop:develop:g1")
    node = goal_db.nodes["loop:develop:g1"]
    assert node["total_iterations"] == 3


def _add_goal(eng, goal_id, status, owner, iterations=None):
    eng.add_node(
        goal_id,
        "Concept",
        properties={
            "loop_kind": "develop",
            "session_id": f"sess-{goal_id}",
            "objective": "obj",
            "owner_host": owner,
            "iterations_json": json.dumps(iterations or []),
            "updated_at": time.time(),
        },
    )
    eng.work_items[loop_work_item_id(goal_id)] = {
        "id": loop_work_item_id(goal_id),
        "kind": "goal_loop",
        "status": status,
        "lease_expires_at": 0.0,
    }


def test_rehydrate_preserves_exact_work_item_status(goal_db):
    _add_goal(goal_db, "dead", WorkItemStatus.RUNNING.value, "worker")
    _add_goal(goal_db, "done", WorkItemStatus.SUCCEEDED.value, "worker")

    stranded = _sessions.rehydrate_goals()
    assert stranded == 1

    assert goal_db.work_items[loop_work_item_id("dead")]["status"] == "running"
    assert goal_db.work_items[loop_work_item_id("done")]["status"] == "succeeded"

    # Visible in the in-memory cache (and therefore in /goals lists).
    assert "dead" in _sessions.active_goals
    assert _sessions.active_goals["dead"]["status"] == WorkItemStatus.RUNNING.value

    # Once per process: a second call is a no-op.
    _add_goal(goal_db, "late", WorkItemStatus.RUNNING.value, "worker")
    assert _sessions.rehydrate_goals() == 0


def test_rehydrate_skips_live_runs(goal_db):
    _add_goal(goal_db, "live", WorkItemStatus.RUNNING.value, "worker")
    _sessions.background_goal_runs["live"] = {"session_id": "sess-live"}
    assert _sessions.rehydrate_goals() == 0
    assert goal_db.nodes["live"]["status"] == "running"


async def test_list_goals_includes_durable_goals(goal_db):
    _add_goal(goal_db, "old", WorkItemStatus.READY.value, "worker")

    class _Req:
        query_params: dict = {}

    resp = await _sessions.list_goals(_Req())
    data = json.loads(resp.body)
    ids = {g["goal_id"] for g in data}
    assert "old" in ids


def test_desired_session_action(goal_db):
    conn = sqlite3.connect(str(_sessions._get_db_path()))
    now = time.time()
    for sid, status in (
        ("s-p", "pause_requested"),
        ("s-k", "kill_requested"),
        ("s-r", "running"),
    ):
        conn.execute(
            "INSERT INTO sessions (id, created_at, updated_at, status) VALUES (?,?,?,?)",
            (sid, now, now, status),
        )
    conn.commit()
    conn.close()

    assert _sessions._desired_session_action("s-p") == "pause"
    assert _sessions._desired_session_action("s-k") == "kill"
    assert _sessions._desired_session_action("s-r") is None
    assert _sessions._desired_session_action("missing") is None


async def test_goal_loop_honors_kill_request(goal_db):
    conn = sqlite3.connect(str(_sessions._get_db_path()))
    now = time.time()
    conn.execute(
        "INSERT INTO sessions (id, created_at, updated_at, status) VALUES (?,?,?,?)",
        ("s-kill", now, now, "kill_requested"),
    )
    conn.commit()
    conn.close()

    await _sessions.run_goal_loop(
        session_id="s-kill",
        goal_id="g-kill",
        objective="obj",
        validation_cmd="",
        max_iterations=5,
        constraints=[],
    )

    assert _sessions.active_goals["g-kill"]["status"] == WorkItemStatus.CANCELLED.value
    conn = sqlite3.connect(str(_sessions._get_db_path()))
    status = conn.execute("SELECT status FROM sessions WHERE id = 's-kill'").fetchone()[
        0
    ]
    conn.close()
    assert status == "cancelled"
    assert goal_db.nodes["g-kill"]["status"] == "cancelled"


async def test_goal_loop_honors_pause_request(goal_db):
    conn = sqlite3.connect(str(_sessions._get_db_path()))
    now = time.time()
    conn.execute(
        "INSERT INTO sessions (id, created_at, updated_at, status) VALUES (?,?,?,?)",
        ("s-pause", now, now, "pause_requested"),
    )
    conn.commit()
    conn.close()

    await _sessions.run_goal_loop(
        session_id="s-pause",
        goal_id="g-pause",
        objective="obj",
        validation_cmd="",
        max_iterations=5,
        constraints=[],
    )

    assert _sessions.active_goals["g-pause"]["status"] == WorkItemStatus.READY.value
    conn = sqlite3.connect(str(_sessions._get_db_path()))
    status = conn.execute(
        "SELECT status FROM sessions WHERE id = 's-pause'"
    ).fetchone()[0]
    conn.close()
    assert status == "ready"
