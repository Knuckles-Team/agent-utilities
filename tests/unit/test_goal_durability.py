"""Durable goal registry + restart rehydration (CONCEPT:ORCH-1.44 / OS-5.18).

``active_goals``/``background_goal_runs`` used to be process memory only — a
gateway restart silently lost every running goal. Goals are now persisted in
the sessions store's ``goals`` table; on restart, this host's non-terminal
goals are surfaced as ``orphaned``, and the supervisory plane's
pause/kill desired-state requests are honored by the owning goal loop.
"""

from __future__ import annotations

import json
import sqlite3
import time

import pytest

from agent_utilities.core import sessions as _sessions
from agent_utilities.models.goal import GoalStatus


@pytest.fixture
def goal_db(tmp_path, monkeypatch):
    db = tmp_path / "sessions.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(_sessions._SQLITE_DDL)
    conn.commit()
    conn.close()
    monkeypatch.setattr(_sessions, "_get_db_path", lambda: db)
    monkeypatch.setattr(_sessions, "_rehydrated", False)
    monkeypatch.setattr(_sessions, "active_goals", {})
    monkeypatch.setattr(_sessions, "background_goal_runs", {})
    return db


def _goal_rows(db):
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute("SELECT * FROM goals").fetchall()]
    conn.close()
    return rows


def test_persist_goal_upserts(goal_db):
    _sessions.active_goals["g1"] = {
        "goal_id": "g1",
        "session_id": "s1",
        "status": GoalStatus.RUNNING,
        "objective": "do the thing",
        "iterations": [],
        "total_iterations": 0,
        "total_duration_ms": 0,
        "total_tool_calls": 0,
        "summary": "",
        "error": "",
    }
    _sessions._persist_goal("g1")
    rows = _goal_rows(goal_db)
    assert len(rows) == 1
    assert rows[0]["status"] == "running"
    assert rows[0]["objective"] == "do the thing"
    assert rows[0]["owner_host"] == _sessions._owner_token()

    # Update path
    _sessions.active_goals["g1"]["status"] = GoalStatus.COMPLETED
    _sessions.active_goals["g1"]["total_iterations"] = 3
    _sessions._persist_goal("g1")
    rows = _goal_rows(goal_db)
    assert len(rows) == 1
    assert rows[0]["status"] == "completed"
    assert rows[0]["total_iterations"] == 3


def _insert_goal(db, goal_id, status, owner, iterations=None):
    conn = sqlite3.connect(str(db))
    conn.execute(
        "INSERT INTO goals (goal_id, session_id, status, objective, owner_host, "
        "iterations_json, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
        (
            goal_id,
            f"sess-{goal_id}",
            status,
            "obj",
            owner,
            json.dumps(iterations or []),
            time.time(),
            time.time(),
        ),
    )
    conn.commit()
    conn.close()


def test_rehydrate_marks_dead_pid_goals_orphaned(goal_db):
    dead_owner = f"{_sessions._HOSTNAME}:999999999"
    _insert_goal(goal_db, "dead", "running", dead_owner)
    _insert_goal(goal_db, "done", "completed", dead_owner)  # terminal: untouched
    _insert_goal(goal_db, "foreign", "running", "other-host:1")  # other host's

    orphaned = _sessions.rehydrate_goals()
    assert orphaned == 1

    rows = {r["goal_id"]: r for r in _goal_rows(goal_db)}
    assert rows["dead"]["status"] == "orphaned"
    assert rows["done"]["status"] == "completed"
    assert rows["foreign"]["status"] == "running"

    # Visible in the in-memory cache (and therefore in /goals lists).
    assert "dead" in _sessions.active_goals
    assert _sessions.active_goals["dead"]["status"] == GoalStatus.ORPHANED

    # Once per process: a second call is a no-op.
    _insert_goal(goal_db, "late", "running", dead_owner)
    assert _sessions.rehydrate_goals() == 0


def test_rehydrate_skips_live_runs(goal_db):
    _insert_goal(goal_db, "live", "running", f"{_sessions._HOSTNAME}:1")
    _sessions.background_goal_runs["live"] = {"session_id": "sess-live"}
    assert _sessions.rehydrate_goals() == 0
    assert _goal_rows(goal_db)[0]["status"] == "running"


async def test_list_goals_includes_durable_goals(goal_db):
    _insert_goal(goal_db, "old", "orphaned", f"{_sessions._HOSTNAME}:2")

    class _Req:
        query_params: dict = {}

    resp = await _sessions.list_goals(_Req())
    data = json.loads(resp.body)
    ids = {g["goal_id"] for g in data}
    assert "old" in ids


def test_desired_session_action(goal_db):
    conn = sqlite3.connect(str(goal_db))
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
    conn = sqlite3.connect(str(goal_db))
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

    assert _sessions.active_goals["g-kill"]["status"] == GoalStatus.CANCELLED
    conn = sqlite3.connect(str(goal_db))
    status = conn.execute(
        "SELECT status FROM sessions WHERE id = 's-kill'"
    ).fetchone()[0]
    conn.close()
    assert status == "cancelled"
    rows = {r["goal_id"]: r for r in _goal_rows(goal_db)}
    assert rows["g-kill"]["status"] == "cancelled"


async def test_goal_loop_honors_pause_request(goal_db):
    conn = sqlite3.connect(str(goal_db))
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

    assert _sessions.active_goals["g-pause"]["status"] == GoalStatus.PAUSED
    conn = sqlite3.connect(str(goal_db))
    status = conn.execute(
        "SELECT status FROM sessions WHERE id = 's-pause'"
    ).fetchone()[0]
    conn.close()
    assert status == "paused"
