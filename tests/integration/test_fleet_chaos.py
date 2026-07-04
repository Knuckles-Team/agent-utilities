"""Multi-agent containment + recovery chaos (CONCEPT:AU-OS.state.unified-durable-state-externalization / OS-5.18).

Exercises the supervisory plane and durable goal loop together, in-process:

* whole-domain pause via the real ``/fleet/pause`` handler contains exactly the
  targeted domain and leaves others running;
* a goal loop honors a supervisor's ``pause_requested`` desired state and, having
  reconciled before doing any work, applies ZERO side effects;
* many concurrent goal loops behave independently — paused ones do nothing,
  unpaused ones run their effect — proving the durable/desired-state wiring holds
  under fan-out, not just for a single loop.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time

import pytest

from agent_utilities.core import sessions as _sessions
from agent_utilities.gateway import fleet
from agent_utilities.models.goal import GoalStatus

pytestmark = pytest.mark.integration


class _Req:
    def __init__(self, body=None, query=None):
        self._body = body or {}
        self.query_params = query or {}

    async def json(self):
        return self._body


async def _payload(resp):
    return json.loads(resp.body)


@pytest.fixture
def fleet_env(tmp_path, monkeypatch):
    db = tmp_path / "sessions.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(_sessions._SQLITE_DDL)
    conn.commit()
    conn.close()
    monkeypatch.setattr(_sessions, "_get_db_path", lambda: db)
    monkeypatch.setattr(fleet._sessions, "_get_db_path", lambda: db)
    monkeypatch.setattr(_sessions, "_rehydrated", True)
    monkeypatch.setattr(_sessions, "active_goals", {})
    monkeypatch.setattr(_sessions, "background_goal_runs", {})
    # Pin sqlite state even if the dev checkout's .env externalizes to Postgres
    # (STATE_DB_URI) — otherwise _connect_db/_select_store bypass the tmp sqlite.
    monkeypatch.delenv("STATE_DB_URI", raising=False)
    monkeypatch.setattr(
        "agent_utilities.core.state_store.postgres_state_enabled", lambda: False
    )
    monkeypatch.setenv("DURABLE_EXECUTION_DB", str(tmp_path / "durable.db"))

    async def _fast_sleep(_delay):
        return None

    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)
    return tmp_path, db


def _insert_session(db, sid, status="active", domain=None):
    meta = json.dumps({"domain": domain}) if domain else "{}"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "INSERT INTO sessions (id, status, created_at, updated_at, metadata_json, turn_count) "
        "VALUES (?,?,?,?,?,0)",
        (sid, status, time.time(), time.time(), meta),
    )
    conn.commit()
    conn.close()


def _session_status(db, sid):
    conn = sqlite3.connect(str(db))
    row = conn.execute("SELECT status FROM sessions WHERE id = ?", (sid,)).fetchone()
    conn.close()
    return row[0] if row else None


async def test_domain_pause_contains_only_target_domain(fleet_env):
    _, db = fleet_env
    _insert_session(db, "fin-1", domain="finance")
    _insert_session(db, "fin-2", domain="finance")
    _insert_session(db, "ops-1", domain="itops")

    resp = await fleet.fleet_pause(_Req(body={"domain": "finance"}))
    data = await _payload(resp)
    assert data["status"] == "success"
    assert set(data["affected"]) == {"fin-1", "fin-2"}
    assert _session_status(db, "fin-1") == "paused"
    assert _session_status(db, "fin-2") == "paused"
    assert _session_status(db, "ops-1") == "active"  # untouched blast radius


async def test_goal_loop_honors_pause_with_zero_effects(fleet_env):
    tmp_path, db = fleet_env
    marker = tmp_path / "paused.txt"
    _insert_session(db, "sp", status="pause_requested")

    await _sessions.run_goal_loop(
        session_id="sp",
        goal_id="gp",
        objective="obj",
        validation_cmd=f"printf x >> {marker}; exit 1",
        max_iterations=5,
        constraints=[],
    )
    # Reconciled to paused at the loop top, before any iteration ran.
    assert _sessions.active_goals["gp"]["status"] == GoalStatus.PAUSED
    assert not marker.exists()


async def test_many_concurrent_goals_independent(fleet_env):
    tmp_path, db = fleet_env
    # 4 agents: even ids preset to pause, odd ids run to completion.
    specs = []
    for i in range(4):
        sid = f"s{i}"
        status = "pause_requested" if i % 2 == 0 else "active"
        _insert_session(db, sid, status=status)
        specs.append((sid, f"g{i}", tmp_path / f"m{i}.txt", i % 2 == 0))

    await asyncio.gather(
        *(
            _sessions.run_goal_loop(
                session_id=sid,
                goal_id=gid,
                objective="obj",
                validation_cmd=f"printf x >> {marker}; exit 1",
                max_iterations=2,
                constraints=[],
            )
            for sid, gid, marker, _paused in specs
        )
    )

    for sid, gid, marker, paused in specs:
        if paused:
            assert _sessions.active_goals[gid]["status"] == GoalStatus.PAUSED
            assert not marker.exists()
        else:
            # Ran both iterations -> two effects, then failed (cmd exit 1).
            assert marker.read_text() == "xx"
            assert _sessions.active_goals[gid]["status"] == GoalStatus.FAILED
