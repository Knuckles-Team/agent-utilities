"""Tests for the native swarm supervisory plane (CONCEPT:OS-5.10).

Exercises the gateway /fleet/* handlers against a temp sqlite session registry —
health/topology aggregation and whole-domain emergency kill.
"""

from __future__ import annotations

import json
import sqlite3
import time

import pytest

from agent_utilities.gateway import fleet


@pytest.fixture
def session_db(tmp_path, monkeypatch):
    db = tmp_path / "sessions.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        """CREATE TABLE sessions (
            id TEXT PRIMARY KEY, status TEXT, background INTEGER DEFAULT 0,
            needs_input INTEGER DEFAULT 0, updated_at REAL, metadata_json TEXT
        )"""
    )
    rows = [
        ("s1", "active", json.dumps({"domain": "finance"})),
        ("s2", "failed", json.dumps({"domain": "finance"})),
        ("s3", "active", json.dumps({"domain": "itops"})),
        ("s4", "completed", "{}"),
    ]
    for sid, status, meta in rows:
        conn.execute(
            "INSERT INTO sessions (id, status, updated_at, metadata_json) VALUES (?,?,?,?)",
            (sid, status, time.time(), meta),
        )
    conn.commit()
    conn.close()
    monkeypatch.setattr(fleet._sessions, "_get_db_path", lambda: db)
    return db


class _Req:
    def __init__(self, body=None):
        self._body = body or {}

    async def json(self):
        return self._body


async def _payload(resp):
    return json.loads(resp.body)


@pytest.mark.asyncio
async def test_fleet_health_per_domain_error_rate(session_db):
    resp = await fleet.fleet_health(_Req())
    data = await _payload(resp)
    assert data["sessions"]["total"] == 4
    # finance: 2 sessions, 1 errored -> 0.5 error rate
    assert data["domains"]["finance"]["error_rate"] == 0.5
    assert data["domains"]["itops"]["active"] == 1


@pytest.mark.asyncio
async def test_fleet_topology_groups_by_domain(session_db):
    resp = await fleet.fleet_topology(_Req())
    data = await _payload(resp)
    domains = {d["domain"] for d in data["domains"]}
    assert {"finance", "itops", "default"} <= domains
    assert data["totals"]["sessions"] == 4


@pytest.mark.asyncio
async def test_fleet_kill_whole_domain(session_db):
    resp = await fleet.fleet_kill(_Req({"domain": "finance"}))
    data = await _payload(resp)
    assert data["count"] == 2
    # Both finance sessions are now cancelled.
    conn = sqlite3.connect(str(session_db))
    statuses = dict(conn.execute("SELECT id, status FROM sessions").fetchall())
    conn.close()
    assert statuses["s1"] == "cancelled"
    assert statuses["s2"] == "cancelled"
    assert statuses["s3"] == "active"  # itops untouched


@pytest.mark.asyncio
async def test_fleet_pause_requires_target(session_db):
    resp = await fleet.fleet_pause(_Req({}))
    assert resp.status_code == 400
