"""Tests for the native swarm supervisory plane (CONCEPT:AU-OS.safety.ontological-guardrail / OS-5.18).

Exercises the gateway /fleet/* handlers against a temp sqlite session registry —
health/topology SQL aggregation, pagination/filtering, whole-domain emergency
kill, and desired-state pause/kill writes under externalized state.
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
    conn.executescript(fleet._sessions._SQLITE_DDL)
    rows = [
        ("s1", "active", json.dumps({"domain": "finance"})),
        ("s2", "failed", json.dumps({"domain": "finance"})),
        ("s3", "active", json.dumps({"domain": "itops"})),
        ("s4", "completed", "{}"),
    ]
    now = time.time()
    for i, (sid, status, meta) in enumerate(rows):
        conn.execute(
            "INSERT INTO sessions (id, status, created_at, updated_at, metadata_json) VALUES (?,?,?,?,?)",
            (sid, status, now, now + i, meta),
        )
    conn.commit()
    conn.close()
    monkeypatch.setattr(fleet._sessions, "_get_db_path", lambda: db)
    monkeypatch.setattr(fleet._sessions, "_rehydrated", True)
    return db


class _Req:
    def __init__(self, body=None, query=None):
        self._body = body or {}
        self.query_params = query or {}

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
async def test_fleet_topology_pagination_and_status_filter(session_db):
    # Page of 2 newest sessions; totals still reflect the whole fleet (OS-5.18).
    resp = await fleet.fleet_topology(_Req(query={"limit": "2", "offset": "0"}))
    data = await _payload(resp)
    returned = sum(len(d["sessions"]) for d in data["domains"])
    assert returned == 2
    assert data["page"] == {"limit": 2, "offset": 0, "returned": 2}
    assert data["totals"]["sessions"] == 4

    # Status filter pushes down to SQL.
    resp = await fleet.fleet_topology(_Req(query={"status": "active"}))
    data = await _payload(resp)
    statuses = {s["status"] for d in data["domains"] for s in d["sessions"]}
    assert statuses == {"active"}


def test_fetch_sessions_filters_in_sql(session_db):
    rows = fleet._fetch_sessions(status="active")
    assert {r["id"] for r in rows} == {"s1", "s3"}
    rows = fleet._fetch_sessions(domain="finance")
    assert {r["id"] for r in rows} == {"s1", "s2"}
    rows = fleet._fetch_sessions(limit=1)
    assert len(rows) == 1
    # Newest first (s4 has the max updated_at).
    assert rows[0]["id"] == "s4"


@pytest.mark.asyncio
async def test_fleet_kill_whole_domain(session_db):
    resp = await fleet.fleet_kill(_Req({"domain": "finance"}))
    data = await _payload(resp)
    assert data["count"] == 2
    # Both finance sessions are now cancelled (single-host: applied directly).
    conn = sqlite3.connect(str(session_db))
    statuses = dict(conn.execute("SELECT id, status FROM sessions").fetchall())
    conn.close()
    assert statuses["s1"] == "cancelled"
    assert statuses["s2"] == "cancelled"
    assert statuses["s3"] == "active"  # itops untouched


@pytest.mark.asyncio
async def test_fleet_kill_desired_state_for_remote_sessions(session_db, monkeypatch):
    # With state externalized (multi-host), a session NOT local to this
    # gateway gets a kill_requested desired-state write that the owning
    # host's loop reconciles (CONCEPT:AU-OS.state.fleet-supervisory-plane-at).
    monkeypatch.setattr(fleet, "_multi_host_state", lambda: True)
    resp = await fleet.fleet_kill(_Req({"session_ids": ["s3"]}))
    data = await _payload(resp)
    assert data["applied"]["s3"] == "kill_requested"
    conn = sqlite3.connect(str(session_db))
    status = conn.execute("SELECT status FROM sessions WHERE id='s3'").fetchone()[0]
    conn.close()
    assert status == "kill_requested"


@pytest.mark.asyncio
async def test_fleet_pause_local_fast_path_under_multi_host(session_db, monkeypatch):
    # A session whose goal loop runs in THIS process is cancelled in-process
    # and finalized immediately even when state is externalized.
    monkeypatch.setattr(fleet, "_multi_host_state", lambda: True)

    class _Task:
        def done(self):
            return True

    monkeypatch.setattr(
        fleet._sessions,
        "background_goal_runs",
        {"g-local": {"session_id": "s1", "task": _Task()}},
    )
    monkeypatch.setattr(fleet._sessions, "active_goals", {})
    resp = await fleet.fleet_pause(_Req({"session_ids": ["s1"]}))
    data = await _payload(resp)
    assert data["applied"]["s1"] == "paused"


@pytest.mark.asyncio
async def test_fleet_pause_requires_target(session_db):
    resp = await fleet.fleet_pause(_Req({}))
    assert resp.status_code == 400


# ── ActionApproval entries in the shared approvals flow (CONCEPT:AU-OS.deployment.fleet-lifecycle-control) ──


class _ApprovalEngine:
    """Engine double exposing one pending ActionApproval node."""

    def __init__(self):
        self.nodes = {
            "action_approval:1": {
                "id": "action_approval:1",
                "kind": "restart_service",
                "target": "caddy-mcp",
                "status": "pending",
            }
        }
        outer = self

        class _Backend:
            def execute(self, query, params=None):
                params = params or {}
                node = outer.nodes.get(params.get("id"))
                if node is not None and node.get("status") == "pending":
                    node["status"] = params.get("status")
                return []

        self.backend = _Backend()

    def query_cypher(self, query, params=None):
        if "ActionApproval" in query:
            return [
                {"a": dict(n)}
                for n in self.nodes.values()
                if n.get("status") == "pending"
            ]
        return []


@pytest.mark.asyncio
async def test_fleet_approvals_lists_action_approvals(monkeypatch):
    import agent_utilities.mcp.kg_server as kg

    async def _no_tasks(tool, **kw):
        return "[]"

    eng = _ApprovalEngine()
    monkeypatch.setattr(kg, "_execute_tool", _no_tasks)
    monkeypatch.setattr(kg, "_get_engine", lambda: eng)
    resp = await fleet.fleet_approvals(_Req())
    data = await _payload(resp)
    assert data["status"] == "success"
    ids = [p.get("id") for p in data["pending"]]
    assert "action_approval:1" in ids


@pytest.mark.asyncio
async def test_fleet_grant_resolves_action_approval_in_place(monkeypatch):
    import agent_utilities.mcp.kg_server as kg

    eng = _ApprovalEngine()
    monkeypatch.setattr(kg, "_get_engine", lambda: eng)
    resp = await fleet.fleet_grant_approval(
        _Req({"job_id": "action_approval:1", "decision": "approved"})
    )
    data = await _payload(resp)
    assert data["status"] == "success"
    assert data["result"]["decision"] == "approved"
    # The node was stamped — the reconciler's drain will execute it next tick.
    assert eng.nodes["action_approval:1"]["status"] == "approved"


@pytest.mark.asyncio
async def test_fleet_grant_denial_stamps_denied(monkeypatch):
    import agent_utilities.mcp.kg_server as kg

    eng = _ApprovalEngine()
    monkeypatch.setattr(kg, "_get_engine", lambda: eng)
    resp = await fleet.fleet_grant_approval(
        _Req({"job_id": "action_approval:1", "decision": "denied"})
    )
    data = await _payload(resp)
    assert data["result"]["decision"] == "denied"
    assert eng.nodes["action_approval:1"]["status"] == "denied"


@pytest.fixture
def tenant_session_db(tmp_path, monkeypatch):
    """Sessions tagged with two different tenants for isolation tests."""
    db = tmp_path / "tenant_sessions.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(fleet._sessions._SQLITE_DDL)
    rows = [
        ("a1", "active", json.dumps({"tenant": "acme", "domain": "finance"})),
        ("a2", "failed", json.dumps({"tenant": "acme", "domain": "finance"})),
        ("b1", "active", json.dumps({"tenant": "globex", "domain": "itops"})),
    ]
    now = time.time()
    for i, (sid, status, meta) in enumerate(rows):
        conn.execute(
            "INSERT INTO sessions (id, status, created_at, updated_at, metadata_json) VALUES (?,?,?,?,?)",
            (sid, status, now, now + i, meta),
        )
    conn.commit()
    conn.close()
    monkeypatch.setattr(fleet._sessions, "_get_db_path", lambda: db)
    monkeypatch.setattr(fleet._sessions, "_rehydrated", True)
    return db


def _actor(actor_id="alice", tenant="acme", roles=()):
    from agent_utilities.models.company_brain import ActorType
    from agent_utilities.security.brain_context import ActorContext

    return ActorContext(
        actor_id=actor_id,
        actor_type=ActorType.HUMAN,
        roles=tuple(roles),
        tenant_id=tenant,
        authenticated=True,
    )


@pytest.mark.asyncio
async def test_fleet_health_scoped_to_caller_tenant(tenant_session_db):
    from agent_utilities.security.brain_context import use_actor

    with use_actor(_actor(tenant="acme")):
        data = await _payload(await fleet.fleet_health(_Req()))
    # Caller in 'acme' sees only its 2 sessions, never globex's.
    assert data["sessions"]["total"] == 2
    assert "globex" not in {d for d in data["domains"]}


@pytest.mark.asyncio
async def test_fleet_topology_isolated_by_tenant(tenant_session_db):
    from agent_utilities.security.brain_context import use_actor

    with use_actor(_actor(tenant="globex", actor_id="bob")):
        data = await _payload(await fleet.fleet_topology(_Req()))
    ids = {s["id"] for d in data["domains"] for s in d["sessions"]}
    assert ids == {"b1"}  # globex caller sees only b1
    assert data["totals"]["sessions"] == 1


@pytest.mark.asyncio
async def test_platform_admin_sees_whole_fleet(tenant_session_db):
    from agent_utilities.security.brain_context import use_actor

    with use_actor(_actor(actor_id="root", tenant="acme", roles=("admin",))):
        data = await _payload(await fleet.fleet_health(_Req()))
    assert data["sessions"]["total"] == 3  # admin sees acme + globex


@pytest.mark.asyncio
async def test_legacy_unscoped_caller_sees_all(tenant_session_db):
    # No actor in scope (ambient SYSTEM_ACTOR, tenant="") → unfiltered, preserving
    # single-tenant/local behaviour.
    data = await _payload(await fleet.fleet_health(_Req()))
    assert data["sessions"]["total"] == 3
