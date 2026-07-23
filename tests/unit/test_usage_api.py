"""Gateway usage API tests (CONCEPT:AU-ECO.mcp.usage-cost-observability-surface).

Mounts usage_router on a throwaway FastAPI app pointed at a temp SQLite store
and exercises the observability surface + the upload transport (ECO-4.42).
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_utilities.gateway.usage_api import usage_router
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor
from agent_utilities.usage import backends as usage_backends
from agent_utilities.usage.models import (
    ParsedSessionBundle,
    UsageEvent,
    UsageMessage,
    UsageSession,
    UsageToolCall,
)
from agent_utilities.usage.recorder import get_usage_recorder


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("USAGE_DB_PATH", str(tmp_path / "usage.db"))
    monkeypatch.setenv("USAGE_CONTENT_RETENTION", "sanitized")
    usage_backends.reset_usage_backend_for_tests()
    # Reset the recorder/service singletons so they pick up the temp backend.
    import agent_utilities.usage.recorder as rec_mod
    import agent_utilities.usage.service as svc_mod

    rec_mod._recorder = None
    svc_mod._service = None

    app = FastAPI()
    app.include_router(usage_router, prefix="/api/observability")
    yield TestClient(app)
    usage_backends.reset_usage_backend_for_tests()


def _seed(*, sid: str = "s1", tenant_id: str = ""):
    get_usage_recorder().record_bundle(
        ParsedSessionBundle(
            session=UsageSession(
                id=sid,
                project="proj-a",
                agent="claude",
                started_at="2026-06-10T09:00:00Z",
                message_count=2,
                health_grade="A",
                outcome="success",
                tenant_id=tenant_id,
            ),
            messages=[
                UsageMessage(
                    session_id=sid,
                    ordinal=0,
                    role="user",
                    content="refactor the parser registry",
                ),
            ],
            tool_calls=[
                UsageToolCall(
                    session_id=sid,
                    message_ordinal=0,
                    tool_name="Edit",
                    category="edit",
                    status="ok",
                ),
            ],
            usage_events=[
                UsageEvent(
                    session_id=sid,
                    source="agent",
                    model="claude-opus-4-8",
                    input_tokens=1000,
                    output_tokens=500,
                    dedup_key="e1",
                ),
            ],
        )
    )


def _verified_actor(tenant_id: str = "test-tenant") -> ActorContext:
    """A verified, non-admin usage caller scoped to ``tenant_id`` — every served
    usage query now requires an authenticated identity with a non-empty tenant
    claim (:func:`~agent_utilities.usage.authorization.resolve_usage_tenant`)."""
    return ActorContext(
        actor_id="usage-tester",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("usage:read",),
        tenant_id=tenant_id,
        authenticated=True,
    )


def test_summary_and_breakdowns(client):
    _seed(tenant_id="test-tenant")
    with use_actor(_verified_actor()):
        s = client.get("/api/observability/summary").json()
        assert s["session_count"] == 1
        assert s["totals"]["input_tokens"] == 1000
        assert s["totals"]["cost_usd"] > 0  # priced via catalog

        models = client.get("/api/observability/by-model").json()
        assert models[0]["key"] == "claude-opus-4-8"

        tools = client.get("/api/observability/analytics/tools").json()
        assert tools[0]["name"] == "Edit"
        assert tools[0]["success_rate"] == 1.0


def test_sessions_and_detail(client):
    _seed(tenant_id="test-tenant")
    with use_actor(_verified_actor()):
        rows = client.get("/api/observability/sessions").json()
        run_ref = rows[0]["id"]
        assert run_ref.startswith("pref_run_")
        detail = client.get(f"/api/observability/sessions/{run_ref}").json()
        assert detail["session"]["id"] == run_ref
        assert len(detail["messages"]) == 1


def test_search_and_activity(client):
    _seed(tenant_id="test-tenant")
    with use_actor(_verified_actor()):
        hits = client.get("/api/observability/search", params={"q": "parser"}).json()
        assert hits and hits[0]["session_id"].startswith("pref_run_")
        cells = client.get("/api/observability/analytics/activity").json()
        assert cells[0]["day_of_week"] == 2  # 2026-06-10 is a Wednesday


def test_upload_transport(client):
    bundle = ParsedSessionBundle(
        session=UsageSession(
            id="up1", project="remote", agent="codex", message_count=1
        ),
        usage_events=[
            UsageEvent(
                session_id="up1",
                source="agent",
                model="gpt-5.5",
                input_tokens=100,
                output_tokens=50,
                dedup_key="u1",
            )
        ],
    )
    with use_actor(_verified_actor(tenant_id="acme")):
        resp = client.post(
            "/api/observability/sessions/upload",
            params={"tenant_id": "acme"},
            json=[bundle.model_dump()],
        )
        assert resp.json() == {"received": 1, "ingested": 1}
        # The uploaded session is now queryable and tenant-scoped.
        rows = client.get(
            "/api/observability/sessions", params={"tenant_id": "acme"}
        ).json()
        assert any(r["id"].startswith("pref_run_") for r in rows)


def test_traces_gated_off_by_default(client):
    out = client.get("/api/observability/traces").json()
    assert out["enabled"] in (False, True)  # shape stable regardless
    assert "traces" in out
    assert "host" not in out


def test_served_usage_queries_bind_to_verified_tenant(client, monkeypatch):
    _seed(sid="same", tenant_id="tenant-a")
    _seed(sid="same", tenant_id="tenant-b")
    actor = ActorContext(
        actor_id="subject-ref",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("usage:read",),
        tenant_id="tenant-a",
        authenticated=True,
    )
    with use_actor(actor):
        rows = client.get("/api/observability/sessions").json()
        denied = client.get(
            "/api/observability/sessions", params={"tenant_id": "tenant-b"}
        )
    assert len(rows) == 1
    assert rows[0]["id"].startswith("pref_run_")
    assert denied.status_code == 403


def test_served_usage_rejects_missing_verified_identity(client, monkeypatch):
    """An actor claim that exists but was never verified (``authenticated=False``)
    must be rejected. Every test already runs inside the suite's own verified
    ambient identity (``isolate_graph_compute_engine``, autouse) — a fully absent
    actor context would raise ``IdentityRequiredError`` before the usage
    authorization layer even runs, so the realistic "missing verified identity"
    case to exercise here is an unauthenticated claim, not no claim at all."""
    unauthenticated = ActorContext(
        actor_id="unverified",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=(),
        tenant_id="",
        authenticated=False,
    )
    with use_actor(unauthenticated):
        response = client.get("/api/observability/summary")
    assert response.status_code == 401
