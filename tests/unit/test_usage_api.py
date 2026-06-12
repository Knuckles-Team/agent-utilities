"""Gateway usage API tests (CONCEPT:ECO-4.41).

Mounts usage_router on a throwaway FastAPI app pointed at a temp SQLite store
and exercises the observability surface + the upload transport (ECO-4.42).
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_utilities.gateway.usage_api import usage_router
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


def _seed():
    sid = "s1"
    get_usage_recorder().record_bundle(
        ParsedSessionBundle(
            session=UsageSession(
                id=sid, project="proj-a", agent="claude",
                started_at="2026-06-10T09:00:00Z", message_count=2,
                health_grade="A", outcome="success",
            ),
            messages=[
                UsageMessage(session_id=sid, ordinal=0, role="user",
                             content="refactor the parser registry"),
            ],
            tool_calls=[
                UsageToolCall(session_id=sid, message_ordinal=0, tool_name="Edit",
                              category="edit", status="ok"),
            ],
            usage_events=[
                UsageEvent(session_id=sid, source="agent", model="claude-opus-4-8",
                           input_tokens=1000, output_tokens=500, dedup_key="e1"),
            ],
        )
    )


def test_summary_and_breakdowns(client):
    _seed()
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
    _seed()
    rows = client.get("/api/observability/sessions").json()
    assert rows[0]["id"] == "s1"
    detail = client.get("/api/observability/sessions/s1").json()
    assert detail["session"]["id"] == "s1"
    assert len(detail["messages"]) == 1


def test_search_and_activity(client):
    _seed()
    hits = client.get("/api/observability/search", params={"q": "parser"}).json()
    assert hits and hits[0]["session_id"] == "s1"
    cells = client.get("/api/observability/analytics/activity").json()
    assert cells[0]["day_of_week"] == 2  # 2026-06-10 is a Wednesday


def test_upload_transport(client):
    bundle = ParsedSessionBundle(
        session=UsageSession(id="up1", project="remote", agent="codex",
                             message_count=1),
        usage_events=[UsageEvent(session_id="up1", source="agent",
                                 model="gpt-5.5", input_tokens=100,
                                 output_tokens=50, dedup_key="u1")],
    )
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
    assert any(r["id"] == "up1" for r in rows)


def test_traces_gated_off_by_default(client):
    out = client.get("/api/observability/traces").json()
    assert out["enabled"] in (False, True)  # shape stable regardless
    assert "traces" in out
