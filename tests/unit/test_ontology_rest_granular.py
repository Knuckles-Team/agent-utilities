"""Granular typed REST surface for ontology/objects (WS2).

Asserts the resource-style GET routes are mounted, appear in OpenAPI, and
round-trip through the SAME ``_execute_tool`` dispatcher the collapsed routes
and MCP tools use (so there is one source of truth, not a parallel impl).
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from agent_utilities.gateway.ontology_api import register_ontology_routes  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    captured: list[tuple[str, dict]] = []

    async def fake_execute_tool(tool, **kwargs):
        captured.append((tool, kwargs))
        import json as _json

        # Echo back a deterministic payload per tool for assertions.
        if tool == "ontology_value_types" and kwargs.get("action") == "describe":
            if kwargs.get("name") == "EmailAddress":
                return _json.dumps({"name": "EmailAddress", "base": "string"})
            return _json.dumps({"error": "unknown value type"})
        if tool == "ontology_function" and kwargs.get("action") == "list":
            return _json.dumps([{"name": "score_risk", "version": "1.0.0"}])
        if tool == "object_edits" and kwargs.get("action") == "history":
            return _json.dumps(
                {"object_id": kwargs.get("object_id"), "history": [{"edit": 1}]}
            )
        return _json.dumps({"ok": True})

    monkeypatch.setattr(
        "agent_utilities.mcp.kg_server._execute_tool", fake_execute_tool
    )
    monkeypatch.setattr(
        "agent_utilities.mcp.kg_server.safe_json_load",
        lambda s: __import__("json").loads(s) if isinstance(s, str) else s,
    )
    app = FastAPI()
    register_ontology_routes(app, prefix="/api")
    return TestClient(app), captured


def test_routes_appear_in_openapi(client):
    tc, _ = client
    spec = tc.get("/openapi.json").json()
    paths = spec["paths"]
    assert "/api/ontology/value-types/{name}" in paths
    assert "/api/objects/{object_id}/history" in paths
    assert "/api/ontology/functions" in paths


def test_get_value_type_round_trips(client):
    tc, captured = client
    resp = tc.get("/api/ontology/value-types/EmailAddress")
    assert resp.status_code == 200
    assert resp.json()["result"]["name"] == "EmailAddress"
    # Dispatched through the shared tool with the right collapsed action.
    assert (
        "ontology_value_types",
        {"action": "describe", "name": "EmailAddress"},
    ) in captured


def test_unknown_value_type_is_404(client):
    tc, _ = client
    resp = tc.get("/api/ontology/value-types/Nope")
    assert resp.status_code == 404


def test_object_history_round_trips(client):
    tc, captured = client
    resp = tc.get("/api/objects/obj-1/history")
    assert resp.status_code == 200
    assert resp.json()["result"]["object_id"] == "obj-1"
    assert ("object_edits", {"action": "history", "object_id": "obj-1"}) in captured


def test_get_function_by_name_filters(client):
    tc, _ = client
    assert (
        tc.get("/api/ontology/functions/score_risk").json()["result"]["version"]
        == "1.0.0"
    )
    assert tc.get("/api/ontology/functions/missing").status_code == 404
