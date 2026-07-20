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
        if tool == "graph_ontology" and kwargs.get("action") == "validate":
            if kwargs.get("source") == "bad turtle":
                return _json.dumps(
                    {"valid": False, "errors": ["parse error"], "warnings": [], "summary": {}}
                )
            return _json.dumps(
                {
                    "valid": True,
                    "errors": [],
                    "warnings": [],
                    "summary": {"n_classes": 1},
                    "shacl_report": {"conforms": True, "text": "Validation Report\nConforms: True"},
                }
            )
        if tool == "ontology_derive" and kwargs.get("action") == "generate":
            return _json.dumps(
                {
                    "domain_hint": kwargs.get("object_type", ""),
                    "interfaces": [{"name": "VetClinic", "description": "", "properties": []}],
                    "link_types": [],
                    "counts": {"interfaces": 1, "link_types": 0},
                    "ttl_proposal": "# PROPOSED ontology extension\n",
                }
            )
        if tool == "graph_ontology" and kwargs.get("action") == "load":
            return _json.dumps(
                {
                    "status": "ok",
                    "idempotent": False,
                    "ontology": {
                        "iri": "http://example.org/pets",
                        "version": "1.0.0",
                        "category": kwargs.get("category", ""),
                        "tags": [],
                    },
                }
            )
        if tool == "graph_ontology" and kwargs.get("action") == "get":
            if kwargs.get("iri") == "urn:missing":
                return _json.dumps({"error": "ontology not hosted: urn:missing"})
            return _json.dumps(
                {
                    "ontology": {
                        "iri": kwargs.get("iri"),
                        "version": "1.0.0",
                        "turtle": "@prefix ex: <http://example.org/pets#> .\nex:Dog a ex:Animal .\n",
                    }
                }
            )
        if tool == "graph_ontology" and kwargs.get("action") == "list":
            return _json.dumps(
                {
                    "count": 1,
                    "ontologies": [
                        {
                            "iri": "http://example.org/pets",
                            "version": "1.0.0",
                            "n_classes": 3,
                            "n_properties": 2,
                            "n_axioms": 6,
                            "category": "animals",
                            "tags": ["demo"],
                        }
                    ],
                }
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


# ── From-scratch ontology generator (coverage row #13) ──────────────────────


def test_generate_ontology_get_round_trips(client):
    tc, captured = client
    resp = tc.get(
        "/api/ontology/generate",
        params={"sample_text": "a vet clinic", "object_type": "veterinary"},
    )
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert result["interfaces"][0]["name"] == "VetClinic"
    assert (
        "ontology_derive",
        {
            "action": "generate",
            "sample_text": "a vet clinic",
            "object_type": "veterinary",
        },
    ) in captured


def test_generate_ontology_post_round_trips(client):
    tc, captured = client
    resp = tc.post(
        "/api/ontology/generate",
        json={"sample_text": "a vet clinic", "object_type": "veterinary"},
    )
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert result["counts"] == {"interfaces": 1, "link_types": 0}
    assert (
        "ontology_derive",
        {
            "action": "generate",
            "sample_text": "a vet clinic",
            "object_type": "veterinary",
        },
    ) in captured


def test_generate_ontology_appears_in_openapi(client):
    tc, _ = client
    spec = tc.get("/openapi.json").json()
    assert "/api/ontology/generate" in spec["paths"]
    assert "get" in spec["paths"]["/api/ontology/generate"]
    assert "post" in spec["paths"]["/api/ontology/generate"]


# ── SHACL validation report (coverage row #9/#97 frontend gap) ──────────────


def test_validate_ontology_round_trips_with_shacl_report(client):
    tc, captured = client
    resp = tc.post(
        "/api/ontology/validate",
        json={"source": "@prefix ex: <http://example.org/> . ex:X a <http://www.w3.org/2002/07/owl#Class> .", "source_type": "text"},
    )
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert result["valid"] is True
    assert result["shacl_report"]["conforms"] is True
    assert (
        "graph_ontology",
        {
            "action": "validate",
            "source": "@prefix ex: <http://example.org/> . ex:X a <http://www.w3.org/2002/07/owl#Class> .",
            "source_type": "text",
        },
    ) in captured


def test_validate_ontology_reports_invalid(client):
    tc, _ = client
    resp = tc.post("/api/ontology/validate", json={"source": "bad turtle"})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert result["valid"] is False
    assert result["errors"]


def test_validate_ontology_requires_source(client):
    tc, _ = client
    resp = tc.post("/api/ontology/validate", json={})
    assert resp.status_code == 400


def test_validate_ontology_appears_in_openapi(client):
    tc, _ = client
    spec = tc.get("/openapi.json").json()
    assert "/api/ontology/validate" in spec["paths"]
    assert "post" in spec["paths"]["/api/ontology/validate"]


# ── Import / export (coverage row #23) ───────────────────────────────────────


def test_load_ontology_round_trips(client):
    tc, captured = client
    resp = tc.post(
        "/api/ontology/load",
        json={"source": "@prefix ex: <http://example.org/pets#> . ex:Dog a ex:Animal .", "source_type": "text"},
    )
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert result["status"] == "ok"
    assert result["ontology"]["iri"] == "http://example.org/pets"
    assert (
        "graph_ontology",
        {
            "action": "load",
            "source": "@prefix ex: <http://example.org/pets#> . ex:Dog a ex:Animal .",
            "source_type": "text",
            "iri": "",
            "version": "",
            "category": "",
            "tags_json": "",
        },
    ) in captured


def test_load_ontology_forwards_category_and_tags(client):
    tc, captured = client
    resp = tc.post(
        "/api/ontology/load",
        json={"source": "x", "category": "finance", "tags": ["draft", "q3"]},
    )
    assert resp.status_code == 200
    assert (
        "graph_ontology",
        {
            "action": "load",
            "source": "x",
            "source_type": "auto",
            "iri": "",
            "version": "",
            "category": "finance",
            "tags_json": '["draft", "q3"]',
        },
    ) in captured


def test_load_ontology_requires_source(client):
    tc, _ = client
    resp = tc.post("/api/ontology/load", json={})
    assert resp.status_code == 400


def test_export_ontology_round_trips(client):
    tc, captured = client
    resp = tc.get("/api/ontology/export", params={"iri": "http://example.org/pets"})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert "ex:Dog" in result["ontology"]["turtle"]
    assert (
        "graph_ontology",
        {
            "action": "get",
            "iri": "http://example.org/pets",
            "version": "",
            "serialize": True,
        },
    ) in captured


def test_export_ontology_not_found(client):
    tc, _ = client
    resp = tc.get("/api/ontology/export", params={"iri": "urn:missing"})
    assert resp.status_code == 404


def test_export_ontology_requires_iri(client):
    tc, _ = client
    resp = tc.get("/api/ontology/export")
    assert resp.status_code == 422  # FastAPI: required query param missing


def test_load_and_export_appear_in_openapi(client):
    tc, _ = client
    spec = tc.get("/openapi.json").json()
    assert "post" in spec["paths"]["/api/ontology/load"]
    assert "get" in spec["paths"]["/api/ontology/export"]


# ── Catalogue (coverage row #4) ───────────────────────────────────────────────


def test_catalogue_default_round_trips(client):
    tc, captured = client
    resp = tc.get("/api/ontology/catalogue")
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert result["count"] == 1
    assert result["ontologies"][0]["category"] == "animals"
    assert (
        "graph_ontology",
        {"action": "list", "search": "", "category": "", "source_type": "", "tag": ""},
    ) in captured


def test_catalogue_forwards_all_filters(client):
    tc, captured = client
    resp = tc.get(
        "/api/ontology/catalogue",
        params={"search": "pets", "category": "animals", "source": "text", "tag": "demo"},
    )
    assert resp.status_code == 200
    assert (
        "graph_ontology",
        {
            "action": "list",
            "search": "pets",
            "category": "animals",
            "source_type": "text",
            "tag": "demo",
        },
    ) in captured


def test_catalogue_appears_in_openapi(client):
    tc, _ = client
    spec = tc.get("/openapi.json").json()
    assert "get" in spec["paths"]["/api/ontology/catalogue"]
