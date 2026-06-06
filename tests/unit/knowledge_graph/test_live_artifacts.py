"""CONCEPT:KG-2.24 — Live Refreshable Artifact tests.

Covers injection-safe interpolation, the bounded-JSON contract, refresh re-derivation from a source,
the "failed refresh preserves prior" rule (bi-temporal valid-time), provenance, and the live gateway
route (create → mutate source → refresh; forced-fail preserves prior).
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.live_artifacts import (
    BoundedJSONError,
    LiveArtifact,
    LiveArtifactStore,
    RefreshService,
    render_template,
    validate_bounded_json,
)
from agent_utilities.knowledge_graph.live_artifacts.models import MAX_ITEMS

pytestmark = pytest.mark.concept(id="KG-2.24")


# ── interpolation (injection-safe) ──────────────────────────────────


def test_scalar_interpolation():
    out = render_template("Hello {{data.name}}", {"name": "world"})
    assert out == "Hello world"


def test_interpolation_escapes_html():
    out = render_template("{{data.x}}", {"x": "<script>alert(1)</script>"})
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_each_directive():
    tmpl = "<ul>{{#each data.items}}<li>{{item.label}}</li>{{/each}}</ul>"
    out = render_template(tmpl, {"items": [{"label": "a"}, {"label": "b"}]})
    assert out == "<ul><li>a</li><li>b</li></ul>"


def test_unknown_path_is_empty():
    assert render_template("[{{data.missing}}]", {}) == "[]"


# ── bounded JSON ────────────────────────────────────────────────────


def test_bounded_json_ok():
    validate_bounded_json({"a": [1, 2, {"b": "ok"}]})  # no raise


def test_bounded_json_too_many_items():
    with pytest.raises(BoundedJSONError):
        validate_bounded_json({"items": list(range(MAX_ITEMS + 1))})


def test_bounded_json_too_deep():
    deep: dict = {}
    cur = deep
    for _ in range(12):
        cur["n"] = {}
        cur = cur["n"]
    with pytest.raises(BoundedJSONError):
        validate_bounded_json(deep)


def test_create_rejects_oversized_data():
    store = LiveArtifactStore()
    art = LiveArtifact(template="x", data={"items": list(range(MAX_ITEMS + 5))})
    with pytest.raises(BoundedJSONError):
        store.create(art)


# ── refresh: re-derive + preserve-prior-on-failure ──────────────────


def _make_store_with_artifact():
    store = LiveArtifactStore()
    art = store.create(
        LiveArtifact(
            name="report",
            template="count={{data.count}}",
            data={"count": 1},
            source_query="MATCH (n) RETURN count(n)",
        )
    )
    return store, art


def test_refresh_rederives_from_source():
    store, art = _make_store_with_artifact()
    svc = RefreshService(store)
    res = svc.refresh(art.artifact_id, lambda a: {"count": 42})
    assert res.ok is True
    assert "count=42" in res.rendered
    assert store.get(art.artifact_id).data["count"] == 42
    assert store.get(art.artifact_id).refresh_count == 1


def test_failed_refresh_preserves_prior():
    store, art = _make_store_with_artifact()
    svc = RefreshService(store)
    # good refresh first
    svc.refresh(art.artifact_id, lambda a: {"count": 7})
    # then a refresh whose source raises → prior (7) must be preserved
    def boom(_a):
        raise RuntimeError("source unavailable")

    res = svc.refresh(art.artifact_id, boom)
    assert res.ok is False
    assert "count=7" in store.get(art.artifact_id).last_rendered
    assert store.get(art.artifact_id).data["count"] == 7


def test_failed_refresh_on_bounded_violation_preserves_prior():
    store, art = _make_store_with_artifact()
    svc = RefreshService(store)
    res = svc.refresh(art.artifact_id, lambda a: {"items": list(range(MAX_ITEMS + 1))})
    assert res.ok is False
    assert store.get(art.artifact_id).data == {"count": 1}  # unchanged
    # both attempts logged
    assert len(svc.refreshes) == 1


def test_provenance_records_model_and_evidence():
    store = LiveArtifactStore()
    art = LiveArtifact(template="x", data={})
    art.provenance.model = "adapter:claude-code"
    art.provenance.evidence_node_ids = ["node:1", "node:2"]
    store.create(art)
    got = store.get(art.artifact_id)
    assert got is not None
    assert got.provenance.model == "adapter:claude-code"
    assert "node:1" in got.provenance.evidence_node_ids


# ── live gateway route ──────────────────────────────────────────────


def test_artifact_routes_end_to_end():
    pytest.importorskip("fastapi")
    import fastapi
    from fastapi.testclient import TestClient

    from agent_utilities.gateway.artifacts_api import artifacts_router

    app = fastapi.FastAPI()
    app.include_router(artifacts_router)
    client = TestClient(app)

    created = client.post(
        "/api/artifacts",
        json={"name": "r", "template": "n={{data.n}}", "data": {"n": 1}, "model": "adapter:ollama"},
    )
    assert created.status_code == 200
    aid = created.json()["artifact_id"]
    assert created.json()["rendered"] == "n=1"

    # manual refresh with new data
    refreshed = client.post(f"/api/artifacts/{aid}/refresh", json={"data": {"n": 99}})
    assert refreshed.status_code == 200
    assert refreshed.json()["ok"] is True
    assert refreshed.json()["rendered"] == "n=99"

    # forced-fail refresh (bounded violation) preserves prior render
    bad = client.post(f"/api/artifacts/{aid}/refresh", json={"data": {"items": list(range(MAX_ITEMS + 1))}})
    assert bad.json()["ok"] is False
    got = client.get(f"/api/artifacts/{aid}")
    assert got.json()["data"]["n"] == 99  # prior preserved

    assert client.post("/api/artifacts/nope/refresh", json={"data": {}}).status_code == 404
