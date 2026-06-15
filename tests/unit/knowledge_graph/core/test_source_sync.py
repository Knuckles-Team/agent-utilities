"""Tests for source-agnostic KG sync: watermark, reconcile, generic fallback (KG-2.9)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.core.source_sync import sync_source

META_MODEL = {
    "factSheets": {
        "Application": {"fields": {}, "relations": {}},
    }
}


class FakeBackend:
    """In-memory stand-in for the graph backend (watermark + reconcile queries)."""

    def __init__(self, leanix_nodes=None):
        self.watermark: str | None = None
        self.archived: list[str] = []
        self._leanix_nodes = leanix_nodes or []

    def execute(self, query, params=None):
        params = params or {}
        if "MATCH (n:SourceSyncState" in query and "RETURN" in query:
            return [{"w": self.watermark}] if self.watermark else []
        if "MERGE (n:SourceSyncState" in query:
            self.watermark = params.get("wm")
            return []
        if "RETURN n.id AS id, n.externalToolId AS guid" in query:
            return self._leanix_nodes
        if "SET n.archived = true" in query:
            self.archived.append(params.get("id"))
            return []
        return []


class FakeEngine:
    def __init__(self, backend):
        self.backend = backend
        self.batches: list[tuple] = []

    def ingest_external_batch(self, domain, entities, relationships=None):
        self.batches.append((domain, entities, relationships))
        return {"status": "success"}


class FakeClient:
    def __init__(self, sheets):
        self._sheets = sheets

    def meta_model(self):
        return META_MODEL

    def factsheets(self, type=None, since=None, ids=None):  # noqa: A002
        items = (
            self._sheets.get(type, [])
            if type
            else [x for v in self._sheets.values() for x in v]
        )
        if since:
            items = [x for x in items if str(x.get("updatedAt") or "") > since]
        return items

    def fact_sheet_ids(self):
        return {x["id"] for v in self._sheets.values() for x in v}


def test_leanix_delta_advances_watermark():
    backend = FakeBackend()
    engine = FakeEngine(backend)
    client = FakeClient(
        {
            "Application": [
                {
                    "id": "a1",
                    "name": "A",
                    "type": "Application",
                    "updatedAt": "2026-01-01",
                },
                {
                    "id": "a2",
                    "name": "B",
                    "type": "Application",
                    "updatedAt": "2026-06-01",
                },
            ]
        }
    )
    out = sync_source(engine, "leanix", mode="delta", client=client)
    assert out["status"] == "ok"
    assert out["source"] == "leanix"
    assert out["delta_capable"] is True
    assert out["nodes_hydrated"] == 2
    assert out["watermark"] == "2026-06-01"
    assert backend.watermark == "2026-06-01"


def test_leanix_delta_second_run_only_fetches_newer():
    backend = FakeBackend()
    backend.watermark = "2026-03-01"
    engine = FakeEngine(backend)
    client = FakeClient(
        {
            "Application": [
                {
                    "id": "a1",
                    "name": "A",
                    "type": "Application",
                    "updatedAt": "2026-01-01",
                },
                {
                    "id": "a2",
                    "name": "B",
                    "type": "Application",
                    "updatedAt": "2026-06-01",
                },
            ]
        }
    )
    out = sync_source(engine, "leanix", mode="delta", client=client)
    assert out["nodes_hydrated"] == 1
    _domain, entities, _ = engine.batches[0]
    assert {e["id"] for e in entities} == {"app:a2"}


def test_leanix_reconcile_tombstones_missing():
    backend = FakeBackend(
        leanix_nodes=[
            {"id": "app:a1", "guid": "a1"},
            {"id": "app:gone", "guid": "gone"},
        ]
    )
    engine = FakeEngine(backend)
    client = FakeClient(
        {"Application": [{"id": "a1", "name": "A", "type": "Application"}]}
    )
    out = sync_source(engine, "leanix", mode="reconcile", client=client)
    assert out["status"] == "completed"
    assert out["tombstoned"] == 1
    assert backend.archived == ["app:gone"]


def test_leanix_no_client_skips():
    out = sync_source(FakeEngine(FakeBackend()), "leanix", mode="delta", client=None)
    assert out["status"] == "skipped"


def test_generic_source_falls_back_to_full_hydrate(monkeypatch):
    """A source without a delta handler syncs via the capability registry (full)."""
    import agent_utilities.knowledge_graph.core.hydration as hyd
    import agent_utilities.knowledge_graph.core.source_sync as ss

    calls: list[tuple] = []

    class FakeManager:
        def hydrate_source(self, engine, source):
            calls.append((engine, source))
            return {"status": "ok", "nodes_hydrated": 3}

    monkeypatch.setattr(hyd, "HydrationManager", FakeManager)

    out = ss.sync_source(object(), "servicenow", mode="delta")
    assert out["status"] == "ok"
    assert out["source"] == "servicenow"
    assert out["delta_capable"] is False
    assert out["mode"] == "full"
    assert calls and calls[0][1] == "servicenow"


def test_generic_reconcile_unsupported():
    out = sync_source(object(), "servicenow", mode="reconcile")
    assert out["status"] == "skipped"
    assert "reconcile not supported" in out["reason"]


def test_materialize_source_routes_through_shared_core(monkeypatch):
    """camunda/aris/egeria route through the shared materialize core, not hydration."""
    import agent_utilities.knowledge_graph.enrichment.materialize as mat

    calls: list[tuple] = []

    def fake_run(engine, category, *, config=None):
        calls.append((category, config))
        return {"status": "materialized", "source": category, "nodes": 4, "edges": 2}

    monkeypatch.setattr(mat, "run_materialize_source", fake_run)

    out = sync_source(object(), "camunda", mode="delta")
    assert out["status"] == "materialized"
    assert out["source"] == "camunda"
    assert out["delta_capable"] is False
    assert calls and calls[0][0] == "camunda"
