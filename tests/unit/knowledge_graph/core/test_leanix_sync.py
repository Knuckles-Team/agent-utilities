"""Tests for LeanIX delta sync: watermark poll, reconcile (CONCEPT:KG-2.9)."""

from __future__ import annotations

from agent_utilities.knowledge_graph.core.leanix_sync import (
    reconcile_leanix,
    sync_leanix,
)

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
        if "MATCH (n:LeanixSyncState" in query and "RETURN" in query:
            return [{"w": self.watermark}] if self.watermark else []
        if "MERGE (n:LeanixSyncState" in query:
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
        items = self._sheets.get(type, []) if type else [
            x for v in self._sheets.values() for x in v
        ]
        if since:
            items = [x for x in items if str(x.get("updatedAt") or "") > since]
        return items

    def fact_sheet_ids(self):
        return {x["id"] for v in self._sheets.values() for x in v}


def test_delta_advances_watermark():
    backend = FakeBackend()
    engine = FakeEngine(backend)
    client = FakeClient(
        {
            "Application": [
                {"id": "a1", "name": "A", "type": "Application", "updatedAt": "2026-01-01"},
                {"id": "a2", "name": "B", "type": "Application", "updatedAt": "2026-06-01"},
            ]
        }
    )
    out = sync_leanix(engine, mode="delta", client=client)
    assert out["status"] == "ok"
    assert out["nodes_hydrated"] == 2
    # Watermark advanced to the newest updatedAt and persisted.
    assert out["watermark"] == "2026-06-01"
    assert backend.watermark == "2026-06-01"


def test_delta_second_run_only_fetches_newer():
    backend = FakeBackend()
    backend.watermark = "2026-03-01"  # prior run's watermark
    engine = FakeEngine(backend)
    client = FakeClient(
        {
            "Application": [
                {"id": "a1", "name": "A", "type": "Application", "updatedAt": "2026-01-01"},
                {"id": "a2", "name": "B", "type": "Application", "updatedAt": "2026-06-01"},
            ]
        }
    )
    out = sync_leanix(engine, mode="delta", client=client)
    # Only the post-watermark fact sheet is ingested.
    assert out["nodes_hydrated"] == 1
    domain, entities, _ = engine.batches[0]
    assert {e["id"] for e in entities} == {"app:a2"}


def test_reconcile_tombstones_missing():
    # KG has app:a1 (live) and app:gone (deleted in LeanIX).
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
    out = reconcile_leanix(engine, client)
    assert out["status"] == "completed"
    assert out["tombstoned"] == 1
    assert backend.archived == ["app:gone"]


def test_no_client_skips():
    out = sync_leanix(FakeEngine(FakeBackend()), mode="delta", client=None)
    assert out["status"] == "skipped"
