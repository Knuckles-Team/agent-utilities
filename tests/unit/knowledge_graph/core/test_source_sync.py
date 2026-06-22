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

    # A generic source name that is NOT delta-capable and NOT a materialize
    # source (a synthetic name, immune to future additions to MATERIALIZE_SOURCES
    # — e.g. 'twenty' became a materialize source) → falls back to full hydrate.
    out = ss.sync_source(object(), "generic_fallback_src", mode="delta")
    assert out["status"] == "ok"
    assert out["source"] == "generic_fallback_src"
    assert out["delta_capable"] is False
    assert out["mode"] == "full"
    assert calls and calls[0][1] == "generic_fallback_src"


def test_generic_reconcile_unsupported():
    out = sync_source(object(), "twenty", mode="reconcile")
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


# ── Fleet sweep: source="all" + sweep_all_sources (KG-2.9) ───────────────────


def test_sync_source_all_fans_out_to_sweep(monkeypatch):
    """source='all' routes through the one entrypoint to the fleet sweep."""
    import agent_utilities.knowledge_graph.core.source_sync as ss

    seen = {}

    def fake_sweep(engine, *, mode="delta", include_materialize=True):
        seen["mode"] = mode
        return {"status": "ok", "swept": 7}

    monkeypatch.setattr(ss, "sweep_all_sources", fake_sweep)
    for alias in ("all", "*", "sweep"):
        res = ss.sync_source(object(), alias, mode="delta")
        assert res["swept"] == 7
    assert seen["mode"] == "delta"


def test_sweep_all_sources_classifies_results(monkeypatch):
    """The sweep isolates each connector and buckets synced/skipped/errors."""
    import agent_utilities.knowledge_graph.core.source_sync as ss
    from agent_utilities.knowledge_graph.core.hydration import HydrationManager

    # Only servicenow env-detects as configured among capability sources.
    monkeypatch.setattr(
        HydrationManager,
        "get_status",
        lambda self: {
            "servicenow": {"configured": True},
            "jira": {"configured": False},
        },
    )

    results = {
        "leanix": {"status": "completed", "nodes": 5},
        "archivebox": {"status": "skipped", "reason": "no new snapshots"},
        "gitlab": {"status": "completed"},
        "servicenow": {"status": "error", "error": "boom"},
    }

    def fake_sync(engine, source, *, mode="delta", ids=None, client=None):
        if source not in results:
            raise RuntimeError(f"{source} not configured")
        return results[source]

    monkeypatch.setattr(ss, "sync_source", fake_sync)

    out = ss.sweep_all_sources(object(), mode="delta", include_materialize=False)
    assert out["status"] == "ok" and out["mode"] == "delta"
    # delta handlers (leanix/archivebox/gitlab) + configured capability (servicenow)
    assert set(out["synced"]) == {"leanix", "gitlab"}
    assert "archivebox" in out["skipped"]
    assert "servicenow" in out["errors"]
    assert "jira" not in out["synced"] and "jira" not in out["errors"]
    # synced: leanix + gitlab; errors: servicenow; skipped: archivebox (no new
    # snapshots) + the unconfigured delta handlers (rss/freshrss/jira/confluence
    # /plane/fleet_connectors) whose fake_sync raises "not configured". ``fleet``
    # is excluded from the sweep, so it never enters any bucket.
    assert out["counts"] == {"synced": 2, "skipped": 7, "errors": 1}


# ── Fleet connectors: every agents/* package in one handler (KG-2.151) ────────


def test_fleet_connectors_registered_as_sweep_candidate():
    """``fleet_connectors`` is a delta handler, so the source='all' sweep fans it out."""
    import agent_utilities.knowledge_graph.core.source_sync as ss

    assert "fleet_connectors" in ss._DELTA_HANDLERS
    assert ss._DELTA_HANDLERS["fleet_connectors"] is ss._sync_fleet_connectors


def test_fleet_connectors_skips_unconfigured_packages(monkeypatch):
    """Packages whose MCP server isn't in mcp_config are skipped, never errored."""
    import agent_utilities.knowledge_graph.core.source_sync as ss

    # No servers registered → every package preset is skipped.
    monkeypatch.setattr(
        "agent_utilities.protocols.source_connectors.connectors.mcp_package._load_mcp_config",
        lambda: {},
    )

    out = ss._sync_fleet_connectors(
        FakeEngine(FakeBackend()), mode="full", ids=None, client=None
    )
    assert out["status"] == "ok"
    assert out["source"] == "fleet_connectors"
    assert out["synced"] == {}
    assert out["counts"]["errors"] == 0
    # every preset reported as skipped (server not in mcp_config)
    assert out["counts"]["skipped"] > 0
    assert all("not in mcp_config" in r for r in out["skipped"].values())


def test_fleet_connectors_drains_configured_package(monkeypatch):
    """A configured package is drained via the mcp connector and processed as Documents."""
    import agent_utilities.knowledge_graph.core.source_sync as ss

    # scholarx-mcp registered → only that package is attempted.
    monkeypatch.setattr(
        "agent_utilities.protocols.source_connectors.connectors.mcp_package._load_mcp_config",
        lambda: {"scholarx-mcp": {"command": "scholarx"}},
    )

    class _Doc:
        def __init__(self, did, text, updated):
            self.id = did
            self.text = text
            self.title = f"T{did}"
            self.source_uri = f"mcp://scholarx/{did}"
            self.updated_at = updated

    class _Conn:
        def poll(self, checkpoint=None):
            class _Batch:
                documents = [_Doc("p1", "alpha", "2026-01-01"), _Doc("p2", "beta", "2026-02-01")]

                class checkpoint:  # noqa: N801
                    has_more = False

            return _Batch()

    monkeypatch.setattr(
        "agent_utilities.protocols.source_connectors.registry.build_connector",
        lambda kind, cfg: _Conn(),
    )

    processed: list[str] = []

    class _Proc:
        def process(self, document, **kw):
            processed.append(kw.get("document_id"))

    monkeypatch.setattr(ss, "_confluence_processor", lambda engine: _Proc())

    out = ss._sync_fleet_connectors(
        FakeEngine(FakeBackend()), mode="full", ids=None, client=None
    )
    assert out["status"] == "ok"
    assert out["synced"]["scholarx"]["documents_ingested"] == 2
    assert out["synced"]["scholarx"]["watermark"] == "2026-02-01"
    assert processed == ["fleet:scholarx:p1", "fleet:scholarx:p2"]
    # all other packages skipped (their *-mcp server not registered)
    assert out["counts"]["errors"] == 0
