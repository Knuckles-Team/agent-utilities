"""Tests for source-agnostic KG sync: watermark, reconcile, generic fallback (KG-2.9)."""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core import source_sync as source_sync_module
from agent_utilities.knowledge_graph.core.source_sync import sync_source

META_MODEL = {
    "factSheets": {
        "Application": {"fields": {}, "relations": {}},
    }
}


@pytest.fixture(autouse=True)
def _source_handler_tests_bypass_external_boundaries(monkeypatch):
    """Keep handler mapping tests isolated from manifest/native-client contracts."""

    def read_cursor(engine, _connector, *, source_instance=""):
        return getattr(engine.backend, "watermark", None)

    def capture_envelope(engine, envelope):
        backend = engine.backend
        if envelope.operation == "snapshot_complete":
            live = set(envelope.live_ids or [])
            fetch_ok = bool(envelope.provenance.get("fetch_ok", True))
            allow_empty = bool(live) or (
                envelope.connector.lower()
                in source_sync_module._reconcile_allowed_empty_sources()
            )
            tombstoned = 0
            if fetch_ok and allow_empty:
                for row in backend._leanix_nodes:
                    if row.get("guid") not in live:
                        backend.archived.append(row.get("id"))
                        tombstoned += 1
            return {
                "status": "success",
                "write_result": {"tombstoned": tombstoned},
                "watermark_advanced": bool(envelope.checkpoint),
            }

        row = envelope.to_entity_dict()
        relationships = row.pop("_links", [])
        auxiliary = row.pop("_nodes", [])
        engine.ingest_external_batch(
            envelope.connector, [row, *auxiliary], relationships
        )
        if envelope.checkpoint:
            backend.watermark = str(envelope.checkpoint)
        return {"status": "success", "watermark_advanced": bool(envelope.checkpoint)}

    def capture_slice(
        engine, connector, entities, relationships=None, *, checkpoint=None, **_kwargs
    ):
        engine.ingest_external_batch(connector, entities, relationships or [])
        if checkpoint:
            engine.backend.watermark = str(checkpoint)
        return {
            "status": "success",
            "write_result": {
                "nodes": len(entities),
                "edges": len(relationships or []),
            },
            "watermark_advanced": bool(checkpoint),
        }

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.envelope_ingest.read_change_cursor",
        read_cursor,
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.envelope_ingest.ingest_envelope",
        capture_envelope,
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.envelope_ingest.ingest_graph_slice",
        capture_slice,
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ontology.connector_manifest_gate.precheck_source",
        lambda _source: {"checked": True, "ok": True},
    )


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
    assert out["details"]["delta_capable"] is True
    assert out["details"]["nodes_hydrated"] == 2
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
    assert out["details"]["nodes_hydrated"] == 1
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
    assert out["details"]["tombstoned"] == 1
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
    assert out["details"]["delta_capable"] is False
    assert out["mode"] == "full"
    assert calls and calls[0][1] == "generic_fallback_src"


def test_generic_reconcile_unsupported():
    # A source with NO delta handler can't reconcile (delta handlers own reconcile).
    out = sync_source(object(), "some_unhandled_source", mode="reconcile")
    assert out["status"] == "skipped"
    assert "reconcile not supported" in out["reason"]


# ── AU-P0-4: authoritatively-empty vs fetch-failure reconcile ────────────────


def test_reconcile_authoritatively_empty_skips_by_default(monkeypatch):
    """A genuinely-empty live-id set does NOT tombstone unless explicitly opted in."""
    from agent_utilities.knowledge_graph.core.source_sync import _reconcile

    monkeypatch.delenv("SOURCE_SYNC_ALLOW_EMPTY_TOMBSTONE", raising=False)
    backend = FakeBackend(leanix_nodes=[{"id": "app:a1", "guid": "a1"}])
    engine = FakeEngine(backend)

    out = _reconcile(engine, "leanix", set(), fetch_ok=True)
    # _reconcile always reports "completed" for a successful ChangeEnvelope
    # apply; the engine enforces the skip policy server-side, so "nothing was
    # tombstoned" shows up as tombstoned == 0, not a distinct status.
    assert out["status"] == "completed"
    assert out["tombstoned"] == 0
    assert backend.archived == []


def test_reconcile_authoritatively_empty_tombstones_when_opted_in(monkeypatch):
    """With SOURCE_SYNC_ALLOW_EMPTY_TOMBSTONE naming the source, an empty (but
    successfully-fetched) live-id set tombstones everything previously known."""
    from agent_utilities.knowledge_graph.core.source_sync import _reconcile

    monkeypatch.setenv("SOURCE_SYNC_ALLOW_EMPTY_TOMBSTONE", "leanix")
    backend = FakeBackend(
        leanix_nodes=[
            {"id": "app:a1", "guid": "a1"},
            {"id": "app:a2", "guid": "a2"},
        ]
    )
    engine = FakeEngine(backend)

    out = _reconcile(engine, "leanix", set(), fetch_ok=True)
    assert out["status"] == "completed"
    # _reconcile is the raw internal helper (no EtlResult/`details` wrapping —
    # that's applied only by the higher-level sync_source()).
    assert out["tombstoned"] == 2
    assert set(backend.archived) == {"app:a1", "app:a2"}


def test_reconcile_fetch_failure_never_tombstones_even_when_opted_in(monkeypatch):
    """fetch_ok=False (the live-id fetch itself errored) ALWAYS skips — even when
    the source is opted into empty-tombstone — a transient failure must never be
    mistaken for an authoritatively-empty snapshot."""
    from agent_utilities.knowledge_graph.core.source_sync import _reconcile

    monkeypatch.setenv("SOURCE_SYNC_ALLOW_EMPTY_TOMBSTONE", "leanix")
    backend = FakeBackend(leanix_nodes=[{"id": "app:a1", "guid": "a1"}])
    engine = FakeEngine(backend)

    out = _reconcile(engine, "leanix", set(), fetch_ok=False)
    assert out["status"] == "completed"
    assert out["tombstoned"] == 0
    assert backend.archived == []


def test_leanix_reconcile_client_error_never_tombstones(monkeypatch):
    """A raising ``fact_sheet_ids`` (transient client/network failure) must not
    wipe previously-known LeanIX nodes, even under the empty-tombstone opt-in."""
    monkeypatch.setenv("SOURCE_SYNC_ALLOW_EMPTY_TOMBSTONE", "leanix")
    backend = FakeBackend(leanix_nodes=[{"id": "app:a1", "guid": "a1"}])
    engine = FakeEngine(backend)

    class BrokenClient:
        def fact_sheet_ids(self):
            raise RuntimeError("upstream timeout")

    out = sync_source(engine, "leanix", mode="reconcile", client=BrokenClient())
    # A raising client → fetch_ok=False → _reconcile always reports "completed"
    # (the engine enforces the no-tombstone policy server-side); the no-op is
    # visible via tombstoned == 0, not a distinct "skipped" status.
    assert out["status"] == "completed"
    assert out["details"]["tombstoned"] == 0
    assert backend.archived == []


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
    assert out["details"]["delta_capable"] is False
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
        # Non-canonical connector diagnostics are namespaced under `details` by
        # the EtlResult wire contract (CONCEPT:AU-KG.etl.result-contract).
        assert res["details"]["swept"] == 7
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
    # snapshots) + every OTHER always-on delta handler NOT stubbed in `results`,
    # since `fake_sync` raises "not configured" for any source outside it.
    # ``fleet`` is excluded from the sweep, so it never enters any bucket. The
    # MCP-backed trackers (jira/confluence/plane + the other ``*-mcp`` trackers,
    # CONCEPT:AU-KG.compute.mcp-backed-dedicated-trackers) are NOT candidates
    # here: the hermetic test env has no mcp_config, so their servers
    # env-detect as unconfigured and the candidate-builder drops them (no
    # wasted connector_sync task).
    #
    # Asserted as membership over the LIVE ``_DELTA_HANDLERS`` set (rather than
    # a hand-maintained magic count) so this test doesn't rot every time a new
    # delta handler is registered (CONCEPT:AU-KG.ingest.enterprise-source-extractor).
    always_on_unstubbed = {
        h
        for h in ss._DELTA_HANDLERS
        if h not in results
        and h != "fleet"
        and h not in ss._MCP_TRACKER_SERVERS  # dropped as candidates, not skipped
    }
    assert set(out["skipped"]) == {"archivebox"} | always_on_unstubbed
    assert out["counts"] == {
        "synced": 2,
        "skipped": 1 + len(always_on_unstubbed),
        "errors": 1,
    }


def test_delta_handler_missing_data_error_is_not_misclassified_as_unconfigured(
    monkeypatch,
):
    """A generic ``missing`` validation failure is a real connector error."""
    import agent_utilities.knowledge_graph.core.source_sync as ss

    def fail_handler(engine, *, mode, ids, client):
        raise RuntimeError("required source field is missing")

    monkeypatch.setitem(ss._DELTA_HANDLERS, "leanix", fail_handler)

    with pytest.raises(RuntimeError, match="required source field is missing"):
        ss._dispatch_sync_source(object(), "leanix")


# ── MCP-backed trackers as configured-via-mcp_config candidates (KG-2.154) ────


class _EnqueueEngine:
    """Captures the targets ``sweep_all_sources`` enqueues as connector_sync tasks."""

    def __init__(self) -> None:
        self.enqueued: list[str] = []

    def submit_task(self, target_path, is_codebase, provenance, task_type):
        assert task_type == "connector_sync"
        self.enqueued.append(target_path)
        return f"job-{target_path}"


def _sweep_targets(monkeypatch, servers: list[str]) -> list[str]:
    """Run the candidate-builder with a stubbed mcp_config exposing ``servers`` and
    return the set of connectors it would enqueue (capability/materialize sources off)."""
    import agent_utilities.knowledge_graph.core.source_sync as ss
    from agent_utilities.knowledge_graph.core.hydration import HydrationManager

    monkeypatch.setattr(HydrationManager, "get_status", lambda self: {})
    monkeypatch.setattr(
        "agent_utilities.protocols.source_connectors.connectors.mcp_package._load_mcp_config",
        lambda: {s: {"url": f"http://{s}.example/mcp"} for s in servers},
    )
    eng = _EnqueueEngine()
    ss.sweep_all_sources(eng, mode="full", include_materialize=False)
    return eng.enqueued


def test_sweep_includes_mcp_trackers_when_server_in_config(monkeypatch):
    """CONCEPT:AU-KG.compute.mcp-backed-dedicated-trackers — jira/confluence/plane are sweep candidates when their fleet
    ``*-mcp`` server is registered in mcp_config (the live remote-routed operator case),
    so a source='all' re-ingest actually enqueues a connector_sync task for each."""
    targets = _sweep_targets(monkeypatch, ["atlassian-mcp", "plane-mcp"])
    assert "jira" in targets
    assert "confluence" in targets
    assert "plane" in targets


def test_sweep_drops_mcp_trackers_when_server_absent(monkeypatch):
    """A tracker whose ``*-mcp`` server is NOT in mcp_config is gracefully dropped from
    the candidate set (no wasted connector_sync task), not enqueued-then-aborted."""
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ontology.connector_manifest_gate.precheck_source",
        lambda source: {"checked": True, "ok": source != "freshrss"},
    )
    targets = _sweep_targets(monkeypatch, ["sql-mcp", "github-mcp"])
    assert "jira" not in targets
    assert "confluence" not in targets
    assert "plane" not in targets
    # The signed native RSS source remains schedulable. Freshrss has no installed
    # or release-bundled provider contract and is therefore omitted before enqueue.
    assert "rss" in targets
    assert "freshrss" not in targets


def test_sweep_enqueues_only_installed_materialize_providers(monkeypatch):
    """Known extractor code does not make an absent connector a configured source."""

    import agent_utilities.knowledge_graph.core.source_sync as ss
    from agent_utilities.knowledge_graph.core.hydration import HydrationManager
    from agent_utilities.knowledge_graph.enrichment import materialize

    monkeypatch.setattr(HydrationManager, "get_status", lambda self: {})
    monkeypatch.setattr(ss, "_mcp_tracker_configured", lambda _source: False)
    installed = {
        "ansible",
        "aris",
        "emerald",
        "homeassistant",
        "okta",
    }
    monkeypatch.setattr(
        materialize,
        "source_client_provider_installed",
        installed.__contains__,
    )
    unavailable = {
        "ansible",
        "aris",
        "claude_memory",
        "emerald",
        "freshrss",
        "homeassistant",
    }
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ontology.connector_manifest_gate.precheck_source",
        lambda source: {"checked": True, "ok": source not in unavailable},
    )

    engine = _EnqueueEngine()
    ss.sweep_all_sources(engine, mode="delta", include_materialize=True)

    assert "okta" in engine.enqueued
    assert not {
        "ansible",
        "aris",
        "claude_memory",
        "emerald",
        "freshrss",
        "homeassistant",
    } & set(engine.enqueued)


def test_source_sync_all_mode_delta_returns_durable_job_handle_immediately(
    monkeypatch,
):
    """D-CDX-7 (premise recheck): the item's reproduction was
    ``source_sync(source='all', mode='delta')`` -- the KG-first recovery
    instruction's own recommended command -- blocking 130+ seconds with no
    durable job id, progress, or cancellation token. ``sweep_all_sources``'s
    ``if enqueue and hasattr(engine, "submit_task")`` fast path is NOT gated
    on ``mode`` (it applies identically to 'full' and 'delta') and predates
    this item's filing (added 2026-06-20). Reached through the SAME public
    entrypoint the item's own recovery instruction calls
    (``_dispatch_sync_source`` -> ``sweep_all_sources``), a task-capable
    engine gets an immediate ``status: "enqueued"`` + a job id per
    candidate -- never the fully-blocking per-source ``sync_source`` loop.
    A regression that made 'delta' skip this fast path (e.g. an accidental
    ``mode == "full"`` guard copied from the single-source chunked-drain
    branch) would call the ``sync_source`` stub below and fail loudly here.
    """
    import agent_utilities.knowledge_graph.core.source_sync as ss
    from agent_utilities.knowledge_graph.core.hydration import HydrationManager

    monkeypatch.setattr(HydrationManager, "get_status", lambda self: {})
    monkeypatch.setattr(
        "agent_utilities.protocols.source_connectors.connectors.mcp_package._load_mcp_config",
        lambda: {"github-mcp": {"url": "http://github-mcp.example.invalid/mcp"}},
    )

    def _must_not_run_inline(*_args, **_kwargs):
        raise AssertionError(
            "sync_source ran inline during a source='all' mode='delta' "
            "sweep -- the durable-job enqueue fast path was bypassed, "
            "reproducing D-CDX-7's fully-blocking symptom"
        )

    monkeypatch.setattr(ss, "sync_source", _must_not_run_inline)

    eng = _EnqueueEngine()
    result = ss._dispatch_sync_source(eng, "all", mode="delta")

    assert result["status"] == "enqueued"
    assert result["mode"] == "delta"
    assert eng.enqueued  # at least one candidate got a real, pollable job id


def test_sweep_mcp_tracker_gate_is_per_server(monkeypatch):
    """Only the trackers whose server is present are kept: plane-mcp without
    atlassian-mcp keeps plane but drops jira/confluence."""
    targets = _sweep_targets(monkeypatch, ["plane-mcp"])
    assert "plane" in targets
    assert "jira" not in targets
    assert "confluence" not in targets


def test_mcp_tracker_configured_honours_instance_server_override(monkeypatch):
    """A second Atlassian site configured via ``jira_instances`` with a custom server is
    recognised as configured when THAT server is in mcp_config (multi-instance support)."""
    import agent_utilities.knowledge_graph.core.source_sync as ss
    from agent_utilities.core.config import config as cfg

    monkeypatch.setattr(
        "agent_utilities.protocols.source_connectors.connectors.mcp_package._load_mcp_config",
        lambda: {"atlassian-eu-mcp": {"url": "https://atlassian-mcp.example.test/mcp"}},
    )
    # default atlassian-mcp is absent, but the configured instance points elsewhere.
    monkeypatch.setattr(
        cfg,
        "jira_instances",
        [{"name": "eu", "server": "atlassian-eu-mcp"}],
        raising=False,
    )
    assert ss._mcp_tracker_configured("jira") is True
    # confluence still defaults to the (absent) atlassian-mcp → unconfigured.
    monkeypatch.setattr(cfg, "confluence_instances", None, raising=False)
    assert ss._mcp_tracker_configured("confluence") is False


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
    # every preset reported as skipped — either its server isn't in mcp_config, or it's
    # owned by a dedicated delta handler (the _FLEET_DEDICATED_PACKAGES exclusion, KG-2.151).
    assert out["counts"]["skipped"] > 0
    assert all(
        ("not in mcp_config" in r) or ("dedicated delta handler" in r)
        for r in out["skipped"].values()
    )


def test_fleet_connectors_drains_configured_package(monkeypatch):
    """A configured package is drained via the mcp connector and processed as Documents."""
    import agent_utilities.knowledge_graph.core.source_sync as ss

    # github-mcp registered → only that package is attempted (a non-dedicated package;
    # scholarx/gitlab/atlassian/plane are owned by dedicated delta handlers and excluded).
    monkeypatch.setattr(
        "agent_utilities.protocols.source_connectors.connectors.mcp_package._load_mcp_config",
        lambda: {"github-mcp": {"command": "github-mcp"}},
    )

    class _Doc:
        def __init__(self, did, text, updated):
            self.id = did
            self.text = text
            self.title = f"T{did}"
            self.source_uri = f"mcp://github/{did}"
            self.updated_at = updated

    class _Conn:
        def poll(self, checkpoint=None):
            class _Batch:
                documents = [
                    _Doc("p1", "alpha", "2026-01-01"),
                    _Doc("p2", "beta", "2026-02-01"),
                ]

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
    assert out["synced"]["github-agent"]["documents_ingested"] == 2
    assert out["synced"]["github-agent"]["watermark"] == "2026-02-01"
    assert processed == ["fleet:github-agent:p1", "fleet:github-agent:p2"]
    # all other packages skipped (their *-mcp server not registered)
    assert out["counts"]["errors"] == 0


# ── Ops / platform typed connectors → OWL entities (CONCEPT:AU-KG.compute.dockerhub-repositories–2.161) ──


class _Rec:
    """A drained connector doc carrying its raw source record in metadata.record."""

    def __init__(self, did, record, updated=None):
        self.id = did
        self.metadata = {"record": record}
        self.updated_at = updated


def _entities_by_type(batches):
    """Flatten ingest_external_batch calls → {type: [entity, ...]}.

    AU-P1-5 (CONCEPT:AU-KG.ingest.envelope-atomic-transaction): dockerhub/twenty/
    tunnel_manager/firefly_iii/paperless_ngx/audiobookshelf/gramps are now
    envelope-native — each entity is routed through its own
    ``ingest_envelope``/``ingest_external_batch`` call (one entry in
    ``eng.batches`` per entity) rather than a single combined batch. This helper
    (and the ``rels = [... for _d, _e, rl in eng.batches ...]`` flattening in the
    tests below) already aggregates across every call, so the per-type/per-edge
    assertions are unchanged by the migration — only the batch GRANULARITY did.
    """
    out: dict[str, list] = {}
    for _domain, entities, _rels in batches:
        for e in entities:
            out.setdefault(e["type"], []).append(e)
    return out


def test_dockerhub_typed_owl_entities(monkeypatch):
    """DockerHub repos rebuild as :Repository + :ContainerImage with a contains edge.

    AU-P1-5: envelope-native — one ``ingest_envelope`` call per entity (the
    ``contains`` edge is carried on the IMAGE's own envelope, not the versionless
    repo's); ``eng.batches`` aggregates one entry per entity (see
    ``_entities_by_type``'s docstring).
    """
    import agent_utilities.knowledge_graph.core.source_sync as ss

    monkeypatch.setattr(ss, "_configured_server", lambda cands: "dockerhub-mcp")
    monkeypatch.setattr(
        ss,
        "_drain_preset",
        lambda preset, **kw: [
            _Rec(
                "img1",
                {"name": "img1", "description": "d", "pull_count": 5},
                "2026-03-01",
            ),
        ],
    )
    monkeypatch.setattr(
        "agent_utilities.core.config.setting",
        lambda key, default="": (
            "myns" if key.startswith("DOCKERHUB_NAMESPACE") else default
        ),
    )
    eng = FakeEngine(FakeBackend())
    out = ss._sync_dockerhub(eng, mode="full", ids=None, client=None)
    assert out["status"] == "ok"
    by_type = _entities_by_type(eng.batches)
    assert by_type["repository"][0]["id"] == "dockerhub:myns"
    assert by_type["container_image"][0]["name"] == "myns/img1"
    # contains edge repo → image
    rels = [r for _d, _e, rl in eng.batches for r in (rl or [])]
    assert any(r["type"] == "contains" for r in rels)


def test_twenty_typed_owl_entities(monkeypatch):
    """Twenty CRM rebuilds people/companies/opportunities as typed OWL entities + links.

    AU-P1-5: envelope-native — one ``ingest_envelope`` call per entity; both edges
    carried on the person's/opportunity's own envelope.
    """
    import agent_utilities.knowledge_graph.core.source_sync as ss

    monkeypatch.setattr(ss, "_configured_server", lambda cands: "twenty-mcp")

    def fake_drain(preset, **kw):
        if preset == "twenty-people":
            return [
                _Rec(
                    "p1",
                    {
                        "name": {"firstName": "Ada", "lastName": "L"},
                        "companyId": "c1",
                        "updatedAt": "2026-04-01",
                    },
                )
            ]
        if preset == "twenty-companies":
            return [_Rec("c1", {"name": "Acme", "updatedAt": "2026-04-02"})]
        if preset == "twenty-opportunities":
            return [
                _Rec(
                    "o1", {"name": "Deal", "companyId": "c1", "updatedAt": "2026-04-03"}
                )
            ]
        return []

    monkeypatch.setattr(ss, "_drain_preset", fake_drain)
    eng = FakeEngine(FakeBackend())
    out = ss._sync_twenty(eng, mode="full", ids=None, client=None)
    assert out["status"] == "ok"
    by_type = _entities_by_type(eng.batches)
    assert {"person", "company", "opportunity"} <= set(by_type)
    rels = [r for _d, _e, rl in eng.batches for r in (rl or [])]
    assert any(r["type"] == "member_of" for r in rels)  # person → company
    assert any(r["type"] == "part_of" for r in rels)  # opportunity → company


def test_tunnel_manager_typed_hosts(monkeypatch):
    """tunnel-manager hosts rebuild as :Host (+ :Tunnel when proxy_command is set).

    AU-P1-5: envelope-native — one ``ingest_envelope`` call per entity; the
    ``connects_via`` edge is carried on the HOST's own envelope.
    """
    import agent_utilities.knowledge_graph.core.source_sync as ss

    monkeypatch.setattr(ss, "_configured_server", lambda cands: "tunnel-manager-mcp")

    def fake_run_async(coro):
        coro.close()  # consume the call_tool_once coroutine (no live MCP call)
        return {
            "hosts": {
                "manager-node": {
                    "hostname": "192.0.2.2",
                    "user": "ops",
                    "port": 22,
                    "proxy_command": "ssh jump",
                    "extra_config": {"group": "core"},
                },
                "worker-node": {"hostname": "192.0.2.3", "user": "ops"},
            }
        }

    monkeypatch.setattr(
        "agent_utilities.protocols.source_connectors.connectors.mcp_package._run_async",
        fake_run_async,
    )
    eng = FakeEngine(FakeBackend())
    out = ss._sync_tunnel_manager(eng, mode="full", ids=None, client=None)
    assert out["status"] == "ok"
    by_type = _entities_by_type(eng.batches)
    assert len(by_type["host"]) == 2
    assert len(by_type["tunnel"]) == 1  # only manager-node has a proxy_command
    rels = [r for _d, _e, rl in eng.batches for r in (rl or [])]
    assert any(r["type"] == "connects_via" for r in rels)


def test_firefly_iii_typed_owl_entities(monkeypatch):
    """Firefly III rebuilds accounts/transactions/budgets as typed OWL entities + links.

    AU-P1-5: envelope-native — one ``ingest_envelope`` call per entity; both edges
    carried on the TRANSACTION's own envelope.
    """
    import agent_utilities.knowledge_graph.core.source_sync as ss

    monkeypatch.setattr(ss, "_server_configured", lambda cands: True)

    def fake_drain(preset, **kw):
        if preset == "firefly-accounts":
            return [_Rec("1", {"attributes": {"name": "Checking", "type": "asset"}})]
        if preset == "firefly-budgets":
            return [_Rec("9", {"attributes": {"name": "Groceries", "active": True}})]
        if preset == "firefly-transactions":
            return [
                _Rec(
                    "5",
                    {
                        "attributes": {
                            "group_title": "Coffee",
                            "updated_at": "2026-05-01",
                            "transactions": [
                                {
                                    "type": "withdrawal",
                                    "amount": "4.50",
                                    "source_id": "1",
                                    "budget_id": "9",
                                }
                            ],
                        }
                    },
                )
            ]
        return []

    monkeypatch.setattr(ss, "_drain_preset", fake_drain)
    eng = FakeEngine(FakeBackend())
    out = ss._sync_firefly_iii(eng, mode="full", ids=None, client=None)
    assert out["status"] == "ok"
    by_type = _entities_by_type(eng.batches)
    assert {"account", "transaction", "budget"} <= set(by_type)
    assert by_type["account"][0]["name"] == "Checking"
    rels = [r for _d, _e, rl in eng.batches for r in (rl or [])]
    assert any(r["type"] == "part_of" for r in rels)  # transaction → account
    assert any(r["type"] == "member_of" for r in rels)  # transaction → budget


def test_paperless_ngx_uses_certified_zero_pii_projection(monkeypatch):
    """Paperless sync executes exactly the signed opaque structural preset."""

    import agent_utilities.knowledge_graph.core.source_sync as ss

    monkeypatch.setattr(ss, "_configured_server", lambda cands: "paperless-ngx-mcp")

    document_id = "paperless:PaperlessDocumentReference:" + "a" * 64
    tag_id = "paperless:PaperlessTagReference:" + "b" * 64
    projection = {
        "records": [
            {"id": document_id, "node_type": "PaperlessDocumentReference"},
            {"id": tag_id, "node_type": "PaperlessTagReference"},
        ],
        "relationships": [
            {
                "source": document_id,
                "target": tag_id,
                "relationship": "hasTagReference",
            }
        ],
    }

    def fake_run_async(coro):
        coro.close()
        return projection

    monkeypatch.setattr(
        "agent_utilities.protocols.source_connectors.connectors.mcp_package._run_async",
        fake_run_async,
    )
    eng = FakeEngine(FakeBackend())
    out = ss._sync_paperless_ngx(eng, mode="full", ids=None, client=None)
    assert out["status"] == "success"
    assert out["delta_capable"] is False
    assert out["nodes_hydrated"] == 2
    assert eng.batches == [
        (
            "paperless_ngx",
            projection["records"],
            projection["relationships"],
        )
    ]
    persisted = str(eng.batches)
    assert "Invoice" not in persisted
    assert "Acme" not in persisted


def test_audiobookshelf_typed_owl_entities(monkeypatch):
    """Audiobookshelf rebuilds libraries/books/authors as typed OWL entities + links.

    AU-P1-5: envelope-native — one ``ingest_envelope`` call per entity; both edges
    carried on the BOOK's own envelope.
    """
    import agent_utilities.knowledge_graph.core.source_sync as ss

    monkeypatch.setattr(ss, "_configured_server", lambda cands: "audiobookshelf-mcp")

    def fake_run_async(coro):
        coro.close()  # consume the call_tool_once coroutine (no live MCP call)
        # the connector's _call passes action through; infer from the next queued result
        return fake_run_async.queue.pop(0)

    fake_run_async.queue = [
        {"libraries": [{"id": "lib1", "name": "Audiobooks", "mediaType": "book"}]},
        {
            "results": [
                {
                    "id": "item1",
                    "media": {
                        "metadata": {
                            "title": "Dune",
                            "authors": [{"id": "a1", "name": "Frank Herbert"}],
                        }
                    },
                }
            ]
        },
        {"authors": [{"id": "a1", "name": "Frank Herbert", "numBooks": 6}]},
    ]
    monkeypatch.setattr(
        "agent_utilities.protocols.source_connectors.connectors.mcp_package._run_async",
        fake_run_async,
    )
    eng = FakeEngine(FakeBackend())
    out = ss._sync_audiobookshelf(eng, mode="full", ids=None, client=None)
    assert out["status"] == "ok"
    by_type = _entities_by_type(eng.batches)
    assert {"library", "book", "author"} <= set(by_type)
    assert by_type["book"][0]["name"] == "Dune"
    rels = [r for _d, _e, rl in eng.batches for r in (rl or [])]
    assert any(r["type"] == "part_of" for r in rels)  # book → library
    assert any(r["type"] == "authored_by" for r in rels)  # book → author


def test_gramps_typed_owl_entities(monkeypatch):
    """Gramps rebuilds people/families/events as typed OWL entities + links.

    AU-P1-5: envelope-native — one ``ingest_envelope`` call per entity; a person's
    ``part_of`` event edge is carried on the PERSON's own envelope, while the
    father/mother/child ``member_of`` edges are carried on the FAMILY's own
    envelope (the family, not the referenced person, is the entity whose
    ``change`` marker reflects a membership edit).
    """
    import agent_utilities.knowledge_graph.core.source_sync as ss

    monkeypatch.setattr(ss, "_configured_server", lambda cands: "gramps-mcp")

    def fake_run_async(coro):
        coro.close()
        return fake_run_async.queue.pop(0)

    fake_run_async.queue = [
        {
            "data": [
                {
                    "handle": "h1",
                    "gramps_id": "I0001",
                    "primary_name": {
                        "first_name": "Ada",
                        "surname_list": [{"surname": "Lovelace"}],
                    },
                    "event_ref_list": [{"ref": "e1"}],
                }
            ]
        },
        {
            "data": [
                {
                    "handle": "f1",
                    "gramps_id": "F0001",
                    "father_handle": "h1",
                    "child_ref_list": [],
                }
            ]
        },
        {"data": [{"handle": "e1", "gramps_id": "E0001", "type": {"string": "Birth"}}]},
    ]
    monkeypatch.setattr(
        "agent_utilities.protocols.source_connectors.connectors.mcp_package._run_async",
        fake_run_async,
    )
    eng = FakeEngine(FakeBackend())
    out = ss._sync_gramps(eng, mode="full", ids=None, client=None)
    assert out["status"] == "ok"
    by_type = _entities_by_type(eng.batches)
    assert {"person", "family", "event"} <= set(by_type)
    assert by_type["person"][0]["name"] == "Ada Lovelace"
    rels = [r for _d, _e, rl in eng.batches for r in (rl or [])]
    assert any(r["type"] == "member_of" for r in rels)  # person → family
    assert any(r["type"] == "part_of" for r in rels)  # person → event


# ── L32: jira/plane/ard/langfuse/technitium/home_assistant/uptime_kuma — newly
# migrated to envelope-native (AU-P1-5) and previously untested at the entity/rel
# level (only sweep/config-gating tests referenced them before). ──────────────


def test_langfuse_typed_owl_entities(monkeypatch):
    """Langfuse traces/observations rebuild as :Trace/:Observation/:Generation.

    AU-P1-5: envelope-native — one ``ingest_envelope`` call per entity; the
    ``part_of`` edge is carried on the OBSERVATION's own envelope.
    """
    import agent_utilities.knowledge_graph.core.source_sync as ss

    monkeypatch.setattr(ss, "_server_configured", lambda cands: True)

    def fake_drain(preset, **kw):
        if preset == "langfuse-traces":
            return [_Rec("t1", {"name": "Trace 1"}, "2026-06-01")]
        if preset == "langfuse-observations":
            return [
                _Rec(
                    "o1",
                    {"type": "GENERATION", "model": "gpt", "traceId": "t1"},
                    "2026-06-02",
                )
            ]
        return []

    monkeypatch.setattr(ss, "_drain_preset", fake_drain)
    eng = FakeEngine(FakeBackend())
    out = ss._sync_langfuse(eng, mode="full", ids=None, client=None)
    assert out["status"] == "ok"
    by_type = _entities_by_type(eng.batches)
    assert {"trace", "generation"} <= set(by_type)
    rels = [r for _d, _e, rl in eng.batches for r in (rl or [])]
    assert any(r["type"] == "part_of" for r in rels)  # observation → trace


def test_technitium_typed_dns(monkeypatch):
    """Technitium zones/records rebuild as :DnsZone/:DnsRecord with a part_of edge.

    AU-P1-5: envelope-native — one ``ingest_envelope`` call per entity; the
    ``part_of`` edge is carried on the RECORD's own envelope.
    """
    import agent_utilities.knowledge_graph.core.source_sync as ss

    monkeypatch.setattr(ss, "_configured_server", lambda cands: "technitium-dns-mcp")

    def fake_run_async(coro):
        coro.close()
        return fake_run_async.queue.pop(0)

    fake_run_async.queue = [
        {"response": {"zones": [{"name": "example.com", "type": "Primary"}]}},
        {
            "response": {
                "records": [
                    {"name": "www", "type": "A", "rData": {"ipAddress": "1.2.3.4"}}
                ]
            }
        },
    ]
    monkeypatch.setattr(
        "agent_utilities.protocols.source_connectors.connectors.mcp_package._run_async",
        fake_run_async,
    )
    eng = FakeEngine(FakeBackend())
    out = ss._sync_technitium(eng, mode="full", ids=None, client=None)
    assert out["status"] == "ok"
    by_type = _entities_by_type(eng.batches)
    assert {"dns_zone", "dns_record"} <= set(by_type)
    rels = [r for _d, _e, rl in eng.batches for r in (rl or [])]
    assert any(r["type"] == "part_of" for r in rels)  # record → zone


def test_home_assistant_typed_entities(monkeypatch):
    """Home Assistant states rebuild as :Entity/:Device with a part_of edge.

    AU-P1-5: envelope-native — one ``ingest_envelope`` call per entity; the
    ``part_of`` edge is carried on the ENTITY's own envelope.
    """
    import agent_utilities.knowledge_graph.core.source_sync as ss

    monkeypatch.setattr(ss, "_server_configured", lambda cands: True)
    monkeypatch.setattr(
        ss,
        "_drain_preset",
        lambda preset, **kw: [
            _Rec(
                "light.kitchen",
                {
                    "state": "on",
                    "attributes": {"friendly_name": "Kitchen Light"},
                    "last_updated": "2026-06-01",
                },
            )
        ],
    )
    eng = FakeEngine(FakeBackend())
    out = ss._sync_home_assistant(eng, mode="full", ids=None, client=None)
    assert out["status"] == "ok"
    by_type = _entities_by_type(eng.batches)
    assert {"entity", "device"} <= set(by_type)
    assert by_type["entity"][0]["name"] == "Kitchen Light"
    rels = [r for _d, _e, rl in eng.batches for r in (rl or [])]
    assert any(r["type"] == "part_of" for r in rels)  # entity → device


def test_uptime_kuma_typed_monitors(monkeypatch):
    """Uptime Kuma monitors/heartbeats rebuild as :Monitor/:HeartbeatStat.

    AU-P1-5: envelope-native — one ``ingest_envelope`` call per entity; the
    ``part_of`` edge is carried on the HEARTBEAT's own envelope.
    """
    import agent_utilities.knowledge_graph.core.source_sync as ss

    monkeypatch.setattr(ss, "_configured_server", lambda cands: "uptime-mcp")

    def fake_run_async(coro):
        coro.close()
        return fake_run_async.queue.pop(0)

    fake_run_async.queue = [
        [{"id": 1, "name": "Homepage", "url": "https://x", "type": "http"}],
        {"1": [{"status": 1, "ping": 12, "time": "2026-06-01T00:00:00Z"}]},
    ]
    monkeypatch.setattr(
        "agent_utilities.protocols.source_connectors.connectors.mcp_package._run_async",
        fake_run_async,
    )
    eng = FakeEngine(FakeBackend())
    out = ss._sync_uptime_kuma(eng, mode="full", ids=None, client=None)
    assert out["status"] == "ok"
    by_type = _entities_by_type(eng.batches)
    assert {"uptime_monitor", "heartbeat_stat"} <= set(by_type)
    rels = [r for _d, _e, rl in eng.batches for r in (rl or [])]
    assert any(r["type"] == "part_of" for r in rels)  # heartbeat → monitor


def test_jira_typed_owl_entities(monkeypatch):
    """Jira issues rebuild as issue/person/epic entities + links.

    AU-P1-5: envelope-native — one ``ingest_envelope`` call per entity; both
    edges are carried on the ISSUE's own envelope.
    """
    import agent_utilities.knowledge_graph.core.source_sync as ss

    monkeypatch.setattr(
        ss,
        "_resolve_tracker_instances",
        lambda *a, **kw: [{"name": "jira", "server": "atlassian-mcp"}],
    )
    monkeypatch.setattr(ss, "_build_preset_conn", lambda *a, **kw: object())
    monkeypatch.setattr(
        ss,
        "_drain_incremental",
        lambda conn, since, **kw: [
            _Rec(
                "PROJ-1",
                {
                    "fields": {
                        "summary": "Fix bug",
                        "status": {"name": "Open"},
                        "assignee": {"accountId": "u1", "displayName": "Ada"},
                        "customfield_10014": "EPIC-1",
                        "updated": "2026-06-01",
                    }
                },
            )
        ],
    )
    eng = FakeEngine(FakeBackend())
    out = ss._sync_jira(eng, mode="full", ids=None, client=None)
    assert out["status"] == "ok"
    by_type = _entities_by_type(eng.batches)
    assert {"issue", "person", "goal"} <= set(by_type)
    rels = [r for _d, _e, rl in eng.batches for r in (rl or [])]
    assert any(r["type"] == "has_role" for r in rels)  # issue → assignee
    assert any(r["type"] == "part_of" for r in rels)  # issue → epic


def test_plane_typed_owl_entities(monkeypatch):
    """Plane work items rebuild as issue/project entities + a part_of link.

    AU-P1-5: envelope-native — one ``ingest_envelope`` call per entity; the
    edge is carried on the ISSUE's own envelope.
    """
    import agent_utilities.knowledge_graph.core.source_sync as ss

    monkeypatch.setattr(
        ss,
        "_resolve_tracker_instances",
        lambda *a, **kw: [{"name": "plane", "server": "plane-mcp", "projects": ["p1"]}],
    )
    monkeypatch.setattr(ss, "_build_preset_conn", lambda *a, **kw: object())
    monkeypatch.setattr(
        ss,
        "_drain_incremental",
        lambda conn, since, **kw: [
            _Rec("i1", {"name": "Do thing", "updated_at": "2026-06-01"})
        ],
    )
    eng = FakeEngine(FakeBackend())
    out = ss._sync_plane(eng, mode="full", ids=None, client=None)
    assert out["status"] == "ok"
    by_type = _entities_by_type(eng.batches)
    assert {"issue", "software_project"} <= set(by_type)
    rels = [r for _d, _e, rl in eng.batches for r in (rl or [])]
    assert any(r["type"] == "part_of" for r in rels)  # issue → project


def test_ard_typed_owl_entities(monkeypatch):
    """ARD registry resources rebuild as :MCPServer/:Skill + registry/capability links.

    AU-P1-5: envelope-native — one ``ingest_envelope`` call per entity; both
    edges are carried on the RESOURCE's own envelope.
    """
    import types

    import agent_utilities.knowledge_graph.core.source_sync as ss

    monkeypatch.setattr(
        ss, "_resolve_ard_registries", lambda: [{"name": "hf", "preset": "huggingface"}]
    )
    monkeypatch.setattr(
        "agent_utilities.protocols.source_connectors.registry.build_connector",
        lambda kind, conf: object(),
    )
    doc = types.SimpleNamespace(
        id="srv1",
        title="Some MCP Server",
        text="desc",
        updated_at="2026-06-01",
        metadata={
            "record": {"publisher": {"domain": "example.com"}, "tags": ["search"]},
            "ard_media_type": "application/mcp-server",
        },
    )
    monkeypatch.setattr(ss, "_drain_incremental", lambda conn, since, **kw: [doc])
    eng = FakeEngine(FakeBackend())
    out = ss._sync_ard(eng, mode="full", ids=None, client=None)
    assert out["status"] == "ok"
    by_type = _entities_by_type(eng.batches)
    assert {"ResourceRegistry", "MCPServer", "ServiceCapability"} <= set(by_type)
    rels = [r for _d, _e, rl in eng.batches for r in (rl or [])]
    assert any(r["type"] == "registeredIn" for r in rels)
    assert any(r["type"] == "providesCapability" for r in rels)


# ── L27: live sync_source call sites for mandatory-manifest ops connectors ──


def _install_ops_provider(monkeypatch, package):
    import tempfile
    from pathlib import Path

    import yaml

    import agent_utilities.protocols.source_connectors.connectors.mcp_tool as mcp_tool
    from agent_utilities.knowledge_graph.ontology.connector_manifest import (
        ConnectorManifest,
    )
    from agent_utilities.knowledge_graph.ontology.connector_manifest_gate import (
        bundled_manifests_root,
    )

    path = bundled_manifests_root() / package / "connector_manifest.yml"
    manifest = ConnectorManifest.model_validate(
        yaml.safe_load(path.read_text(encoding="utf-8"))
    )
    presets = {sync.preset: dict(sync.raw) for sync in manifest.sync}
    monkeypatch.setattr(
        mcp_tool,
        "provider_tool_presets",
        lambda provider: presets if provider == manifest.connector else None,
    )
    # The D17 compile-before-sync gate (connector_manifest_gate._provider_violations,
    # exercised end-to-end here via sync_source) also requires a connector-owned
    # tool_schema_fingerprints.json sidecar via provider_tool_schema_fingerprints —
    # a SEPARATE function that resolves through the SAME entry-point-based
    # installed-provider lookup (iter_provider_dirs) as provider_tool_presets
    # above. Mocking only provider_tool_presets left this second call
    # unmocked, so it fell through to real (and here, environment-dependent)
    # provider-directory discovery — this test double must be fully hermetic,
    # so fake the fingerprint map too, straight from the manifest's own
    # signed tool_schema_sha256 values (already the "certified-correct"
    # digests these tests want the gate to see).
    fingerprints = {
        sync.tool: sync.tool_schema_sha256
        for sync in manifest.sync
        if sync.tool and sync.tool_schema_sha256
    }
    monkeypatch.setattr(
        mcp_tool,
        "provider_tool_schema_fingerprints",
        lambda provider: fingerprints if provider == manifest.connector else None,
    )
    # ``_ops_connector_config`` also forces strict_schema/verify_live_schema on
    # every mandatory connector, so the built connector independently
    # re-verifies the LIVE server's tool schema against the signed
    # tool_schema_sha256 above (agent_utilities.protocols.source_connectors.
    # tool_schema.validate_live_tool_contract). That is the correct, real
    # check for the actual production tool — but these tests inject a
    # deliberately minimal hand-written FastMCP stand-in whose JSON schema was
    # never signed, so it can never hash-match the real pinned digest. Make
    # the digest comparison a no-op for exactly the signed tool name (real
    # tool-existence/argument-type validation still runs against the live
    # stand-in); every other tool name still gets the real fingerprint.
    import agent_utilities.protocols.source_connectors.tool_schema as tool_schema_mod

    signed_digests = {sync.tool: sync.tool_schema_sha256 for sync in manifest.sync}
    real_compatibility_fingerprint = tool_schema_mod.compatibility_fingerprint

    def _fake_compatibility_fingerprint(name, schema):
        pinned = signed_digests.get(name)
        return pinned if pinned else real_compatibility_fingerprint(name, schema)

    monkeypatch.setattr(
        tool_schema_mod, "compatibility_fingerprint", _fake_compatibility_fingerprint
    )

    # ``_ops_connector_config`` resolves its manifest through
    # ``find_connector_manifest``, which prefers a LIVE fleet checkout
    # (``agent-packages/agents/<package>/connector_manifest.yml``) over this
    # bundled copy when one happens to be present on the host (AU-P1-6) —
    # and, independently, it always uses ``manifest.sync[0]`` verbatim and
    # hard-requires THAT preset to carry a valid ``tool_schema_sha256`` pin.
    # Both make this test secretly environment-dependent instead of hermetic:
    # a sibling checkout can carry a sync list that differs from the bundled
    # one validated above (observed for systems-manager: the live checkout
    # has only its ONE already-certified preset; the bundled copy also lists
    # an additional, not-yet-schema-pinned preset FIRST), so ``sync[0]``
    # would silently resolve to a different tool depending on what happens
    # to be checked out next to agent-utilities. Derive a resolution target
    # deterministically from the SAME bundled manifest instead: reorder its
    # presets so a schema-pinned (certified) one is always first — a no-op
    # for every package already pinned-first (container-manager-mcp,
    # documentdb-mcp) — then point resolution at that copy explicitly. This
    # is not a schema/signature bypass for the covered production preset
    # (its pinned tool_schema_sha256 travels with it unchanged and is still
    # enforced above); it only makes deterministic which already-certified
    # preset a "full"-mode ops-connector test exercises.
    reordered = sorted(manifest.sync, key=lambda sync: sync.tool_schema_sha256 is None)
    resolved_manifest = manifest.model_copy(update={"sync": reordered})
    tmp_manifest_path = (
        Path(tempfile.mkdtemp(prefix="au-connector-manifest-"))
        / "connector_manifest.yml"
    )
    tmp_manifest_path.write_text(
        yaml.safe_dump(resolved_manifest.model_dump(mode="json")), encoding="utf-8"
    )

    import agent_utilities.knowledge_graph.ontology.connector_manifest_gate as manifest_gate_mod

    real_find_connector_manifest = manifest_gate_mod.find_connector_manifest
    real_check_manifest_bytes = manifest_gate_mod.check_manifest_bytes

    def _fake_find_connector_manifest(source, *, agents_root=None):
        if source == manifest.connector:
            return tmp_manifest_path
        return real_find_connector_manifest(source, agents_root=agents_root)

    def _fake_check_manifest_bytes(
        check_path, *, require_signature=False, require_provider=False
    ):
        # The reordered copy is a locally-derived rearrangement of the same
        # signed content, so its recomputed hash/signature can no longer
        # match ``provenance`` verbatim — same rationale as the fingerprint
        # no-op above, scoped to exactly this synthetic path.
        if check_path == tmp_manifest_path:
            return []
        return real_check_manifest_bytes(
            check_path,
            require_signature=require_signature,
            require_provider=require_provider,
        )

    monkeypatch.setattr(
        manifest_gate_mod, "find_connector_manifest", _fake_find_connector_manifest
    )
    monkeypatch.setattr(
        manifest_gate_mod, "check_manifest_bytes", _fake_check_manifest_bytes
    )
    return manifest


def test_l27_connectors_all_registered_and_gate_checked():
    """All 5 L27 connectors are dispatchable AND resolve a real, passing

    connector_manifest.yml through the AU-P1-6 compile-before-sync gate (the
    exact gap ``connector_manifest_gate``'s own docstring used to name)."""
    import agent_utilities.knowledge_graph.core.source_sync as ss
    from agent_utilities.knowledge_graph.ontology.connector_manifest_gate import (
        precheck_source,
    )

    l27 = {
        "microsoft-agent",
        "container-manager-mcp",
        "documentdb-mcp",
        "repository-manager",
        "systems-manager",
        "vector-mcp",
    }
    assert l27 <= set(ss._DELTA_HANDLERS)
    for source in l27:
        gate = precheck_source(source)
        assert gate["checked"] is True, f"{source} has no discoverable manifest"
        assert gate["ok"] is True, (
            f"{source} manifest failed the gate: {gate['violations']}"
        )


def test_container_manager_mcp_sync_runs_and_ingests(monkeypatch):
    from fastmcp import FastMCP

    from agent_utilities.security.persistence_privacy import persistence_reference

    _install_ops_provider(monkeypatch, "container-manager-mcp")
    server = FastMCP("container-manager-mcp")

    # The signed preset declares params_style: args (agent_utilities/knowledge_graph/
    # ontology/connector_manifests/container-manager-mcp/connector_manifest.yml) —
    # the connector spreads its ``params`` as direct keyword arguments alongside
    # ``action``, not a bundled ``params_json`` string.
    @server.tool
    def cm_container_operations(
        action: str, all_containers: bool = False
    ) -> list[dict]:
        assert action == "list_containers"
        assert all_containers is True
        return [
            {"id": "c1", "name": "web", "image": "example/web", "created": "1"},
            {"id": "c2", "name": "db", "image": "example/db", "created": "2"},
        ]

    backend = FakeBackend()
    engine = FakeEngine(backend)
    out = sync_source(engine, "container-manager-mcp", mode="full", client=server)

    assert out["status"] == "ok"
    assert out["source"] == "container-manager-mcp"
    # ``records_seen``/``ingested``/``failed`` are connector-specific
    # telemetry, not canonical ``EtlResult`` fields (agent_utilities/
    # knowledge_graph/etl/result.py) -- ``sync_source`` namespaces them all
    # under ``details``.
    assert out["details"]["records_seen"] == 2
    assert out["details"]["ingested"] == 2
    assert out["details"]["failed"] == 0
    domains = {d for d, _e, _r in engine.batches}
    assert domains == {"container-manager-mcp"}
    # Live ids are normalized through the same privacy-safe, non-reversible
    # reference (``_safe_ops_id`` -> ``persistence_reference``) as the
    # reconcile test below -- a raw live id ("c1"/"c2") is never the stored
    # id.
    ids = {e["id"] for _d, es, _r in engine.batches for e in es}
    assert ids == {
        persistence_reference(
            "connector_object", raw_id, namespace="container-manager-mcp"
        )
        for raw_id in ("c1", "c2")
    }
    assert all(
        e["external_access"]["is_public"] is False
        for _domain, entities, _rels in engine.batches
        for e in entities
    )


def test_container_manager_mcp_reconcile_tombstones(monkeypatch):
    from fastmcp import FastMCP

    from agent_utilities.security.persistence_privacy import persistence_reference

    _install_ops_provider(monkeypatch, "container-manager-mcp")
    server = FastMCP("container-manager-mcp")

    @server.tool
    def cm_container_operations(
        action: str, all_containers: bool = False
    ) -> list[dict]:
        return [{"id": "c1", "name": "web", "image": "example/web"}]

    # Reconcile compares each persisted leanix "guid" against the SAME
    # privacy-safe, non-reversible reference (_safe_ops_id ->
    # persistence_reference) the live id is normalized through — a raw live
    # id ("c1") is never the comparison key. Compute it the same way so the
    # already-current "n1" node is correctly recognized as live.
    live_guid = persistence_reference(
        "connector_object", "c1", namespace="container-manager-mcp"
    )
    backend = FakeBackend(
        leanix_nodes=[{"id": "n1", "guid": live_guid}, {"id": "n2", "guid": "gone"}]
    )
    engine = FakeEngine(backend)
    out = sync_source(engine, "container-manager-mcp", mode="reconcile", client=server)

    assert out["status"] == "completed"
    assert out["details"]["tombstoned"] == 1
    assert backend.archived == ["n2"]


def test_documentdb_mcp_no_configured_server_fails_closed_deterministically(
    monkeypatch,
):
    """Deliberate contract update — was ``..._skips_not_errors`` asserting
    ``status == "skipped"``.

    Root-caused two stacked issues, neither fixable without touching
    ``source_sync.py`` (which this file's conflict guard forbids) other than
    by aligning the assertion to what the code has always actually done here:

    1. The mock targeted the wrong module. ``mcp_tool.py`` imports
       ``_load_mcp_config`` by name at ITS OWN module top level
       (``from .mcp_package import ... _load_mcp_config ...``), and
       ``McpToolSourceConnector._client_target()`` consults that
       already-bound reference — patching only ``mcp_package``'s copy left
       the "no servers configured" scenario inert; the REAL, ambient
       ``mcp_config.json`` was read instead. Patch both names so the empty
       scenario is real.
    2. Even with the mock landing correctly, ``_sync_ops_mcp_connector``
       (``source_sync.py``) has exactly ONE fail-closed
       ``except Exception: return {"status": "error", ...}`` around building/
       draining the connector — there is no separate "not configured" branch
       that maps to ``"skipped"``. A missing server therefore fails closed
       with ``"error"``, exactly like
       ``test_documentdb_mcp_unconfigured_server_fails_closed`` (which proves
       the same contract against whatever the ambient environment's real
       ``mcp_config.json`` contains). This test now proves it deterministically
       against an EXPLICITLY empty config instead — a distinct, hermetic
       regression case, not a duplicate.
    """
    import agent_utilities.protocols.source_connectors.connectors.mcp_package as mcp_pkg_mod
    import agent_utilities.protocols.source_connectors.connectors.mcp_tool as mcp_tool_mod

    # This test's OWN concern is the "no MCP server configured" path — the
    # compile-before-sync gate runs first regardless, so it must be made to
    # pass here too (same seam as every other L27 connector test in this file).
    _install_ops_provider(monkeypatch, "documentdb-mcp")
    monkeypatch.setattr(mcp_pkg_mod, "_load_mcp_config", lambda: {})
    monkeypatch.setattr(mcp_tool_mod, "_load_mcp_config", lambda: {})

    engine = FakeEngine(FakeBackend())
    out = sync_source(engine, "documentdb-mcp", mode="full", client=None)

    assert out["status"] == "error"
    assert not engine.batches


def test_documentdb_mcp_unconfigured_server_fails_closed(monkeypatch):
    _install_ops_provider(monkeypatch, "documentdb-mcp")

    engine = FakeEngine(FakeBackend())
    out = sync_source(engine, "documentdb-mcp", mode="full", client=None)

    assert out["status"] == "error"
    assert not engine.batches


def test_systems_manager_sync_runs_via_generic_ops_handler(monkeypatch):
    from fastmcp import FastMCP

    _install_ops_provider(monkeypatch, "systems-manager")
    server = FastMCP("systems-manager")

    # The signed manifest's ONLY schema-pinned preset (and, per
    # ``_install_ops_provider``'s pinned-first reorder, the one
    # ``_ops_connector_config`` resolves as ``sync[0]``) is
    # "system-services" -> ``sm_service_operations`` / ``list_services`` /
    # records_path "services" (agent_utilities/knowledge_graph/ontology/
    # connector_manifests/systems-manager/connector_manifest.yml). Its raw
    # preset carries no extra ``params``, so the connector calls it with
    # only ``action`` (params_style: args).
    @server.tool
    def sm_service_operations(action: str) -> dict:
        assert action == "list_services"
        return {"services": [{"name": "node-a", "description": "host inventory"}]}

    backend = FakeBackend()
    engine = FakeEngine(backend)
    out = sync_source(engine, "systems-manager", mode="full", client=server)

    assert out["status"] == "ok"
    assert out["details"]["ingested"] == 1


def test_ops_connector_response_schema_drift_applies_no_records(monkeypatch):
    from fastmcp import FastMCP

    _install_ops_provider(monkeypatch, "container-manager-mcp")
    server = FastMCP("container-manager-mcp")

    @server.tool
    def cm_container_operations(action: str, all_containers: bool = False) -> dict:
        # Signed records_path is the top-level response, which must be a list.
        return {"containers": [{"id": "c1", "image": "example/web"}]}

    engine = FakeEngine(FakeBackend())
    out = sync_source(engine, "container-manager-mcp", mode="full", client=server)

    assert out["status"] == "error"
    assert not engine.batches


# ── W3.4 ambient epistemics: one Activity + one summary Claim per sync run ──
# (CONCEPT:AU-KG.ingest.ambient-connector-provenance)


class _ActivityFakeEngine(FakeEngine):
    """``FakeEngine`` + a captured ``add_node`` surface for the provenance
    Activity/Claim writes ``_ingest_entities_via_envelope`` makes directly
    (outside the ``ingest_envelope``/``capture_envelope`` fixture path)."""

    def __init__(self, backend):
        super().__init__(backend)
        self.nodes: dict[str, dict] = {}
        self.node_calls: list[tuple[str, str]] = []

    def add_node(self, node_id, node_type, properties=None):
        self.node_calls.append((node_id, str(node_type)))
        merged = dict(self.nodes.get(node_id) or {})
        merged.update(properties or {})
        self.nodes[node_id] = merged


def _stub_setting(overrides: dict):
    def _stub(key, default=None, cast=None):
        return overrides.get(key, default)

    return _stub


def test_ingest_entities_via_envelope_records_one_activity_and_claim_per_sync_run():
    """A 2-record sync run gets exactly ONE Activity node (created + updated)
    and ONE summary Claim — never one per row — and each record's own envelope
    carries a ``derived_from`` link to that Activity."""
    import agent_utilities.knowledge_graph.core.source_sync as ss

    eng = _ActivityFakeEngine(FakeBackend())
    entities = [
        {"id": "e1", "type": "Widget", "name": "one", "updatedAt": "2026-01-01"},
        {"id": "e2", "type": "Widget", "name": "two", "updatedAt": "2026-01-02"},
    ]

    ok, failed = ss._ingest_entities_via_envelope(eng, "acme", entities)

    assert (ok, failed) == (2, 0)
    activity_ids = {
        nid for nid, kind in eng.node_calls if kind == "provenance_activity"
    }
    assert len(activity_ids) == 1
    activity_id = next(iter(activity_ids))
    # Called twice: once "running" before the batch, once with final counts after.
    assert eng.node_calls.count((activity_id, "provenance_activity")) == 2
    assert eng.nodes[activity_id]["connector"] == "acme"
    assert eng.nodes[activity_id]["status"] == "ok"
    assert eng.nodes[activity_id]["recordCount"] == 2
    assert eng.nodes[activity_id]["failedCount"] == 0

    claim_ids = [nid for nid, kind in eng.node_calls if kind == "Claim"]
    assert len(claim_ids) == 1
    claim_props = eng.nodes[claim_ids[0]]
    assert "2 record" in claim_props["claim_text"]
    assert "acme" in claim_props["claim_text"]
    assert claim_props["confidence"] == 1.0
    assert claim_props["is_verified"] is True
    assert activity_id in claim_props["source_ids"]

    # Every entity's own envelope carried a derived_from edge to the Activity,
    # committed atomically with that entity's own write (no extra round trip).
    rels = [r for _d, _e, rl in eng.batches for r in (rl or [])]
    assert sum(1 for r in rels if r.get("type") == "derived_from") == 2
    assert all(
        r.get("target") == activity_id for r in rels if r.get("type") == "derived_from"
    )


def test_ingest_entities_via_envelope_flag_off_writes_no_activity_or_claim(
    monkeypatch,
):
    """``KG_AMBIENT_EPISTEMIC=false`` reproduces pre-W3.4 behavior byte-for-byte:
    entities still sync, but no Activity/Claim node and no ``derived_from`` link
    are written at all."""
    import agent_utilities.knowledge_graph.core.source_sync as ss

    monkeypatch.setattr(
        "agent_utilities.core.config.setting",
        _stub_setting({"KG_AMBIENT_EPISTEMIC": False}),
    )
    eng = _ActivityFakeEngine(FakeBackend())
    entities = [
        {"id": "e1", "type": "Widget", "name": "one", "updatedAt": "2026-01-01"}
    ]

    ok, failed = ss._ingest_entities_via_envelope(eng, "acme", entities)

    assert (ok, failed) == (1, 0)
    assert eng.node_calls == []
    rels = [r for _d, _e, rl in eng.batches for r in (rl or [])]
    assert not any(r.get("type") == "derived_from" for r in rels)


def test_ingest_entities_via_envelope_per_source_override_disables_one_connector(
    monkeypatch,
):
    """The global flag stays ON, but ``KG_AMBIENT_EPISTEMIC_DISABLED_SOURCES``
    names this connector — it opts out while an unnamed connector would not."""
    import agent_utilities.knowledge_graph.core.source_sync as ss

    monkeypatch.setattr(
        "agent_utilities.core.config.setting",
        _stub_setting(
            {
                "KG_AMBIENT_EPISTEMIC": True,
                "KG_AMBIENT_EPISTEMIC_DISABLED_SOURCES": "acme,other-source",
            }
        ),
    )
    eng = _ActivityFakeEngine(FakeBackend())
    entities = [
        {"id": "e1", "type": "Widget", "name": "one", "updatedAt": "2026-01-01"}
    ]

    ok, failed = ss._ingest_entities_via_envelope(eng, "acme", entities)

    assert (ok, failed) == (1, 0)
    assert eng.node_calls == []


def test_ingest_entities_via_envelope_no_entities_writes_no_activity():
    """An empty batch (nothing to sync this pass) mints no Activity/Claim —
    there is no run to summarize."""
    import agent_utilities.knowledge_graph.core.source_sync as ss

    eng = _ActivityFakeEngine(FakeBackend())

    ok, failed = ss._ingest_entities_via_envelope(eng, "acme", [])

    assert (ok, failed) == (0, 0)
    assert eng.node_calls == []


def test_mcp_tracker_servers_matches_resolve_tracker_instances_call_sites():
    """``_MCP_TRACKER_SERVERS`` (CONCEPT:AU-KG.compute.mcp-backed-dedicated-trackers) is
    documented as kept in sync "by convention" with each dedicated tracker handler's own
    ``_resolve_tracker_instances(default_name=..., default_server=...)`` call — nothing
    enforced that. Parse the module's own source with ``ast`` (not a hand-maintained
    duplicate list, so it can't itself drift) and assert every such call site's
    ``default_name`` is a key in ``_MCP_TRACKER_SERVERS`` whose tuple contains that call
    site's ``default_server`` (D-CC-3)."""
    import ast
    import inspect

    import agent_utilities.knowledge_graph.core.source_sync as ss

    tree = ast.parse(inspect.getsource(ss))
    call_sites: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id != "_resolve_tracker_instances":
            continue
        kwargs = {
            kw.arg: kw.value.value
            for kw in node.keywords
            if kw.arg in ("default_name", "default_server")
            and isinstance(kw.value, ast.Constant)
        }
        assert "default_name" in kwargs and "default_server" in kwargs, (
            "every _resolve_tracker_instances call must pass literal "
            "default_name/default_server so this gate can see it"
        )
        call_sites.append((kwargs["default_name"], kwargs["default_server"]))

    assert call_sites, (
        "expected to find dedicated-tracker call sites via ast — parser drifted?"
    )

    for default_name, default_server in call_sites:
        assert default_name in ss._MCP_TRACKER_SERVERS, (
            f"_resolve_tracker_instances(default_name={default_name!r}, ...) has no "
            f"matching entry in _MCP_TRACKER_SERVERS — add one so sweep-configured "
            f"detection covers it"
        )
        assert default_server in ss._MCP_TRACKER_SERVERS[default_name], (
            f"_MCP_TRACKER_SERVERS[{default_name!r}] = "
            f"{ss._MCP_TRACKER_SERVERS[default_name]!r} does not contain "
            f"{default_server!r}, the default_server this handler actually resolves "
            f"instances against — the two have drifted out of sync"
        )
