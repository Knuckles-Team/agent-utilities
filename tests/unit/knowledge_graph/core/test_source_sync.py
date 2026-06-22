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
    # A source with NO delta handler can't reconcile (delta handlers own reconcile).
    out = sync_source(object(), "some_unhandled_source", mode="reconcile")
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
    # snapshots) + the unconfigured delta handlers (rss/freshrss/fleet_connectors)
    # whose fake_sync raises "not configured". ``fleet`` is excluded from the sweep,
    # so it never enters any bucket. The MCP-backed trackers (jira/confluence/plane,
    # CONCEPT:KG-2.154) are NOT candidates here: the hermetic test env has no
    # mcp_config, so their ``*-mcp`` servers env-detect as unconfigured and the
    # candidate-builder drops them (no wasted connector_sync task).
    assert out["counts"] == {"synced": 2, "skipped": 4, "errors": 1}


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
        lambda: {s: {"url": f"http://{s}.arpa/mcp"} for s in servers},
    )
    eng = _EnqueueEngine()
    ss.sweep_all_sources(eng, mode="full", include_materialize=False)
    return eng.enqueued


def test_sweep_includes_mcp_trackers_when_server_in_config(monkeypatch):
    """CONCEPT:KG-2.154 — jira/confluence/plane are sweep candidates when their fleet
    ``*-mcp`` server is registered in mcp_config (the live remote-routed operator case),
    so a source='all' re-ingest actually enqueues a connector_sync task for each."""
    targets = _sweep_targets(monkeypatch, ["atlassian-mcp", "plane-mcp"])
    assert "jira" in targets
    assert "confluence" in targets
    assert "plane" in targets


def test_sweep_drops_mcp_trackers_when_server_absent(monkeypatch):
    """A tracker whose ``*-mcp`` server is NOT in mcp_config is gracefully dropped from
    the candidate set (no wasted connector_sync task), not enqueued-then-aborted."""
    targets = _sweep_targets(monkeypatch, ["sql-mcp", "github-mcp"])
    assert "jira" not in targets
    assert "confluence" not in targets
    assert "plane" not in targets
    # the always-local feed handlers are unaffected by the tracker gate
    assert "rss" in targets and "freshrss" in targets


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
        lambda: {"atlassian-eu-mcp": {"url": "http://atlassian-eu-mcp.arpa/mcp"}},
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


# ── Ops / platform typed connectors → OWL entities (CONCEPT:KG-2.155–2.161) ──


class _Rec:
    """A drained connector doc carrying its raw source record in metadata.record."""

    def __init__(self, did, record, updated=None):
        self.id = did
        self.metadata = {"record": record}
        self.updated_at = updated


def _entities_by_type(batches):
    """Flatten ingest_external_batch calls → {type: [entity, ...]}."""
    out: dict[str, list] = {}
    for _domain, entities, _rels in batches:
        for e in entities:
            out.setdefault(e["type"], []).append(e)
    return out


def test_dockerhub_typed_owl_entities(monkeypatch):
    """DockerHub repos rebuild as :Repository + :ContainerImage with a contains edge."""
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
    """Twenty CRM rebuilds people/companies/opportunities as typed OWL entities + links."""
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
    """tunnel-manager hosts rebuild as :Host (+ :Tunnel when proxy_command is set)."""
    import agent_utilities.knowledge_graph.core.source_sync as ss

    monkeypatch.setattr(ss, "_configured_server", lambda cands: "tunnel-manager-mcp")

    def fake_run_async(coro):
        coro.close()  # consume the call_tool_once coroutine (no live MCP call)
        return {
            "hosts": {
                "r820": {
                    "hostname": "10.0.0.2",
                    "user": "ops",
                    "port": 22,
                    "proxy_command": "ssh jump",
                    "extra_config": {"group": "core"},
                },
                "rw710": {"hostname": "10.0.0.3", "user": "ops"},
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
    assert len(by_type["tunnel"]) == 1  # only r820 has a proxy_command
    rels = [r for _d, _e, rl in eng.batches for r in (rl or [])]
    assert any(r["type"] == "connects_via" for r in rels)


def test_firefly_iii_typed_owl_entities(monkeypatch):
    """Firefly III rebuilds accounts/transactions/budgets as typed OWL entities + links."""
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


def test_paperless_ngx_typed_owl_entities(monkeypatch):
    """Paperless-ngx rebuilds documents/correspondents/tags as typed OWL entities + links."""
    import agent_utilities.knowledge_graph.core.source_sync as ss

    monkeypatch.setattr(ss, "_server_configured", lambda cands: True)

    def fake_drain(preset, **kw):
        if preset == "paperless-correspondents":
            return [_Rec("3", {"name": "Acme Corp"})]
        if preset == "paperless-tags":
            return [_Rec("7", {"name": "invoice"})]
        if preset == "paperless-documents":
            return [
                _Rec(
                    "11",
                    {
                        "title": "Invoice Q1",
                        "correspondent": 3,
                        "tags": [7],
                        "modified": "2026-05-02",
                    },
                )
            ]
        return []

    monkeypatch.setattr(ss, "_drain_preset", fake_drain)
    eng = FakeEngine(FakeBackend())
    out = ss._sync_paperless_ngx(eng, mode="full", ids=None, client=None)
    assert out["status"] == "ok"
    by_type = _entities_by_type(eng.batches)
    assert {"document", "correspondent", "tag"} <= set(by_type)
    rels = [r for _d, _e, rl in eng.batches for r in (rl or [])]
    assert any(r["type"] == "member_of" for r in rels)  # document → correspondent
    assert any(r["type"] == "tagged_with" for r in rels)  # document → tag


def test_audiobookshelf_typed_owl_entities(monkeypatch):
    """Audiobookshelf rebuilds libraries/books/authors as typed OWL entities + links."""
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


def test_gramps_web_typed_owl_entities(monkeypatch):
    """Gramps Web rebuilds people/families/events as typed OWL entities + links."""
    import agent_utilities.knowledge_graph.core.source_sync as ss

    monkeypatch.setattr(ss, "_configured_server", lambda cands: "gramps-web-mcp")

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
    out = ss._sync_gramps_web(eng, mode="full", ids=None, client=None)
    assert out["status"] == "ok"
    by_type = _entities_by_type(eng.batches)
    assert {"person", "family", "event"} <= set(by_type)
    assert by_type["person"][0]["name"] == "Ada Lovelace"
    rels = [r for _d, _e, rl in eng.batches for r in (rl or [])]
    assert any(r["type"] == "member_of" for r in rels)  # person → family
    assert any(r["type"] == "part_of" for r in rels)  # person → event
