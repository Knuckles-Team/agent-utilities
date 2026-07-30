"""Fleet capability elevation: tools → Tool capability nodes (CONCEPT:AU-KG.ontology.capability-node-aliases-lexical).

These cover the data half of ontology-native classification — that the served
multiplexer catalog becomes ``Tool`` capability nodes carrying the schema the
classification gate and the dispatcher's specialist routing both query, without
spawning any MCP servers (the catalog is injected).
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core.source_sync import (
    _reconcile_declared_fleet,
    _sync_fleet,
    _write_fleet_nodes,
    derive_capability_synonyms,
    sync_source,
)


@pytest.fixture(autouse=True)
def _capture_native_graph_slice(monkeypatch: pytest.MonkeyPatch) -> None:
    def capture(engine, connector, entities, relationships=None, **_kwargs):
        engine.ingest_external_batch(connector, entities, relationships)
        return {
            "status": "success",
            "write_result": {"nodes": len(entities), "edges": len(relationships or [])},
        }

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.envelope_ingest.ingest_graph_slice",
        capture,
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ontology.connector_manifest_gate.precheck_source",
        lambda _source: {"checked": True, "ok": True},
    )


class FakeEngine:
    """Records add_node / link_nodes so we can assert what was written."""

    def __init__(self) -> None:
        self.nodes: dict[str, tuple[str, dict]] = {}
        self.edges: list[tuple[str, str, str]] = []

    def add_node(self, node_id, node_type, properties=None):
        self.nodes[node_id] = (node_type, dict(properties or {}))

    def link_nodes(self, source_id, target_id, rel_type, properties=None):
        self.edges.append((source_id, target_id, rel_type))

    def query_cypher(self, query, params=None):
        return []

    def ingest_external_batch(self, domain, entities, relationships=None):
        for entity in entities:
            row = dict(entity)
            node_id = row.pop("id")
            node_type = row.pop("node_type")
            self.add_node(node_id, node_type, properties=row)
        for edge in relationships or []:
            self.link_nodes(edge["source"], edge["target"], edge["relationship"])
        return {"status": "success"}


CATALOG = {
    "portainer-agent": {
        "tools": [
            {"name": "list_stacks", "description": "List Portainer stacks"},
            {"name": "deploy_stack", "description": "Deploy a stack"},
        ],
        "error": None,
    },
    "github-mcp": {
        "tools": [{"name": "list_issues", "description": "List open issues"}],
        "error": None,
    },
    "broken-mcp": {"tools": [], "error": "timeout after 10s"},
}


# ── synonym derivation ───────────────────────────────────────────────────────


def test_synonyms_recover_product_from_server_name():
    # The validation cases: a turn says "portainer"/"github", servers are *-agent/*-mcp.
    assert "portainer" in derive_capability_synonyms("portainer-agent")
    assert "github" in derive_capability_synonyms("github-mcp")
    assert "servicenow" in derive_capability_synonyms("servicenow-api")


def test_synonyms_keep_multitoken_products():
    syns = derive_capability_synonyms("data-science-mcp")
    assert "data-science" in syns  # de-suffixed product
    assert "data" in syns and "science" in syns  # individual tokens
    assert "mcp" not in syns  # generic suffix dropped


def test_synonyms_empty_for_blank():
    assert derive_capability_synonyms("") == []


# ── node writing ─────────────────────────────────────────────────────────────


def test_write_fleet_nodes_creates_tool_nodes_with_dispatcher_schema():
    engine = FakeEngine()
    counts = _write_fleet_nodes(engine, CATALOG)

    assert counts["tools_written"] == 3
    assert counts["servers_written"] == 2  # broken server is skipped
    assert "broken-mcp" in counts["unreachable"]

    node_type, props = engine.nodes["tool_portainer-agent_list_stacks"]
    assert node_type == "Tool"
    # The exact fields config._fetch_tools reads back: MATCH (t:Tool) RETURN
    # t.name, t.description, t.mcp_server, t.relevance_score, t.tags, t.requires_approval
    assert props["name"] == "list_stacks"
    assert props["mcp_server"] == "portainer-agent"
    assert props["tags"] == ["portainer"]  # == dispatcher's derived server_tag
    assert props["requires_approval"] is False
    assert isinstance(props["relevance_score"], (int, float))
    # gate vocabulary
    assert "portainer" in props["synonyms"]


def test_write_fleet_nodes_links_tool_to_server():
    engine = FakeEngine()
    _write_fleet_nodes(engine, CATALOG)
    assert (
        "mcp_server_portainer-agent",
        "tool_portainer-agent_list_stacks",
        "SERVES",
    ) in engine.edges
    # server node defensively upserted so the edge always resolves
    assert engine.nodes["mcp_server_portainer-agent"][0] == "MCPServer"


def test_unreachable_server_writes_no_tools_but_is_recorded():
    engine = FakeEngine()
    _write_fleet_nodes(engine, CATALOG)
    assert not any(nid.startswith("tool_broken-mcp_") for nid in engine.nodes)


# ── handler + routing surface ────────────────────────────────────────────────


def test_sync_fleet_accepts_injected_catalog():
    engine = FakeEngine()
    res = _sync_fleet(engine, mode="full", client=CATALOG)
    assert res["status"] == "ok"
    assert res["source"] == "fleet"
    assert res["tools_written"] == 3
    assert res["servers_seen"] == 3


def test_sync_source_routes_fleet_to_handler():
    # The two-surface contract: `source_sync source=fleet` (and the REST twin)
    # dispatch into _sync_fleet through the one entrypoint.
    engine = FakeEngine()
    res = sync_source(engine, "fleet", mode="full", client=CATALOG)
    assert res["status"] == "ok"
    # Non-canonical connector diagnostics are namespaced under `details` by the
    # EtlResult wire contract (CONCEPT:AU-KG.etl.result-contract).
    assert res["details"]["tools_written"] == 3


# ── Skills-over-MCP ingestion (CONCEPT:AU-ECO.mcp.skills-over-mcp-provider) ──


CATALOG_WITH_SKILLS = {
    "docs-mcp": {
        "tools": [{"name": "search_docs", "description": "Search the docs"}],
        "skills": [
            {
                "name": "release-notes-writer",
                "uri": "skill://release-notes-writer/SKILL.md",
                "description": "Draft release notes from a changelog",
            }
        ],
        "error": None,
    },
    "skills-only-mcp": {
        "tools": [],
        "skills": [
            {
                "name": "onboarding-guide",
                "uri": "skill://onboarding-guide/SKILL.md",
                "description": "Walk a new hire through setup",
            }
        ],
        "error": None,
    },
}


def test_write_fleet_nodes_creates_skill_nodes_with_ranker_schema():
    engine = FakeEngine()
    counts = _write_fleet_nodes(engine, CATALOG_WITH_SKILLS)

    assert counts["skills_written"] == 2
    assert counts["tools_written"] == 1
    assert counts["servers_written"] == 2

    node_type, props = engine.nodes["skill_docs-mcp_release-notes-writer"]
    assert node_type == "Skill"
    assert props["name"] == "release-notes-writer"
    assert props["mcp_server"] == "docs-mcp"
    assert props["kind"] == "mcp_skill"
    assert props["source_ref"] == "skill://release-notes-writer"
    assert props["requires_approval"] is False
    assert isinstance(props["relevance_score"], (int, float))


def test_write_fleet_nodes_links_skill_to_server():
    engine = FakeEngine()
    _write_fleet_nodes(engine, CATALOG_WITH_SKILLS)
    assert (
        "mcp_server_docs-mcp",
        "skill_docs-mcp_release-notes-writer",
        "SERVES",
    ) in engine.edges


def test_write_fleet_nodes_writes_server_with_only_skills_and_no_tools():
    """A server that serves ZERO tools but at least one skill:// resource must
    still be written (previously any server with an empty tool list was
    dropped outright)."""
    engine = FakeEngine()
    _write_fleet_nodes(engine, CATALOG_WITH_SKILLS)
    assert engine.nodes["mcp_server_skills-only-mcp"][0] == "MCPServer"
    assert "skill_skills-only-mcp_onboarding-guide" in engine.nodes


def test_fleet_skill_id_is_namespaced_away_from_in_loop_skill_identity():
    """A fleet-probed skill must NEVER collide with the canonical
    ``skill:<slug>`` identity a richer in-loop ``ingest_runnable_skill`` writes
    (body/instruction) — a per-server-namespaced id keeps a thin fleet re-probe
    from ever overwriting those fields."""
    from agent_utilities.knowledge_graph.ingestion.skill_workflow_ingest import (
        skill_reference,
    )

    engine = FakeEngine()
    _write_fleet_nodes(engine, CATALOG_WITH_SKILLS)
    assert "skill:release-notes-writer" not in engine.nodes
    assert "skill_docs-mcp_release-notes-writer" in engine.nodes
    # The canonical reference is still recorded so ranking/binding can relate
    # the two identities without merging them.
    props = engine.nodes["skill_docs-mcp_release-notes-writer"][1]
    assert props["source_ref"] == skill_reference("release-notes-writer")


def test_sync_fleet_reports_skills_written(monkeypatch):
    import agent_utilities.orchestration.fleet_reconciler as fleet_reconciler

    monkeypatch.setattr(fleet_reconciler, "resolve_registry_path", lambda: None)
    engine = FakeEngine()
    res = _sync_fleet(engine, mode="full", client=CATALOG_WITH_SKILLS)
    assert res["status"] == "ok"
    assert res["skills_written"] == 2


def test_derive_tool_mode_classifies_variant():
    """CONCEPT:AU-KG.ontology.capability-node-aliases-lexical — condensed = action+params_json schema; verbose = typed params."""
    from agent_utilities.knowledge_graph.core.source_sync import _derive_tool_mode

    assert (
        _derive_tool_mode({"properties": {"action": {}, "params_json": {}}})
        == "condensed"
    )
    assert _derive_tool_mode({"properties": {"owner": {}, "repo": {}}}) == "verbose"
    assert _derive_tool_mode({}) == "verbose"
    assert _derive_tool_mode(None) == "verbose"


def test_both_tool_variants_ingested_with_mode():
    """A server serving BOTH a condensed (action-routed) and a verbose (1:1 typed) tool
    ingests BOTH as distinct Tool nodes, each tagged with its variant."""
    catalog = {
        "github-mcp": {
            "tools": [
                {
                    "name": "github_issues",
                    "description": "Manage GitHub issues",
                    "inputSchema": {"properties": {"action": {}, "params_json": {}}},
                },
                {
                    "name": "github_search_issues",
                    "description": "Search issues",
                    "inputSchema": {"properties": {"q": {}, "sort": {}}},
                },
            ],
            "error": None,
        }
    }
    eng = FakeEngine()
    _write_fleet_nodes(eng, catalog)
    assert eng.nodes["tool_github-mcp_github_issues"][1]["tool_mode"] == "condensed"
    assert (
        eng.nodes["tool_github-mcp_github_search_issues"][1]["tool_mode"] == "verbose"
    )


# ── declared-vs-probed reconcile against deploy/mcp-fleet.registry.yml ───────
# The registry's ``name`` (always ``<pkg>-mcp``-shaped) and a probed
# ``mcp_config.json`` server key frequently diverge (e.g. registry
# ``github-mcp``/package ``github-agent`` vs. a config key of either) — the
# reconcile checks a registry entry's ``name`` OR ``package`` against the
# probed catalog, mirroring ``_sync_fleet_connectors``'s own membership check
# for this same naming mismatch.


def _write_registry(tmp_path, entries: list[tuple[str, str]]):
    """entries: list of (name, package) -> a minimal mcp-fleet.registry.yml."""
    lines = ["services:"]
    for name, package in entries:
        lines.append(f"  - name: {name}")
        lines.append(f"    package: {package}")
    path = tmp_path / "mcp-fleet.registry.yml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_reconcile_declared_fleet_covers_by_name_or_package(tmp_path, monkeypatch):
    import agent_utilities.orchestration.fleet_reconciler as fleet_reconciler

    registry = _write_registry(
        tmp_path,
        [
            ("portainer-agent", "portainer-agent"),  # covered: name is a catalog key
            ("gh-declared-mcp", "github-mcp"),  # covered: package is a catalog key
            ("never-probed-mcp", "never-probed"),  # covered by neither
        ],
    )
    monkeypatch.setattr(fleet_reconciler, "resolve_registry_path", lambda: registry)

    out = _reconcile_declared_fleet(
        CATALOG
    )  # CATALOG keys: portainer-agent/github-mcp/broken-mcp

    assert out == {"declared_total": 3, "declared_uncovered": ["never-probed-mcp"]}


def test_reconcile_declared_fleet_none_when_registry_missing(monkeypatch):
    import agent_utilities.orchestration.fleet_reconciler as fleet_reconciler

    monkeypatch.setattr(fleet_reconciler, "resolve_registry_path", lambda: None)
    assert _reconcile_declared_fleet(CATALOG) is None


def test_reconcile_declared_fleet_none_on_unparsable_registry(tmp_path, monkeypatch):
    import agent_utilities.orchestration.fleet_reconciler as fleet_reconciler

    broken = tmp_path / "mcp-fleet.registry.yml"
    broken.write_text("not: [valid, yaml, :::", encoding="utf-8")
    monkeypatch.setattr(fleet_reconciler, "resolve_registry_path", lambda: broken)
    assert _reconcile_declared_fleet(CATALOG) is None


def test_sync_fleet_merges_declared_reconcile_info(tmp_path, monkeypatch):
    """``_sync_fleet``'s result carries the reconcile info additively — the
    existing ``tools_written``/``servers_seen`` contract is unchanged."""
    import agent_utilities.orchestration.fleet_reconciler as fleet_reconciler

    registry = _write_registry(
        tmp_path,
        [("portainer-agent", "portainer-agent"), ("never-probed-mcp", "never-probed")],
    )
    monkeypatch.setattr(fleet_reconciler, "resolve_registry_path", lambda: registry)

    engine = FakeEngine()
    res = _sync_fleet(engine, mode="full", client=CATALOG)

    assert res["status"] == "ok"
    assert res["tools_written"] == 3
    assert res["servers_seen"] == 3
    assert res["declared_total"] == 2
    assert res["declared_uncovered"] == ["never-probed-mcp"]
