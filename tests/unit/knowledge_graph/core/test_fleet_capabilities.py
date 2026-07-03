"""Fleet capability elevation: tools → Tool capability nodes (CONCEPT:KG-2.133).

These cover the data half of ontology-native classification — that the served
multiplexer catalog becomes ``Tool`` capability nodes carrying the schema the
classification gate and the dispatcher's specialist routing both query, without
spawning any MCP servers (the catalog is injected).
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.core.source_sync import (
    _sync_fleet,
    _write_fleet_nodes,
    derive_capability_synonyms,
    sync_source,
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
    assert res["tools_written"] == 3


def test_derive_tool_mode_classifies_variant():
    """CONCEPT:KG-2.133 — condensed = action+params_json schema; verbose = typed params."""
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
