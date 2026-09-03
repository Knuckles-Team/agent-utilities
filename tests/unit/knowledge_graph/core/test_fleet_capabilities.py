"""Fleet capability elevation: tools → Tool capability nodes (CONCEPT:AU-KG.ontology.capability-node-aliases-lexical).

These cover the data half of ontology-native classification — that the served
multiplexer catalog becomes ``Tool`` capability nodes carrying the schema the
classification gate and the dispatcher's specialist routing both query, without
spawning any MCP servers (the catalog is injected).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_utilities.knowledge_graph.core.source_sync import (
    _sync_fleet,
    _write_fleet_nodes,
    _write_fleet_slice,
    derive_capability_synonyms,
    sync_source,
)


@pytest.fixture(autouse=True)
def _capture_native_graph_slice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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
    # D-SH-2's rejected-row cache (source_sync._write_fleet_slice) persists to
    # the XDG cache dir — isolate it to a tmp dir so these tests never read or
    # write the developer's real ~/.cache/agent-utilities/.
    monkeypatch.setenv("AGENT_UTILITIES_CACHE_DIR", str(tmp_path / "cache"))


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
    assert props["relevance_score"] == 50
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


def test_sync_fleet_reports_skills_written():
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


# ---------------------------------------------------------------------------
# D-SH-2 (reports/deferred/lane-skill-harvest.md): the fleet catalog's ONE
# atomic ChangeEnvelope commit rejects the whole slice when a single row trips
# the engine's persistence-privacy policy; _write_fleet_slice bisects to
# isolate and drop only the offender. Without a cache, that bisection cost is
# paid again on EVERY sync for the SAME already-known offender. These tests
# exercise the cache directly (bypassing the autouse fixture's always-succeeds
# ingest_graph_slice patch) so a specific row can be made to fail.
# ---------------------------------------------------------------------------


def _patch_ingest_graph_slice(monkeypatch: pytest.MonkeyPatch, bad_ids: set[str]):
    """Make ``ingest_graph_slice`` raise iff the slice contains a row in ``bad_ids``."""

    def maybe_fail(engine, connector, entities, relationships=None, **_kwargs):
        if any(e["id"] in bad_ids for e in entities):
            raise RuntimeError("persistence privacy policy rejected inline text")
        engine.ingest_external_batch(connector, entities, relationships)
        return {"status": "success"}

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.envelope_ingest.ingest_graph_slice",
        maybe_fail,
    )


def test_write_fleet_slice_caches_a_rejected_row_and_skips_bisection_next_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entities = [
        {"id": "row-good", "type": "Tool", "name": "good"},
        {"id": "row-bad", "type": "Tool", "name": "bad"},
    ]

    # --- First sync: row-bad is genuinely rejected, discovered by bisection.
    calls: list[int] = []

    def counting_maybe_fail(engine, connector, entities, relationships=None, **kw):
        calls.append(len(entities))
        if any(e["id"] == "row-bad" for e in entities):
            raise RuntimeError("persistence privacy policy rejected inline text")
        engine.ingest_external_batch(connector, entities, relationships)
        return {"status": "success"}

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.envelope_ingest.ingest_graph_slice",
        counting_maybe_fail,
    )
    engine = FakeEngine()
    rejected, pending = _write_fleet_slice(engine, entities, [])
    assert rejected == ["row-bad"]
    assert pending == []
    assert len(calls) >= 2  # the full slice failed, then bisection ran

    # --- Second sync, SAME content: row-bad is pre-excluded entirely -- the
    # remaining (clean) rows commit in ONE shot, no bisection re-discovery.
    calls.clear()
    engine2 = FakeEngine()
    rejected2, pending2 = _write_fleet_slice(engine2, entities, [])
    assert rejected2 == ["row-bad"]
    assert pending2 == []
    assert calls == [1]  # exactly one attempt: the single clean remaining row
    assert "row-good" in engine2.nodes  # the clean row still landed


def test_write_fleet_slice_re_attempts_a_known_bad_row_once_its_content_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entities = [{"id": "row-bad", "type": "Tool", "name": "bad", "v": 1}]
    _patch_ingest_graph_slice(monkeypatch, {"row-bad"})
    engine = FakeEngine()
    assert _write_fleet_slice(engine, entities, []) == (["row-bad"], [])

    # Content changed (e.g. the offending description was edited) -- and this
    # edit happens to have fixed it. Re-attempted rather than blindly reusing
    # the stale verdict, because the content hash no longer matches.
    fixed_entities = [{"id": "row-bad", "type": "Tool", "name": "bad", "v": 2}]

    def always_succeeds(engine, connector, entities, relationships=None, **kw):
        engine.ingest_external_batch(connector, entities, relationships)
        return {"status": "success"}

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.envelope_ingest.ingest_graph_slice",
        always_succeeds,
    )
    engine2 = FakeEngine()
    assert _write_fleet_slice(engine2, fixed_entities, []) == ([], [])
    assert "row-bad" in engine2.nodes


# ---------------------------------------------------------------------------
# Retryable PARTIAL_MATERIALIZATION at the fleet-catalog-slice level (the
# production defect this closes: the engine's own bounded resume in
# ``ingest_envelope`` can still exhaust its budget on a slow rebuild — when it
# does, ``_write_fleet_slice`` must NOT bisect further, since every half would
# just hit the same shared, transient, still-materializing engine state again;
# it must treat the whole still-pending batch as "gave up this sync only",
# never cache it as a permanent rejection, and report it separately from a
# genuine content rejection.
# ---------------------------------------------------------------------------


def _materialization_exhausted_error(row_ids: set[str]) -> RuntimeError:
    """The RuntimeError ``ingest_graph_slice`` raises once ``ingest_envelope``
    itself gave up on a retryable PARTIAL_MATERIALIZATION signal — carries the
    marker ``_write_fleet_slice`` greps for, same as production."""
    from agent_utilities.knowledge_graph.ingestion.envelope_ingest import (
        PARTIAL_MATERIALIZATION_RETRIES_EXHAUSTED_MARKER,
    )

    return RuntimeError(
        "native ChangeEnvelope graph slice failed: "
        "_PartialMaterializationRetriesExhausted "
        f"({PARTIAL_MATERIALIZATION_RETRIES_EXHAUSTED_MARKER}: rows "
        f"{sorted(row_ids)} did not finish materializing within budget)"
    )


def test_write_fleet_slice_does_not_bisect_on_materialization_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entities = [
        {"id": "tool_a", "type": "Tool", "name": "a"},
        {"id": "tool_b", "type": "Tool", "name": "b"},
        {"id": "tool_c", "type": "Tool", "name": "c"},
        {"id": "tool_d", "type": "Tool", "name": "d"},
    ]
    calls: list[int] = []

    def always_still_materializing(
        engine, connector, entities, relationships=None, **kw
    ):
        calls.append(len(entities))
        raise _materialization_exhausted_error({e["id"] for e in entities})

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.envelope_ingest.ingest_graph_slice",
        always_still_materializing,
    )
    engine = FakeEngine()

    rejected, pending = _write_fleet_slice(engine, entities, [])

    assert rejected == []  # never treated as a genuine content rejection
    assert sorted(pending) == ["tool_a", "tool_b", "tool_c", "tool_d"]
    # ONE attempt for the whole slice — bisection would have produced
    # len(entities)*2-1 = 7 calls; paying that here would turn one shared,
    # transient condition into an O(n log n) hammering of a still-recovering
    # engine.
    assert calls == [4]


def test_write_fleet_slice_materialization_pending_rows_are_not_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entities = [{"id": "tool_a", "type": "Tool", "name": "a"}]

    def still_materializing(engine, connector, entities, relationships=None, **kw):
        raise _materialization_exhausted_error({e["id"] for e in entities})

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.envelope_ingest.ingest_graph_slice",
        still_materializing,
    )
    engine = FakeEngine()
    rejected, pending = _write_fleet_slice(engine, entities, [])
    assert rejected == []
    assert pending == ["tool_a"]

    # Next sync: the engine finished materializing. Because a
    # materialization-pending row is never written to the known-bad cache,
    # it is re-attempted with a clean slate (unlike a genuinely rejected row,
    # which would be pre-excluded — see the sibling cache tests above).
    def now_succeeds(engine, connector, entities, relationships=None, **kw):
        engine.ingest_external_batch(connector, entities, relationships)
        return {"status": "success"}

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.envelope_ingest.ingest_graph_slice",
        now_succeeds,
    )
    engine2 = FakeEngine()
    rejected2, pending2 = _write_fleet_slice(engine2, entities, [])
    assert rejected2 == []
    assert pending2 == []
    assert "tool_a" in engine2.nodes


def test_write_fleet_nodes_reports_materialization_pending_separately_from_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def always_still_materializing(
        engine, connector, entities, relationships=None, **kw
    ):
        raise _materialization_exhausted_error({e["id"] for e in entities})

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.envelope_ingest.ingest_graph_slice",
        always_still_materializing,
    )
    engine = FakeEngine()

    res = _write_fleet_nodes(engine, CATALOG)

    assert res["catalog_rows_rejected"] == 0
    assert res["catalog_rejected_ids"] == []
    assert res["catalog_rows_materialization_pending"] == 5  # 2 servers + 3 tools
    assert sorted(res["catalog_materialization_pending_ids"]) == [
        "mcp_server_github-mcp",
        "mcp_server_portainer-agent",
        "tool_github-mcp_list_issues",
        "tool_portainer-agent_deploy_stack",
        "tool_portainer-agent_list_stacks",
    ]
