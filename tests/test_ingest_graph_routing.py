"""Ingestion graph routing + unified-query preservation (CONCEPT:KG-2.269).

Two layers:

* **Unit** — the routing policy (:func:`route_graph`) is deterministic, honours the
  ``KG_INGEST_GRAPH_ROUTING`` gate, and spreads a realistic source set across many
  graph names (so the engine's ``FNV-1a(name) % K`` redb shard writers, EG-026,
  parallelise instead of all funnelling into ``__commons__``).

* **Engine** (``@pytest.mark.engine``) — a node written to a routed content graph
  (``code:<x>``) is still returned by the normal unified query path. This is the
  correctness guard: routing must NOT make content invisible to existing reads.
"""

from __future__ import annotations

import uuid

import pytest

from agent_utilities.core.config import AgentConfig
from agent_utilities.knowledge_graph.core import ingest_routing


def _cfg(routing: bool) -> AgentConfig:
    # Set via the alias — AgentConfig fields are alias-only (populate_by_name off).
    return AgentConfig(KG_INGEST_GRAPH_ROUTING=routing)


# ── FNV-1a (the engine's EG-026 shard key) — replicated to assert spread ──────
def _fnv1a(s: str) -> int:
    h = 0xCBF29CE484222325
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def _shard(name: str, k: int) -> int:
    return _fnv1a(name) % k


# ── Routing policy: deterministic + gated ─────────────────────────────────────
def test_routing_disabled_is_legacy_commons() -> None:
    c = _cfg(False)
    assert ingest_routing.route_graph(repo="agent-utilities", config=c) == "__commons__"
    assert (
        ingest_routing.route_graph(kind="connector", source_type="servicenow", config=c)
        == "__commons__"
    )


def test_routing_disabled_still_honours_tenant() -> None:
    c = _cfg(False)
    # Tenant routing (CONCEPT:KG-2.58) is independent of the new flag.
    assert (
        ingest_routing.route_graph(tenant="acme", config=c)
        == "tenant__acme____commons__"
    )


def test_route_graph_per_source_names() -> None:
    c = _cfg(True)
    assert (
        ingest_routing.route_graph(repo="agent-utilities", config=c)
        == "code:agent-utilities"
    )
    assert (
        ingest_routing.route_graph(kind="code", repo="Epistemic Graph", config=c)
        == "code:epistemic-graph"
    )
    assert (
        ingest_routing.route_graph(kind="connector", source_type="servicenow", config=c)
        == "src:servicenow"
    )
    assert (
        ingest_routing.route_graph(kind="chat", agent="planner", config=c)
        == "chat:planner"
    )
    assert (
        ingest_routing.route_graph(kind="research", source_type="arXiv", config=c)
        == "research:arxiv"
    )
    # Tenant wins over source kind.
    assert (
        ingest_routing.route_graph(tenant="acme", repo="x", config=c)
        == "tenant__acme____commons__"
    )
    # No natural owner → default graph.
    assert ingest_routing.route_graph(config=c) == "__commons__"


def test_route_graph_deterministic() -> None:
    c = _cfg(True)
    a = ingest_routing.route_graph(kind="connector", source_type="gitlab", config=c)
    b = ingest_routing.route_graph(kind="connector", source_type="gitlab", config=c)
    assert a == b == "src:gitlab"


def test_empty_slug_falls_back_to_default() -> None:
    c = _cfg(True)
    # A slug that sanitises to empty must not emit a degenerate ``code:`` name.
    assert ingest_routing.route_graph(repo="///", config=c) == "__commons__"


def test_is_content_graph() -> None:
    assert ingest_routing.is_content_graph("code:foo")
    assert ingest_routing.is_content_graph("src:servicenow")
    assert not ingest_routing.is_content_graph("__commons__")
    assert not ingest_routing.is_content_graph(None)


def test_routing_spreads_across_shards() -> None:
    """A realistic source set must occupy >1 of the K redb shard writers.

    The whole point of CONCEPT:KG-2.269: distinct graph names hash to distinct
    shards, so K cores commit in parallel rather than 1.
    """
    c = _cfg(True)
    repos = [
        "agent-utilities",
        "epistemic-graph",
        "gitlab-api",
        "servicenow-api",
        "onetrust-api",
    ]
    connectors = ["servicenow", "gitlab", "freshrss", "rss", "web"]
    names = [ingest_routing.route_graph(repo=r, config=c) for r in repos]
    names += [
        ingest_routing.route_graph(kind="connector", source_type=s, config=c)
        for s in connectors
    ]

    assert len(set(names)) == len(names)  # every source → its own graph
    for k in (2, 4, 8):
        buckets = {_shard(n, k) for n in names}
        # All-into-one (the __commons__ bottleneck) would be a single bucket.
        assert len(buckets) > 1, f"K={k}: names collapsed onto one shard {buckets}"


def test_read_graph_targets_gating() -> None:
    ingest_routing._reset_for_tests()
    # Disabled → single default graph (legacy fast path).
    assert ingest_routing.read_graph_targets(_cfg(False)) == ["__commons__"]
    # Enabled but nothing routed yet → still single default (no needless fan-out).
    assert ingest_routing.read_graph_targets(_cfg(True)) == ["__commons__"]
    # After content is registered → default first, then the content graphs.
    ingest_routing.register_content_graph("code:foo")
    ingest_routing.register_content_graph("src:bar")
    ingest_routing.register_content_graph(
        "__commons__"
    )  # ignored (not a content graph)
    targets = ingest_routing.read_graph_targets(_cfg(True))
    assert targets[0] == "__commons__"
    assert set(targets) == {"__commons__", "code:foo", "src:bar"}
    ingest_routing._reset_for_tests()


# ── Unified query still sees routed content (REAL engine) ─────────────────────
@pytest.mark.engine
def test_node_in_routed_graph_found_by_unified_query(engine_graph, monkeypatch) -> None:
    """A node written to ``code:<x>`` is returned by the normal graph_query path.

    The key correctness point of CONCEPT:KG-2.269: spreading writes across graphs
    must not make content invisible to existing reads.
    """
    from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
        EpistemicGraphBackend,
    )

    monkeypatch.setenv("KG_INGEST_GRAPH_ROUTING", "true")
    ingest_routing._reset_for_tests()

    # Write a uniquely-identifiable node into a routed CONTENT graph.
    routed = f"code:routetest-{uuid.uuid4().hex[:10]}"
    node_id = f"RouteProbe::{uuid.uuid4().hex[:12]}"
    backend = EpistemicGraphBackend(graph_name=routed)
    try:
        backend.add_node(node_id, label="RouteProbe", name="route-probe")
        ingest_routing.register_content_graph(routed)

        # The unified read resolver must fan across the content-graph set.
        import agent_utilities.mcp.kg_server as kg_server

        entries, _errors, fanout = kg_server._resolve_read_engines("")
        assert fanout, (
            "implicit-default read should fan across content graphs when routing is on"
        )
        resolved = {name for name, _ in entries}
        assert routed in resolved, (
            f"routed graph {routed} missing from read set {resolved}"
        )

        # Run the actual query across the union and confirm the node surfaces.
        found = False
        for _name, engine in entries:
            try:
                rows = engine.query_cypher("MATCH (n:RouteProbe) RETURN n.id AS id", {})
            except Exception:
                continue
            if any((r or {}).get("id") == node_id for r in rows or []):
                found = True
                break
        assert found, (
            f"node {node_id} in {routed} not visible via the unified query path"
        )
    finally:
        try:
            client = getattr(backend._graph, "_client", None)
            if client is not None:
                client.tenants.delete(routed)
        except Exception:
            pass
        ingest_routing._reset_for_tests()
