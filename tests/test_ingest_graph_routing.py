"""Ingestion graph routing + unified-query preservation (CONCEPT:AU-KG.ingest.unified-query-routing).

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

import json
import uuid
from typing import Any

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
    # Tenant routing (CONCEPT:AU-KG.sharding.tenant-partitioned-sharding-hrw) is independent of the new flag.
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


def _cfg_fanout(routing: bool, fanout: bool) -> AgentConfig:
    return AgentConfig(
        KG_INGEST_GRAPH_ROUTING=routing, KG_INGEST_SHARD_FANOUT=fanout
    )


# ── Per-shard content-keyed fanout (CONCEPT:AU-KG.ingest.batched-cross-graph-writer) ──
def test_shard_fanout_off_is_one_graph_per_source() -> None:
    """Fanout OFF (default): a source stays on ONE graph regardless of content_key."""
    c = _cfg_fanout(routing=True, fanout=False)
    assert not ingest_routing.shard_fanout_enabled(c)
    g1 = ingest_routing.route_graph(source_type="freshrss", content_key="a", config=c)
    g2 = ingest_routing.route_graph(source_type="freshrss", content_key="b", config=c)
    assert g1 == g2 == "src:freshrss"


def test_shard_fanout_requires_routing() -> None:
    # Fanout without routing is inert (routing is the prerequisite).
    c = _cfg_fanout(routing=False, fanout=True)
    assert not ingest_routing.shard_fanout_enabled(c)


def test_shard_fanout_spreads_a_single_source_across_k() -> None:
    """Fanout ON: many content keys for ONE source fan across ``#0..#K-1`` sub-graphs,
    so a high-volume source's writes spread over the K redb shard writers instead of
    pinning one. All sub-graphs keep the ``src:`` prefix (still content graphs)."""
    from unittest import mock

    c = _cfg_fanout(routing=True, fanout=True)
    assert ingest_routing.shard_fanout_enabled(c)
    with mock.patch(
        "agent_utilities.knowledge_graph.core.worker_scheduler.durable_shard_writers",
        return_value=4,
    ):
        graphs = {
            ingest_routing.route_graph(
                source_type="freshrss", content_key=f"item-{i}", config=c
            )
            for i in range(200)
        }
    # More than one distinct sub-graph in flight for the SAME source.
    assert len(graphs) > 1
    # Every sub-graph is a recognised content graph under the source prefix.
    for g in graphs:
        assert g.startswith("src:freshrss#")
        assert ingest_routing.is_content_graph(g)


def test_shard_fanout_is_deterministic_per_content_key() -> None:
    from unittest import mock

    c = _cfg_fanout(routing=True, fanout=True)
    with mock.patch(
        "agent_utilities.knowledge_graph.core.worker_scheduler.durable_shard_writers",
        return_value=4,
    ):
        a = ingest_routing.route_graph(
            source_type="freshrss", content_key="stable", config=c
        )
        b = ingest_routing.route_graph(
            source_type="freshrss", content_key="stable", config=c
        )
    assert a == b
    assert 0 <= ingest_routing.shard_bucket_for("stable", 4) < 4


def test_shard_fanout_leaves_codebase_and_tenant_whole() -> None:
    """Codebase is already per-repo sharded and a tenant must stay whole — neither
    is fanned out even with a content_key."""
    from unittest import mock

    c = _cfg_fanout(routing=True, fanout=True)
    with mock.patch(
        "agent_utilities.knowledge_graph.core.worker_scheduler.durable_shard_writers",
        return_value=4,
    ):
        assert (
            ingest_routing.route_graph(repo="agent-utilities", content_key="x", config=c)
            == "code:agent-utilities"
        )
        assert (
            ingest_routing.route_graph(tenant="acme", content_key="x", config=c)
            == "tenant__acme____commons__"
        )


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

    The whole point of CONCEPT:AU-KG.ingest.unified-query-routing: distinct graph names hash to distinct
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

    The key correctness point of CONCEPT:AU-KG.ingest.unified-query-routing: spreading writes across graphs
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


# ── Fan-out merge correctness (CONCEPT:AU-KG.backend.fanout-dedup) ──────────────────────────────
# The unified read fans an implicit-default query across the active content graphs
# and merges. Two bugs the merge must not have: (1) an AGGREGATION row (count/sum)
# has no node id, so the legacy id-dedup leaves one copy of every group row PER
# graph — the live evidence was a Task count repeated ~24×; (2) the fan-out must
# never query the SAME backend twice (the default ``__commons__`` showing up once
# per graph). Driven through the real ``graph_query`` tool + ``_resolve_read_engines``.


class _RowsBackend:
    cypher_support = "full"
    supports_sparql = False

    def __init__(self, graph_name: str | None = None) -> None:
        self.graph_name = graph_name

    def close(self) -> None:  # pragma: no cover - parity with real backend
        pass


class _RowsEngine:
    """A fake engine whose ``query_cypher`` returns a fixed row set."""

    def __init__(
        self, label: str, graph_name: str | None = None, rows: Any = None
    ) -> None:
        self.label = label
        self.backend = _RowsBackend(graph_name)
        self._rows = rows if rows is not None else [{"engine": label}]

    def query_cypher(self, cypher, params=None, as_of=None):
        return list(self._rows)


def test_is_aggregation_cypher_detection() -> None:
    from agent_utilities.mcp.tools.query_tools import is_aggregation_cypher

    # Aggregates collapse rows → must be detected.
    assert is_aggregation_cypher("MATCH (t:Task) RETURN t.lane AS lane, count(*) AS n")
    assert is_aggregation_cypher("MATCH (n) RETURN sum(n.cost) AS total")
    assert is_aggregation_cypher("MATCH (n) RETURN avg(n.score), max(n.score)")
    assert is_aggregation_cypher("MATCH (n) RETURN collect(n.id)")
    # Plain row queries are NOT aggregations.
    assert not is_aggregation_cypher("MATCH (n:Task) RETURN n.id AS id")
    assert not is_aggregation_cypher("MATCH (f:Function {name:'probe'}) RETURN f")
    # False-positive guards: a property whose name contains an agg word, and an
    # aggregate word inside a string literal, must NOT trip detection.
    assert not is_aggregation_cypher("MATCH (n) RETURN n.max_depth AS d")
    assert not is_aggregation_cypher("MATCH (n) WHERE n.note = 'count(*)' RETURN n")


def test_resolve_read_engines_dedups_duplicate_backends(monkeypatch) -> None:
    """The fan-out target set must never include the SAME backend twice.

    A content graph that resolves onto the default ``__commons__`` store collapses
    to one entry, so a node living only in ``__commons__`` is queried exactly once.
    """
    import agent_utilities.mcp.kg_server as kg_server
    from agent_utilities.knowledge_graph.core import ingest_routing as ir
    from agent_utilities.knowledge_graph.core.shard_topology import default_graph_name

    default = default_graph_name()
    default_engine = _RowsEngine("default", graph_name=default)
    # ``code:dupcommons`` mis-resolves to the SAME __commons__ store (duplicate);
    # ``code:real`` is a genuinely distinct backend.
    dup = _RowsEngine("dup", graph_name=default)
    real = _RowsEngine("real", graph_name="code:real")
    by_name = {"code:dupcommons": dup, "code:real": real}

    monkeypatch.setattr(kg_server, "_get_engine", lambda: default_engine)
    monkeypatch.setattr(ir, "routing_enabled", lambda *a, **k: True)
    monkeypatch.setattr(
        ir,
        "read_graph_targets",
        lambda *a, **k: [default, "code:dupcommons", "code:real"],
    )
    monkeypatch.setattr(ir, "safe_engine_for_graph", lambda name: (by_name[name], None))

    entries, errors, fanout = kg_server._resolve_read_engines("")
    assert fanout
    keys = [getattr(e.backend, "graph_name", None) for _, e in entries]
    assert keys.count(default) == 1, f"__commons__ queried >1×: {keys}"
    assert "code:real" in keys
    assert len(entries) == 2  # the duplicate __commons__ target was dropped


async def test_aggregation_query_merges_not_duplicates(monkeypatch) -> None:
    """The live-evidence bug: a Task count fanned across ~24 graphs returned the
    same aggregate row 24×. It must now return ONE row per (lane,status)."""
    import agent_utilities.mcp.kg_server as kg_server

    kg_server.ensure_tools_registered()
    canonical_rows = [
        {"lane": "connectors", "status": "completed", "n": 7},
        {"lane": "ingestion", "status": "completed", "n": 13},
    ]
    canonical = _RowsEngine("default", graph_name="__commons__", rows=canonical_rows)
    # The substrate of the bug: the resolver fans across many graphs that all see
    # the same __commons__ aggregate row (no node id to dedup on).
    fan = [(f"code:g{i}", _RowsEngine(f"g{i}", rows=canonical_rows)) for i in range(24)]
    monkeypatch.setattr(
        kg_server, "_resolve_read_engines", lambda target: (fan, {}, True)
    )
    monkeypatch.setattr(kg_server, "_get_engine", lambda: canonical)

    out = await kg_server._execute_tool(
        "graph_query",
        cypher="MATCH (t:Task) RETURN t.lane AS lane, t.status AS status, count(*) AS n",
        target="",
    )
    rows = json.loads(out)
    assert rows == canonical_rows  # exactly one row per group, NOT 24 duplicates


async def test_routed_node_query_still_fans_and_dedups_once(monkeypatch) -> None:
    """Non-aggregation queries keep the fan + id-dedup: a node routed to ``code:*``
    is still found via the unified path, returned exactly once even when overlapping
    backends echo it, and other routed content is still gathered (CONCEPT:AU-KG.ingest.unified-query-routing)."""
    import agent_utilities.mcp.kg_server as kg_server

    kg_server.ensure_tools_registered()
    probe = {"id": "Func::probe", "name": "probe"}
    other = {"id": "Func::other", "name": "other"}
    entries = [
        ("default", _RowsEngine("default", rows=[])),
        ("code:a", _RowsEngine("a", rows=[probe])),
        ("code:b", _RowsEngine("b", rows=[probe])),  # overlap → must dedup by id
        ("code:c", _RowsEngine("c", rows=[other])),
    ]
    monkeypatch.setattr(
        kg_server, "_resolve_read_engines", lambda target: (entries, {}, True)
    )

    out = await kg_server._execute_tool(
        "graph_query",
        cypher="MATCH (f:Function {name:'probe'}) RETURN f.id AS id, f.name AS name",
        target="",
    )
    rows = json.loads(out)
    ids = [r["id"] for r in rows]
    assert ids.count("Func::probe") == 1  # routed node returned exactly once
    assert "Func::other" in ids  # fan still gathers other routed content
