"""Ingestion graph routing + unified-query preservation (CONCEPT:AU-KG.ingest.unified-query-routing).

Two layers:

* **Unit** — the routing policy (:func:`route_graph`) is deterministic and spreads
  a realistic source set across many
  graph names (so the engine's ``FNV-1a(name) % K`` redb shard writers, EG-026,
  parallelise instead of all funnelling into ``__commons__``).

* **Engine** (``@pytest.mark.engine``) — a node written to a routed content graph
  (``code:<x>``) is still returned by the normal unified query path. This is the
  correctness guard: routing must NOT make content invisible to existing reads.
"""

from __future__ import annotations

from typing import Any

from agent_utilities.core.config import AgentConfig
from agent_utilities.knowledge_graph.core import ingest_routing


def _cfg() -> AgentConfig:
    return AgentConfig()


# ── FNV-1a (the engine's EG-026 shard key) — replicated to assert spread ──────
def _fnv1a(s: str) -> int:
    h = 0xCBF29CE484222325
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def _shard(name: str, k: int) -> int:
    return _fnv1a(name) % k


# ── Routing policy: deterministic ──────────────────────────────────────────────
def test_routing_honours_tenant() -> None:
    c = _cfg()
    assert (
        ingest_routing.route_graph(tenant="acme", config=c)
        == "tenant__acme____commons__"
    )


def test_route_graph_per_source_names() -> None:
    c = _cfg()
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


def _cfg_fanout(fanout: bool) -> AgentConfig:
    return AgentConfig(KG_INGEST_SHARD_FANOUT=fanout)


# ── Per-shard content-keyed fanout (CONCEPT:AU-KG.ingest.batched-cross-graph-writer) ──
def test_shard_fanout_off_is_one_graph_per_source() -> None:
    """Fanout OFF (default): a source stays on ONE graph regardless of content_key."""
    c = _cfg_fanout(fanout=False)
    assert not ingest_routing.shard_fanout_enabled(c)
    g1 = ingest_routing.route_graph(source_type="freshrss", content_key="a", config=c)
    g2 = ingest_routing.route_graph(source_type="freshrss", content_key="b", config=c)
    assert g1 == g2 == "src:freshrss"


def test_shard_fanout_spreads_a_single_source_across_k() -> None:
    """Fanout ON: many content keys for ONE source fan across ``#0..#K-1`` sub-graphs,
    so a high-volume source's writes spread over the K redb shard writers instead of
    pinning one. All sub-graphs keep the ``src:`` prefix (still content graphs)."""
    from unittest import mock

    c = _cfg_fanout(fanout=True)
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

    c = _cfg_fanout(fanout=True)
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

    c = _cfg_fanout(fanout=True)
    with mock.patch(
        "agent_utilities.knowledge_graph.core.worker_scheduler.durable_shard_writers",
        return_value=4,
    ):
        assert (
            ingest_routing.route_graph(
                repo="agent-utilities", content_key="x", config=c
            )
            == "code:agent-utilities"
        )
        assert (
            ingest_routing.route_graph(tenant="acme", content_key="x", config=c)
            == "tenant__acme____commons__"
        )


def test_route_graph_deterministic() -> None:
    c = _cfg()
    a = ingest_routing.route_graph(kind="connector", source_type="gitlab", config=c)
    b = ingest_routing.route_graph(kind="connector", source_type="gitlab", config=c)
    assert a == b == "src:gitlab"


def test_empty_slug_falls_back_to_default() -> None:
    c = _cfg()
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
    c = _cfg()
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


def test_read_graph_targets() -> None:
    ingest_routing._reset_for_tests()
    # Nothing routed yet → single default (no needless fan-out).
    assert ingest_routing.read_graph_targets(_cfg()) == ["__commons__"]
    # After content is registered → default first, then the content graphs.
    ingest_routing.register_content_graph("code:foo")
    ingest_routing.register_content_graph("src:bar")
    ingest_routing.register_content_graph(
        "__commons__"
    )  # ignored (not a content graph)
    targets = ingest_routing.read_graph_targets(_cfg())
    assert targets[0] == "__commons__"
    assert set(targets) == {"__commons__", "code:foo", "src:bar"}
    ingest_routing._reset_for_tests()


# ── Unified query still sees routed content (REAL engine) ─────────────────────


# ── Fan-out merge correctness (CONCEPT:AU-KG.backend.fanout-dedup) ──────────────────────────────
# The unified read fans an implicit-default query across the active content graphs
# and merges. Two bugs the merge must not have: (1) an AGGREGATION row (count/sum)
# has no node id, so the legacy id-dedup leaves one copy of every group row PER
# graph — the live evidence was an aggregate row repeated ~24×; (2) the fan-out must
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
