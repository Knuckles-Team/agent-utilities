"""Tests for the unbounded global-analytics OOM guard on ``engine_analytics``
(``mcp/tools/engine_tools.py``).

Context (exhaustive kg-* validation): ``engine_analytics(action="pagerank",
params_json="{}")`` — an unbounded global PageRank over the live ~139k-node
graph — OOM-killed the epistemic-graph engine (exitCode 137).
``AnalyticsClient.pagerank``/``betweenness_centrality``/``degree_centrality_all``
take no kwarg that scopes them below the entire graph, so any call is
unbounded by construction; ``personalized_pagerank`` is bounded only when it
carries a non-empty ``seed_nodes`` frontier.

These tests assert the guard rejects the unbounded calls with a clear, typed
error BEFORE an engine client is even acquired (so a misbehaving call can
never reach the wire), while every bounded/legitimate analytics call (a
seeded ``personalized_pagerank``, a single-node ``degree_centrality``, or any
non-``analytics`` domain) still passes through unaffected.

No live engine is required — mirrors the fake sub-client pattern in
``tests/unit/test_engine_tools_scope_policy.py``.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools import engine_tools


def _fake_client_factory():
    """A fake ``SyncEpistemicGraphClient`` with recording sub-clients."""
    calls: list[tuple[str, str, dict]] = []

    def _sub(domain: str):
        def _make(name):
            def _call(**kwargs):
                calls.append((domain, name, kwargs))
                return {"ok": True, "domain": domain, "method": name}

            return _call

        class _Sub:
            def __getattr__(self, name):
                return _make(name)

        return _Sub()

    class _Client:
        def __getattr__(self, name):
            return _sub(name)

    return _Client(), calls


@pytest.fixture(autouse=True)
def _fresh_client_pool(monkeypatch):
    """Isolate the module-level client-pool singleton across tests."""
    monkeypatch.setattr(engine_tools, "_CLIENT_POOL", None)
    yield
    monkeypatch.setattr(engine_tools, "_CLIENT_POOL", None)


# ── unit-level guard function ────────────────────────────────────────────────
@pytest.mark.parametrize(
    "action", ["pagerank", "betweenness_centrality", "degree_centrality_all"]
)
def test_guard_rejects_always_unbounded_actions(action):
    msg = engine_tools._reject_unbounded_analytics("analytics", action, {})
    assert msg is not None
    assert "OOM" in msg
    assert "personalized_pagerank" in msg


def test_guard_rejects_personalized_pagerank_without_seed_nodes():
    msg = engine_tools._reject_unbounded_analytics(
        "analytics", "personalized_pagerank", {}
    )
    assert msg is not None
    assert "seed_nodes" in msg


def test_guard_allows_personalized_pagerank_with_seed_nodes():
    msg = engine_tools._reject_unbounded_analytics(
        "analytics",
        "personalized_pagerank",
        {"seed_nodes": [["n1", 1.0]]},
    )
    assert msg is None


def test_guard_allows_single_node_degree_centrality():
    """``degree_centrality`` (singular) is a bounded, single-node read — never
    gated, unlike its whole-graph sibling ``degree_centrality_all``."""
    msg = engine_tools._reject_unbounded_analytics(
        "analytics", "degree_centrality", {"node_id": "n1"}
    )
    assert msg is None


def test_guard_ignores_non_analytics_domains():
    msg = engine_tools._reject_unbounded_analytics("nodes", "pagerank", {})
    assert msg is None


# ── live-path: through the registered MCP tool + _dispatch ───────────────────
def test_unbounded_pagerank_rejected_via_dispatch_without_touching_client(
    monkeypatch,
):
    """The guard must fire BEFORE ``_client_for`` is called — an unbounded
    pagerank must never reach the wire, let alone OOM the engine."""
    kg_server.ensure_tools_registered()
    client, calls = _fake_client_factory()
    client_acquired = []

    def _tracking_client_for(graph):
        client_acquired.append(graph)
        return client

    monkeypatch.setattr(engine_tools, "_client_for", _tracking_client_for)

    tool = kg_server.REGISTERED_TOOLS["engine_analytics"]
    out = json.loads(asyncio.run(tool(action="pagerank", params_json="{}", graph="")))

    assert "error" in out
    assert "OOM" in out["error"]
    assert out.get("guard") == "RESULT_TOO_LARGE"
    assert client_acquired == []  # never touched the engine client/pool
    assert calls == []  # never reached the fake wire method


def test_bounded_personalized_pagerank_passes_through(monkeypatch):
    kg_server.ensure_tools_registered()
    client, calls = _fake_client_factory()
    monkeypatch.setattr(engine_tools, "_client_for", lambda graph: client)

    tool = kg_server.REGISTERED_TOOLS["engine_analytics"]
    params = {"seed_nodes": [["n1", 1.0]]}
    out = json.loads(
        asyncio.run(
            tool(
                action="personalized_pagerank",
                params_json=json.dumps(params),
                graph="",
            )
        )
    )

    assert out.get("ok") is True, out
    assert calls == [("analytics", "personalized_pagerank", params)]


def test_bounded_single_node_degree_centrality_passes_through(monkeypatch):
    kg_server.ensure_tools_registered()
    client, calls = _fake_client_factory()
    monkeypatch.setattr(engine_tools, "_client_for", lambda graph: client)

    tool = kg_server.REGISTERED_TOOLS["engine_analytics"]
    params = {"node_id": "n1"}
    out = json.loads(
        asyncio.run(
            tool(
                action="degree_centrality",
                params_json=json.dumps(params),
                graph="",
            )
        )
    )

    assert out.get("ok") is True, out
    assert calls == [("analytics", "degree_centrality", params)]
