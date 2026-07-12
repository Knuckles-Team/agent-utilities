"""`graph_search` default-scope fan-out must stay fast (CONCEPT:AU-KG.ingest.unified-query-routing
follow-up).

Before this, an implicit-default ``graph_search`` (no ``target``) fanned across
the *entire* routed content-graph set — ``default`` plus every active
``code:<repo>``/``src:<repo>`` graph, which can be dozens of per-repo
connections — under the SAME 30s per-backend budget used for an explicit
``target='all'`` request. When most of those per-repo graphs are idle/
unreachable, the default call blocked up to 30s on each one, flooding the
result with "timed out after 30s" entries even though the primary/default
backend answered in milliseconds.

These tests prove: (1) an implicit-default fan-out applies the SHORT
``DEFAULT_CONTENT_FANOUT_TIMEOUT_S`` skip-budget to the SUPPLEMENTARY content
backends, not the full 30s ``DEFAULT_FANOUT_TIMEOUT_S``; (2) an explicit
``target='all'`` (a deliberate cross-repo search) keeps the full budget; and
(3), end-to-end with real ``fanout_execute`` concurrency, that a default search
grounds the primary backend's real hits and returns quickly even when the
content-graph backends never respond.

Regression guard (the second fix): the PRIMARY/``default`` backend is queried
SEPARATELY at the normal budget and is NEVER part of the short-timeout
supplementary fan-out. Under a wide implicit fan-out (~70 ``code:*`` graphs) a
single shared short wall-clock across all targets starved the primary — it was
queued behind the hung code backends and timed out too, returning ZERO results.
So we also assert the primary grounds even when every supplementary backend
times out AND even when the hung supplementary backends saturate the fan-out
worker pool.

Mirrors the ``kg_server._resolve_read_engines`` monkeypatch pattern of
``test_graph_query_evidence_bundle.py`` / ``test_graph_query_include_epistemic.py``
— no live engine required.
"""

from __future__ import annotations

import asyncio
import json
import time

from agent_utilities.mcp import kg_server


def _register_query_tools():
    from fastmcp import FastMCP

    from agent_utilities.mcp.tools.query_tools import register_query_tools

    register_query_tools(FastMCP("test"))


class _FakeSearchEngine:
    """Stands in for an ``IntelligenceGraphEngine`` on the ``search_hybrid`` path
    ``graph_search``'s default ``mode='hybrid'`` calls."""

    def __init__(self, results=None, delay: float = 0.0):
        self._results = results or []
        self._delay = delay

    def search_hybrid(self, query, top_k, self_correct=False, as_of=None):
        if self._delay:
            time.sleep(self._delay)
        return self._results


_GROUNDED_HIT = [
    {
        "score": 0.87,
        "node": {
            "type": "Concept",
            "name": "DelegationRouter",
            "description": "Routes agent delegation across the fleet.",
            "id": "concept:delegation-router",
        },
    }
]


def _fake_resolve_read_engines_multi(entries, fanout=True):
    def _resolve(target):
        return (entries, {}, fanout)

    return _resolve


# ── (1) implicit-default target → SHORT per-backend timeout ────────────────


def test_graph_search_default_target_uses_short_fanout_timeout(monkeypatch):
    """No ``target`` passed at all → the resolved content-graph union fan-out
    must run under the SHORT ``DEFAULT_CONTENT_FANOUT_TIMEOUT_S`` budget, not
    the full 30s ``DEFAULT_FANOUT_TIMEOUT_S`` — this is the fix under test."""
    _register_query_tools()
    entries = [
        ("default", _FakeSearchEngine(_GROUNDED_HIT)),
        ("code:repo-a", _FakeSearchEngine()),
        ("code:repo-b", _FakeSearchEngine()),
    ]
    monkeypatch.setattr(
        kg_server, "_resolve_read_engines", _fake_resolve_read_engines_multi(entries)
    )

    seen: dict[str, object] = {}
    real_fanout_execute = kg_server.fanout_execute

    def _spy(entries_arg, fn, *, timeout=None):
        seen["timeout"] = timeout
        seen["names"] = [n for n, _ in entries_arg]
        return real_fanout_execute(entries_arg, fn, timeout=timeout)

    monkeypatch.setattr(kg_server, "fanout_execute", _spy)

    asyncio.run(
        kg_server._execute_tool("graph_search", query="delegation router")
    )

    assert seen["timeout"] == kg_server.DEFAULT_CONTENT_FANOUT_TIMEOUT_S
    assert seen["timeout"] < kg_server.DEFAULT_FANOUT_TIMEOUT_S
    # The short skip-timeout fan-out covers ONLY the supplementary content
    # backends — the primary/default is queried separately and must NOT be in it.
    assert "default" not in seen["names"]
    assert seen["names"] == ["code:repo-a", "code:repo-b"]


def test_graph_search_target_default_string_also_uses_short_timeout(monkeypatch):
    """Explicitly passing ``target='default'`` is equivalent to omitting it —
    still the implicit single-connection intent, so it must also get the short
    budget when the resolver reports a fan-out (routing spread it)."""
    _register_query_tools()
    entries = [
        ("default", _FakeSearchEngine(_GROUNDED_HIT)),
        ("code:repo-a", _FakeSearchEngine()),
    ]
    monkeypatch.setattr(
        kg_server, "_resolve_read_engines", _fake_resolve_read_engines_multi(entries)
    )
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        kg_server,
        "fanout_execute",
        lambda entries_arg, fn, *, timeout=None: (seen.update(timeout=timeout), ({}, {}))[
            1
        ],
    )

    asyncio.run(
        kg_server._execute_tool(
            "graph_search", query="delegation router", target="default"
        )
    )

    assert seen["timeout"] == kg_server.DEFAULT_CONTENT_FANOUT_TIMEOUT_S


# ── (2) explicit cross-repo opt-in → full timeout preserved ────────────────


def test_graph_search_explicit_all_target_keeps_full_fanout_timeout(monkeypatch):
    """``target='all'`` is a deliberate cross-repo request — it must keep the
    full ``DEFAULT_FANOUT_TIMEOUT_S`` budget (``timeout=None`` → fanout_execute
    falls back to its own default), not the short content-graph budget."""
    _register_query_tools()
    entries = [
        ("default", _FakeSearchEngine(_GROUNDED_HIT)),
        ("code:repo-a", _FakeSearchEngine()),
    ]
    monkeypatch.setattr(
        kg_server, "_resolve_read_engines", _fake_resolve_read_engines_multi(entries)
    )
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        kg_server,
        "fanout_execute",
        lambda entries_arg, fn, *, timeout=None: (seen.update(timeout=timeout), ({}, {}))[
            1
        ],
    )

    asyncio.run(
        kg_server._execute_tool("graph_search", query="delegation router", target="all")
    )

    assert seen["timeout"] is None  # fanout_execute's own (full) default applies


# ── (3) end-to-end: grounds fast despite unreachable content-graph backends ─


def test_graph_search_default_grounds_primary_backend_and_skips_slow_backends(
    monkeypatch,
):
    """Real ``fanout_execute`` concurrency: the primary/default backend answers
    instantly with a real grounded hit; several ``code:<repo>`` backends never
    return. The default search must ground the real hit and come back within
    the SHORT budget — not block for the full fan-out timeout per dead graph."""
    _register_query_tools()
    # Patch the short budget down for a fast, deterministic test while proving
    # the exact same wiring the production constant uses.
    monkeypatch.setattr(kg_server, "DEFAULT_CONTENT_FANOUT_TIMEOUT_S", 0.3)

    entries = [
        ("default", _FakeSearchEngine(_GROUNDED_HIT)),
    ] + [
        (f"code:repo-{i}", _FakeSearchEngine(delay=1.5)) for i in range(5)
    ]
    monkeypatch.setattr(
        kg_server, "_resolve_read_engines", _fake_resolve_read_engines_multi(entries)
    )

    started = time.monotonic()
    out = asyncio.run(
        kg_server._execute_tool("graph_search", query="delegation router")
    )
    elapsed = time.monotonic() - started

    # Bounded by the short per-backend budget, nowhere near 5 * 30s (or even
    # 5 * 1.5s if the slow backends were awaited sequentially).
    assert elapsed < 2.0

    # The primary/default backend's real hit is grounded in the output.
    assert "DelegationRouter" in out
    assert "concept:delegation-router" in out
    assert "=== default ===" in out

    # The unreachable content-graph backends are reported as timeouts, not
    # silently missing and not blocking the call.
    for i in range(5):
        assert f"=== code:repo-{i} (error) ===" in out
        assert "timed out" in out


# ── (4) regression: primary grounds even when supplementary starve the pool ─


def test_graph_search_primary_grounds_when_all_supplementary_time_out(monkeypatch):
    """The reported regression: under a wide implicit fan-out where EVERY
    supplementary ``code:*`` backend hangs — enough of them to saturate the
    fan-out worker pool (min(8, N)) — the primary/``default`` backend must STILL
    ground its real ranked hits. Before the primary was split out of the
    short-timeout set, it was queued behind the hung backends and timed out too,
    so the search returned zero results even though the engine was healthy."""
    _register_query_tools()
    monkeypatch.setattr(kg_server, "DEFAULT_CONTENT_FANOUT_TIMEOUT_S", 0.5)

    # 20 hung supplementary backends (each 5s) >> the 8-thread fan-out pool, so a
    # primary placed INSIDE this set could never be scheduled within 0.5s.
    entries = [("default", _FakeSearchEngine(_GROUNDED_HIT))] + [
        (f"code:repo-{i}", _FakeSearchEngine(delay=5.0)) for i in range(20)
    ]
    monkeypatch.setattr(
        kg_server, "_resolve_read_engines", _fake_resolve_read_engines_multi(entries)
    )

    started = time.monotonic()
    out = asyncio.run(
        kg_server._execute_tool("graph_search", query="delegation router")
    )
    elapsed = time.monotonic() - started

    # Primary grounded — NOT zero results — and the whole call stayed fast.
    assert "=== default ===" in out
    assert "DelegationRouter" in out
    assert "concept:delegation-router" in out
    assert elapsed < 3.0
    # Every supplementary backend timed out; none blocked the primary.
    assert out.count("timed out") >= 20


def test_graph_search_primary_grounds_when_no_default_named_entry(monkeypatch):
    """Robustness: if the resolver ever produces a fan-out with no entry named
    ``default``, the first entry is treated as primary and queried at the normal
    budget so SOMETHING always grounds (never a fully-skipped, empty result)."""
    _register_query_tools()
    monkeypatch.setattr(kg_server, "DEFAULT_CONTENT_FANOUT_TIMEOUT_S", 0.3)
    entries = [
        ("code:primary", _FakeSearchEngine(_GROUNDED_HIT)),
        ("code:repo-a", _FakeSearchEngine(delay=1.5)),
    ]
    monkeypatch.setattr(
        kg_server, "_resolve_read_engines", _fake_resolve_read_engines_multi(entries)
    )
    out = asyncio.run(
        kg_server._execute_tool("graph_search", query="delegation router")
    )
    assert "=== code:primary ===" in out
    assert "DelegationRouter" in out
    assert "=== code:repo-a (error) ===" in out
