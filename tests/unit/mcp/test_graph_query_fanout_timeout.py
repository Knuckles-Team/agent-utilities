"""`graph_query` default-scope fan-out must stay fast (CONCEPT:AU-KG.ingest.unified-query-routing
follow-up — the `graph_query` twin of `test_graph_search_fanout_timeout.py`).

Before this, an implicit-default `graph_query` (no `target`, local Cypher scope)
shared ONE `DEFAULT_FANOUT_TIMEOUT_S` (30s) budget across every resolved
content-graph connection, including the `default` connection — unlike
`graph_search`, which already splits the primary/`default` connection out at
the full budget and only applies the short `DEFAULT_CONTENT_FANOUT_TIMEOUT_S`
to the supplementary `code:*`/`src:*` connections. Under a wide implicit
fan-out (dozens of idle/unreachable per-repo graphs) a plain Cypher read
against the live default graph could take minutes (measured: a `graph_query`
via the intent surface took 320s while the identical `engine_query
action=cypher` against the same default graph returned instantly) even though
the engine itself was healthy.

These tests prove `graph_query`'s local Cypher fan-out now mirrors
`graph_search`'s: (1) an implicit-default fan-out applies the SHORT
`DEFAULT_CONTENT_FANOUT_TIMEOUT_S` skip-budget to the SUPPLEMENTARY content
backends only, never the `default` connection; (2) an explicit `target='all'`
(a deliberate cross-repo search) keeps the full budget; and (3), end-to-end
with real `fanout_execute` concurrency, that a default query grounds the
primary backend's real rows and returns quickly even when every content-graph
backend never responds.
"""

from __future__ import annotations

import asyncio
import time

from agent_utilities.mcp import kg_server


def _register_query_tools():
    from fastmcp import FastMCP

    from agent_utilities.mcp.tools.query_tools import register_query_tools

    register_query_tools(FastMCP("test"))


class _FakeQueryEngine:
    """Stand-in for `IntelligenceGraphEngine` on the local Cypher path."""

    def __init__(self, rows=None, delay: float = 0.0):
        self._rows = rows if rows is not None else []
        self._delay = delay

    def query_cypher(self, cypher, params, as_of=None, include_epistemic=False):
        if self._delay:
            time.sleep(self._delay)
        return self._rows


class _DenyingQueryEngine:
    """Stand-in for a backend whose OWN ACL/tenant enforcement rejects this
    query (e.g. `engine.query_cypher` raising because the actor lacks
    `kg:read` on this tenant/graph) — never returns rows."""

    def __init__(self, message: str = "actor lacks kg:read on this tenant"):
        self._message = message

    def query_cypher(self, cypher, params, as_of=None, include_epistemic=False):
        raise PermissionError(self._message)


_GROUNDED_ROW = [{"id": "concept:delegation-router", "name": "DelegationRouter"}]
_OTHER_TENANT_ROW = [{"id": "concept:other-tenant-secret", "name": "ShouldNeverLeak"}]


def _fake_resolve_read_engines_multi(entries, fanout=True):
    def _resolve(target):
        return (entries, {}, fanout)

    return _resolve


# ── (1) implicit-default target → SHORT per-backend timeout ────────────────


def test_graph_query_default_target_uses_short_fanout_timeout(monkeypatch):
    """No `target` passed at all → the resolved content-graph union fan-out
    must run under the SHORT `DEFAULT_CONTENT_FANOUT_TIMEOUT_S` budget for the
    SUPPLEMENTARY connections, not the full 30s `DEFAULT_FANOUT_TIMEOUT_S` —
    this is the fix under test."""
    _register_query_tools()
    entries = [
        ("default", _FakeQueryEngine(_GROUNDED_ROW)),
        ("code:repo-a", _FakeQueryEngine()),
        ("code:repo-b", _FakeQueryEngine()),
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

    out = asyncio.run(
        kg_server._execute_tool("graph_query", cypher="MATCH (n) RETURN n")
    ).model_dump()

    assert seen["timeout"] == kg_server.DEFAULT_CONTENT_FANOUT_TIMEOUT_S
    assert seen["timeout"] < kg_server.DEFAULT_FANOUT_TIMEOUT_S
    # The short skip-timeout fan-out covers ONLY the supplementary content
    # backends — the primary/default is queried separately and must NOT be in it.
    assert "default" not in seen["names"]
    assert seen["names"] == ["code:repo-a", "code:repo-b"]
    # The primary's real row is still grounded in the merged union output.
    rows = out["reasoning_trace"][-1]["payload"]["rows"]
    assert {"id": "concept:delegation-router", "name": "DelegationRouter"} in rows


def test_graph_query_target_default_string_also_uses_short_timeout(monkeypatch):
    """Explicitly passing `target='default'` is equivalent to omitting it —
    still the implicit single-connection intent, so it must also get the short
    budget for supplementary connections when the resolver reports a fan-out
    (routing spread it)."""
    _register_query_tools()
    entries = [
        ("default", _FakeQueryEngine(_GROUNDED_ROW)),
        ("code:repo-a", _FakeQueryEngine()),
    ]
    monkeypatch.setattr(
        kg_server, "_resolve_read_engines", _fake_resolve_read_engines_multi(entries)
    )
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        kg_server,
        "fanout_execute",
        lambda entries_arg, fn, *, timeout=None: (
            seen.update(timeout=timeout),
            ({}, {}),
        )[1],
    )

    asyncio.run(
        kg_server._execute_tool(
            "graph_query", cypher="MATCH (n) RETURN n", target="default"
        )
    )

    assert seen["timeout"] == kg_server.DEFAULT_CONTENT_FANOUT_TIMEOUT_S


# ── (2) explicit cross-repo opt-in → full timeout preserved ────────────────


def test_graph_query_explicit_all_target_keeps_full_fanout_timeout(monkeypatch):
    """`target='all'` is a deliberate cross-repo request — it must keep the
    full `DEFAULT_FANOUT_TIMEOUT_S` budget (`timeout=None` → `fanout_execute`
    falls back to its own default), not the short content-graph budget, and
    must NOT be split into a primary/supplementary set."""
    _register_query_tools()
    entries = [
        ("default", _FakeQueryEngine(_GROUNDED_ROW)),
        ("code:repo-a", _FakeQueryEngine()),
    ]
    monkeypatch.setattr(
        kg_server, "_resolve_read_engines", _fake_resolve_read_engines_multi(entries)
    )
    seen: dict[str, object] = {}

    def _spy(entries_arg, fn, *, timeout=None):
        seen["timeout"] = timeout
        seen["names"] = [n for n, _ in entries_arg]
        return ({}, {})

    monkeypatch.setattr(kg_server, "fanout_execute", _spy)

    asyncio.run(
        kg_server._execute_tool(
            "graph_query", cypher="MATCH (n) RETURN n", target="all"
        )
    )

    assert seen["timeout"] is None  # fanout_execute's own (full) default applies
    # Both connections went through the SAME fanout_execute call — never split.
    assert seen["names"] == ["default", "code:repo-a"]


# ── (3) end-to-end: grounds fast despite unreachable content-graph backends ─


def test_graph_query_default_grounds_primary_backend_and_skips_slow_backends(
    monkeypatch,
):
    """Real `fanout_execute` concurrency: the primary/default backend answers
    instantly with a real row; several `code:<repo>` backends never return.
    The default query must ground the real row and come back within the SHORT
    budget — not block for the full fan-out timeout per dead graph."""
    _register_query_tools()
    monkeypatch.setattr(kg_server, "DEFAULT_CONTENT_FANOUT_TIMEOUT_S", 0.3)

    entries = [
        ("default", _FakeQueryEngine(_GROUNDED_ROW)),
    ] + [(f"code:repo-{i}", _FakeQueryEngine(delay=1.5)) for i in range(5)]
    monkeypatch.setattr(
        kg_server, "_resolve_read_engines", _fake_resolve_read_engines_multi(entries)
    )

    started = time.monotonic()
    out = asyncio.run(
        kg_server._execute_tool("graph_query", cypher="MATCH (n) RETURN n")
    ).model_dump()
    elapsed = time.monotonic() - started

    # Bounded by the short per-backend budget, nowhere near 5 * 30s (or even
    # 5 * 1.5s if the slow backends were awaited sequentially).
    assert elapsed < 2.0

    rows = out["reasoning_trace"][-1]["payload"]["rows"]
    assert {"id": "concept:delegation-router", "name": "DelegationRouter"} in rows


def test_graph_query_primary_grounds_when_all_supplementary_time_out(monkeypatch):
    """The `graph_query` analog of the `graph_search` regression guard: under a
    wide implicit fan-out where EVERY supplementary `code:*` backend hangs —
    enough of them to saturate the fan-out worker pool (min(8, N)) — the
    primary/`default` backend must STILL ground its real rows."""
    _register_query_tools()
    monkeypatch.setattr(kg_server, "DEFAULT_CONTENT_FANOUT_TIMEOUT_S", 0.5)

    entries = [("default", _FakeQueryEngine(_GROUNDED_ROW))] + [
        (f"code:repo-{i}", _FakeQueryEngine(delay=5.0)) for i in range(20)
    ]
    monkeypatch.setattr(
        kg_server, "_resolve_read_engines", _fake_resolve_read_engines_multi(entries)
    )

    started = time.monotonic()
    out = asyncio.run(
        kg_server._execute_tool("graph_query", cypher="MATCH (n) RETURN n")
    ).model_dump()
    elapsed = time.monotonic() - started

    rows = out["reasoning_trace"][-1]["payload"]["rows"]
    assert {"id": "concept:delegation-router", "name": "DelegationRouter"} in rows
    assert elapsed < 3.0


def test_graph_query_primary_grounds_when_no_default_named_entry(monkeypatch):
    """Robustness: if the resolver ever produces a fan-out with no entry named
    `default`, the first entry is treated as primary and queried at the normal
    budget so SOMETHING always grounds (never a fully-skipped, empty result)."""
    _register_query_tools()
    monkeypatch.setattr(kg_server, "DEFAULT_CONTENT_FANOUT_TIMEOUT_S", 0.3)
    entries = [
        ("code:primary", _FakeQueryEngine(_GROUNDED_ROW)),
        ("code:repo-a", _FakeQueryEngine(delay=1.5)),
    ]
    monkeypatch.setattr(
        kg_server, "_resolve_read_engines", _fake_resolve_read_engines_multi(entries)
    )
    out = asyncio.run(
        kg_server._execute_tool("graph_query", cypher="MATCH (n) RETURN n")
    ).model_dump()
    rows = out["reasoning_trace"][-1]["payload"]["rows"]
    assert {"id": "concept:delegation-router", "name": "DelegationRouter"} in rows


def test_graph_query_aggregation_still_bypasses_fanout_entirely(monkeypatch):
    """Sanity: the pre-existing aggregation short-circuit (line ~454, run
    against the canonical default graph only) is untouched by this split — an
    aggregation Cypher never enters the new primary/supplementary branch at
    all, so it must not call `fanout_execute`."""
    _register_query_tools()
    entries = [
        ("default", _FakeQueryEngine([{"count": 3}])),
        ("code:repo-a", _FakeQueryEngine(delay=5.0)),
    ]
    monkeypatch.setattr(
        kg_server, "_resolve_read_engines", _fake_resolve_read_engines_multi(entries)
    )
    # The aggregation short-circuit resolves its OWN engine via `_get_engine()`
    # rather than the fan-out `entries` above — mock that too so the assertion
    # below observes the aggregation engine's real row, not the ambient default.
    monkeypatch.setattr(
        kg_server, "_get_engine", lambda: _FakeQueryEngine([{"count": 3}])
    )
    called = {"fanout": False}

    def _spy(*a, **kw):
        called["fanout"] = True
        return ({}, {})

    monkeypatch.setattr(kg_server, "fanout_execute", _spy)

    out = asyncio.run(
        kg_server._execute_tool(
            "graph_query", cypher="MATCH (n) RETURN count(n) AS count"
        )
    ).model_dump()

    assert called["fanout"] is False
    rows = out["reasoning_trace"][-1]["payload"]["rows"]
    assert rows == [{"count": 3}]


# ── (5) correctness: an ACL/tenant denial never leaks rows or crashes ──────
# Hard constraint: a faster read that returns different rows — or that lets
# one backend's rejection crash the whole call instead of degrading like
# every other fan-out target — is a security/availability bug, not an
# optimization. These prove the primary/supplementary split (which queries
# the primary DIRECTLY, bypassing `fanout_execute`'s own try/except, to skip
# its concurrency/timeout machinery) still gets the SAME partial-success error
# handling every other fan-out target gets.


def test_graph_query_primary_denial_is_labeled_error_not_a_crash_or_leak(monkeypatch):
    """The PRIMARY/default backend's own ACL enforcement denying this query
    (cross-tenant / insufficient scope) must be caught and labeled, never
    crash `graph_query`, and must never let a DIFFERENT backend's rows stand
    in for the denied one."""
    _register_query_tools()
    entries = [
        ("default", _DenyingQueryEngine("actor lacks kg:read on tenant X")),
        ("code:repo-a", _FakeQueryEngine(_GROUNDED_ROW)),
    ]
    monkeypatch.setattr(
        kg_server, "_resolve_read_engines", _fake_resolve_read_engines_multi(entries)
    )

    # Must not raise — a denied primary degrades like any other fan-out
    # target, it does not crash the whole tool call.
    out = asyncio.run(
        kg_server._execute_tool("graph_query", cypher="MATCH (n) RETURN n")
    ).model_dump()

    rows = out["reasoning_trace"][-1]["payload"]["rows"]
    # The denied primary contributed NOTHING — in particular nothing that
    # would only exist if its PermissionError were silently swallowed into an
    # empty-but-successful result.
    assert rows == _GROUNDED_ROW


def test_graph_query_supplementary_denial_never_leaks_into_primary_rows(
    monkeypatch,
):
    """A supplementary `code:*` backend's ACL denial must not contaminate the
    merged union — the primary's own rows are the only rows returned, and the
    denied backend's rows (had it ignored the denial) never appear."""
    _register_query_tools()
    entries = [
        ("default", _FakeQueryEngine(_GROUNDED_ROW)),
        ("code:other-tenant", _DenyingQueryEngine("cross-tenant denied")),
    ]
    monkeypatch.setattr(
        kg_server, "_resolve_read_engines", _fake_resolve_read_engines_multi(entries)
    )

    out = asyncio.run(
        kg_server._execute_tool("graph_query", cypher="MATCH (n) RETURN n")
    ).model_dump()

    rows = out["reasoning_trace"][-1]["payload"]["rows"]
    assert rows == _GROUNDED_ROW
    assert _OTHER_TENANT_ROW[0] not in rows


def test_graph_query_explicit_target_denial_also_never_crashes(monkeypatch):
    """The explicit `target='all'` path (untouched by the primary/supplementary
    split, still routed through plain `fanout_execute`) must keep the same
    denial-is-labeled-not-fatal contract — a regression guard that the split
    didn't accidentally change the OTHER branch's error handling either."""
    _register_query_tools()
    entries = [
        ("default", _FakeQueryEngine(_GROUNDED_ROW)),
        ("code:other-tenant", _DenyingQueryEngine("cross-tenant denied")),
    ]
    monkeypatch.setattr(
        kg_server, "_resolve_read_engines", _fake_resolve_read_engines_multi(entries)
    )

    out = asyncio.run(
        kg_server._execute_tool(
            "graph_query", cypher="MATCH (n) RETURN n", target="all"
        )
    ).model_dump()

    payload = out["reasoning_trace"][-1]["payload"]
    # target='all' is NOT a union read — per-connection results/errors map.
    assert payload["targets"]["default"] == _GROUNDED_ROW
    assert "code:other-tenant" in payload["errors"]
    assert _OTHER_TENANT_ROW[0] not in payload["targets"].get("code:other-tenant", [])
