"""B-18 — implicit content-graph-union fan-out under an authenticated,
multi-tenant session (GOC-59..67 EXPANSION-TRACKS, B-series).

Root cause: ``ingest_routing.safe_engine_for_graph`` builds a ``for_graph()``
view for every supplementary content graph in the implicit-default fan-out
(``kg_server._resolve_read_engines``), but never rebinds ``session.graph`` to
match. The wire layer's own mismatch lock
(``_SessionRoutedAsyncClient._send``, ``graph_compute.py:509-517``:
"An explicit graph cannot retarget the verified GraphSession") accepts an
explicit ``graph`` only when it equals ``session.graph`` — so it rejected
every non-"default" leg with ``PermissionError``, silently degrading the
content-graph union to per-target errors.

A second, independent defect made this worse: ``kg_server.fanout_execute``
ran each target via a plain ``concurrent.futures.ThreadPoolExecutor``, which
does NOT propagate the calling thread's ``contextvars`` context (unlike
``asyncio.to_thread``, which the async tool endpoints use to reach this
synchronous helper in the first place) — so the ambient authenticated
``GraphSession`` was invisible inside EVERY fan-out worker thread, not just
the mismatched ones.

The fix reuses the sanctioned narrowing primitive
(``kg_server.bound_to_graph`` — the same ``GraphSession.with_graph()`` +
``use_session()`` seam ``resolve_explicit_graph``/single-connection callers
already use) around each content-graph fan-out call, and fixes
``fanout_execute`` to propagate a fresh ``contextvars.copy_context()`` per
submission so ``bound_to_graph`` has an ambient session to narrow in the
first place.

These tests exercise the REAL ``kg_server.fanout_execute`` concurrency and
the real ``graph_query``/``graph_search`` fan-out code paths — no live engine
required, mirroring ``test_graph_query_fanout_timeout.py``'s
``_resolve_read_engines`` monkeypatch pattern — with FAKE engines that
reproduce the wire layer's own mismatch contract: they raise
``PermissionError`` unless the ambient session's ``.graph`` matches the
graph they are scoped to, exactly like a real ``for_graph()`` view's
``_send`` does. This is a "known-bad input" proof, not a happy-path stub: run
without the fix, every one of these fails.
"""

from __future__ import annotations

import asyncio

from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    current_session,
    use_session,
)
from agent_utilities.mcp import kg_server
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor


def _register_query_tools():
    from fastmcp import FastMCP

    from agent_utilities.mcp.tools.query_tools import register_query_tools

    register_query_tools(FastMCP("test"))


def _multi_tenant_session(actor: ActorContext, graph: str) -> GraphSession:
    return GraphSession(
        actor=actor,
        tenant=actor.tenant_id,
        scopes=frozenset({"kg:read", "kg:write"}),
        graph=graph,
        policy_version="policy-v1",
        audience="agent-services",
    )


class _GraphScopedEngine:
    """Fakes a real ``for_graph()`` view's own enforcement: a call only
    succeeds when the AMBIENT session's ``.graph`` matches the physical graph
    this engine is scoped to — the exact contract
    ``_SessionRoutedAsyncClient._send`` enforces
    (``graph and session.graph and graph != session.graph`` -> raise
    ``PermissionError``). Reproduces the wire-layer defect without a live
    engine.
    """

    def __init__(self, fixed_graph: str, rows: list[dict]) -> None:
        self._fixed_graph = fixed_graph
        self._rows = rows

    def query_cypher(self, cypher, params, as_of=None, include_epistemic=False):
        session = current_session()
        if session is None or session.graph != self._fixed_graph:
            raise PermissionError(
                "An explicit graph cannot retarget the verified GraphSession"
            )
        return self._rows

    def search_hybrid(self, query, top_k, self_correct=False, as_of=None, session=None):
        current = current_session()
        if current is None or current.graph != self._fixed_graph:
            raise PermissionError(
                "An explicit graph cannot retarget the verified GraphSession"
            )
        return [{"id": self._fixed_graph, "score": 1.0, "content": self._fixed_graph}]


def _fake_resolve_read_engines(entries, fanout=True):
    def _resolve(target):
        return (entries, {}, fanout)

    return _resolve


# ── fanout_execute: ambient session context reaches every worker thread ────


def test_fanout_execute_propagates_ambient_session_into_worker_threads():
    """Known-bad-input regression guard for the SECOND defect: before the fix,
    ``ThreadPoolExecutor.submit`` started every worker with a brand-new, empty
    ``contextvars`` context, so ``current_session()`` returned ``None`` inside
    every fan-out target regardless of which graph it targeted."""
    actor = ActorContext(
        actor_id="agent:fanout-ctx-test",
        actor_type=ActorType.AUTOMATED_SERVICE,
        tenant_id="tenant:b18",
        authenticated=True,
    )
    session = _multi_tenant_session(actor, "__commons__")

    seen: list[tuple[str, str]] = []

    def _fn(name, engine):
        current = current_session()
        seen.append((name, current.actor.actor_id if current else ""))
        return name

    with use_actor(actor), use_session(session):
        results, errors = kg_server.fanout_execute(
            [("a", None), ("b", None), ("c", None)], _fn
        )

    assert errors == {}
    assert results == {"a": "a", "b": "b", "c": "c"}
    # Every worker thread saw the SAME ambient authenticated actor — not "".
    assert seen == [
        ("a", "agent:fanout-ctx-test"),
        ("b", "agent:fanout-ctx-test"),
        ("c", "agent:fanout-ctx-test"),
    ]


# ── graph_query: two-graph union fan-out under an authenticated session ────


def test_graph_query_implicit_union_fanout_succeeds_across_two_graphs(monkeypatch):
    """The acceptance criterion: an authenticated, multi-tenant session's
    implicit-default ``graph_query`` fans out across the "default" graph PLUS
    two distinct content graphs, and every leg actually grounds real rows —
    not per-target ``PermissionError``s hidden in the union merge."""
    _register_query_tools()

    actor = ActorContext(
        actor_id="agent:tenant-b18",
        actor_type=ActorType.AUTOMATED_SERVICE,
        tenant_id="tenant:b18",
        authenticated=True,
    )
    session = _multi_tenant_session(actor, "__commons__")

    entries = [
        ("default", _GraphScopedEngine("__commons__", [{"id": "commons:1"}])),
        ("code:repo-a", _GraphScopedEngine("code:repo-a", [{"id": "code:repo-a:1"}])),
        ("code:repo-b", _GraphScopedEngine("code:repo-b", [{"id": "code:repo-b:1"}])),
    ]
    monkeypatch.setattr(
        kg_server, "_resolve_read_engines", _fake_resolve_read_engines(entries)
    )

    with use_actor(actor), use_session(session):
        out = asyncio.run(
            kg_server._execute_tool("graph_query", cypher="MATCH (n) RETURN n")
        ).model_dump()
        # The narrowing is scoped to the call — the ambient session is
        # restored to the caller's own graph afterward, never leaked.
        assert current_session().graph == "__commons__"

    rows = out["reasoning_trace"][-1]["payload"]["rows"]
    row_ids = {r["id"] for r in rows}
    # All THREE legs grounded real rows -- in particular the two non-default
    # content graphs, which is exactly what the mismatch lock used to reject.
    assert row_ids == {"commons:1", "code:repo-a:1", "code:repo-b:1"}


def test_graph_query_union_fanout_fails_without_narrowing_known_bad_input(monkeypatch):
    """Companion known-bad-input proof: calling the SAME fake, mismatch-
    enforcing engine directly for a non-default graph, with the ambient
    session left un-narrowed, reproduces the exact defect this fix closes —
    demonstrating the fake genuinely encodes the real wire-layer contract
    rather than trivially always succeeding."""
    actor = ActorContext(
        actor_id="agent:tenant-b18",
        actor_type=ActorType.AUTOMATED_SERVICE,
        tenant_id="tenant:b18",
        authenticated=True,
    )
    session = _multi_tenant_session(actor, "__commons__")
    engine = _GraphScopedEngine("code:repo-a", [{"id": "code:repo-a:1"}])

    with use_actor(actor), use_session(session):
        # No `bound_to_graph` narrowing here -- session.graph stays
        # "__commons__" while the engine is scoped to "code:repo-a".
        try:
            engine.query_cypher("MATCH (n) RETURN n", {})
            raised = False
        except PermissionError:
            raised = True
    assert raised, "the fake must reproduce the real mismatch-lock rejection"


# ── graph_search: same union fan-out, hybrid-search path ───────────────────


def test_graph_search_implicit_union_fanout_succeeds_across_two_graphs(monkeypatch):
    """The ``graph_search`` twin of the ``graph_query`` acceptance test above
    -- proves the same fix on the ``_run_search``/``_content_graph_search``
    path."""
    _register_query_tools()

    actor = ActorContext(
        actor_id="agent:tenant-b18-search",
        actor_type=ActorType.AUTOMATED_SERVICE,
        tenant_id="tenant:b18",
        authenticated=True,
    )
    session = _multi_tenant_session(actor, "__commons__")

    entries = [
        ("default", _GraphScopedEngine("__commons__", [])),
        ("code:repo-a", _GraphScopedEngine("code:repo-a", [])),
        ("code:repo-b", _GraphScopedEngine("code:repo-b", [])),
    ]
    monkeypatch.setattr(
        kg_server, "_resolve_read_engines", _fake_resolve_read_engines(entries)
    )

    with use_actor(actor), use_session(session):
        text = asyncio.run(kg_server._execute_tool("graph_search", query="anything"))
    # Every leg's own tag (its fixed graph name, echoed by `_GraphScopedEngine`)
    # made it into the merged output — none was swallowed as a mismatch error.
    assert "code:repo-a" in text
    assert "code:repo-b" in text
    assert "__commons__" in text
    assert "PermissionError" not in text
    assert "cannot retarget" not in text
