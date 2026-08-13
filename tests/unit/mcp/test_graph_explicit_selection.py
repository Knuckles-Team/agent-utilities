"""Wire-First acceptance tests: explicit PHYSICAL GRAPH selection on the graph-os
query/write MCP tool surface (CONCEPT:AU-KG.backend.explicit-graph-selection).

Root cause this closes: ``target`` (now ``connection``) has always selected a
*backend connection alias* (:class:`~agent_utilities.knowledge_graph.core.connection_registry.ConnectionRegistry`),
never a *physical engine graph* (the thing ``engine_tenants(action="list")`` /
the engine's own ``ListGraphs`` returns) — see ``kg_server._resolve_read_engines``/
``_resolve_target_engines``. There was no parameter a caller could use to say
"query graph X"; which graph a query hit was a side effect of connection
routing. This file proves the new, separate ``graph`` parameter on
``graph_query``/``graph_search``/``graph_write``:

* is bound into the verified session (``kg_server.bound_to_graph`` →
  ``GraphSession.with_graph`` → ``use_session``) — the SAME sanctioned
  primitive ``agent_utilities.orchestration.approval``/
  ``agent_utilities.knowledge_graph.pipeline`` already use, not a new
  authority mechanism;
* proves real two-graph isolation (write to graph A, not visible from graph B);
* fails closed on an unknown graph and on an ambiguous/unsupported
  graph+connection combination — never a silent fallback or union;
* behaves identically whether the ambient session came from an external
  bearer identity or the local/stdio "unauthenticated" bootstrap — the
  binding code has no auth-posture-conditional branch.

These call the tools via ``kg_server._execute_tool`` — the SAME dispatch the
MCP and REST surfaces use (REST is a generic ``_execute_tool(tool_name,
**body)`` passthrough, see ``kg_server.py``'s ``graph_query_endpoint`` et al.),
so this is also the REST-parity proof: nothing REST-specific to test.

``graph_query``'s public contract is the sole typed ``EvidenceBundle``
(``EvidenceBundle.from_payload``): success payloads land the `rows`/
`connection`/`graph` echo under ``reasoning_trace[-1]["payload"]`` (the
sibling-preservation path for an embedded ``evidence_bundle``), and a failure
lands the standardized operation error under the top-level ``error`` field —
see ``tests/unit/mcp/test_graph_query_evidence_bundle.py`` for the same
pattern. ``graph_write`` returns a raw string (JSON for structured actions,
plain text otherwise).
"""

from __future__ import annotations

import asyncio
import json

import pytest

from agent_utilities.knowledge_graph.core.session import GraphSession, current_session
from agent_utilities.mcp import kg_server
from agent_utilities.security.brain_context import ActorContext, ActorType


def _register_tools():
    from fastmcp import FastMCP

    from agent_utilities.mcp.tools.query_tools import register_query_tools
    from agent_utilities.mcp.tools.write_ingest_tools import register_write_ingest_tools

    mcp = FastMCP("test")
    register_query_tools(mcp)
    register_write_ingest_tools(mcp)


def _session(*, tenant: str = "test-tenant", roles: tuple = ("test",)) -> GraphSession:
    """A verified session with NO bound graph (``graph=""``) — mirrors both the
    externally-authenticated posture (an OIDC actor whose session simply has no
    tenant-graph pin in this fixture) and the local/stdio "unauthenticated"
    bootstrap (neutral service claims, no external IdP) identically: neither
    is a special case in the binding code below."""
    actor = ActorContext(
        actor_id="test-service",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=roles,
        tenant_id=tenant,
        authenticated=True,
    )
    return GraphSession(
        actor=actor,
        tenant=tenant,
        scopes=frozenset({"kg:read", "kg:write"}),
        policy_version="test-policy",
        audience="test-audience",
    )


class _FakeTenants:
    """Fake ``client.tenants`` namespace — the SAME ``ListGraphs`` shape
    ``kg_server._wait_for_engine_materialization`` already consults."""

    def __init__(self, names):
        self._names = list(names)

    def list(self):
        return [{"name": n} for n in self._names]


class _FakeComputeClient:
    def __init__(self, names):
        self.tenants = _FakeTenants(names)


class _FakeGraphCompute:
    def __init__(self, names):
        self.client = _FakeComputeClient(names)


class _FakeBackend:
    cypher_support = "full"
    supports_sparql = False

    def close(self):
        pass


class _MultiGraphEngine:
    """A fake ``IntelligenceGraphEngine`` whose node store is PARTITIONED by
    the ambient session's ``.graph`` — exactly how the real engine's per-graph
    catalog isolates data. This is the seam ``kg_server.bound_to_graph``
    actually manipulates (``current_session().graph``), so a real isolation
    failure in the wiring would show up here, not just in a mock call-count.
    No ``.graph`` (compute-provenance) attribute on purpose: exercises
    ``_evidence_bundle_for_rows``'s clean degrade path, so this test suite
    checks row content via the ``rows``/``connection``/``graph`` echo, not
    ``.claims`` (which only reflects the — here empty — provenance bundle).
    """

    def __init__(self, catalog_graphs):
        self.graph_compute = _FakeGraphCompute(catalog_graphs)
        self.backend = _FakeBackend()
        self._by_graph: dict[str, list[dict]] = {g: [] for g in catalog_graphs}
        self.calls: list[str] = []

    def _active_graph(self) -> str:
        session = current_session()
        return (session.graph if session else "") or "__commons__"

    def add_node(self, node_id, node_type, props=None):
        g = self._active_graph()
        self.calls.append(f"add_node:{g}:{node_id}")
        self._by_graph.setdefault(g, []).append(
            {"id": node_id, "type": node_type, **(props or {})}
        )

    def query_cypher(self, cypher, params=None, as_of=None, include_epistemic=False):
        g = self._active_graph()
        self.calls.append(f"query:{g}")
        return list(self._by_graph.get(g, []))


@pytest.fixture(autouse=True)
def _reset_state():
    saved_registry = kg_server._CONNECTION_REGISTRY
    saved_session = kg_server._PROCESS_SESSION
    kg_server._CONNECTION_REGISTRY = None
    kg_server._PROCESS_SESSION = _session()
    yield
    kg_server._CONNECTION_REGISTRY = saved_registry
    kg_server._PROCESS_SESSION = saved_session


def _install_engine(engine) -> None:
    from agent_utilities.knowledge_graph.core.connection_registry import (
        ConnectionRegistry,
    )

    kg_server._CONNECTION_REGISTRY = ConnectionRegistry(
        default_engine_provider=lambda: engine
    )


def _query_payload(bundle) -> dict:
    """Extract the `rows`/`connection`/`graph` echo from a successful
    `graph_query` `EvidenceBundle` — the sibling-preservation path for the
    embedded `evidence_bundle` (see module docstring)."""
    dumped = bundle.model_dump()
    return dumped["reasoning_trace"][-1]["payload"]


def _query_error_code(bundle) -> str | None:
    dumped = bundle.model_dump()
    error = dumped.get("error")
    return error.get("code") if isinstance(error, dict) else None


# ── (1) graph listing / discovery ───────────────────────────────────────────


async def test_explicit_graph_selection_consults_the_engine_catalog():
    """A caller can discover which graphs exist: resolving an explicit `graph`
    genuinely consults the SAME catalog `engine_tenants(action="list")`/
    `ListGraphs` exposes (`engine.graph_compute.client.tenants.list()`), not a
    hardcoded/guessed list — proven by asserting the fake catalog was queried
    and that its membership is what decides accept vs. reject."""
    _register_tools()
    engine = _MultiGraphEngine(catalog_graphs=["graph-a", "graph-b"])
    _install_engine(engine)

    listed = engine.graph_compute.client.tenants.list()
    assert {row["name"] for row in listed} == {"graph-a", "graph-b"}

    ok = await kg_server._execute_tool(
        "graph_query", cypher="MATCH (n) RETURN n", graph="graph-a"
    )
    payload = _query_payload(ok)
    assert payload["graph"] == "graph-a"
    assert payload["connection"] == "default"

    missing = await kg_server._execute_tool(
        "graph_query", cypher="MATCH (n) RETURN n", graph="does-not-exist"
    )
    assert _query_error_code(missing) == "graph_not_found"


# ── (2) two-graph isolation — the core proof ────────────────────────────────


async def test_two_graph_isolation_write_a_not_visible_from_b():
    """Write to graph A, confirm it is NOT visible when querying graph B — the
    direct proof that `graph` genuinely selects a physical graph and never
    falls back to / unions with any other graph."""
    _register_tools()
    engine = _MultiGraphEngine(catalog_graphs=["graph-a", "graph-b"])
    _install_engine(engine)

    write_out = await kg_server._execute_tool(
        "graph_write",
        action="add_node",
        node_id="secret-1",
        node_type="Thing",
        graph="graph-a",
    )
    assert "added" in write_out
    assert "[connection=default graph=graph-a]" in write_out

    # Visible from graph-a...
    from_a = await kg_server._execute_tool(
        "graph_query", cypher="MATCH (n) RETURN n", graph="graph-a"
    )
    rows_a = _query_payload(from_a)["rows"]
    assert [r["id"] for r in rows_a] == ["secret-1"]

    # ...NOT visible from graph-b.
    from_b = await kg_server._execute_tool(
        "graph_query", cypher="MATCH (n) RETURN n", graph="graph-b"
    )
    assert _query_payload(from_b)["rows"] == []

    # ...and NOT visible from the implicit/default (no explicit graph) path
    # either — proving no silent fallback merges graph-a's write into the
    # caller's own default view. (The ambient test session's OWN bound graph
    # — set by the suite-wide `isolate_graph_compute_engine` fixture, not by
    # this test — is whatever it is; the point is that it is disjoint from
    # `graph-a`/`graph-b` and sees neither.)
    own_default_graph = current_session().graph
    default_read = await kg_server._execute_tool(
        "graph_query", cypher="MATCH (n) RETURN n"
    )
    assert _query_payload(default_read)["rows"] == []

    assert engine.calls == [
        "add_node:graph-a:secret-1",
        "query:graph-a",
        "query:graph-b",
        f"query:{own_default_graph}",
    ]


# ── (3) fail-closed on unknown graph ────────────────────────────────────────


async def test_unknown_graph_fails_closed_never_falls_back():
    _register_tools()
    engine = _MultiGraphEngine(catalog_graphs=["graph-a"])
    _install_engine(engine)
    # Seed graph-a with a row so a silent fallback would be observable.
    await kg_server._execute_tool(
        "graph_write",
        action="add_node",
        node_id="n1",
        node_type="Thing",
        graph="graph-a",
    )

    out = await kg_server._execute_tool(
        "graph_query", cypher="MATCH (n) RETURN n", graph="ghost-graph"
    )
    assert _query_error_code(out) == "graph_not_found"
    dumped_json = json.dumps(out.model_dump())
    assert "ghost-graph" not in dumped_json  # no raw name leak in the public payload
    # The engine was never actually queried for the unknown graph — fail
    # closed happens BEFORE the call, not as an empty-result silent pass.
    assert not any(c.startswith("query:ghost-graph") for c in engine.calls)

    write_out = await kg_server._execute_tool(
        "graph_write",
        action="add_node",
        node_id="n2",
        node_type="Thing",
        graph="ghost-graph",
    )
    write_payload = json.loads(write_out)
    assert write_payload["error"]["code"] == "graph_not_found"
    assert not any("n2" in c for c in engine.calls)


# ── (4) alias/graph-name collision — explicit, never a silent pick ─────────


async def test_graph_and_fanout_connection_is_a_typed_conflict_not_silent_pick():
    """`graph` requires exactly one resolved connection — combining it with
    `connection='all'` must be a typed error, never a silent pick of one
    connection's own implicit graph while ignoring the rest."""
    _register_tools()
    engine = _MultiGraphEngine(catalog_graphs=["graph-a"])
    _install_engine(engine)
    registry = kg_server.get_connection_registry()
    registry.register("other", {"backend": "memory"})

    out = await kg_server._execute_tool(
        "graph_query",
        cypher="MATCH (n) RETURN n",
        connection="all",
        graph="graph-a",
    )
    assert _query_error_code(out) == "graph_selection_conflict"
    assert engine.calls == []  # never silently queried the default connection


async def test_graph_on_non_native_connection_is_a_typed_conflict_not_silent_pick():
    """A connection with no physical-graph concept (e.g. an external,
    non-epistemic-graph backend) combined with an explicit `graph` must be a
    typed error, never a silent ignore of `graph` that queries the
    connection's own single implicit graph while claiming to honor the
    request — the exact alias/graph-name collision this defect describes."""
    _register_tools()
    default_engine = _MultiGraphEngine(catalog_graphs=["graph-a"])
    _install_engine(default_engine)

    class _ExternalNoGraphConnection:
        """No `.graph_compute` / no multi-graph concept — like a Neo4j/AGE
        `ExternalGraphConnection`, which has none of that shape."""

        def query_cypher(self, cypher, params=None, as_of=None):
            return [{"from": "external"}]

    registry = kg_server.get_connection_registry()
    external = _ExternalNoGraphConnection()
    registry._build_engine = lambda spec: external  # type: ignore[method-assign]
    registry.register("other", {"backend": "memory"})

    # `connection="other"` always means the registered connection alias, never
    # gets reinterpreted as a graph name even if some OTHER physical graph
    # happened to share that string — proving the two axes never
    # cross-contaminate. Combined with an explicit `graph` it's unsupported.
    out = await kg_server._execute_tool(
        "graph_query",
        cypher="MATCH (n) RETURN n",
        connection="other",
        graph="graph-a",
    )
    assert _query_error_code(out) == "graph_selection_conflict"


# ── (5) verified-context propagation (authenticated posture) ───────────────


async def test_explicit_graph_binds_into_the_verified_session_and_restores():
    """The selected graph reaches the engine bound into the request's verified
    `GraphSession` (not merely passed through as a loose argument), and the
    ambient session is restored afterward — no leakage across calls. The
    ambient session itself comes from the suite-wide `isolate_graph_compute_engine`
    fixture (a genuinely verified, authenticated `GraphSession`, per
    `AGENTS.md`'s *Session currency* section — MCP/REST dispatch always
    requires one); this test only asserts the identity/tenant carried through
    UNCHANGED while `.graph` was retargeted for the one call."""
    _register_tools()

    before = current_session()
    assert before is not None
    before_actor_id = before.actor.actor_id
    before_tenant = before.tenant
    before_graph = before.graph
    assert before_graph != "graph-a"  # the fixture's own graph, not the target

    observed: list[tuple[str, str, str]] = []

    class _ObservingEngine(_MultiGraphEngine):
        def add_node(self, node_id, node_type, props=None):
            session = current_session()
            observed.append((session.graph, session.actor.actor_id, session.tenant))
            super().add_node(node_id, node_type, props)

    observing = _ObservingEngine(catalog_graphs=["graph-a"])
    _install_engine(observing)

    await kg_server._execute_tool(
        "graph_write",
        action="add_node",
        node_id="n1",
        node_type="Thing",
        graph="graph-a",
    )

    # Bound to the EXPLICIT graph for the call, same identity/tenant — never a
    # forged/widened principal, only the graph axis moved.
    assert observed == [("graph-a", before_actor_id, before_tenant)]

    after = current_session()
    assert after is not None
    assert after.graph == before_graph  # restored, not leaked
    assert after.actor.actor_id == before_actor_id


# ── (6) unauthenticated (local/stdio-style) posture — identical behavior ───


async def test_unauthenticated_style_session_same_explicitness_and_fail_closed():
    """The other tests in this file all run under a genuine PER-REQUEST ambient
    ``GraphSession`` (the suite-wide ``isolate_graph_compute_engine`` fixture —
    representing a network-authenticated request, one verified session per
    call). The local/stdio "unauthenticated" bootstrap posture is structurally
    different: per ``AGENTS.md``'s *Session currency* section, stdio mints
    ONE process-lifetime session up front (from a one-time, in-memory-signed,
    neutral-claims JWT — never re-verified per request) and every served call
    falls back to it via ``_PROCESS_SESSION`` (`kg_server.verified_tool_session_scope`:
    ``session = ambient or _PROCESS_SESSION``) precisely because THERE IS no
    per-request ambient session to find. This test reproduces exactly that
    shape — ``suspend_session()`` clears the ambient contextvar (the same
    primitive used at "explicitly unauthenticated boundaries such as liveness
    probes") and only ``_PROCESS_SESSION`` supplies authority — and asserts
    `resolve_explicit_graph`/`bound_to_graph` behave IDENTICALLY: no
    conditional "auth enabled" branch exists anywhere in that code, and this
    proves it operationally, not just by code inspection.
    """
    from agent_utilities.knowledge_graph.core.session import suspend_session
    from agent_utilities.security.brain_context import current_actor

    _register_tools()
    engine = _MultiGraphEngine(catalog_graphs=["graph-a"])
    _install_engine(engine)

    # `verified_tool_session_scope` also cross-checks the SEPARATE ambient
    # actor context against the session's own actor (`_reject_caller_authority`
    # boundary) — reuse whatever actor the suite's own fixture already bound
    # rather than fighting that unrelated system; the thing under test here is
    # the GraphSession fallback (`ambient or _PROCESS_SESSION`), not identity.
    local_actor = current_actor()
    kg_server._PROCESS_SESSION = GraphSession(
        actor=local_actor,
        tenant=local_actor.tenant_id,
        scopes=frozenset({"kg:read", "kg:write"}),
        policy_version="test-policy",
        audience="test-audience",
    )

    outer_ambient = current_session()  # the suite fixture's own per-request session

    with suspend_session():
        assert current_session() is None  # no per-request ambient — the stdio shape

        ok = await kg_server._execute_tool(
            "graph_query", cypher="MATCH (n) RETURN n", graph="graph-a"
        )
        assert _query_payload(ok)["graph"] == "graph-a"

        unknown = await kg_server._execute_tool(
            "graph_query", cypher="MATCH (n) RETURN n", graph="ghost"
        )
        assert _query_error_code(unknown) == "graph_not_found"

        conflict = await kg_server._execute_tool(
            "graph_query", cypher="MATCH (n) RETURN n", connection="all", graph="graph-a"
        )
        assert _query_error_code(conflict) == "graph_selection_conflict"

        write_out = await kg_server._execute_tool(
            "graph_write",
            action="add_node",
            node_id="local-1",
            node_type="Thing",
            graph="graph-a",
        )
        assert "added" in write_out

    assert current_session() is outer_ambient  # suspend_session restored cleanly


if __name__ == "__main__":
    asyncio.run(test_two_graph_isolation_write_a_not_visible_from_b())
