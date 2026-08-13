"""GOC-67 (R-23): low-level engine-tool graph narrowing.

Root cause: ``engine_tools._dispatch`` built ``_client_for(graph)`` -- a
client VIEW fixed to the caller's explicit ``graph`` -- while the ambient
verified ``GraphSession`` still named whatever graph the calling boundary
authenticated it under (typically the deployment default). The mismatch
between a client fixed to ``graph`` and a session naming something else is
correctly rejected deep in the wire layer
(``_SessionRoutedAsyncClient._send``: "A graph-scoped view cannot retarget
the verified GraphSession") -- but only AFTER the client was already built,
so a legitimate, RBAC-authorized ``graph=`` selection on ``POST
/engine/nodes`` (or the equivalent MCP call) always failed with
``PermissionError`` before the engine's own RBAC/RLS ever evaluated it.

The fix narrows the SESSION first (via ``kg_server.bound_to_graph`` -- the
exact ``GraphSession.with_graph()``/``use_session()`` primitive
``query_tools.py``'s ``resolve_explicit_graph``/``bound_to_graph`` callers
already use) and re-enters ``_dispatch`` once. This file proves: the
narrowing actually happens before the client is constructed, recursion
terminates after EXACTLY one narrowing, the exact
``engine_tenants(list)`` -> ``engine_nodes(list_by_label, graph=...)``
scenario from the acceptance brief succeeds, identity/tenant/policy
authority is preserved unchanged through the narrowing (only ``.graph``
moves), and the behavior is identical whether the ambient session came from
a per-request authenticated context or the local/stdio "unauthenticated"
process-session fallback.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    current_session,
    use_session,
)
from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools import engine_tools
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor

ADMIN_ACTOR = ActorContext(
    actor_id="principal:ops",
    actor_type=ActorType.AI_AGENT,
    roles=(),
    tenant_id="acme",
    authenticated=True,
)

READER_ACTOR = ActorContext(
    actor_id="principal:reader",
    actor_type=ActorType.AI_AGENT,
    roles=(),
    tenant_id="acme",
    authenticated=True,
)


def _session(actor: ActorContext, graph: str, *scopes: str) -> GraphSession:
    return GraphSession(
        actor=actor,
        tenant=actor.tenant_id,
        scopes=frozenset(scopes),
        graph=graph,
        policy_version="policy-v1",
        audience="agent-services",
    )


def _observing_client():
    """A fake ``SyncEpistemicGraphClient`` recording the AMBIENT session's
    ``.graph`` at the moment each sub-client method actually runs -- proves
    the session was already narrowed by the time the client is used, not
    merely by the time it's constructed."""
    observed: list[tuple[str, str, str]] = []  # (domain, action, session.graph)

    def _sub(domain: str):
        def _make(name):
            def _call(**kwargs):
                session = current_session()
                observed.append((domain, name, session.graph if session else ""))
                return {"ok": True, "domain": domain, "method": name, "kwargs": kwargs}

            return _call

        class _Sub:
            def __getattr__(self, name):
                return _make(name)

        return _Sub()

    class _Client:
        def __getattr__(self, name):
            return _sub(name)

    return _Client(), observed


@pytest.fixture(autouse=True)
def _fresh_client_pool(monkeypatch):
    monkeypatch.setattr(engine_tools, "_CLIENT_POOL", None)
    yield
    monkeypatch.setattr(engine_tools, "_CLIENT_POOL", None)


# ── (1) narrowing happens before the client is constructed ─────────────────


def test_explicit_graph_mismatch_narrows_session_before_client_construction(
    monkeypatch,
):
    kg_server.ensure_tools_registered()
    client, observed = _observing_client()

    client_for_calls: list[tuple[str, str]] = []

    def _client_for(graph: str):
        session = current_session()
        client_for_calls.append((graph, session.graph if session else ""))
        return client

    monkeypatch.setattr(engine_tools, "_client_for", _client_for)

    session = _session(READER_ACTOR, "tenant-acme-graph", "kg:read", "kg:write")
    with use_actor(READER_ACTOR), use_session(session):
        out = json.loads(
            engine_tools._dispatch(
                "nodes", {"has"}, "has", json.dumps({"node_id": "n1"}), "tenant-other-graph"
            )
        )
        # The ambient session is restored after `_dispatch` returns -- no leak.
        assert current_session().graph == "tenant-acme-graph"

    assert out["ok"] is True
    # `_client_for` was invoked exactly once, and by the time it ran the
    # AMBIENT session had already been narrowed to match the requested graph
    # -- proving the narrowing happened strictly BEFORE client construction.
    assert client_for_calls == [("tenant-other-graph", "tenant-other-graph")]
    assert observed == [("nodes", "has", "tenant-other-graph")]


def test_no_narrowing_when_graph_matches_or_is_empty(monkeypatch):
    kg_server.ensure_tools_registered()
    client, observed = _observing_client()

    client_for_calls: list[str] = []

    def _client_for(graph: str):
        client_for_calls.append(graph)
        return client

    monkeypatch.setattr(engine_tools, "_client_for", _client_for)
    session = _session(READER_ACTOR, "tenant-acme-graph", "kg:read")

    with use_actor(READER_ACTOR), use_session(session):
        # graph == session.graph: no narrowing needed.
        engine_tools._dispatch("nodes", {"has"}, "has", "{}", "tenant-acme-graph")
        # graph == "": no narrowing needed either.
        engine_tools._dispatch("nodes", {"has"}, "has", "{}", "")

    assert client_for_calls == ["tenant-acme-graph", ""]
    assert [g for _, _, g in observed] == ["tenant-acme-graph", "tenant-acme-graph"]


# ── (2) recursion terminates after EXACTLY one narrowing ───────────────────


def test_recursion_terminates_after_exactly_one_narrowing(monkeypatch):
    kg_server.ensure_tools_registered()
    client, _observed = _observing_client()
    monkeypatch.setattr(engine_tools, "_client_for", lambda graph: client)

    original_dispatch = engine_tools._dispatch
    call_count = {"n": 0}
    seen_graphs: list[str] = []

    def counting_dispatch(domain, methods, action, params_json, graph):
        call_count["n"] += 1
        session = current_session()
        seen_graphs.append(session.graph if session else "")
        return original_dispatch(domain, methods, action, params_json, graph)

    # `_dispatch`'s own recursive self-call is a module-global name lookup
    # (`return _dispatch(...)` inside its own body), so patching the module
    # attribute is observed on the recursive re-entry too -- this is what
    # lets the counter see BOTH the initial call and the one recursion.
    monkeypatch.setattr(engine_tools, "_dispatch", counting_dispatch)

    session = _session(READER_ACTOR, "tenant-acme-graph", "kg:read")
    with use_actor(READER_ACTOR), use_session(session):
        out = json.loads(
            counting_dispatch("nodes", {"has"}, "has", '{"node_id": "n1"}', "tenant-other-graph")
        )

    assert out["ok"] is True
    # Exactly two invocations: the original call (session still mismatched)
    # plus ONE recursive re-entry (session now narrowed, mismatch branch
    # skipped, proceeds to construct the client). A structural regression
    # that recursed unboundedly would never reach this assertion (or would
    # blow the stack first); a regression that dropped the narrowing
    # entirely would leave `call_count == 1` and `out["ok"]` would never be
    # reached because the underlying wire layer would raise instead.
    assert call_count["n"] == 2
    assert seen_graphs == ["tenant-acme-graph", "tenant-other-graph"]


# ── (3) the acceptance brief's exact scenario ───────────────────────────────


def test_engine_tenants_list_then_engine_nodes_list_by_label_succeeds(monkeypatch):
    """``engine_tenants(list)`` (ADMIN domain) discovers a graph; a
    subsequent ``engine_nodes(list_by_label, graph=<listed-graph>)`` (a
    normal domain) against a session still naming the deployment default
    must succeed via the one-step narrowing -- not raise ``PermissionError``
    for a graph the caller just legitimately discovered."""
    kg_server.ensure_tools_registered()

    listed_graphs = ["tenant-acme-graph", "tenant-other-graph"]

    def _sub(domain: str):
        def _make(name):
            def _call(**kwargs):
                if domain == "tenants" and name == "list":
                    return [{"name": g} for g in listed_graphs]
                if domain == "nodes" and name == "list_by_label":
                    session = current_session()
                    return {
                        "label": kwargs.get("label"),
                        "graph": session.graph if session else "",
                        "nodes": ["n1", "n2"],
                    }
                raise AssertionError(f"unexpected call {domain}.{name}({kwargs})")

            return _call

        class _Sub:
            def __getattr__(self, name):
                return _make(name)

        return _Sub()

    class _Client:
        def __getattr__(self, name):
            return _sub(name)

    client = _Client()
    monkeypatch.setattr(engine_tools, "_client_for", lambda graph: client)

    admin_session = _session(ADMIN_ACTOR, "tenant-acme-graph", "kg:admin", "kg:read", "kg:write")

    with use_actor(ADMIN_ACTOR), use_session(admin_session):
        tenants_out = json.loads(
            engine_tools._dispatch("tenants", {"list"}, "list", "{}", "")
        )
        graphs = [row["name"] for row in tenants_out]
        assert graphs == listed_graphs

        # Ambient session still names the DEFAULT graph -- unchanged.
        assert current_session().graph == "tenant-acme-graph"

        # Select the OTHER listed graph -- the mismatch case this defect covers.
        target_graph = "tenant-other-graph"
        assert target_graph != current_session().graph

        nodes_out = json.loads(
            engine_tools._dispatch(
                "nodes",
                {"list_by_label"},
                "list_by_label",
                json.dumps({"label": "Thing"}),
                target_graph,
            )
        )
        assert nodes_out["graph"] == target_graph
        assert nodes_out["nodes"] == ["n1", "n2"]

        # Restored after the call.
        assert current_session().graph == "tenant-acme-graph"


# ── (4) identity/tenant/policy preserved through narrowing (incl. redirects) ──


def test_verified_identity_preserved_through_narrowing(monkeypatch):
    """Only ``.graph`` moves during narrowing -- actor, tenant, scopes, and
    policy_version are carried through unchanged (``with_graph`` is a
    ``dataclasses.replace`` of ONE field)."""
    kg_server.ensure_tools_registered()
    client, observed_calls = _observing_client()
    monkeypatch.setattr(engine_tools, "_client_for", lambda graph: client)

    session = _session(READER_ACTOR, "tenant-acme-graph", "kg:read")
    observed_sessions: list[GraphSession] = []

    real_bound_to_graph = kg_server.bound_to_graph

    def _spy_bound_to_graph(graph):
        cm = real_bound_to_graph(graph)

        class _Spy:
            def __enter__(self_inner):
                cm.__enter__()
                observed_sessions.append(current_session())
                return None

            def __exit__(self_inner, *exc):
                return cm.__exit__(*exc)

        return _Spy()

    monkeypatch.setattr(kg_server, "bound_to_graph", _spy_bound_to_graph)

    with use_actor(READER_ACTOR), use_session(session):
        engine_tools._dispatch("nodes", {"has"}, "has", '{"node_id": "n1"}', "tenant-other-graph")

    assert len(observed_sessions) == 1
    narrowed = observed_sessions[0]
    assert narrowed.graph == "tenant-other-graph"
    assert narrowed.actor.actor_id == session.actor.actor_id
    assert narrowed.tenant == session.tenant
    assert narrowed.scopes == session.scopes
    assert narrowed.policy_version == session.policy_version
    assert narrowed.audience == session.audience

    # `engine_verified_context()` -- the claims payload threaded to BOTH the
    # local RPC branch and a redirected-connection RPC branch inside the
    # wire layer's `_invoke_at` (`graph_compute.py`) -- carries no `graph`
    # claim at all, so it is byte-for-byte identical before and after
    # narrowing. Whichever physical placement/redirect the wire layer
    # chooses for this call, the SAME verified identity is what it signs.
    assert narrowed.engine_verified_context() == session.engine_verified_context()


# ── (5) identical behavior in the authenticated vs. unauthenticated posture ──


def test_narrowing_behaves_identically_under_stdio_process_session(monkeypatch):
    """Mirrors ``test_graph_explicit_selection.py``'s unauthenticated-posture
    proof: the local/stdio bootstrap has no per-request ambient session, only
    the process-lifetime ``_PROCESS_SESSION`` fallback
    (``kg_server.verified_tool_session_scope``:
    ``session = ambient or _PROCESS_SESSION``). Narrowing must work
    identically there -- no auth-posture-conditional branch anywhere in
    ``_dispatch``."""
    from agent_utilities.knowledge_graph.core.session import suspend_session

    kg_server.ensure_tools_registered()
    client, observed = _observing_client()
    monkeypatch.setattr(engine_tools, "_client_for", lambda graph: client)

    saved_registry = kg_server._CONNECTION_REGISTRY
    saved_process_session = kg_server._PROCESS_SESSION
    kg_server._CONNECTION_REGISTRY = None
    kg_server._PROCESS_SESSION = _session(
        READER_ACTOR, "tenant-acme-graph", "kg:read", "kg:write"
    )
    try:
        with use_actor(READER_ACTOR), suspend_session():
            assert current_session() is None  # the stdio shape: no per-request ambient

            out = asyncio.run(
                kg_server._execute_tool(
                    "engine_nodes",
                    action="has",
                    params_json=json.dumps({"node_id": "n1"}),
                    graph="tenant-other-graph",
                )
            )
            payload = json.loads(out)
            assert payload["ok"] is True
            assert current_session() is None  # restored, no leak into the caller
    finally:
        kg_server._CONNECTION_REGISTRY = saved_registry
        kg_server._PROCESS_SESSION = saved_process_session

    assert observed == [("nodes", "has", "tenant-other-graph")]
