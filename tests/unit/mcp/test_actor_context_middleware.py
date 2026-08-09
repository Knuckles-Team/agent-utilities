"""ActorContextMiddleware bridges a validated JWT → current_actor (fleet-wide) +
GraphSession, and (``require_verified_session=True``, graph-os only) fails
closed instead of proceeding unauthenticated (BUG-036/GOC-15).

CONCEPT:AU-OS.identity.idp-agnostic-role-inheritance
CONCEPT:AU-KG.identity.verified-carrier-required-federation
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_utilities.knowledge_graph.core.session import (
    SessionRequiredError,
    current_session,
)
from agent_utilities.mcp.middlewares import ActorContextMiddleware
from agent_utilities.security.brain_context import current_actor


class _Auth:
    def __init__(self, claims):
        self.claims = claims


@pytest.mark.asyncio
@pytest.mark.concept("CONCEPT:AU-OS.identity.idp-agnostic-role-inheritance")
async def test_bridge_scopes_call_to_okta_identity_and_resets(monkeypatch):
    from agent_utilities.core.config import config

    monkeypatch.setattr(config, "auth_jwt_audience", "agent-services")
    monkeypatch.setattr(config, "kg_policy_version", "policy-v1")
    captured = {}
    ambient = current_actor()

    async def call_next(_ctx):
        actor = current_actor()
        captured["actor"] = actor
        captured["session"] = current_session()
        return "ok"

    ctx = SimpleNamespace(
        auth=_Auth(
            {
                "sub": "principal:verified",
                "tenant_id": "tenant-a",
                "groups": ["k8s:prod"],
                "email": "a@b.c",
            }
        )
    )
    result = await ActorContextMiddleware().on_call_tool(ctx, call_next)

    assert result == "ok"
    actor = captured["actor"]
    assert actor.authenticated is True
    assert actor.actor_id == "principal:verified"
    # Okta group folded into the capability set + retained as a raw group.
    assert "k8s:prod" in actor.roles
    assert actor.groups == ("k8s:prod",)
    assert captured["session"].tenant == "tenant-a"
    assert captured["session"].audience == "agent-services"
    assert captured["session"].policy_version == "policy-v1"
    # Contextvar restored after the call (no leakage).
    assert current_actor() == ambient
    # The request's principal specifically did not leak into the ambient actor.
    # (This used to assert `authenticated is False`, which encoded an assumption
    # about the *suite's* ambient actor rather than about leakage; the suite
    # fixture now binds an authenticated service principal. The assertion was
    # never executed before D-SP-1 because minting reached the engine, so the
    # whole test skipped in an engine-less environment.)
    assert current_actor().actor_id != "principal:verified"


@pytest.mark.asyncio
@pytest.mark.concept("CONCEPT:AU-OS.identity.idp-agnostic-role-inheritance")
async def test_bridge_is_noop_without_claims_by_default():
    """Default (``require_verified_session=False``) mode: every non-graph-os
    fleet server. Unchanged from before BUG-036's fix — this middleware is
    opportunistic identity *enrichment* for those ~60 independently-owned
    packages, not a gate; each keeps its own authorization contract.
    """
    ambient = current_actor()

    async def call_next(_ctx):
        return current_actor()

    ctx = SimpleNamespace(auth=None)
    actor = await ActorContextMiddleware().on_call_tool(ctx, call_next)
    # No claims -> genuine no-op (no set_actor/reset_actor at all): whatever
    # actor was already ambient is untouched. This suite's autouse
    # ``isolate_graph_compute_engine`` fixture (tests/conftest.py) binds an
    # AUTHENTICATED ambient test actor for every test, so "ambient" here is
    # not the unauthenticated default a production process with no configured
    # auth would have — the no-op invariant itself (``actor == ambient``) is
    # what this test proves.
    assert actor == ambient


def test_bridge_fails_closed_without_claims_when_verified_session_required():
    """BUG-036/GOC-15: graph-os's mode (``require_verified_session=True``)
    refuses an unauthenticated call outright instead of silently proceeding —
    this is the exact regression the federation reader's fail-open (BUG-036)
    exposed: the MCP-native dispatch path (FastMCP -> this middleware -> the
    ``@mcp.tool``-decorated function) had nothing else guarding it.

    Proven against a genuinely empty context (no ambient GraphSession/actor at
    all) via the same ``contextvars.Context().run(...)`` technique the
    committed federation carrier-authority test
    (``tests/unit/knowledge_graph/test_federation_carrier_authority.py``) uses
    to escape this suite's autouse session-binding fixture — this is the shape
    a real unauthenticated MCP-native tool call reaches this middleware in: no
    claims, no prior ambient authority, no local process bootstrap.
    """
    import asyncio
    import contextvars

    called = {"count": 0}

    async def call_next(_ctx):
        called["count"] += 1
        return "should not run"

    ctx = SimpleNamespace(auth=None)

    def isolated() -> None:
        assert current_session() is None, "test precondition: no ambient session"
        middleware = ActorContextMiddleware(require_verified_session=True)
        with pytest.raises(SessionRequiredError):
            asyncio.run(middleware.on_call_tool(ctx, call_next))

    contextvars.Context().run(isolated)
    assert called["count"] == 0, (
        "BUG-036 regression: the tool body ran despite no verified caller "
        "identity being bound anywhere in the call"
    )


def test_bridge_exempts_bound_local_process_authority():
    """The ONE narrow, named exemption in ``require_verified_session=True``
    mode: the tiny-profile local stdio process bootstrap.

    stdio has no per-call bearer token by protocol —
    ``kg_server._PROCESS_SESSION`` (a genuine, already-verified,
    process-lifetime ``GraphSession`` minted by
    :func:`~agent_utilities.security.request_identity.mint_local_process_session`
    at server start) IS that transport's carrier, not an absence of one, so a
    claim-less call must still proceed when it is bound.
    """
    import asyncio
    import contextvars

    from agent_utilities.knowledge_graph.core.session import GraphSession
    from agent_utilities.models.company_brain import ActorType
    from agent_utilities.security.brain_context import ActorContext

    process_actor = ActorContext(
        actor_id="graph-os:local-process",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("kg:admin",),
        tenant_id="local",
        authenticated=True,
    )
    process_session = GraphSession(
        actor=process_actor,
        tenant="local",
        scopes=frozenset({"kg:admin"}),
        graph="",
        policy_version="local-ephemeral-v1",
        audience="graph-os-local",
    )

    async def call_next(_ctx):
        return "ok"

    ctx = SimpleNamespace(auth=None)

    def isolated() -> None:
        assert current_session() is None, "test precondition: no ambient session"
        from agent_utilities.mcp import kg_server

        old = kg_server._PROCESS_SESSION
        kg_server._PROCESS_SESSION = process_session
        try:
            middleware = ActorContextMiddleware(require_verified_session=True)
            result = asyncio.run(middleware.on_call_tool(ctx, call_next))
        finally:
            kg_server._PROCESS_SESSION = old
        assert result == "ok"

    contextvars.Context().run(isolated)


def test_bridge_exempts_already_ambient_session():
    """The second exemption: an authority already bound ambiently (a nested
    call under an already-authenticated caller) must not be re-demanded.
    """
    import asyncio
    import contextvars

    from agent_utilities.knowledge_graph.core.session import (
        GraphSession,
        reset_session,
        set_session,
    )
    from agent_utilities.models.company_brain import ActorType
    from agent_utilities.security.brain_context import ActorContext

    actor = ActorContext(
        actor_id="nested:caller",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=("kg:read",),
        tenant_id="tenant-a",
        authenticated=True,
    )
    session = GraphSession(
        actor=actor,
        tenant="tenant-a",
        scopes=frozenset({"kg:read"}),
        graph="",
        policy_version="policy-v1",
        audience="agent-services",
    )

    async def call_next(_ctx):
        return "ok"

    ctx = SimpleNamespace(auth=None)

    def isolated() -> None:
        assert current_session() is None, "test precondition: no ambient session"
        token = set_session(session)
        try:
            middleware = ActorContextMiddleware(require_verified_session=True)
            result = asyncio.run(middleware.on_call_tool(ctx, call_next))
        finally:
            reset_session(token)
        assert result == "ok"

    contextvars.Context().run(isolated)
