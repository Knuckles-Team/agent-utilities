"""Tests for the unified GraphSession currency (workstream AU-P0-1)."""

from __future__ import annotations

import contextvars

import pytest

from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    ScopeError,
    current_session,
    use_session,
)
from agent_utilities.observability import correlation as corr
from agent_utilities.security.brain_context import (
    ActorContext,
    SYSTEM_ACTOR,
    use_actor,
)
from agent_utilities.models.company_brain import ActorType


def _fresh():
    """Run each assertion in an isolated contextvars context (no cross-test leakage)."""
    return contextvars.copy_context()


def test_graph_session_defaults_are_unscoped_like_today():
    session = GraphSession()
    assert session.actor == SYSTEM_ACTOR
    assert session.tenant == ""
    assert session.scopes == frozenset()
    assert session.graph == ""
    assert session.txn is None
    assert session.endpoint is None
    assert session.catalog_epoch is None


def test_graph_session_defaults_tenant_from_actor():
    actor = ActorContext(
        actor_id="agent:marketing",
        actor_type=ActorType.AI_AGENT,
        roles=("marketing",),
        tenant_id="acme",
    )
    session = GraphSession(actor=actor)
    assert session.tenant == "acme"


def test_from_ambient_carries_tenant_actor_and_trace():
    ctx = _fresh()

    def body():
        actor = ActorContext(
            actor_id="agent:acme",
            actor_type=ActorType.AI_AGENT,
            roles=("reader",),
            tenant_id="acme",
        )
        with use_actor(actor):
            cid = corr.ensure_correlation_id()
            session = GraphSession.from_ambient(graph="acme:__commons__")
            assert session.actor == actor
            assert session.tenant == "acme"
            assert session.graph == "acme:__commons__"
            assert session.trace_context == cid

    ctx.run(body)


def test_from_ambient_defaults_to_system_actor_with_no_ambient_scope():
    ctx = _fresh()

    def body():
        session = GraphSession.from_ambient()
        assert session.actor == SYSTEM_ACTOR
        assert session.tenant == ""
        assert session.trace_context  # a correlation id was ensured

    ctx.run(body)


def test_from_ambient_overrides_are_respected():
    ctx = _fresh()

    def body():
        session = GraphSession.from_ambient(
            graph="g1", policy_version="v7", scopes=frozenset({"kg:write"})
        )
        assert session.graph == "g1"
        assert session.policy_version == "v7"
        assert session.scopes == frozenset({"kg:write"})

    ctx.run(body)


def test_current_session_contextvar_propagation():
    ctx = _fresh()

    def body():
        assert current_session() is None
        session = GraphSession(tenant="acme")
        with use_session(session) as active:
            assert active is session
            assert current_session() is session
            # A copy of the currently-running context (e.g. handed to a
            # spawned asyncio Task) inherits the ambient session, exactly like
            # ActorContext's own use_actor/current_actor propagation.
            child_ctx = _fresh()

            def child():
                assert current_session() is session

            child_ctx.run(child)
        assert current_session() is None

    ctx.run(body)


def test_use_session_restores_previous_on_exit_and_exception():
    ctx = _fresh()

    def body():
        outer = GraphSession(tenant="outer")
        inner = GraphSession(tenant="inner")
        with use_session(outer):
            assert current_session().tenant == "outer"
            try:
                with use_session(inner):
                    assert current_session().tenant == "inner"
                    raise RuntimeError("boom")
            except RuntimeError:
                pass
            assert current_session().tenant == "outer"

    ctx.run(body)


def test_require_scope_raises_when_missing():
    actor = ActorContext(
        actor_id="agent:acme",
        actor_type=ActorType.AI_AGENT,
        roles=("reader",),
        tenant_id="acme",
    )
    session = GraphSession(actor=actor, scopes=frozenset({"kg:read"}))
    session.require_scope("kg:read")  # does not raise
    with pytest.raises(ScopeError):
        session.require_scope("kg:write")


def test_require_scope_admin_actor_is_exempt():
    session = GraphSession(actor=SYSTEM_ACTOR, scopes=frozenset())
    # SYSTEM_ACTOR carries the "admin" role — exempt from scope enforcement,
    # matching today's unscoped-by-default behaviour.
    session.require_scope("kg:write")


def test_with_graph_with_txn_with_scopes_are_immutable_copies():
    session = GraphSession()
    graphed = session.with_graph("tenant:g1")
    assert graphed.graph == "tenant:g1"
    assert session.graph == ""

    txned = session.with_txn(object())
    assert txned.txn is not None
    assert session.txn is None

    scoped = session.with_scopes("kg:read", "kg:write")
    assert scoped.scopes == frozenset({"kg:read", "kg:write"})
    assert session.scopes == frozenset()


def test_with_actor_updates_tenant():
    session = GraphSession(tenant="old")
    actor = ActorContext(actor_id="a", actor_type=ActorType.AI_AGENT, tenant_id="new")
    rescoped = session.with_actor(actor)
    assert rescoped.actor == actor
    assert rescoped.tenant == "new"
