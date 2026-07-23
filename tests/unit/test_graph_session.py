"""Current-contract tests for verified graph-session authority."""

from __future__ import annotations

import contextvars
import time

import pytest

from agent_utilities.knowledge_graph.core.session import (
    GraphSession,
    ScopeError,
    SessionExpiredError,
    SessionRequiredError,
    current_session,
    graph_session_required,
    resolve_session,
    use_session,
)
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext


def _actor(
    *,
    actor_id: str = "principal:verified",
    tenant: str = "tenant-a",
    roles: tuple[str, ...] = ("kg:read",),
    authenticated: bool = True,
) -> ActorContext:
    return ActorContext(
        actor_id=actor_id,
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=roles,
        tenant_id=tenant,
        authenticated=authenticated,
    )


def _session(**overrides) -> GraphSession:
    values = {
        "actor": _actor(),
        "tenant": "tenant-a",
        "scopes": frozenset({"kg:read"}),
        "graph": "tenant-a-graph",
        "policy_version": "policy-1",
        "audience": "agent-services",
    }
    values.update(overrides)
    return GraphSession(**values)


def _fresh():
    """Run each assertion in an isolated contextvars context (no cross-test leakage)."""
    return contextvars.copy_context()


def test_graph_session_enforcement_is_invariant():
    assert graph_session_required() is True


@pytest.mark.parametrize(
    "actor,tenant",
    [
        (_actor(authenticated=False), "tenant-a"),
        (_actor(actor_id=""), "tenant-a"),
        (_actor(tenant=""), "tenant-a"),
        (_actor(tenant="tenant-b"), "tenant-a"),
        (_actor(), ""),
    ],
)
def test_session_construction_rejects_unverified_or_mismatched_authority(actor, tenant):
    with pytest.raises(SessionRequiredError):
        GraphSession(actor=actor, tenant=tenant)


def test_no_ambient_session_is_never_synthesized():
    def isolated():
        assert current_session() is None
        with pytest.raises(SessionRequiredError, match="verified ambient"):
            GraphSession.from_ambient()
        with pytest.raises(SessionRequiredError, match="verified ambient"):
            resolve_session()

    contextvars.Context().run(isolated)


def test_bound_session_is_the_only_resolvable_authority():
    verified = _session()
    with use_session(verified):
        assert GraphSession.from_ambient() is verified
        assert resolve_session() is verified


def test_current_session_contextvar_propagation():
    # A genuinely blank context (not ``_fresh()``/copy_context()) — the suite's
    # own autouse ``isolate_graph_compute_engine`` fixture binds a verified
    # ambient session for every test, so a *copy* of the running context would
    # inherit it; this test's starting-state assertion needs a context with no
    # contextvar values at all, matching ``test_graph_session_enforcement_is_invariant``.
    ctx = contextvars.Context()

    def body():
        assert current_session() is None
        session = _session()
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
        outer = _session()
        inner = _session()
        with use_session(outer):
            assert current_session() is outer
            try:
                with use_session(inner):
                    assert current_session() is inner
                    raise RuntimeError("boom")
            except RuntimeError:
                pass
            assert current_session() is outer

    ctx.run(body)


def test_expired_bearer_session_fails_every_runtime_authority_boundary():
    expired = _session(
        actor=ActorContext(
            actor_id="principal:verified",
            actor_type=ActorType.AUTOMATED_SERVICE,
            roles=("kg:read",),
            tenant_id="tenant-a",
            authenticated=True,
            credential_expires_at=int(time.time()) - 1,
        )
    )
    with pytest.raises(SessionExpiredError):
        expired.engine_verified_context()
    with pytest.raises(SessionExpiredError):
        use_session(expired).__enter__()


def test_supplied_authority_must_match_ambient_except_transaction():
    verified = _session()
    transaction = object()
    with use_session(verified):
        assert resolve_session(verified.with_txn(transaction)).txn is transaction
        with pytest.raises(SessionRequiredError, match="does not match"):
            resolve_session(_session(scopes=frozenset({"kg:admin"})))
        with pytest.raises(SessionRequiredError, match="retarget"):
            resolve_session(graph="other-graph")


def test_route_and_transaction_helpers_do_not_widen_authority():
    original = _session()
    routed = original.with_route(
        endpoint="transport-internal",
        placement_group=3,
        catalog_epoch=17,
    )
    assert routed.actor == original.actor
    assert routed.tenant == original.tenant
    assert routed.scopes == original.scopes
    assert routed.graph == original.graph
    assert routed.endpoint == "transport-internal"
    assert routed.placement_group == 3
    assert routed.catalog_epoch == 17


def test_scope_hierarchy_requires_explicit_scope_not_role():
    writer = _session(scopes=frozenset({"kg:write"}))
    writer.require_scope("kg:read")
    writer.require_scope("kg:write")
    with pytest.raises(ScopeError):
        writer.require_scope("kg:admin")

    role_only_admin = _session(actor=_actor(roles=("admin",)), scopes=frozenset())
    with pytest.raises(ScopeError):
        role_only_admin.require_scope("kg:admin")

    administrator = _session(scopes=frozenset({"kg:admin"}))
    administrator.require_scope("kg:read")
    administrator.require_scope("kg:write")
    administrator.require_scope("kg:admin")


def test_fine_grained_scopes_remain_exact():
    session = _session(scopes=frozenset({"node:read"}))
    session.require_scope("node:read")
    with pytest.raises(ScopeError):
        session.require_scope("node:write")


def test_engine_context_is_verified_and_contains_no_local_metadata():
    session = _session(
        actor=_actor(actor_id="principal:opaque-123", roles=("kg:write",)),
        scopes=frozenset({"kg:read", "kg:write"}),
        endpoint="transport-internal",
        trace_context="trace-opaque",
    )
    context = session.engine_verified_context()
    assert set(context) == {
        "principal",
        "tenant",
        "audience",
        "agent_id",
        "roles",
        "scopes",
        "delegation",
        "policy_version",
    }
    assert context["principal"] == "principal:opaque-123"
    assert "endpoint" not in context
    assert "graph" not in context
    assert "path" not in context
    assert "display_name" not in context
    assert "traceparent" not in context


@pytest.mark.parametrize("field", ["audience", "policy_version"])
def test_engine_context_rejects_incomplete_verified_authority(field):
    with pytest.raises(SessionRequiredError):
        _session(**{field: ""}).engine_verified_context()
