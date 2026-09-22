"""Public API exports the single verified AU session authority."""

from __future__ import annotations

import pytest

from agent_utilities import api
from agent_utilities.api import session
from agent_utilities.knowledge_graph.core import session as core_session
from agent_utilities.knowledge_graph.core.session import suspend_session
from agent_utilities.security.actor_identity import ActorType
from agent_utilities.security.brain_context import ActorContext


def test_public_session_exports_are_the_canonical_implementation() -> None:
    assert api.GraphSession is core_session.GraphSession
    assert api.resolve_session is core_session.resolve_session
    assert api.use_session is core_session.use_session
    assert session.GraphSession is core_session.GraphSession


def test_public_resolver_requires_the_ambient_verified_session() -> None:
    with suspend_session(), pytest.raises(api.SessionRequiredError):
        api.resolve_session()


def test_public_session_scope_binds_and_resolves_exact_authority() -> None:
    actor = ActorContext(
        actor_id="agent:api-session-test",
        actor_type=ActorType.AI_AGENT,
        tenant_id="tenant:api-session-test",
        authenticated=True,
    )
    bound = api.GraphSession(
        actor=actor,
        tenant="tenant:api-session-test",
        scopes=frozenset({"kg:read"}),
        policy_version="policy:test",
        audience="graph-os",
    )

    with api.use_session(bound):
        assert api.current_session() is bound
        assert api.resolve_session(bound, required_scope="kg:read") is bound
        with pytest.raises(api.ScopeError):
            api.resolve_session(bound, required_scope="kg:write")
