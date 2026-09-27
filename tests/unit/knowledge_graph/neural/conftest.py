"""Verified neural test authority; production callers receive this from middleware."""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.security.brain_context import ActorContext, ActorType


@pytest.fixture(autouse=True)
def neural_session():
    session = GraphSession(
        actor=ActorContext(
            actor_id="neural-test-reviewer",
            actor_type=ActorType.AUTOMATED_SERVICE,
            tenant_id="acme",
            authenticated=True,
        ),
        tenant="acme",
        scopes=frozenset({"kg:read", "kg:write"}),
        graph="tenant-acme",
    )
    with use_session(session):
        yield session
