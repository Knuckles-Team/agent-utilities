"""ActorContextMiddleware bridges a validated JWT → current_actor (fleet-wide) +
GraphSession.

CONCEPT:AU-OS.identity.idp-agnostic-role-inheritance
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_utilities.knowledge_graph.core.session import current_session
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
    assert current_actor().authenticated is False


@pytest.mark.asyncio
@pytest.mark.concept("CONCEPT:AU-OS.identity.idp-agnostic-role-inheritance")
async def test_bridge_is_noop_without_claims():
    ambient = current_actor()

    async def call_next(_ctx):
        return current_actor()

    ctx = SimpleNamespace(auth=None)
    actor = await ActorContextMiddleware().on_call_tool(ctx, call_next)
    # No claims -> genuine no-op (no set_actor/reset_actor at all): whatever
    # actor was already ambient is untouched. Non-graph servers retain their
    # own boundary; graph-os rejects the later tool call because no verified
    # session was minted. This suite's autouse ``isolate_graph_compute_engine``
    # fixture (tests/conftest.py) binds an AUTHENTICATED ambient test actor for
    # every test, so "ambient" here is not the unauthenticated default a
    # production process with no configured auth would have — the no-op
    # invariant itself (``actor == ambient``) is what this test proves.
    assert actor == ambient
