"""ActorContextMiddleware bridges a validated JWT → current_actor (fleet-wide).

CONCEPT:AU-OS.identity.idp-agnostic-role-inheritance
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_utilities.mcp.middlewares import ActorContextMiddleware
from agent_utilities.security.brain_context import current_actor


class _Auth:
    def __init__(self, claims):
        self.claims = claims


@pytest.mark.asyncio
@pytest.mark.concept("CONCEPT:AU-OS.identity.idp-agnostic-role-inheritance")
async def test_bridge_scopes_call_to_okta_identity_and_resets():
    captured = {}

    async def call_next(_ctx):
        actor = current_actor()
        captured["actor"] = actor
        return "ok"

    ctx = SimpleNamespace(
        auth=_Auth({"sub": "user:ada", "groups": ["k8s:prod"], "email": "a@b.c"})
    )
    result = await ActorContextMiddleware().on_call_tool(ctx, call_next)

    assert result == "ok"
    actor = captured["actor"]
    assert actor.authenticated is True
    assert actor.actor_id == "user:ada"
    # Okta group folded into the capability set + retained as a raw group.
    assert "k8s:prod" in actor.roles
    assert actor.groups == ("k8s:prod",)
    # Contextvar restored after the call (no leakage).
    assert current_actor().authenticated is False


@pytest.mark.asyncio
@pytest.mark.concept("CONCEPT:AU-OS.identity.idp-agnostic-role-inheritance")
async def test_bridge_is_noop_without_claims():
    async def call_next(_ctx):
        return current_actor()

    ctx = SimpleNamespace(auth=None)
    actor = await ActorContextMiddleware().on_call_tool(ctx, call_next)
    # Unauthenticated → ambient SYSTEM_ACTOR unchanged (back-compat).
    assert actor.authenticated is False
