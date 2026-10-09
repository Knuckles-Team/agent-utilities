"""Pack-provisioning engine capabilities project only from their exact JWT role."""

from __future__ import annotations

from agent_utilities.security.brain_context import ActorContext
from agent_utilities.security.request_identity import _resolve_authenticated_scopes

PROVISIONING = {
    "connector:catalog-attest",
    "admin:connector-pack",
    "agent:pack-control",
    "security:admin",
}


def _actor(*roles: str) -> ActorContext:
    return ActorContext(actor_id="svc:graph-os", authenticated=True, roles=roles)


def test_provisioning_roles_project_exactly() -> None:
    scopes = _resolve_authenticated_scopes(_actor("kg:admin", *sorted(PROVISIONING)))

    assert PROVISIONING <= scopes
    assert {"kg:admin", "kg:read", "kg:write"} <= scopes


def test_no_other_role_implies_provisioning_scopes() -> None:
    scopes = _resolve_authenticated_scopes(
        _actor("kg:admin", "admin", "agent-services")
    )

    assert scopes.isdisjoint(PROVISIONING)
    assert scopes == frozenset({"kg:admin", "kg:read", "kg:write"})


def test_unlisted_roles_never_project() -> None:
    # "identity:admin" used to stand in for an unlisted role here, but it is a
    # genuine EG-registered scope (agent_utilities/security/scope_registry.py)
    # -- not a representative "unlisted" example. "bogus:unlisted" is not, and
    # is not a member of PROVISIONING either, so it exercises the same intent
    # without asserting against a real scope.
    scopes = _resolve_authenticated_scopes(_actor("kg:read", "bogus:unlisted", "x:y"))

    assert scopes == frozenset({"kg:read"})
