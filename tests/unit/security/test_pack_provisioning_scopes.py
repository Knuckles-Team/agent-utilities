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
    scopes = _resolve_authenticated_scopes(_actor("kg:read", "identity:admin", "x:y"))

    assert scopes == frozenset({"kg:read"})
