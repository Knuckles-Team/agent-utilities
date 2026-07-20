"""One tenant-authorization authority for usage REST, MCP, and services."""

from __future__ import annotations

from dataclasses import dataclass

from agent_utilities.knowledge_graph.core.session import graph_session_required
from agent_utilities.security.brain_context import ActorContext, current_actor

__all__ = [
    "UsageAuthorizationError",
    "is_usage_admin",
    "require_usage_admin",
    "resolve_usage_tenant",
]

_USAGE_ADMIN_CAPABILITIES = frozenset(
    {"admin", "system", "usage:admin", "usage:read:any"}
)


@dataclass(frozen=True)
class UsageAuthorizationError(PermissionError):
    """Safe authorization failure carrying an HTTP-compatible status code."""

    detail: str
    status_code: int = 403

    def __str__(self) -> str:
        return self.detail


def is_usage_admin(actor: ActorContext) -> bool:
    actor.ensure_credential_current()
    return bool(_USAGE_ADMIN_CAPABILITIES.intersection(actor.roles))


def resolve_usage_tenant(requested: str | None = None) -> str | None:
    """Resolve a caller filter to the verified ambient tenant.

    Development mode remains compatible with an explicit tenant filter.  In
    the served profile, a verified identity and non-empty tenant claim are
    mandatory.  A non-admin caller can never select another tenant; an admin
    must still select it explicitly, so an omitted value is never an implicit
    all-tenant query.
    """

    actor = current_actor()
    if not graph_session_required():
        return requested or actor.tenant_id or None
    if not actor.authenticated:
        raise UsageAuthorizationError(
            "authenticated usage identity required", status_code=401
        )
    if not actor.tenant_id:
        raise UsageAuthorizationError("usage tenant claim required")
    if requested and requested != actor.tenant_id:
        if not is_usage_admin(actor):
            raise UsageAuthorizationError("cross-tenant usage access denied")
        return requested
    return actor.tenant_id


def require_usage_admin() -> None:
    """Require verified usage administration when enforcement is active."""

    if not graph_session_required():
        return
    actor = current_actor()
    if not actor.authenticated:
        raise UsageAuthorizationError(
            "authenticated usage identity required", status_code=401
        )
    if not is_usage_admin(actor):
        raise UsageAuthorizationError("usage administration required")
