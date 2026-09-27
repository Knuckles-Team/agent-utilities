"""Verified tenant authority for governed neural graph operations."""

from __future__ import annotations

from agent_utilities.knowledge_graph.core.session import resolve_session


def require_neural_tenant(tenant: str, *, write: bool) -> str:
    """Bind a neural operation to the server-verified graph session.

    The tenant argument is a consistency assertion, never an authority source.
    A graph must be explicit so an operation cannot fall into a default graph.
    """
    session = resolve_session(required_scope="kg:write" if write else "kg:read")
    if not session.graph or not session.tenant:
        raise PermissionError("neural graph operations require a tenant graph")
    if tenant != session.tenant:
        raise PermissionError("neural tenant does not match GraphSession")
    return session.tenant
