"""JWT-principal Eunomia authorization for the central MCP multiplexer.

The stock ``eunomia_mcp`` middleware derives the authorization principal from the
``x-agent-id`` request **header** (``agent:<x-agent-id>``). On a shared, internet-
adjacent gateway that header is **client-supplied and spoofable** — any caller
could claim ``x-agent-id: claude-code`` and inherit its policy. That defeats
zero-trust.

This module binds the principal to the **cryptographically verified JWT** instead:
the ``JWTVerifier`` (plan Phase 2) has already validated the Bearer token by the
time middleware runs, so we read the verified ``client_id`` (Keycloak ``azp``) via
``get_access_token()`` and use ``agent:<client_id>`` as the principal — overriding
any header a caller tries to inject. With no valid token the principal stays the
header/``unknown`` value, which a ``default_effect: deny`` policy rejects.

Evaluation runs **embedded** (in-process PDP over a local policy file) so the auth
hot path has no dependency on a remote Eunomia server being reachable — an outage
can never lock the gateway out (plan Phase 3 robustness). CONCEPT:ECO-4.36.
"""

from __future__ import annotations

import logging

from eunomia_core import schemas

try:  # eunomia_mcp is an optional dep (only installed where authz is enabled)
    from eunomia_mcp.middleware import EunomiaMcpMiddleware

    _EUNOMIA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the extra
    EunomiaMcpMiddleware = object  # type: ignore[assignment,misc]
    _EUNOMIA_AVAILABLE = False

logger = logging.getLogger(__name__)


class JwtPrincipalEunomiaMiddleware(EunomiaMcpMiddleware):  # type: ignore[valid-type,misc]
    """Eunomia middleware whose principal is the verified JWT, not a header."""

    def _extract_principal(self) -> schemas.PrincipalCheck:
        # Start from the stock extraction (captures user_agent / api_key / header
        # fallback), then override the identity with the verified token if present.
        principal = super()._extract_principal()

        client_id = None
        try:
            from fastmcp.server.dependencies import get_access_token

            token = get_access_token()
            if token is not None:
                client_id = getattr(token, "client_id", None)
        except Exception:  # no auth context (e.g. local stdio) → header fallback
            token = None

        if client_id:
            principal.uri = f"agent:{client_id}"
            principal.attributes["agent_id"] = client_id
            principal.attributes["jwt_verified"] = True
        return principal


def create_jwt_eunomia_middleware(
    policy_file: str,
    enable_audit_logging: bool = True,
) -> JwtPrincipalEunomiaMiddleware:
    """Build an embedded (in-process) JWT-principal Eunomia middleware.

    Mirrors ``eunomia_mcp.create_eunomia_middleware``'s embedded setup (no SQL
    persistence, no dynamic fetchers, policy loaded from ``policy_file``) but
    returns the JWT-principal subclass.
    """
    if not _EUNOMIA_AVAILABLE:
        raise ImportError(
            "eunomia_mcp is required for JWT-principal authorization "
            "(pip install 'eunomia-mcp')."
        )

    from eunomia.config import settings
    from eunomia.server import EunomiaServer
    from eunomia_mcp.bridge import EunomiaMode
    from eunomia_mcp.utils import load_policy_config

    # Embedded PDP: in-process evaluation, no external dependency.
    settings.ENGINE_SQL_DATABASE = False
    settings.FETCHERS = {}
    server = EunomiaServer()
    server.engine.add_policy(load_policy_config(policy_file))

    return JwtPrincipalEunomiaMiddleware(
        mode=EunomiaMode.SERVER,
        eunomia_server=server,
        enable_audit_logging=enable_audit_logging,
    )
