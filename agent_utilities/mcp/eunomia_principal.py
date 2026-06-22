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
from typing import TYPE_CHECKING, Any

from eunomia_core import schemas

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

try:  # eunomia_mcp is an optional dep (only installed where authz is enabled)
    from eunomia_mcp.middleware import EunomiaMcpMiddleware

    _EUNOMIA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the extra
    EunomiaMcpMiddleware = object  # type: ignore[assignment,misc]
    _EUNOMIA_AVAILABLE = False

logger = logging.getLogger(__name__)

# The remote Eunomia server enforces a hard per-request cap on ``POST /check/bulk``
# (HTTP 400 ``Too many requests. Maximum allowed: 100``). The third-party caller
# chain — ``EunomiaMcpMiddleware._authorize_listing`` → ``EunomiaBridge.bulk_check``
# → ``eunomia_sdk.EunomiaClient.bulk_check`` (``POST /check/bulk`` with the FULL
# list) — sends every component in one request, so a server exposing >100 tools
# (e.g. the plane-mcp ingestion surface fronted by the multiplexer) gets a 400 and
# the whole ``tools/list`` fails. Chunk under the cap and merge results.
# CONCEPT:ECO-4.88.
EUNOMIA_BULK_CHECK_MAX = 100


class _ChunkingBulkCheckBridge:
    """Wrap a ``EunomiaBridge`` so ``bulk_check`` batches under the server cap.

    Delegates every attribute to the wrapped bridge except ``bulk_check``, which it
    splits into chunks of at most ``max_batch`` (default 100, the remote server's
    hard cap) and concatenates the per-chunk responses in order. A response list is
    positionally aligned with its request list, so concatenating chunk responses in
    request order reproduces exactly the single-request result — a resource is
    allowed iff the (single) check covering it within its batch allowed it.
    """

    def __init__(self, bridge: Any, max_batch: int = EUNOMIA_BULK_CHECK_MAX) -> None:
        if max_batch < 1:
            raise ValueError("max_batch must be >= 1")
        self._bridge = bridge
        self._max_batch = max_batch

    def __getattr__(self, name: str) -> Any:
        # Anything that isn't ``bulk_check`` (``check``, ``mode``, ``_client`` …)
        # passes straight through to the real bridge.
        return getattr(self._bridge, name)

    async def bulk_check(
        self, requests: Sequence[schemas.CheckRequest]
    ) -> list[schemas.CheckResponse]:
        requests = list(requests)
        if len(requests) <= self._max_batch:
            return await self._bridge.bulk_check(requests)

        merged: list[schemas.CheckResponse] = []
        for start in range(0, len(requests), self._max_batch):
            chunk = requests[start : start + self._max_batch]
            merged.extend(await self._bridge.bulk_check(chunk))
        return merged


def apply_bulk_check_chunking(
    middleware: Any, max_batch: int = EUNOMIA_BULK_CHECK_MAX
) -> Any:
    """Make ``middleware`` chunk its ``/check/bulk`` calls under the server cap.

    Wraps the middleware's ``_eunomia`` bridge with :class:`_ChunkingBulkCheckBridge`
    so listing authorization (``_authorize_listing``) never exceeds the remote
    Eunomia server's 100-item limit. Idempotent and a no-op if the middleware has no
    ``_eunomia`` attribute (defensive against upstream refactors). Returns the same
    middleware for call-chaining.
    """
    bridge = getattr(middleware, "_eunomia", None)
    if bridge is None or isinstance(bridge, _ChunkingBulkCheckBridge):
        return middleware
    middleware._eunomia = _ChunkingBulkCheckBridge(bridge, max_batch=max_batch)
    return middleware


def apply_fastmcp_enabled_compat() -> None:
    """Give fastmcp 3.x components the ``enabled`` attribute eunomia-mcp expects.

    ``eunomia_mcp`` (0.3.x) gates every authorized call on ``component.enabled``
    (``EunomiaMcpMiddleware._authorize_execution``), an attribute that existed on
    fastmcp **2.x** tool/resource/prompt components. fastmcp **3.x** removed it —
    in 3.x there is no per-component disable flag, every *registered* component is
    live — so the access raises ``'FunctionTool' object has no attribute 'enabled'``
    and turns every tool call on a eunomia-enforced server into an internal error.

    Until eunomia-mcp targets fastmcp 3.x, expose the attribute it reads on the
    shared component base with the value 3.x semantics already guarantee (always
    enabled). Idempotent; a no-op on fastmcp 2.x (where the field exists) or when
    fastmcp is absent.
    """
    try:
        from fastmcp.utilities.components import FastMCPComponent
    except ImportError:  # pragma: no cover - fastmcp not installed
        return
    if "enabled" in getattr(FastMCPComponent, "model_fields", {}):
        return  # fastmcp 2.x already provides it as a real field
    if not hasattr(FastMCPComponent, "enabled"):
        FastMCPComponent.enabled = True  # type: ignore[attr-defined]


# Applied on import so the embedded (JWT) path below is covered; the remote path
# calls it explicitly from the server factory.
apply_fastmcp_enabled_compat()


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

    middleware = JwtPrincipalEunomiaMiddleware(
        mode=EunomiaMode.SERVER,
        eunomia_server=server,
        enable_audit_logging=enable_audit_logging,
    )
    return apply_bulk_check_chunking(middleware)
