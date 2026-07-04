#!/usr/bin/python
"""OIDC client-credentials token provider for the MCP multiplexer (CONCEPT:AU-OS.identity.so-jwt-protected-children).

The multiplexer aggregates many child MCP servers. When a child enforces JWT auth
(``AUTH_TYPE=jwt``), the multiplexer must present a valid bearer token or its calls
are rejected (401). Children are configured per-entry in ``mcp_config.json`` and
historically carried no ``Authorization`` header, so flipping a child to jwt made
it unreachable through the multiplexer.

This module gives the multiplexer ONE service identity: it mints a Keycloak
service-account token via the OAuth2 ``client_credentials`` grant (audience
``agent-services`` — the same audience the children's ``JWTVerifier`` checks),
caches it, refreshes it before expiry, and the multiplexer attaches it to every
remote child that does not declare its own ``Authorization`` header.

Opt-in via ``MCP_CLIENT_AUTH=oidc-client-credentials``; otherwise every helper is
an inert no-op (returns ``None`` / ``{}``), so behaviour is unchanged when unset.

Configuration (env):
  MCP_CLIENT_AUTH        = oidc-client-credentials   # enable
  OIDC_CLIENT_ID         = mcp-multiplexer
  OIDC_CLIENT_SECRET     = <secret>                  # injected from OpenBao at deploy
  OIDC_AUDIENCE          = agent-services            # default
  OIDC_TOKEN_URL         = http://keycloak.arpa/realms/master/protocol/openid-connect/token
                           # default derived from FASTMCP_SERVER_AUTH_JWT_ISSUER
  OIDC_SCOPE             = <space-separated>          # optional
  OIDC_TLS_VERIFY        = true|false                # default true
"""

from __future__ import annotations

import functools
import threading
import time

import anyio
import httpx
import requests
from fastmcp.utilities.logging import get_logger

from agent_utilities.core._env import setting

logger = get_logger(name="MultiplexerClientAuth")

# Refresh this many seconds before the token actually expires.
_EXPIRY_SKEW_S = 30.0

# Floor for the derived child-session lifetime — never recycle a child more
# aggressively than this even if the IdP hands out a pathologically short TTL.
_MIN_SESSION_MAX_AGE = 20.0

# Fallback access-token lifetime (s) assumed before the first mint reveals the
# IdP's real ``expires_in`` — a conservative middle value.
_DEFAULT_TOKEN_TTL_S = 300.0


def _enabled() -> bool:
    return setting("MCP_CLIENT_AUTH", "none").strip().lower() == (
        "oidc-client-credentials"
    )


def _derive_token_url() -> str | None:
    # Explicit pin wins (any provider, or discovery-less environments).
    explicit = setting("OIDC_TOKEN_URL", None)
    if explicit:
        return explicit
    # Provider-agnostic (CONCEPT:AU-OS.identity.resolve-token-endpoint-from): resolve the token endpoint from the issuer's
    # OIDC discovery doc instead of a vendor-specific path. ``OIDC_ISSUER`` is the
    # canonical var; fall back to the JWT issuer. For multi-issuer trust (comma list,
    # CONCEPT:AU-OS.identity.native-multi-realm-jwt) the service mints from its primary (first) issuer.
    issuer = setting("OIDC_ISSUER", None) or setting(
        "FASTMCP_SERVER_AUTH_JWT_ISSUER", None
    )
    if not issuer:
        return None
    primary = issuer.split(",")[0].strip()
    from agent_utilities.security.oidc_discovery import token_endpoint_for

    return token_endpoint_for(primary)


class ClientCredentialsTokenProvider:
    """Thread-safe, self-refreshing OAuth2 client-credentials token cache."""

    def __init__(
        self,
        token_url: str,
        client_id: str,
        client_secret: str,
        audience: str | None = None,
        scope: str | None = None,
        verify: bool = True,
        timeout: int = 15,
    ) -> None:
        self.token_url = token_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.audience = audience
        self.scope = scope
        self.verify = verify
        self.timeout = timeout
        self._lock = threading.Lock()
        self._token: str | None = None
        self._expires_at = 0.0
        self._ttl_seconds = 0.0  # last observed access-token lifetime (expires_in)

    @property
    def access_token_ttl(self) -> float:
        """Last observed access-token lifetime in seconds (``expires_in``).

        Returns a conservative default until the first successful mint reveals
        the IdP's real value — drives how often child sessions must be recycled
        so they never outlive their bearer."""
        return self._ttl_seconds if self._ttl_seconds > 0 else _DEFAULT_TOKEN_TTL_S

    def get_token(self, *, force: bool = False) -> str:
        """Return a cached access token, refreshing it if missing or near expiry.

        ``force=True`` bypasses the cache and mints a fresh token — used when a
        child returns 401 (the cached token rotated/expired between the skew
        check and the request reaching the child)."""
        with self._lock:
            now = time.monotonic()
            if not force and self._token and now < self._expires_at - _EXPIRY_SKEW_S:
                return self._token
            data = {"grant_type": "client_credentials"}
            if self.audience:
                data["audience"] = self.audience
            if self.scope:
                data["scope"] = self.scope
            resp = requests.post(
                self.token_url,
                data=data,
                auth=(self.client_id, self.client_secret),
                verify=self.verify,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
            self._token = payload["access_token"]
            self._ttl_seconds = float(payload.get("expires_in", 300))
            self._expires_at = now + self._ttl_seconds
            logger.info(
                "Minted multiplexer service token (audience=%s, ttl=%ss)",
                self.audience,
                int(payload.get("expires_in", 300)),
            )
            return self._token


_provider: ClientCredentialsTokenProvider | None = None
_provider_lock = threading.Lock()


def get_provider() -> ClientCredentialsTokenProvider | None:
    """Return the configured provider, or ``None`` when disabled/misconfigured.

    Self-healing: a missing-creds result is NOT cached. A first call that races
    ahead of the secret bridge / OIDC discovery (e.g. at server startup) must not
    permanently disable fleet auth — the next call retries and succeeds once the
    creds are present. Only the successfully-built provider is memoized.
    """
    global _provider
    if not _enabled():
        return None
    if _provider is not None:
        return _provider
    with _provider_lock:
        if _provider is not None:
            return _provider
        token_url = _derive_token_url()
        client_id = setting("OIDC_CLIENT_ID", None)
        client_secret = setting("OIDC_CLIENT_SECRET", None)
        if not (token_url and client_id and client_secret):
            logger.error(
                "MCP_CLIENT_AUTH=oidc-client-credentials but OIDC_TOKEN_URL/"
                "FASTMCP_SERVER_AUTH_JWT_ISSUER, OIDC_CLIENT_ID or "
                "OIDC_CLIENT_SECRET is missing — children stay unauthenticated "
                "(will retry on the next call)."
            )
            return None
        verify = setting("OIDC_TLS_VERIFY", "true").strip().lower() not in (
            "false",
            "0",
            "no",
        )
        _provider = ClientCredentialsTokenProvider(
            token_url=token_url,
            client_id=client_id,
            client_secret=client_secret,
            audience=setting("OIDC_AUDIENCE", "agent-services"),
            scope=setting("OIDC_SCOPE", None) or None,
            verify=verify,
        )
        return _provider


def bearer_header(existing: dict | None) -> dict:
    """Authorization header for a remote child, or ``{}`` if not applicable.

    Never overrides an explicitly-configured ``Authorization`` header, and never
    raises — a token-mint failure degrades to no header (the child then 401s,
    which is visible in metrics/logs rather than crashing the multiplexer).
    """
    if existing and any(k.lower() == "authorization" for k in existing):
        return {}
    provider = get_provider()
    if provider is None:
        return {}
    try:
        return {"Authorization": f"Bearer {provider.get_token()}"}
    except Exception as exc:  # pragma: no cover - network/permission failure
        logger.warning("Could not mint multiplexer service token: %s", exc)
        return {}


class ClientCredentialsAuth(httpx.Auth):
    """Per-request service-bearer injection for a long-lived child transport.

    The multiplexer holds ONE pooled streamable-http/SSE session per child for
    the child's whole life. Baking a static bearer into the session headers
    (:func:`bearer_header`) freezes a short-lived Keycloak access token: once it
    expires the child returns 401, and because a streamable-http tool result is
    delivered asynchronously over the SSE stream, the in-flight call never
    receives its result — it hangs to the call-timeout instead of erroring,
    which surfaces as "flaky" child calls. Pulling the token from the
    self-refreshing provider on *every* request keeps the session authenticated
    for its whole life; a 401 forces a one-shot re-mint + retry for the rare
    case the token rotates mid-flight. Never raises: a mint failure degrades to
    an unauthenticated request (the child then 401s, visible in logs/metrics).
    """

    def __init__(self, provider: ClientCredentialsTokenProvider) -> None:
        self._provider = provider

    def _flow(self, request: httpx.Request, token: str | None):
        if token is not None:
            request.headers["Authorization"] = f"Bearer {token}"
        response = yield request
        if token is not None and response.status_code == 401:
            try:
                fresh = self._provider.get_token(force=True)
            except Exception:  # pragma: no cover - degrade to the 401
                return
            request.headers["Authorization"] = f"Bearer {fresh}"
            yield request

    def auth_flow(self, request: httpx.Request):
        try:
            token: str | None = self._provider.get_token()
        except Exception as exc:  # pragma: no cover - degrade to no header
            logger.warning("Could not mint multiplexer service token: %s", exc)
            token = None
        yield from self._flow(request, token)

    async def async_auth_flow(self, request: httpx.Request):
        # The mint does blocking I/O (``requests.post``); offload it so a
        # token refresh never stalls the multiplexer's event loop.
        try:
            token: str | None = await anyio.to_thread.run_sync(self._provider.get_token)
        except Exception as exc:  # pragma: no cover - degrade to no header
            logger.warning("Could not mint multiplexer service token: %s", exc)
            token = None
        if token is not None:
            request.headers["Authorization"] = f"Bearer {token}"
        response = yield request
        if token is not None and response.status_code == 401:
            try:
                fresh = await anyio.to_thread.run_sync(
                    functools.partial(self._provider.get_token, force=True)
                )
            except Exception:  # pragma: no cover - degrade to the 401
                return
            request.headers["Authorization"] = f"Bearer {fresh}"
            yield request


def bearer_auth(existing: dict | None) -> ClientCredentialsAuth | None:
    """An :class:`httpx.Auth` that authenticates a remote child per request.

    Preferred over :func:`bearer_header` for the multiplexer's long-lived child
    transports: a per-request token never goes stale, so a pooled session keeps
    working across token expiry instead of wedging on a 401. Never overrides a
    child's explicit ``Authorization`` header; returns ``None`` when the service
    identity is disabled/misconfigured (the child stays unauthenticated, exactly
    as with ``bearer_header``)."""
    if existing and any(k.lower() == "authorization" for k in existing):
        return None
    provider = get_provider()
    if provider is None:
        # Never silently leave a child unauthenticated: log WHY so a fleet-wide
        # 401 is diagnosable (enabled=False → config; enabled=True → creds/discovery).
        logger.warning(
            "[OS-5.32] no service bearer for child (MCP_CLIENT_AUTH enabled=%s) — "
            "request goes unauthenticated; a jwt-protected child will 401.",
            _enabled(),
        )
        return None
    return ClientCredentialsAuth(provider)


def service_session_max_age(existing: dict | None) -> float | None:
    """Seconds a child session may live before it must be recycled, or ``None``.

    A streamable-http/SSE child session is authenticated once at connect time and
    its result stream stays open for the session's whole life; once the bearer
    expires the child stops serving that stream and in-flight calls wedge. So a
    service-authenticated child must reconnect (re-mint) before its token's
    lifetime elapses. Returns that safe lifetime (token TTL minus the refresh
    skew and a small buffer, floored), or ``None`` when the child carries its own
    ``Authorization`` (we don't manage its token) or the service identity is
    disabled. Primes the provider once so the IdP's real TTL is known."""
    if existing and any(k.lower() == "authorization" for k in existing):
        return None
    provider = get_provider()
    if provider is None:
        return None
    try:
        provider.get_token()  # prime (cached) so access_token_ttl is real
    except Exception:  # pragma: no cover - mint failure; fall back to default TTL
        pass
    return max(_MIN_SESSION_MAX_AGE, provider.access_token_ttl - _EXPIRY_SKEW_S - 5.0)
