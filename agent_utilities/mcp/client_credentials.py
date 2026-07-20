#!/usr/bin/python
"""Outbound child-auth provider for the MCP multiplexer (CONCEPT:AU-OS.identity.so-jwt-protected-children).

The multiplexer aggregates many child MCP servers. When a child enforces auth (JWT
bearer, or a reverse proxy demanding HTTP Basic) the multiplexer must present a
valid credential or its calls are rejected (401). Children are configured per-entry
in ``mcp_config.json`` and historically carried no ``Authorization`` header, so
flipping a child to enforce auth made it unreachable through the multiplexer.

This module gives the multiplexer ONE service identity, in either of two schemes,
selected by ``MCP_CLIENT_AUTH``:

  * ``oidc-client-credentials`` — mint an OIDC provider service token via the
    OAuth2 ``client_credentials`` grant (with the audience the child verifies),
    cache it, refresh it before
    expiry, and attach ``Authorization: Bearer <token>`` to every remote child that
    does not declare its own header.
  * ``basic`` — attach a static ``Authorization: Basic <base64(user:pass)>`` from
    ``MCP_BASIC_AUTH_USERNAME`` and the referenced password (for a child, or an
    upstream proxy, that authenticates with HTTP Basic rather than a bearer token).

Either way the credential is never applied over a child's explicit ``Authorization``
header. Once a service-auth mode is selected, incomplete configuration or a mint
failure aborts the outbound request; it never silently becomes anonymous. Unset
(``MCP_CLIENT_AUTH=none``, the default) makes every helper an inert no-op.

Configuration (env):
  MCP_CLIENT_AUTH        = oidc-client-credentials | basic | none   # scheme (default none)
  # --- oidc-client-credentials ---
  OIDC_CLIENT_ID         = <runtime client identifier>
  OIDC_CLIENT_SECRET_REF = <runtime secret reference> # required
  OIDC_AUDIENCE          = <child verifier audience>
  OIDC_TOKEN_URL         = https://<identity-provider>/oauth2/token
  OIDC_ISSUER            = https://<identity-provider> # discovery alternative
  OIDC_SCOPE             = <space-separated>          # optional
  OIDC_TLS_PROFILE[_REF] = <runtime TLS profile>     # private trust and optional mTLS
  # --- basic ---
  MCP_BASIC_AUTH_USERNAME     = <runtime user>
  MCP_BASIC_AUTH_PASSWORD_REF = <runtime secret reference> # required
"""

from __future__ import annotations

import base64
import functools
import json
import re
import threading
import time

import anyio
import httpx
from fastmcp.utilities.logging import get_logger

from agent_utilities.core._env import setting

logger = get_logger(name="MultiplexerClientAuth")

# Recognized MCP_CLIENT_AUTH schemes.
_MODE_OIDC = "oidc-client-credentials"
_MODE_BASIC = "basic"
_MODE_NONE = "none"

# Refresh this many seconds before the token actually expires.
_EXPIRY_SKEW_S = 30.0

# Floor for the derived child-session lifetime — never recycle a child more
# aggressively than this even if the IdP hands out a pathologically short TTL.
_MIN_SESSION_MAX_AGE = 20.0

# Fallback access-token lifetime (s) assumed before the first mint reveals the
# IdP's real ``expires_in`` — a conservative middle value.
_DEFAULT_TOKEN_TTL_S = 300.0
_MAX_TOKEN_RESPONSE_BYTES = 1024 * 1024
_SECRET_REF_RE = re.compile(r"^(?:env|secret|vault)://[A-Za-z0-9_./#-]+$")


def _configured_text(name: str) -> str:
    """Read one canonical AgentConfig/XDG projection as normalized text."""
    return str(setting(name, "") or "").strip()


def _runtime_secret_reference(ref_name: str) -> str | None:
    """Return one validated reference without resolving credential material."""
    ref = _configured_text(ref_name)
    if not ref:
        return None
    if len(ref) > 2_048 or not _SECRET_REF_RE.fullmatch(ref):
        raise RuntimeError("outbound MCP credential reference is invalid")
    return ref


def _resolve_runtime_secret(ref_name: str) -> str | None:
    """Resolve a referenced credential only at the outbound request boundary."""
    ref = _runtime_secret_reference(ref_name)
    if not ref:
        return None
    scheme, _, target = ref.partition("://")
    if scheme == "env":
        # Environment references are already materialized at the process
        # boundary. Resolving them must not construct the durable secret
        # backend: doing so can recursively autostart an engine merely to
        # read this process's own environment.
        if not target or target[0].isdigit() or not target.replace("_", "a").isalnum():
            raise RuntimeError("outbound MCP credential reference is invalid")
        value = str(setting(target, "") or "")
    else:
        from agent_utilities.security.secrets_client import create_secrets_client

        value = str(create_secrets_client().resolve_ref(ref) or "")
    if not value:
        return None
    if len(value) > 16_384 or any(character in value for character in "\r\n\x00"):
        raise RuntimeError("outbound MCP credential is invalid")
    return value


def _auth_mode() -> str:
    """The configured outbound child-auth scheme.

    ``oidc-client-credentials`` | ``basic`` | ``none``. An unrecognized value
    is a configuration error; a typo must not silently disable authentication.
    """
    mode = str(setting("MCP_CLIENT_AUTH", _MODE_NONE) or _MODE_NONE).strip().lower()
    if mode in (_MODE_OIDC, _MODE_BASIC, _MODE_NONE):
        return mode
    raise RuntimeError("MCP_CLIENT_AUTH has an unsupported value")


def _derive_token_url() -> str | None:
    # Explicit pin wins (any provider, or discovery-less environments).
    explicit = _configured_text("OIDC_TOKEN_URL")
    if explicit:
        return explicit
    # Provider-agnostic (CONCEPT:AU-OS.identity.resolve-token-endpoint-from):
    # resolve the token endpoint from the explicitly configured issuer's OIDC
    # discovery document instead of borrowing an inbound-verifier setting.
    issuer = _configured_text("OIDC_ISSUER")
    if not issuer:
        return None
    from agent_utilities.security.oidc_discovery import token_endpoint_for

    return token_endpoint_for(issuer)


def outbound_auth_configuration_status() -> dict[str, object]:
    """Return redacted outbound-auth readiness without resolving any secret."""
    mode = _auth_mode()
    missing: list[str] = []
    invalid: list[str] = []
    if mode == _MODE_OIDC:
        required = {
            "OIDC_CLIENT_ID": _configured_text("OIDC_CLIENT_ID"),
            "OIDC_CLIENT_SECRET_REF": _configured_text("OIDC_CLIENT_SECRET_REF"),
            "OIDC_AUDIENCE": _configured_text("OIDC_AUDIENCE"),
        }
        missing.extend(name for name, value in required.items() if not value)
        invalid.extend(
            name
            for name in ("OIDC_CLIENT_ID", "OIDC_AUDIENCE")
            if required[name]
            and (
                len(required[name]) > 4_096
                or any(character in required[name] for character in "\r\n\x00")
            )
        )
        if not (_configured_text("OIDC_TOKEN_URL") or _configured_text("OIDC_ISSUER")):
            missing.append("OIDC_TOKEN_URL_OR_OIDC_ISSUER")
        try:
            _runtime_secret_reference("OIDC_CLIENT_SECRET_REF")
        except RuntimeError:
            invalid.append("OIDC_CLIENT_SECRET_REF")
    elif mode == _MODE_BASIC:
        required = {
            "MCP_BASIC_AUTH_USERNAME": _configured_text("MCP_BASIC_AUTH_USERNAME"),
            "MCP_BASIC_AUTH_PASSWORD_REF": _configured_text(
                "MCP_BASIC_AUTH_PASSWORD_REF"
            ),
        }
        missing.extend(name for name, value in required.items() if not value)
        username = required["MCP_BASIC_AUTH_USERNAME"]
        if username and (
            len(username) > 4_096
            or any(character in username for character in "\r\n\x00:")
        ):
            invalid.append("MCP_BASIC_AUTH_USERNAME")
        try:
            _runtime_secret_reference("MCP_BASIC_AUTH_PASSWORD_REF")
        except RuntimeError:
            invalid.append("MCP_BASIC_AUTH_PASSWORD_REF")
    return {
        "mode": mode,
        "ready": not missing and not invalid,
        "missing": tuple(sorted(missing)),
        "invalid": tuple(sorted(invalid)),
        "redacted": True,
    }


def validate_outbound_auth_configuration() -> None:
    """Fail closed when an enabled outbound authentication mode is incomplete."""
    status = outbound_auth_configuration_status()
    if status["ready"]:
        return
    if status["mode"] == _MODE_BASIC:
        raise RuntimeError("Outbound MCP basic identity is incomplete")
    raise RuntimeError("Outbound MCP service identity is incomplete")


class ClientCredentialsTokenProvider:
    """Thread-safe, self-refreshing OAuth2 client-credentials token cache."""

    def __init__(
        self,
        token_url: str,
        client_id: str,
        client_secret: str,
        audience: str,
        scope: str | None = None,
        timeout: int = 15,
    ) -> None:
        self.token_url = token_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.audience = audience
        self.scope = scope
        if not 1 <= len(client_id) <= 4_096 or any(
            character in client_id for character in "\r\n\x00"
        ):
            raise ValueError("OIDC client identifier is invalid")
        if not 1 <= len(client_secret) <= 16_384 or any(
            character in client_secret for character in "\r\n\x00"
        ):
            raise ValueError("OIDC client secret is invalid")
        if not 1 <= len(audience) <= 4_096 or any(
            character in audience for character in "\r\n\x00"
        ):
            raise ValueError("OIDC audience is invalid")
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
            from agent_utilities.security.oidc_discovery import oidc_http_client

            with oidc_http_client(timeout=float(self.timeout)) as client:
                with client.stream(
                    "POST",
                    self.token_url,
                    data=data,
                    auth=(self.client_id, self.client_secret),
                ) as response:
                    response.raise_for_status()
                    body = bytearray()
                    for chunk in response.iter_bytes():
                        body.extend(chunk)
                        if len(body) > _MAX_TOKEN_RESPONSE_BYTES:
                            raise RuntimeError(
                                "OIDC token response exceeded its safety boundary"
                            )
            payload = json.loads(body)
            if not isinstance(payload, dict):
                raise RuntimeError("OIDC token response had an invalid shape")
            token = payload.get("access_token")
            if not isinstance(token, str) or not 1 <= len(token) <= 65_536:
                raise RuntimeError("OIDC token response had an invalid token")
            try:
                ttl = float(payload.get("expires_in", 300))
            except (TypeError, ValueError):
                ttl = 300.0
            if not 1.0 <= ttl <= 86_400.0:
                raise RuntimeError("OIDC token response had an invalid lifetime")
            self._token = token
            self._ttl_seconds = ttl
            self._expires_at = now + self._ttl_seconds
            logger.info("Minted multiplexer service token")
            return self._token


_provider: ClientCredentialsTokenProvider | None = None
_provider_lock = threading.Lock()


def get_provider() -> ClientCredentialsTokenProvider | None:
    """Return the configured provider, or ``None`` only when auth is disabled.

    Self-healing: a missing-creds result is NOT cached. A first call that races
    ahead of the secret bridge / OIDC discovery (e.g. at server startup) must not
    permanently disable fleet auth — the next call retries and succeeds once the
    creds are present. Only the successfully-built provider is memoized.
    """
    global _provider
    if _auth_mode() != _MODE_OIDC:
        return None
    if _provider is not None:
        return _provider
    with _provider_lock:
        if _provider is not None:
            return _provider
        validate_outbound_auth_configuration()
        token_url = _derive_token_url()
        client_id = _configured_text("OIDC_CLIENT_ID")
        audience = _configured_text("OIDC_AUDIENCE")
        client_secret = _resolve_runtime_secret("OIDC_CLIENT_SECRET_REF")
        if not (token_url and client_id and client_secret and audience):
            logger.error("Outbound MCP service identity is incomplete")
            raise RuntimeError("Outbound MCP service identity is incomplete")
        _provider = ClientCredentialsTokenProvider(
            token_url=token_url,
            client_id=client_id,
            client_secret=client_secret,
            audience=audience,
            scope=setting("OIDC_SCOPE", None) or None,
        )
        return _provider


def _basic_credentials() -> tuple[str, str] | None:
    """The configured HTTP Basic ``(username, password)``, or ``None``.

    Active only under ``MCP_CLIENT_AUTH=basic``; both vars must be present (a
    missing pair is a hard configuration error rather than an anonymous fallback).
    """
    if _auth_mode() != _MODE_BASIC:
        return None
    validate_outbound_auth_configuration()
    user = _configured_text("MCP_BASIC_AUTH_USERNAME")
    password = _resolve_runtime_secret("MCP_BASIC_AUTH_PASSWORD_REF")
    if not (user and password):
        logger.error("Outbound MCP basic identity is incomplete")
        raise RuntimeError("Outbound MCP basic identity is incomplete")
    user = str(user)
    if not 1 <= len(user) <= 4_096 or any(
        character in user for character in "\r\n\x00:"
    ):
        raise RuntimeError("Outbound MCP basic identity is invalid")
    return (user, password)


def _basic_header_value(user: str, password: str) -> str:
    token = base64.b64encode(f"{user}:{password}".encode()).decode("ascii")
    return f"Basic {token}"


def child_auth_header(existing: dict | None) -> dict:
    """Authorization header for a remote child, or ``{}`` if not applicable.

    Returns the configured outbound credential — ``Bearer <token>`` under
    ``oidc-client-credentials`` or ``Basic <base64>`` under ``basic``. Never
    overrides an explicitly configured ``Authorization`` header. A configured
    service-auth mode fails closed when its credential cannot be produced.
    """
    if existing and any(k.lower() == "authorization" for k in existing):
        return {}
    mode = _auth_mode()
    if mode == _MODE_NONE:
        return {}
    basic = _basic_credentials()
    if mode == _MODE_BASIC and basic is not None:
        return {"Authorization": _basic_header_value(*basic)}
    provider = get_provider()
    if provider is None:  # defensive: mode=OIDC always returns or raises
        raise RuntimeError("Outbound MCP service identity is unavailable")
    try:
        return {"Authorization": f"Bearer {provider.get_token()}"}
    except Exception as exc:  # pragma: no cover - network/permission failure
        logger.warning(
            "Could not mint multiplexer service token (exception_type=%s)",
            type(exc).__name__,
        )
        raise RuntimeError("Could not mint outbound MCP credential") from None


class ClientCredentialsAuth(httpx.Auth):
    """Per-request service-bearer injection for a long-lived child transport.

    The multiplexer holds ONE pooled streamable-http/SSE session per child for
    the child's whole life. Baking a static bearer into the session headers
    (:func:`child_auth_header`) freezes a short-lived OIDC access token: once it
    expires the child returns 401, and because a streamable-http tool result is
    delivered asynchronously over the SSE stream, the in-flight call never
    receives its result — it hangs to the call-timeout instead of erroring,
    which surfaces as "flaky" child calls. Pulling the token from the
    self-refreshing provider on *every* request keeps the session authenticated
    for its whole life; a 401 forces a one-shot re-mint + retry for the rare
    case the token rotates mid-flight. A mint failure aborts the request rather
    than silently retrying without authentication.
    """

    def __init__(self, provider: ClientCredentialsTokenProvider) -> None:
        self._provider = provider

    def _flow(self, request: httpx.Request, token: str):
        request.headers["Authorization"] = f"Bearer {token}"
        response = yield request
        if response.status_code == 401:
            try:
                fresh = self._provider.get_token(force=True)
            except Exception:  # pragma: no cover - degrade to the 401
                return
            request.headers["Authorization"] = f"Bearer {fresh}"
            yield request

    def auth_flow(self, request: httpx.Request):
        try:
            token = self._provider.get_token()
        except Exception as exc:  # pragma: no cover - network/permission failure
            logger.warning(
                "Could not mint multiplexer service token (exception_type=%s)",
                type(exc).__name__,
            )
            raise RuntimeError("Could not mint outbound MCP credential") from None
        yield from self._flow(request, token)

    async def async_auth_flow(self, request: httpx.Request):
        # The mint does blocking I/O; offload it so a
        # token refresh never stalls the multiplexer's event loop.
        try:
            token = await anyio.to_thread.run_sync(self._provider.get_token)
        except Exception as exc:  # pragma: no cover - network/permission failure
            logger.warning(
                "Could not mint multiplexer service token (exception_type=%s)",
                type(exc).__name__,
            )
            raise RuntimeError("Could not mint outbound MCP credential") from None
        request.headers["Authorization"] = f"Bearer {token}"
        response = yield request
        if response.status_code == 401:
            try:
                fresh = await anyio.to_thread.run_sync(
                    functools.partial(self._provider.get_token, force=True)
                )
            except Exception:  # pragma: no cover - degrade to the 401
                return
            request.headers["Authorization"] = f"Bearer {fresh}"
            yield request


def child_auth(existing: dict | None) -> httpx.Auth | None:
    """An :class:`httpx.Auth` that authenticates a remote child.

    Preferred over :func:`child_auth_header` for the multiplexer's long-lived child
    transports. Under ``basic`` it returns a static :class:`httpx.BasicAuth`; under
    ``oidc-client-credentials`` a :class:`ClientCredentialsAuth` that pulls a fresh
    token per request (so a pooled session keeps working across token expiry instead
    of wedging on a 401). Never overrides a child's explicit ``Authorization``
    header; returns ``None`` only when service identity is explicitly disabled.
    A selected auth mode fails closed if it is incomplete.
    """
    if existing and any(k.lower() == "authorization" for k in existing):
        return None
    mode = _auth_mode()
    if mode == _MODE_NONE:
        return None
    basic = _basic_credentials()
    if mode == _MODE_BASIC and basic is not None:
        return httpx.BasicAuth(*basic)
    provider = get_provider()
    if provider is None:
        raise RuntimeError("Outbound MCP service identity is unavailable")
    return ClientCredentialsAuth(provider)


def service_session_max_age(existing: dict | None) -> float | None:
    """Seconds a child session may live before it must be recycled, or ``None``.

    A streamable-http/SSE child session is authenticated once at connect time and
    its result stream stays open for the session's whole life; once the bearer
    expires the child stops serving that stream and in-flight calls wedge. So a
    service-authenticated child must reconnect (re-mint) before its token's
    lifetime elapses. Returns that safe lifetime (token TTL minus the refresh
    skew and a small buffer, floored), or ``None`` when the child carries its own
    ``Authorization`` (we don't manage its token), the service identity is
    disabled, or the scheme is ``basic`` (a static credential never expires, so no
    recycle is needed — ``get_provider`` is ``None`` for basic). Primes the OIDC
    provider once so the IdP's real TTL is known."""
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
