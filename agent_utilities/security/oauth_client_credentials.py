#!/usr/bin/python
from __future__ import annotations

"""OAuth2 ``client_credentials`` token lifecycle for outbound LLM + embedding endpoints.

CONCEPT:AU-OS.identity.oauth2-client-credentials-lifecycle — machine-to-machine auth for
enterprise OpenAI-compatible / Azure OpenAI endpoints.

Today graph-os's outbound LLM/embedding clients (:mod:`agent_utilities.core.model_factory`,
:mod:`agent_utilities.core.embedding_utilities`) only know how to carry a **static** ``api_key``
as a Bearer/API-key header. Enterprise gateways front their OpenAI-compatible or Azure OpenAI
endpoints with a machine-to-machine OAuth2 ``client_credentials`` grant instead: the caller
exchanges a ``client_id``/``client_secret`` for a **short-lived** access token at a ``token_url``
and must mint a fresh one before every expiry — a static key never satisfies that.

This module is the ONE token-minting seam for that grant, generalizing the multiplexer's
existing OIDC client-credentials pattern
(:class:`agent_utilities.mcp.client_credentials.ClientCredentialsTokenProvider`, OS-5.32 — built
for authenticating to *child MCP servers*) to outbound *provider* calls instead:

* :class:`OAuth2ClientCredentialsConfig` — the declarative config block a chat/embedding model
  config carries (``token_url``/``client_id``/``client_secret``/``scope``/``audience``/
  ``extra_params``), mutually exclusive with a static ``api_key`` at the config layer
  (:class:`agent_utilities.core.config.ChatModelConfig` /
  :class:`agent_utilities.core.config.EmbeddingModelConfig` /
  :class:`agent_utilities.models.model_registry.ModelDefinition`).
* :class:`OAuthClientCredentialsProvider` — mints over a DNS-pinned, bounded HTTPS transport,
  caches by the complete audience/security configuration, and proactively renews the bearer
  (thread-safe; renews before expiry with a configurable skew; one-shot force-refresh on 401).
* :class:`OAuth2ClientCredentialsAuth` — an :class:`httpx.Auth` that injects
  ``Authorization: Bearer <token>`` into every request on the client it is attached to (sync +
  async), so a caller just builds its ``httpx.Client``/``AsyncClient`` with
  ``auth=httpx_auth_from_config(model_cfg.oauth2)`` and the mint/refresh is transparent.

**Provider coverage**: works against any generic OIDC/OAuth2 token endpoint. Azure AD v2 is
supported via ``scope="api://<resource>/.default"`` (the ``.default`` static scope Azure AD
requires for client-credentials); Azure AD v1 / resource-indicator IdPs (Auth0, Okta) via
``audience`` or ``extra_params={"resource": "..."}``.

**Secrets**: ``client_secret`` (and ``client_id`` when sensitive) are secret-REFERENCES
(``vault://``/``env://``/``secret://``), resolved through the existing
:class:`agent_utilities.security.secrets_client.SecretsClient` (Vault/OpenBao/the engine-backed
encrypted store) at token-mint time — never a plaintext value in config. A plaintext
``client_secret`` is rejected at config-validation time (see
:meth:`OAuth2ClientCredentialsConfig._reject_plaintext_secret`).

**Never logged**: token endpoints, client identity, token material, secret references, response
bodies, and raw transport exceptions do not cross this boundary.
"""

import functools
import hashlib
import json
import logging
import math
import threading
import time
from collections.abc import Callable
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

__all__ = [
    "OAuth2ClientCredentialsConfig",
    "OAuthClientCredentialsProvider",
    "OAuth2ClientCredentialsAuth",
    "DEFAULT_REFRESH_SKEW_SECONDS",
    "DEFAULT_REFRESH_SKEW_FRACTION",
    "resolve_effective_skew",
    "get_client_credentials_provider",
    "reset_client_credentials_cache",
    "build_provider_from_config",
    "httpx_auth_from_config",
]

#: Recognized secret-reference URI schemes (mirrors
#: :meth:`agent_utilities.security.secrets_client.SecretsClient.resolve_ref`).
_SECRET_REF_SCHEMES = ("vault://", "env://", "secret://")

#: Renew this many seconds before actual expiry when no explicit skew is configured, floored at
#: this constant — short-TTL tokens (e.g. 120s) still get a meaningful renewal buffer.
DEFAULT_REFRESH_SKEW_SECONDS = 60.0

#: ... or this fraction of the observed TTL, whichever is larger — long-TTL tokens (e.g. 3600s)
#: get a proportionally bigger buffer instead of cutting it close every time.
DEFAULT_REFRESH_SKEW_FRACTION = 0.20

#: Conservative assumed token lifetime before the first mint reveals the IdP's real
#: ``expires_in`` (mirrors ``mcp.client_credentials._DEFAULT_TOKEN_TTL_S``).
_DEFAULT_TOKEN_TTL_S = 300.0
_MAX_TOKEN_RESPONSE_BYTES = 1024 * 1024
_MAX_TOKEN_BYTES = 64 * 1024
_RESERVED_TOKEN_PARAMS = frozenset(
    {"grant_type", "client_id", "client_secret", "scope", "audience"}
)


class _ParsedTokenResponse:
    """Minimal response shape kept for deterministic transport injection tests."""

    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


class _BoundedTokenRequester:
    """Requests-shaped adapter backed exclusively by the pinned HTTPX client."""

    @staticmethod
    def post(
        url: str,
        *,
        data: dict[str, str],
        auth: tuple[str, str] | None,
        timeout: float,
        _client: httpx.Client,
    ) -> _ParsedTokenResponse:
        del timeout  # already enforced by the constructed pinned client
        with _client.stream("POST", url, data=data, auth=auth) as response:
            response.raise_for_status()
            response_body = bytearray()
            for chunk in response.iter_bytes(chunk_size=64 * 1024):
                response_body.extend(chunk)
                if len(response_body) > _MAX_TOKEN_RESPONSE_BYTES:
                    raise ValueError("OAuth2 token response exceeded its boundary")
        return _ParsedTokenResponse(json.loads(response_body))


# Kept as a tiny injectable seam for deterministic tests; this is not the
# third-party Requests package and never bypasses the pinned HTTPX transport.
requests = _BoundedTokenRequester()


def _looks_like_secret_ref(value: str) -> bool:
    """Whether ``value`` is a secret-reference URI rather than a literal."""
    return str(value).casefold().startswith(_SECRET_REF_SCHEMES)


def _bounded_runtime_text(
    value: Any,
    *,
    label: str,
    maximum_bytes: int,
    reject_space: bool = False,
    allow_empty: bool = False,
) -> str:
    """Validate a resolved credential/config value without reflecting it."""

    if not isinstance(value, str) or (not value and not allow_empty):
        raise ValueError(f"OAuth2 {label} is invalid")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError:
        raise ValueError(f"OAuth2 {label} is invalid") from None
    if len(encoded) > maximum_bytes or any(
        ord(character) < (33 if reject_space else 32) or ord(character) == 127
        for character in value
    ):
        raise ValueError(f"OAuth2 {label} is invalid")
    return value


class OAuth2ClientCredentialsConfig(BaseModel):
    """Declarative OAuth2 ``client_credentials`` block for a chat/embedding provider.

    CONCEPT:AU-OS.identity.oauth2-client-credentials-lifecycle. Mutually exclusive with a static
    ``api_key`` at the owning config's level (enforced there, not here, since that field lives on
    :class:`ChatModelConfig`/:class:`EmbeddingModelConfig`/:class:`ModelDefinition`).
    """

    token_url: str = Field(
        min_length=1,
        max_length=8_192,
        description="OIDC/OAuth2 token endpoint (e.g. Azure AD /oauth2/v2.0/token).",
    )
    client_id: str = Field(
        min_length=1,
        max_length=4_096,
        description=(
            "Client id. A literal value, or a secret-reference URI "
            "(vault://, env://, secret://) when the client id itself is sensitive."
        ),
    )
    client_secret: str = Field(
        min_length=1,
        max_length=4_096,
        description=(
            "Secret-reference URI resolving to the client secret — vault://, env://, "
            "or secret://. A plaintext value is rejected at validation time."
        ),
    )
    scope: str | None = Field(
        default=None,
        max_length=8_192,
        description="Space-separated scopes. Azure AD v2: '<resource>/.default'.",
    )
    audience: str | None = Field(
        default=None,
        max_length=4_096,
        description="Audience/resource-indicator parameter (Auth0/Okta-style IdPs).",
    )
    extra_params: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Extra form params merged into the token request body verbatim "
            "(e.g. Azure AD v1 'resource')."
        ),
    )
    token_auth_style: Literal["body", "basic"] = Field(
        default="body",
        description=(
            "How the client credentials are presented AT the token endpoint (RFC 6749 §2.3.1). "
            "'body' (default) sends client_id/client_secret as form params in the POST body; "
            "'basic' sends them via HTTP Basic auth (Authorization: Basic base64(id:secret)) "
            "and omits them from the body — required by IdPs that mandate "
            "'client_secret_basic' token-endpoint auth."
        ),
    )
    tls_profile: str | None = Field(
        default=None,
        max_length=64,
        description="Runtime TLS profile name; certificate material stays outside this block.",
    )
    tls_profile_ref: str | None = Field(
        default=None,
        max_length=4_096,
        description="Secret reference containing a runtime TLS profile.",
    )
    timeout: float = Field(
        default=15.0,
        gt=0,
        le=60,
        description="Token request timeout, in seconds.",
    )
    refresh_skew_seconds: float | None = Field(
        default=None,
        ge=0,
        le=7 * 24 * 60 * 60,
        description=(
            "Explicit renew-before-expiry skew, in seconds. Unset (None) auto-derives "
            "max(60s, 20% of the observed access-token TTL)."
        ),
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("client_secret")
    @classmethod
    def _reject_plaintext_secret(cls, v: str) -> str:
        """Refuse a plaintext ``client_secret`` — it must be a secret-reference URI.

        Fleet secret-handling convention (CONCEPT:AU-OS.config.secrets-authentication): a client
        secret never lives as a literal string in config.json / env-derived settings; it is
        stored in the secrets backend (Vault/OpenBao/the engine-backed encrypted store) and
        referenced by URI, resolved at mint time via
        :class:`agent_utilities.security.secrets_client.SecretsClient`.
        """
        if not v or not _looks_like_secret_ref(v):
            raise ValueError(
                "oauth2.client_secret must be a secret reference "
                f"({', '.join(_SECRET_REF_SCHEMES)}) — plaintext client secrets are forbidden "
                "in config. Store the secret in the secrets backend and reference it by URI "
                "(e.g. 'vault://llm/azure/client_secret')."
            )
        return v

    @field_validator("tls_profile_ref")
    @classmethod
    def _validate_tls_profile_ref(cls, value: str | None) -> str | None:
        if value is not None and not _looks_like_secret_ref(value):
            raise ValueError("oauth2.tls_profile_ref must be a secret reference")
        return value

    @field_validator("extra_params")
    @classmethod
    def _validate_extra_params(cls, value: dict[str, str]) -> dict[str, str]:
        if len(value) > 32:
            raise ValueError("oauth2.extra_params contains too many fields")
        total = 0
        normalized: dict[str, str] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key).strip()
            item = str(raw_value)
            if (
                not key
                or len(key) > 128
                or key.casefold() in _RESERVED_TOKEN_PARAMS
                or any(
                    ord(character) < 33 or ord(character) == 127 for character in key
                )
                or any(
                    ord(character) < 32 or ord(character) == 127 for character in item
                )
            ):
                raise ValueError("oauth2.extra_params contains a rejected field")
            total += len(key.encode("utf-8")) + len(item.encode("utf-8"))
            if total > 32 * 1024:
                raise ValueError("oauth2.extra_params exceeded its size boundary")
            normalized[key] = item
        return normalized

    @model_validator(mode="after")
    def _validate_transport(self) -> OAuth2ClientCredentialsConfig:
        parsed = urlsplit(self.token_url)
        if (
            parsed.scheme.casefold() != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise ValueError(
                "oauth2.token_url must be an HTTPS endpoint without credentials"
            )
        if self.tls_profile and self.tls_profile_ref:
            raise ValueError("oauth2 TLS profile source is ambiguous")
        return self


def resolve_effective_skew(ttl_seconds: float, explicit_skew: float | None) -> float:
    """Renew-before-expiry skew: the explicit override, or ``max(60s, 20% of ttl)``.

    An explicit ``refresh_skew_seconds`` always wins (an operator's considered choice); the
    auto-derived default scales with the IdP's actual token lifetime so a short-TTL token (say
    120s) still gets a meaningful buffer (the 60s floor) while a long-TTL one (3600s) renews
    proportionally earlier (720s) instead of cutting every refresh close.
    """
    if not math.isfinite(float(ttl_seconds)) or ttl_seconds <= 0:
        raise ValueError("OAuth2 token TTL is invalid")
    if explicit_skew is not None:
        if not math.isfinite(float(explicit_skew)):
            raise ValueError("OAuth2 refresh skew is invalid")
        return max(0.0, explicit_skew)
    return max(
        DEFAULT_REFRESH_SKEW_SECONDS, ttl_seconds * DEFAULT_REFRESH_SKEW_FRACTION
    )


class OAuthClientCredentialsProvider:
    """Thread-safe, self-refreshing OAuth2 ``client_credentials`` token cache.

    Mints via ``POST token_url`` with ``grant_type=client_credentials`` (+
    ``client_id``/``client_secret``/``scope``/``audience``/``extra_params``), parses
    ``access_token``/``expires_in``, and proactively renews before expiry
    (:func:`resolve_effective_skew`). ``clock`` is injectable so tests drive expiry
    deterministically instead of sleeping real time.

    Never logs token material, endpoint identity, client identity, or raw failures.
    """

    def __init__(
        self,
        token_url: str,
        client_id: str,
        client_secret: str,
        *,
        scope: str | None = None,
        audience: str | None = None,
        extra_params: dict[str, str] | None = None,
        tls_profile: str | None = None,
        tls_profile_ref: str | None = None,
        timeout: float = 15.0,
        refresh_skew_seconds: float | None = None,
        token_auth_style: Literal["body", "basic"] = "body",
        clock: Callable[[], float] | None = None,
    ) -> None:
        token_url = _bounded_runtime_text(
            token_url,
            label="token endpoint",
            maximum_bytes=8_192,
            reject_space=True,
        )
        parsed_token_url = urlsplit(token_url)
        if (
            len(str(token_url)) > 8_192
            or parsed_token_url.scheme.casefold() != "https"
            or not parsed_token_url.hostname
            or parsed_token_url.username is not None
            or parsed_token_url.password is not None
            or parsed_token_url.fragment
        ):
            raise ValueError("OAuth2 token endpoint was rejected")
        client_id = _bounded_runtime_text(
            client_id, label="client identity", maximum_bytes=4_096
        )
        client_secret = _bounded_runtime_text(
            client_secret, label="client credential", maximum_bytes=64 * 1024
        )
        if scope is not None:
            scope = _bounded_runtime_text(scope, label="scope", maximum_bytes=8_192)
        if audience is not None:
            audience = _bounded_runtime_text(
                audience, label="audience", maximum_bytes=4_096
            )
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or not 0 < float(timeout) <= 60
        ):
            raise ValueError("OAuth2 token timeout is invalid")
        if refresh_skew_seconds is not None and (
            not math.isfinite(float(refresh_skew_seconds))
            or not 0 <= float(refresh_skew_seconds) <= 7 * 24 * 60 * 60
        ):
            raise ValueError("OAuth2 refresh skew is invalid")
        if tls_profile and tls_profile_ref:
            raise ValueError("OAuth2 TLS profile source is ambiguous")
        if tls_profile is not None:
            tls_profile = _bounded_runtime_text(
                tls_profile,
                label="TLS profile",
                maximum_bytes=64,
                reject_space=True,
            )
        if tls_profile_ref is not None:
            tls_profile_ref = _bounded_runtime_text(
                tls_profile_ref,
                label="TLS profile reference",
                maximum_bytes=4_096,
                reject_space=True,
            )
            if not _looks_like_secret_ref(tls_profile_ref):
                raise ValueError("OAuth2 TLS profile reference is invalid")
        if token_auth_style not in {"body", "basic"}:
            raise ValueError("OAuth2 token authentication style is invalid")
        if (
            not isinstance(extra_params, (dict, type(None)))
            or len(extra_params or {}) > 32
        ):
            raise ValueError("OAuth2 extra parameters were rejected")
        normalized_extra: dict[str, str] = {}
        extra_size = 0
        for raw_key, raw_value in (extra_params or {}).items():
            key = _bounded_runtime_text(
                raw_key,
                label="extra parameter",
                maximum_bytes=128,
                reject_space=True,
            ).strip()
            value = _bounded_runtime_text(
                raw_value,
                label="extra parameter",
                maximum_bytes=32 * 1024,
                allow_empty=True,
            )
            if key.casefold() in _RESERVED_TOKEN_PARAMS:
                raise ValueError("OAuth2 extra parameter was rejected")
            extra_size += len(key.encode("utf-8")) + len(value.encode("utf-8"))
            if extra_size > 32 * 1024:
                raise ValueError("OAuth2 extra parameters exceeded their size boundary")
            normalized_extra[key] = value
        self.token_url = token_url
        self.client_id = client_id
        self._client_secret = client_secret
        self.scope = scope
        self.audience = audience
        self.extra_params = normalized_extra
        self.token_auth_style = token_auth_style
        self.tls_profile = tls_profile
        self.tls_profile_ref = tls_profile_ref
        self.timeout = timeout
        self._explicit_skew = refresh_skew_seconds
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._token: str | None = None
        self._expires_at = 0.0
        self._ttl_seconds = 0.0  # last observed expires_in
        self._http: httpx.Client | None = None
        self._resolved_tls: Any | None = None

    @property
    def access_token_ttl(self) -> float:
        """Last observed access-token lifetime, or a conservative default before the first mint."""
        return self._ttl_seconds if self._ttl_seconds > 0 else _DEFAULT_TOKEN_TTL_S

    def _effective_skew(self) -> float:
        return resolve_effective_skew(self.access_token_ttl, self._explicit_skew)

    def _http_client(self) -> httpx.Client:
        if self._http is not None:
            return self._http

        from agent_utilities.core.config import config
        from agent_utilities.core.http_client import create_http_client
        from agent_utilities.core.transport_security import (
            TransportSecurityError,
            resolve_configured_tls_profile,
        )

        # A provider-local selector is one atomic trust decision.  Do not fill
        # its missing half from the service default: a local profile reference
        # plus an unrelated global profile name can otherwise be interpreted as
        # a named lookup inside the referenced profile.
        if self.tls_profile is not None or self.tls_profile_ref is not None:
            profile_name = self.tls_profile
            profile_ref = self.tls_profile_ref
        else:
            profile_name = config.oauth2_token_tls_profile
            profile_ref = config.oauth2_token_tls_profile_ref

        trust = resolve_configured_tls_profile(
            "oauth2-token",
            profile_name=profile_name,
            profile_ref=profile_ref,
            config=config,
        )
        if trust.proxy_url:
            trust.cleanup()
            raise TransportSecurityError("oauth2_proxy_incompatible_with_dns_pinning")
        try:
            client = create_http_client(
                timeout=self.timeout,
                verify=trust.ssl_context,
                pin_egress=True,
                allowed_private_hosts=config.model_http_allowed_private_hosts,
                allow_loopback=False,
                trust_env=False,
                follow_redirects=False,
            )
        except Exception:
            trust.cleanup()
            raise
        self._resolved_tls = trust
        self._http = client
        return client

    def close(self) -> None:
        """Release pooled transport resources and runtime TLS material."""
        if self._http is not None:
            self._http.close()
            self._http = None
        if self._resolved_tls is not None:
            self._resolved_tls.cleanup()
            self._resolved_tls = None

    def get_token(self, *, force: bool = False) -> str:
        """Return a cached bearer, minting/renewing it first when missing, stale, or ``force``.

        ``force=True`` bypasses the cache unconditionally — used for the one-shot re-mint after
        a 401 (the cached token rotated/was revoked between the skew check and the request
        reaching the endpoint).
        """
        with self._lock:
            now = self._clock()
            if (
                not force
                and self._token
                and now < self._expires_at - self._effective_skew()
            ):
                return self._token
            self._mint(now)
            assert self._token is not None  # noqa: S101 - _mint always sets it or raises
            return self._token

    def _mint(self, now: float) -> None:
        """POST the client-credentials grant and populate ``_token``/``_expires_at``.

        Caller holds ``self._lock``. Raises on any transport/HTTP/shape failure rather than
        degrading silently — a caller with no bearer must see a loud auth failure, not a
        request that quietly goes out unauthenticated.
        """
        data: dict[str, str] = {"grant_type": "client_credentials"}
        # client_secret_basic (HTTP Basic at the token endpoint) vs client_secret_post (body
        # params) — RFC 6749 §2.3.1. 'basic' presents id/secret via HTTP Basic and keeps them
        # OUT of the body; 'body' (default) sends them as form params exactly as before.
        basic_auth: tuple[str, str] | None = None
        if self.token_auth_style == "basic":
            basic_auth = (self.client_id, self._client_secret)
        else:
            data["client_id"] = self.client_id
            data["client_secret"] = self._client_secret
        if self.scope:
            data["scope"] = self.scope
        if self.audience:
            data["audience"] = self.audience
        data.update(self.extra_params)
        try:
            resp = requests.post(
                self.token_url,
                data=data,
                auth=basic_auth,
                timeout=self.timeout,
                _client=self._http_client(),
            )
            resp.raise_for_status()
            payload = resp.json()
        except (httpx.HTTPError, UnicodeError, ValueError, TypeError):
            raise RuntimeError("OAuth2 token request failed") from None
        if not isinstance(payload, dict) or "access_token" not in payload:
            # Never echo the response body — it may itself carry token-adjacent material.
            raise ValueError("OAuth2 token response is missing access_token")
        token = payload["access_token"]
        if (
            not isinstance(token, str)
            or not token
            or len(token.encode("utf-8")) > _MAX_TOKEN_BYTES
            or any(ord(character) < 33 or ord(character) == 127 for character in token)
        ):
            raise ValueError("OAuth2 token response contained an invalid access token")
        try:
            ttl = float(payload.get("expires_in", _DEFAULT_TOKEN_TTL_S))
        except (TypeError, ValueError):
            raise ValueError(
                "OAuth2 token response contained an invalid expiry"
            ) from None
        if not math.isfinite(ttl) or not 1 <= ttl <= 7 * 24 * 60 * 60:
            raise ValueError("OAuth2 token response contained an invalid expiry")
        self._token = token
        self._ttl_seconds = ttl
        self._expires_at = now + self._ttl_seconds
        logger.info("OAuth2 client credential renewed")


class OAuth2ClientCredentialsAuth(httpx.Auth):
    """Per-request bearer injection for a long-lived LLM/embedding client.

    Mirrors :class:`agent_utilities.mcp.client_credentials.ClientCredentialsAuth` (OS-5.32),
    generalized for outbound provider clients instead of child-MCP sessions. Pulling the token
    from the self-refreshing :class:`OAuthClientCredentialsProvider` on *every* request (rather
    than baking a static header once) keeps a pooled ``httpx.Client``/``AsyncClient``
    authenticated across token expiry; a 401 forces one re-mint + retry for the rare case the
    token rotates mid-flight. A mint failure fails closed before the provider request is sent.
    """

    def __init__(self, provider: OAuthClientCredentialsProvider) -> None:
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
        except Exception:
            logger.warning("OAuth2 client credential unavailable")
            raise httpx.ProtocolError("OAuth2 client credential unavailable") from None
        yield from self._flow(request, token)

    async def async_auth_flow(self, request: httpx.Request):
        import anyio

        try:
            token = await anyio.to_thread.run_sync(self._provider.get_token)
        except Exception:
            logger.warning("OAuth2 client credential unavailable")
            raise httpx.ProtocolError("OAuth2 client credential unavailable") from None
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


# ---------------------------------------------------------------------------
# Process-wide, complete-configuration-keyed provider cache
# ---------------------------------------------------------------------------

_PROVIDERS: dict[str, OAuthClientCredentialsProvider] = {}
_PROVIDERS_LOCK = threading.Lock()


def _provider_cache_key(
    *,
    token_url: str,
    client_id: str,
    client_secret: str,
    scope: str | None,
    audience: str | None,
    extra_params: dict[str, str] | None,
    tls_profile: str | None,
    tls_profile_ref: str | None,
    timeout: float,
    refresh_skew_seconds: float | None,
    token_auth_style: str,
) -> str:
    material = {
        "token_url": token_url,
        "client_id": client_id,
        "client_secret_digest": hashlib.sha256(
            client_secret.encode("utf-8")
        ).hexdigest(),
        "scope": scope,
        "audience": audience,
        "extra_params": dict(sorted((extra_params or {}).items())),
        "tls_profile": tls_profile,
        "tls_profile_ref_digest": (
            hashlib.sha256(tls_profile_ref.encode("utf-8")).hexdigest()
            if tls_profile_ref
            else None
        ),
        "timeout": float(timeout),
        "refresh_skew_seconds": refresh_skew_seconds,
        "token_auth_style": token_auth_style,
    }
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def get_client_credentials_provider(
    token_url: str,
    client_id: str,
    client_secret: str,
    *,
    scope: str | None = None,
    audience: str | None = None,
    extra_params: dict[str, str] | None = None,
    tls_profile: str | None = None,
    tls_profile_ref: str | None = None,
    timeout: float = 15.0,
    refresh_skew_seconds: float | None = None,
    token_auth_style: Literal["body", "basic"] = "body",
    clock: Callable[[], float] | None = None,
) -> OAuthClientCredentialsProvider:
    """Return a provider shared only by an identical security configuration."""
    key = _provider_cache_key(
        token_url=token_url,
        client_id=client_id,
        client_secret=client_secret,
        scope=scope,
        audience=audience,
        extra_params=extra_params,
        tls_profile=tls_profile,
        tls_profile_ref=tls_profile_ref,
        timeout=timeout,
        refresh_skew_seconds=refresh_skew_seconds,
        token_auth_style=token_auth_style,
    )
    with _PROVIDERS_LOCK:
        existing = _PROVIDERS.get(key)
        if existing is not None:
            return existing
        provider = OAuthClientCredentialsProvider(
            token_url,
            client_id,
            client_secret,
            scope=scope,
            audience=audience,
            extra_params=extra_params,
            tls_profile=tls_profile,
            tls_profile_ref=tls_profile_ref,
            timeout=timeout,
            refresh_skew_seconds=refresh_skew_seconds,
            token_auth_style=token_auth_style,
            clock=clock,
        )
        _PROVIDERS[key] = provider
        return provider


def reset_client_credentials_cache() -> None:
    """Drop every cached provider (test isolation, or a deliberate live secret rotation)."""
    with _PROVIDERS_LOCK:
        for provider in _PROVIDERS.values():
            provider.close()
        _PROVIDERS.clear()


def build_provider_from_config(
    cfg: OAuth2ClientCredentialsConfig | dict[str, Any],
    secrets: Any | None = None,
) -> OAuthClientCredentialsProvider:
    """Resolve ``client_secret``/``client_id`` refs and return the cached provider.

    Reuses the fleet's one runtime secret-reference resolver.  ``env://`` references are
    intentionally resolved directly from the runtime environment so model authentication can
    bootstrap before the graph-backed secret store is available; Vault/OpenBao and engine-backed
    references still use the configured secrets backend.  A caller with no usable secret fails
    loudly at construction time rather than silently minting an unauthenticated client.
    """
    if not isinstance(cfg, OAuth2ClientCredentialsConfig):
        cfg = OAuth2ClientCredentialsConfig.model_validate(cfg)

    def _resolve(reference: str) -> str:
        # Environment references are process-bound bootstrap material.  They
        # must never be delegated to an injected durable/graph-backed resolver:
        # doing so can recursively open or autostart the engine before outbound
        # model authentication is available.  Keep this path on the fleet's
        # canonical runtime-reference resolver even when a backend resolver was
        # supplied for vault:// or secret:// references.
        if reference.startswith("env://"):
            from agent_utilities.security.cli_secrets import (
                resolve_runtime_secret_reference,
            )

            return resolve_runtime_secret_reference(reference)
        if secrets is not None:
            return secrets.resolve_ref(reference)

        # This canonical resolver creates a SecretsClient only for references
        # that genuinely require the configured backend.
        from agent_utilities.security.cli_secrets import (
            resolve_runtime_secret_reference,
        )

        return resolve_runtime_secret_reference(reference)

    try:
        client_secret = _resolve(cfg.client_secret)
    except Exception:
        raise ValueError("oauth2.client_secret did not resolve") from None
    if not client_secret:
        raise ValueError(
            "oauth2.client_secret did not resolve through the configured secrets backend"
        )
    try:
        client_id = (
            _resolve(cfg.client_id)
            if _looks_like_secret_ref(cfg.client_id)
            else cfg.client_id
        )
    except Exception:
        raise ValueError("oauth2.client_id did not resolve") from None
    if not client_id:
        raise ValueError("oauth2.client_id did not resolve")

    return get_client_credentials_provider(
        cfg.token_url,
        client_id,
        client_secret,
        scope=cfg.scope,
        audience=cfg.audience,
        extra_params=cfg.extra_params,
        tls_profile=cfg.tls_profile,
        tls_profile_ref=cfg.tls_profile_ref,
        timeout=cfg.timeout,
        refresh_skew_seconds=cfg.refresh_skew_seconds,
        token_auth_style=cfg.token_auth_style,
    )


def httpx_auth_from_config(
    cfg: OAuth2ClientCredentialsConfig | dict[str, Any] | None,
    secrets: Any | None = None,
) -> OAuth2ClientCredentialsAuth | None:
    """One-shot helper: an oauth2 config block → a ready-to-attach ``httpx.Auth``, or ``None``.

    The single injection seam the LLM (:mod:`agent_utilities.core.model_factory`) and embedding
    (:mod:`agent_utilities.core.embedding_utilities`) client builders — and the graph-os
    registry path (:mod:`agent_utilities.server.dependencies`) — call so a request mints/renews
    its bearer transparently. Returns ``None`` for a falsy ``cfg`` so callers can unconditionally
    do ``httpx.AsyncClient(..., auth=httpx_auth_from_config(model_cfg.oauth2))``.
    """
    if not cfg:
        return None
    provider = build_provider_from_config(cfg, secrets=secrets)
    return OAuth2ClientCredentialsAuth(provider)
