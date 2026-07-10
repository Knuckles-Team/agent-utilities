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
* :class:`OAuthClientCredentialsProvider` — mints, caches, and proactively renews the bearer
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
(``vault://``/``env://``/``sqlite://``/``secret://``), resolved through the existing
:class:`agent_utilities.security.secrets_client.SecretsClient` (Vault/OpenBao/the engine-backed
encrypted store) at token-mint time — never a plaintext value in config. A plaintext
``client_secret`` is rejected at config-validation time (see
:meth:`OAuth2ClientCredentialsConfig._reject_plaintext_secret`).

**Never logged**: log lines below carry only ``token_url``/``client_id``/TTL — never the token
or the secret.
"""

import functools
import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import httpx
import requests
from pydantic import BaseModel, Field, field_validator

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
_SECRET_REF_SCHEMES = ("vault://", "env://", "sqlite://", "secret://")

#: Renew this many seconds before actual expiry when no explicit skew is configured, floored at
#: this constant — short-TTL tokens (e.g. 120s) still get a meaningful renewal buffer.
DEFAULT_REFRESH_SKEW_SECONDS = 60.0

#: ... or this fraction of the observed TTL, whichever is larger — long-TTL tokens (e.g. 3600s)
#: get a proportionally bigger buffer instead of cutting it close every time.
DEFAULT_REFRESH_SKEW_FRACTION = 0.20

#: Conservative assumed token lifetime before the first mint reveals the IdP's real
#: ``expires_in`` (mirrors ``mcp.client_credentials._DEFAULT_TOKEN_TTL_S``).
_DEFAULT_TOKEN_TTL_S = 300.0


def _looks_like_secret_ref(value: str) -> bool:
    """Whether ``value`` is a secret-reference URI rather than a literal."""
    return "://" in value


class OAuth2ClientCredentialsConfig(BaseModel):
    """Declarative OAuth2 ``client_credentials`` block for a chat/embedding provider.

    CONCEPT:AU-OS.identity.oauth2-client-credentials-lifecycle. Mutually exclusive with a static
    ``api_key`` at the owning config's level (enforced there, not here, since that field lives on
    :class:`ChatModelConfig`/:class:`EmbeddingModelConfig`/:class:`ModelDefinition`).
    """

    token_url: str = Field(description="OIDC/OAuth2 token endpoint (e.g. Azure AD /oauth2/v2.0/token).")
    client_id: str = Field(
        description=(
            "Client id. A literal value, or a secret-reference URI "
            "(vault://, env://, sqlite://, secret://) when the client id itself is sensitive."
        )
    )
    client_secret: str = Field(
        description=(
            "Secret-reference URI resolving to the client secret — vault://, env://, "
            "sqlite://, or secret://. A plaintext value is rejected at validation time."
        )
    )
    scope: str | None = Field(
        default=None,
        description="Space-separated scopes. Azure AD v2: '<resource>/.default'.",
    )
    audience: str | None = Field(
        default=None,
        description="Audience/resource-indicator parameter (Auth0/Okta-style IdPs).",
    )
    extra_params: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Extra form params merged into the token request body verbatim "
            "(e.g. Azure AD v1 'resource')."
        ),
    )
    verify: bool = Field(default=True, description="TLS verification for the token request.")
    timeout: float = Field(default=15.0, description="Token request timeout, in seconds.")
    refresh_skew_seconds: float | None = Field(
        default=None,
        description=(
            "Explicit renew-before-expiry skew, in seconds. Unset (None) auto-derives "
            "max(60s, 20% of the observed access-token TTL)."
        ),
    )

    model_config = {"extra": "forbid"}

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


def resolve_effective_skew(ttl_seconds: float, explicit_skew: float | None) -> float:
    """Renew-before-expiry skew: the explicit override, or ``max(60s, 20% of ttl)``.

    An explicit ``refresh_skew_seconds`` always wins (an operator's considered choice); the
    auto-derived default scales with the IdP's actual token lifetime so a short-TTL token (say
    120s) still gets a meaningful buffer (the 60s floor) while a long-TTL one (3600s) renews
    proportionally earlier (720s) instead of cutting every refresh close.
    """
    if explicit_skew is not None:
        return max(0.0, explicit_skew)
    return max(DEFAULT_REFRESH_SKEW_SECONDS, ttl_seconds * DEFAULT_REFRESH_SKEW_FRACTION)


class OAuthClientCredentialsProvider:
    """Thread-safe, self-refreshing OAuth2 ``client_credentials`` token cache.

    Mints via ``POST token_url`` with ``grant_type=client_credentials`` (+
    ``client_id``/``client_secret``/``scope``/``audience``/``extra_params``), parses
    ``access_token``/``expires_in``, and proactively renews before expiry
    (:func:`resolve_effective_skew`). ``clock`` is injectable so tests drive expiry
    deterministically instead of sleeping real time.

    Never logs the token or the secret — only ``token_url``/``client_id``/TTL.
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
        verify: bool = True,
        timeout: float = 15.0,
        refresh_skew_seconds: float | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.token_url = token_url
        self.client_id = client_id
        self._client_secret = client_secret
        self.scope = scope
        self.audience = audience
        self.extra_params = dict(extra_params or {})
        self.verify = verify
        self.timeout = timeout
        self._explicit_skew = refresh_skew_seconds
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._token: str | None = None
        self._expires_at = 0.0
        self._ttl_seconds = 0.0  # last observed expires_in

    @property
    def access_token_ttl(self) -> float:
        """Last observed access-token lifetime, or a conservative default before the first mint."""
        return self._ttl_seconds if self._ttl_seconds > 0 else _DEFAULT_TOKEN_TTL_S

    def _effective_skew(self) -> float:
        return resolve_effective_skew(self.access_token_ttl, self._explicit_skew)

    def get_token(self, *, force: bool = False) -> str:
        """Return a cached bearer, minting/renewing it first when missing, stale, or ``force``.

        ``force=True`` bypasses the cache unconditionally — used for the one-shot re-mint after
        a 401 (the cached token rotated/was revoked between the skew check and the request
        reaching the endpoint).
        """
        with self._lock:
            now = self._clock()
            if not force and self._token and now < self._expires_at - self._effective_skew():
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
        data: dict[str, str] = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self._client_secret,
        }
        if self.scope:
            data["scope"] = self.scope
        if self.audience:
            data["audience"] = self.audience
        data.update(self.extra_params)
        resp = requests.post(
            self.token_url,
            data=data,
            verify=self.verify,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        if not isinstance(payload, dict) or "access_token" not in payload:
            # Never echo the response body — it may itself carry token-adjacent material.
            raise ValueError(
                f"OAuth2 token response from {self.token_url} is missing 'access_token'."
            )
        self._token = str(payload["access_token"])
        self._ttl_seconds = float(payload.get("expires_in", _DEFAULT_TOKEN_TTL_S))
        self._expires_at = now + self._ttl_seconds
        logger.info(
            "Minted OAuth2 client-credentials token (token_url=%s, client_id=%s, ttl=%ss)",
            self.token_url,
            self.client_id,
            int(self._ttl_seconds),
        )


class OAuth2ClientCredentialsAuth(httpx.Auth):
    """Per-request bearer injection for a long-lived LLM/embedding client.

    Mirrors :class:`agent_utilities.mcp.client_credentials.ClientCredentialsAuth` (OS-5.32),
    generalized for outbound provider clients instead of child-MCP sessions. Pulling the token
    from the self-refreshing :class:`OAuthClientCredentialsProvider` on *every* request (rather
    than baking a static header once) keeps a pooled ``httpx.Client``/``AsyncClient``
    authenticated across token expiry; a 401 forces one re-mint + retry for the rare case the
    token rotates mid-flight. Never raises: a mint failure degrades to an unauthenticated
    request (the provider then 401s — visible in logs/metrics rather than crashing the caller).
    """

    def __init__(self, provider: OAuthClientCredentialsProvider) -> None:
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
            logger.warning(
                "Could not mint OAuth2 client-credentials token for %s: %s",
                self._provider.token_url,
                exc,
            )
            token = None
        yield from self._flow(request, token)

    async def async_auth_flow(self, request: httpx.Request):
        import anyio

        try:
            token: str | None = await anyio.to_thread.run_sync(self._provider.get_token)
        except Exception as exc:  # pragma: no cover - degrade to no header
            logger.warning(
                "Could not mint OAuth2 client-credentials token for %s: %s",
                self._provider.token_url,
                exc,
            )
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


# ---------------------------------------------------------------------------
# Process-wide, (token_url, client_id, scope)-keyed provider cache
# ---------------------------------------------------------------------------

_PROVIDERS: dict[tuple[str, str, str | None], OAuthClientCredentialsProvider] = {}
_PROVIDERS_LOCK = threading.Lock()


def get_client_credentials_provider(
    token_url: str,
    client_id: str,
    client_secret: str,
    *,
    scope: str | None = None,
    audience: str | None = None,
    extra_params: dict[str, str] | None = None,
    verify: bool = True,
    timeout: float = 15.0,
    refresh_skew_seconds: float | None = None,
    clock: Callable[[], float] | None = None,
) -> OAuthClientCredentialsProvider:
    """Return the process-wide provider for ``(token_url, client_id, scope)``, building it once.

    The cache key deliberately excludes ``client_secret``/``audience``/``extra_params`` — every
    LLM/embedding client built for the same ``(token_url, client_id, scope)`` triple shares one
    token cache and one in-flight renewal, instead of each caller minting its own. Rotating the
    secret for an already-cached key needs :func:`reset_client_credentials_cache` (tests / a rare
    live secret rotation).
    """
    key = (token_url, client_id, scope)
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
            verify=verify,
            timeout=timeout,
            refresh_skew_seconds=refresh_skew_seconds,
            clock=clock,
        )
        _PROVIDERS[key] = provider
        return provider


def reset_client_credentials_cache() -> None:
    """Drop every cached provider (test isolation, or a deliberate live secret rotation)."""
    with _PROVIDERS_LOCK:
        _PROVIDERS.clear()


def build_provider_from_config(
    cfg: OAuth2ClientCredentialsConfig | dict[str, Any],
    secrets: Any | None = None,
) -> OAuthClientCredentialsProvider:
    """Resolve ``client_secret``/``client_id`` refs and return the cached provider.

    Reuses :class:`agent_utilities.security.secrets_client.SecretsClient` — the fleet's ONE
    secret-reference resolver (Vault/OpenBao/the engine-backed encrypted store/``env://``) —
    rather than inventing a second one. Raises if a required ref does not resolve: a caller with
    no usable secret must fail loudly at construction time, not silently mint an unauthenticated
    client.
    """
    if not isinstance(cfg, OAuth2ClientCredentialsConfig):
        cfg = OAuth2ClientCredentialsConfig.model_validate(cfg)

    if secrets is None:
        from agent_utilities.security.secrets_client import create_secrets_client

        secrets = create_secrets_client()

    client_secret = secrets.resolve_ref(cfg.client_secret)
    if not client_secret:
        raise ValueError(
            f"oauth2.client_secret ref {cfg.client_secret!r} did not resolve to a value via "
            "the secrets backend — check the secret exists and the reference is correct."
        )
    client_id = (
        secrets.resolve_ref(cfg.client_id)
        if _looks_like_secret_ref(cfg.client_id)
        else cfg.client_id
    )
    if not client_id:
        raise ValueError(f"oauth2.client_id ref {cfg.client_id!r} did not resolve to a value.")

    return get_client_credentials_provider(
        cfg.token_url,
        client_id,
        client_secret,
        scope=cfg.scope,
        audience=cfg.audience,
        extra_params=cfg.extra_params,
        verify=cfg.verify,
        timeout=cfg.timeout,
        refresh_skew_seconds=cfg.refresh_skew_seconds,
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
