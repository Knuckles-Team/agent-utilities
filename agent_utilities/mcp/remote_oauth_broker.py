#!/usr/bin/python
from __future__ import annotations

"""Remote browser-OAuth MCP broker — provider-agnostic per-principal token custody.

CONCEPT:AU-ECO.mcp.remote-oauth-broker

GOC-85 (U-11 / U-41 / U-43 / U-44 / U-45). Builds the credential class this repo has
never had: a per-user, per-provider, per-remote-resource OAuth token, minted through a
real browser-mediated authorization flow and owned by a server-side broker rather than
the single-process local helper (:mod:`agent_utilities.security.browser_auth`, U-43 —
never imported here) or the fleet-global multiplexer child/session maps (U-44/U-45 —
never touched here; see "What this module deliberately does NOT do" below).

Confirmed by source inspection before this module was written (matches the fail-closed
regression pins in ``tests/unit/mcp/test_remote_oauth_fail_closed.py``, landed in
``6c1606cbb``): no ``TokenStorage``/``OAuthClientProvider`` consumer, no server callback
route, and no per-principal token store existed anywhere in this repo. Those pins are
left completely untouched by this change — this module adds a new, self-contained
broker; it does not wire a delegated user identity into
:mod:`agent_utilities.mcp.multiplexer` or :mod:`agent_utilities.mcp.server_factory`.

Ten-part architecture (lane doc ``plans/graph-os-completion-program/lanes/
GOC-85-remote-browser-oauth-mcp-broker.md``), what is implemented here and what is
deliberately deferred:

1. Provider registration — :class:`ProviderDescriptor` / :class:`ProviderRegistry`.
   Administrator-populated only; never caller-supplied at request time.               DONE
2. RFC 9728 + RFC 8414 discovery — :func:`discover_protected_resource` /
   :func:`discover_authorization_server`. HTTPS-only, bounded, issuer-consistent.     DONE
3. Client registration — pre-registered public-PKCE metadata on the descriptor.
   Dynamic client registration is a pluggable hook (:class:`DynamicClientRegistrar`)
   with no default implementation.                                        SPECIFIED, NOT WIRED
4. Authorization transaction — :class:`OAuthTransaction` / :class:`TransactionStore`,
   encrypted, session-bound, short TTL.                                               DONE
5. Callback validation — exact redirect, single-use state, same-principal/session
   binding, scope-widening rejection — :meth:`RemoteOAuthBroker.callback`.            DONE
6. Token store — :class:`OAuthTokenStore`, versioned-key encrypted, keyed by
   tenant/principal/provider/resource/audience.                                       DONE
7. Refresh/revocation — per-token lock, atomic rotation, fail-closed revoke.           DONE
8. Per-principal remote MCP session (U-44) — attaching this broker's tokens to the
   multiplexer's outbound child/session model.                              NOT IMPLEMENTED
9. Per-call authorization in the multiplexer (U-44/U-45 wiring).            NOT IMPLEMENTED
10. Sanitized audit — :func:`_audit`, reusing this repo's "never log the secret" norm
    (the U-54 hygiene filter in :mod:`agent_utilities.mcp.oauth_log_hygiene` covers the
    *vendored SDK's* loggers; this module's own audit calls never pass code/state/
    verifier/token values to a log call in the first place, so there is nothing for a
    filter to redact).                                                                DONE

What this module deliberately does NOT do, and why
----------------------------------------------------
GOC-85's own lane contract hard-blocks this lane on GOC-15 (verified identity carrier —
the contract this broker's per-principal session isolation "must be consistent with")
and GOC-51 (identity/secrets/supply-chain closure this lane's token-custody design "must
fit rather than duplicate"). Both are, at the time of writing, still status ``PROPOSED``
in the program ledger — neither has landed a single commit. Wiring per-principal remote
MCP sessions (W06) or per-call authorization (W08) into the fleet-global
:mod:`agent_utilities.mcp.multiplexer` today would mean inventing a session/principal
model in a vacuum, in the exact file the U-44/U-45 canary tests in
``test_remote_oauth_fail_closed.py`` fence specifically because an *incidental* wiring
there — done without GOC-15's carrier contract to be consistent with — is the unsafe
shortcut this whole theme rules out ("that needs a deliberate design decision, not an
incidental import").

So this module ships the broker CORE — discovery, transaction, callback validation,
encrypted per-principal token custody, refresh, revocation, sanitized audit — as a
free-standing capability with no caller yet. It reuses, rather than reinvents, the ONE
verified-identity primitive this repo already has
(:class:`agent_utilities.security.brain_context.ActorContext`, server-minted by
:mod:`agent_utilities.security.request_identity`'s ``ActorIdentityMiddleware``): every
broker entrypoint requires an ``authenticated=True`` actor with a non-empty
``actor_id``/``tenant_id`` and derives storage identity from it — never from a
caller-supplied string. That satisfies W07 ("the broker process authenticates each
browser user individually") using GOC-15's *existing* AU-side carrier rather than
inventing a second one; full cross-surface (EG/WebUI) carrier uniformity is GOC-15's own
remaining scope.

Multiplexer attachment (W06/W08) is intentionally left unwired and specified only: a
:class:`RemoteOAuthBroker` instance exposes :meth:`RemoteOAuthBroker.bearer_headers_for`
(endpoint-bound: the resource URL passed in must equal the provider's registered exact
resource URL or it raises) as the seam a future per-principal multiplexer child would
call per outbound request — but nothing here reaches into
:mod:`agent_utilities.mcp.multiplexer`'s children/session maps.

Real-provider validation stops at the authorization URL. ``begin()`` is the only method
exercised in this lane against anything that could be a real provider, and even that is
only ever exercised in tests against a mock authorization server — no real provider is
registered, enabled, or contacted by this change. ``callback()``/``refresh()``/
``revoke()`` are fully implemented and tested exclusively against mock HTTP transports;
enabling any of them against a live third-party IdP is explicitly out of scope here
("stop at a user-mediated consent URL for real-provider validation").
"""

import base64
import hashlib
import json
import logging
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlencode, urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent_utilities.security.brain_context import ActorContext

logger = logging.getLogger(__name__)
_audit_logger = logging.getLogger("agent_utilities.mcp.remote_oauth_broker.audit")

__all__ = [
    "AuthorizationServerMetadata",
    "DynamicClientRegistrar",
    "OAuthBindingError",
    "OAuthDiscoveryError",
    "OAuthProviderError",
    "OAuthRedirectNotAllowlistedError",
    "OAuthRefreshRaceError",
    "OAuthRevokedError",
    "OAuthScopeError",
    "OAuthStateError",
    "OAuthTokenAbsentError",
    "OAuthTransaction",
    "ProtectedResourceMetadata",
    "ProviderDescriptor",
    "ProviderRegistry",
    "RemoteOAuthBroker",
    "StoredToken",
    "TransactionStore",
    "OAuthTokenStore",
    "discover_authorization_server",
    "discover_protected_resource",
]

# ---------------------------------------------------------------------------
# Bounds — explicit, numeric, never open-ended (lane contract: "Algorithmic
# and resource budget").
# ---------------------------------------------------------------------------
_MAX_DISCOVERY_RESPONSE_BYTES = 256 * 1024
_DISCOVERY_TIMEOUT_S = 10.0
_MAX_TOKEN_RESPONSE_BYTES = 1024 * 1024
_TOKEN_EXCHANGE_TIMEOUT_S = 20.0
_TRANSACTION_TTL_S = 300.0
_MAX_STATE_LEN = 256
_CONSUMED_TOMBSTONE = "__consumed__"


# ---------------------------------------------------------------------------
# Errors — one exception type per fail-closed boundary, never a bare bool.
# ---------------------------------------------------------------------------
class OAuthProviderError(RuntimeError):
    """Unknown, disabled, or otherwise rejected provider registration."""


class OAuthDiscoveryError(RuntimeError):
    """RFC 9728 / RFC 8414 discovery failed, was malformed, or was inconsistent."""


class OAuthStateError(RuntimeError):
    """The transaction ``state`` was missing, unknown, replayed, or expired."""


class OAuthBindingError(RuntimeError):
    """The callback did not bind to the transaction's principal/tenant/session."""


class OAuthRedirectNotAllowlistedError(RuntimeError):
    """The requested redirect URI is not the provider's exact registered value."""


class OAuthScopeError(RuntimeError):
    """The authorization server granted a scope outside what was requested."""


class OAuthTokenAbsentError(RuntimeError):
    """No stored token/refresh-token exists for this principal/provider/resource."""


class OAuthRevokedError(RuntimeError):
    """The stored record is revoked (or its revocation status is ambiguous)."""


class OAuthRefreshRaceError(RuntimeError):
    """The stored record changed underneath a refresh; retry."""


# ---------------------------------------------------------------------------
# 1. Provider registration — administrator-approved only.
# ---------------------------------------------------------------------------
class ProviderDescriptor(BaseModel):
    """One administrator-approved remote-MCP OAuth provider.

    Never constructed from caller-supplied request data — a
    :class:`ProviderRegistry` is populated only by deployment/administration code.
    Abstraction seam for "abstraction-first, no opinionation": nothing here is
    specific to one vendor's quirks; a provider is fully described by this model.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_id: str = Field(min_length=1, max_length=128)
    resource_url: str = Field(
        min_length=1,
        max_length=2048,
        description="Exact remote MCP resource URL. Never a caller-supplied value.",
    )
    authorization_server_url: str | None = Field(
        default=None,
        max_length=2048,
        description=(
            "Explicit authorization-server issuer to use when the protected-resource "
            "metadata lists more than one. Unset picks the resource metadata's first "
            "entry."
        ),
    )
    client_id: str = Field(
        min_length=1,
        max_length=512,
        description="Pre-registered public PKCE client id for the deployment callback.",
    )
    redirect_uri: str = Field(
        min_length=1,
        max_length=2048,
        description="Exact deployment callback route. The sole allowlisted redirect.",
    )
    scopes: tuple[str, ...] = Field(default_factory=tuple)
    enabled: bool = Field(
        default=False,
        description="Staged rollout gate — disabled providers reject begin()/callback().",
    )

    @field_validator("resource_url", "redirect_uri")
    @classmethod
    def _require_https(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme.casefold() != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError("must be an exact HTTPS URL without embedded credentials")
        return value

    @field_validator("authorization_server_url")
    @classmethod
    def _optional_https(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return cls._require_https(value)


class ProviderRegistry:
    """Administrator-populated provider set. No request-time registration exists."""

    def __init__(self) -> None:
        self._providers: dict[str, ProviderDescriptor] = {}

    def register(self, descriptor: ProviderDescriptor) -> None:
        """Administrator-only call — never reachable from an MCP tool argument."""
        self._providers[descriptor.provider_id] = descriptor

    def get(self, provider_id: str) -> ProviderDescriptor | None:
        return self._providers.get(provider_id)

    def require_enabled(self, provider_id: str) -> ProviderDescriptor:
        provider = self._providers.get(str(provider_id or ""))
        if provider is None or not provider.enabled:
            raise OAuthProviderError("unknown or disabled provider")
        return provider


# ---------------------------------------------------------------------------
# 2. Discovery — RFC 9728 protected-resource, RFC 8414 authorization-server.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProtectedResourceMetadata:
    resource: str
    authorization_servers: tuple[str, ...]


@dataclass(frozen=True)
class AuthorizationServerMetadata:
    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    code_challenge_methods_supported: tuple[str, ...]
    grant_types_supported: tuple[str, ...]


def _bounded_json_get(client: httpx.Client, url: str) -> dict[str, Any]:
    parsed = urlsplit(url)
    if parsed.scheme.casefold() != "https" or not parsed.hostname:
        raise OAuthDiscoveryError("discovery endpoint must be an exact HTTPS URL")
    try:
        with client.stream("GET", url, headers={"Accept": "application/json"}) as resp:
            resp.raise_for_status()
            body = bytearray()
            for chunk in resp.iter_bytes(65536):
                body.extend(chunk)
                if len(body) > _MAX_DISCOVERY_RESPONSE_BYTES:
                    raise OAuthDiscoveryError("discovery response exceeded its bound")
    except httpx.HTTPError as exc:
        raise OAuthDiscoveryError("discovery request failed") from exc
    try:
        payload = json.loads(bytes(body))
    except (ValueError, UnicodeDecodeError):
        raise OAuthDiscoveryError("discovery response was not valid JSON") from None
    if not isinstance(payload, dict):
        raise OAuthDiscoveryError("discovery response had an unexpected shape")
    return payload


def _protected_resource_metadata_url(resource_url: str) -> str:
    """RFC 9728 §3.1 well-known URI construction."""
    parsed = urlsplit(resource_url)
    path = parsed.path.rstrip("/")
    well_known = f"/.well-known/oauth-protected-resource{path}"
    return f"{parsed.scheme}://{parsed.netloc}{well_known}"


def _authorization_server_metadata_url(issuer: str) -> str:
    """RFC 8414 §3.1 well-known URI construction (insert before the issuer's path)."""
    parsed = urlsplit(issuer)
    path = parsed.path.rstrip("/")
    well_known = f"/.well-known/oauth-authorization-server{path}"
    return f"{parsed.scheme}://{parsed.netloc}{well_known}"


def discover_protected_resource(
    client: httpx.Client, resource_url: str
) -> ProtectedResourceMetadata:
    """RFC 9728: fetch the protected-resource metadata for ``resource_url``."""
    metadata_url = _protected_resource_metadata_url(resource_url)
    payload = _bounded_json_get(client, metadata_url)
    resource = payload.get("resource")
    servers = payload.get("authorization_servers")
    if (
        not isinstance(resource, str)
        or not resource
        or not isinstance(servers, list)
        or not servers
        or not all(isinstance(s, str) and s for s in servers)
    ):
        raise OAuthDiscoveryError(
            "protected-resource metadata is missing resource/authorization_servers"
        )
    return ProtectedResourceMetadata(
        resource=resource, authorization_servers=tuple(servers)
    )


def discover_authorization_server(
    client: httpx.Client,
    resource_metadata: ProtectedResourceMetadata,
    *,
    authorization_server_url: str | None = None,
) -> AuthorizationServerMetadata:
    """RFC 8414: fetch + validate the authorization-server metadata.

    Requires HTTPS, a matching issuer (issuer consistency — the returned ``issuer``
    field must equal the URL that was queried), the authorization_code grant, and
    PKCE S256 support. Any violation fails closed with :class:`OAuthDiscoveryError`.
    """
    issuer_url = authorization_server_url or resource_metadata.authorization_servers[0]
    if issuer_url not in resource_metadata.authorization_servers:
        raise OAuthDiscoveryError(
            "configured authorization server is not among the resource's declared servers"
        )
    metadata_url = _authorization_server_metadata_url(issuer_url)
    payload = _bounded_json_get(client, metadata_url)

    issuer = payload.get("issuer")
    authorization_endpoint = payload.get("authorization_endpoint")
    token_endpoint = payload.get("token_endpoint")
    challenge_methods = payload.get("code_challenge_methods_supported") or []
    grant_types = payload.get("grant_types_supported") or [
        "authorization_code",
        "implicit",
    ]

    if not isinstance(issuer, str) or issuer.rstrip("/") != issuer_url.rstrip("/"):
        raise OAuthDiscoveryError("authorization-server issuer is inconsistent")
    if not isinstance(authorization_endpoint, str) or not isinstance(
        token_endpoint, str
    ):
        raise OAuthDiscoveryError("authorization-server metadata is incomplete")
    for endpoint in (authorization_endpoint, token_endpoint):
        parsed = urlsplit(endpoint)
        if parsed.scheme.casefold() != "https" or not parsed.hostname:
            raise OAuthDiscoveryError("authorization-server endpoints must be HTTPS")
    if not isinstance(challenge_methods, list) or "S256" not in challenge_methods:
        raise OAuthDiscoveryError("authorization server does not support PKCE S256")
    if not isinstance(grant_types, list) or "authorization_code" not in grant_types:
        raise OAuthDiscoveryError(
            "authorization server does not support the authorization_code grant"
        )
    return AuthorizationServerMetadata(
        issuer=issuer,
        authorization_endpoint=authorization_endpoint,
        token_endpoint=token_endpoint,
        code_challenge_methods_supported=tuple(challenge_methods),
        grant_types_supported=tuple(grant_types),
    )


class DynamicClientRegistrar(Protocol):
    """Pluggable RFC 7591 dynamic-client-registration hook.

    No default implementation ships — the safer, unopinionated default is
    pre-registered ``client_id``/``redirect_uri`` on :class:`ProviderDescriptor`.
    A deployment that needs DCR supplies an implementation of this protocol; the
    broker never invents or silently switches a client id/issuer on its own.
    """

    def register(
        self, provider: ProviderDescriptor, as_metadata: AuthorizationServerMetadata
    ) -> str:
        """Return the registered ``client_id`` for ``provider`` at this issuer."""
        ...


# ---------------------------------------------------------------------------
# 4. Authorization transaction — encrypted, session-bound, short TTL.
# ---------------------------------------------------------------------------
def _generate_pkce() -> tuple[str, str]:
    """PKCE S256 verifier/challenge. Deliberately not imported from
    ``security.browser_auth`` — see the module docstring's U-43 rationale."""
    verifier = secrets.token_urlsafe(64)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("utf-8")).digest())
        .decode("utf-8")
        .rstrip("=")
    )
    return verifier, challenge


@dataclass(frozen=True)
class OAuthTransaction:
    state: str
    nonce: str
    verifier: str
    provider_id: str
    tenant_id: str
    principal_id: str
    browser_session_id: str
    redirect_uri: str
    scope: str
    created_at: float
    expires_at: float

    def is_expired(self, now: float) -> bool:
        return now >= self.expires_at

    def to_json(self) -> str:
        return json.dumps(
            {
                "state": self.state,
                "nonce": self.nonce,
                "verifier": self.verifier,
                "provider_id": self.provider_id,
                "tenant_id": self.tenant_id,
                "principal_id": self.principal_id,
                "browser_session_id": self.browser_session_id,
                "redirect_uri": self.redirect_uri,
                "scope": self.scope,
                "created_at": self.created_at,
                "expires_at": self.expires_at,
            },
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, raw: str) -> OAuthTransaction:
        data = json.loads(raw)
        return cls(**data)


class TransactionStore:
    """Session-bound encrypted transaction storage, single-use by atomic claim.

    Backed by the same :class:`~agent_utilities.security.secrets_client.SecretsClient`
    every other secret in this repo uses (engine-backed encrypted store, or
    OpenBao/Vault when ``SECRETS_BACKEND=vault`` — never a new store). Single-use is
    enforced with :meth:`SecretsClient.compare_and_set` (already atomic on the durable
    engine backend, the same primitive signing-key rotation uses) rather than a
    read-then-delete race: exactly one caller's compare-and-set from the exact bytes it
    read to a tombstone can win, so a concurrent replay of the same ``state`` fails
    closed even under real concurrency, not just under one process's lock.
    """

    def __init__(self, secrets_client: Any) -> None:
        self._secrets = secrets_client

    @staticmethod
    def _key(state: str) -> str:
        return f"oauth-tx:{state}"

    def put(self, tx: OAuthTransaction) -> None:
        self._secrets.set(self._key(tx.state), tx.to_json(), kind="oauth-transaction")

    def consume_once(self, state: str, *, now: float | None = None) -> OAuthTransaction:
        if (
            not isinstance(state, str)
            or not state
            or len(state) > _MAX_STATE_LEN
            or any(ord(character) < 32 or ord(character) == 127 for character in state)
        ):
            raise OAuthStateError("malformed state")
        key = self._key(state)
        raw = self._secrets.get(key)
        if raw is None or raw == _CONSUMED_TOMBSTONE:
            raise OAuthStateError("unknown, expired, or already-consumed state")
        claimed = self._secrets.compare_and_set(key, raw, _CONSUMED_TOMBSTONE)
        if not claimed:
            raise OAuthStateError("state was concurrently consumed (replay)")
        self._secrets.delete(key)
        try:
            tx = OAuthTransaction.from_json(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            raise OAuthStateError("stored transaction was malformed") from None
        if tx.state != state or tx.is_expired(now if now is not None else time.time()):
            raise OAuthStateError("authorization transaction expired or mismatched")
        return tx


# ---------------------------------------------------------------------------
# 6/7. Token store — encrypted, versioned-key, per-principal; refresh/revoke.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StoredToken:
    access_token: str
    refresh_token: str | None
    token_type: str
    expires_at: float
    granted_scope: str
    key_version: int
    audience: str


_TOKEN_LOCKS: dict[str, threading.Lock] = {}
_TOKEN_LOCKS_GUARD = threading.Lock()


def _lock_for(storage_key: str) -> threading.Lock:
    """Per-token lock, never a global lock — concurrent refreshes for different
    tokens must not serialize against each other (lane resource budget)."""
    with _TOKEN_LOCKS_GUARD:
        lock = _TOKEN_LOCKS.get(storage_key)
        if lock is None:
            lock = threading.Lock()
            _TOKEN_LOCKS[storage_key] = lock
        return lock


class OAuthTokenStore:
    """Encrypted token custody keyed by tenant/principal/provider/resource/audience.

    The storage key is DERIVED ONLY from a verified, authenticated
    :class:`ActorContext` — never from a caller-supplied tenant/principal string.
    This is the structural mechanism, not a policy statement, behind "one user must
    never be able to reach another's token": there is no code path in this class that
    accepts an arbitrary principal identifier, so a caller can only ever address ITS
    OWN storage key.
    """

    CURRENT_KEY_VERSION = 1

    def __init__(self, secrets_client: Any) -> None:
        self._secrets = secrets_client

    @staticmethod
    def _require_verified(actor: ActorContext) -> tuple[str, str]:
        tenant = str(actor.tenant_id or "").strip()
        principal = str(actor.actor_id or "").strip()
        if not actor.authenticated or not tenant or not principal:
            raise PermissionError(
                "OAuth token custody requires a verified, authenticated principal "
                "with a tenant claim"
            )
        return tenant, principal

    @staticmethod
    def _storage_key(
        tenant: str, principal: str, provider_id: str, resource_url: str, audience: str
    ) -> str:
        # Key NAMES stay queryable plaintext metadata in the secrets backend (see
        # SecretsClient's own docstring); hash the principal-identifying components so
        # the key namespace itself never carries a readable actor id / tenant name.
        digest = hashlib.sha256(
            "\n".join((tenant, principal, provider_id, resource_url, audience)).encode(
                "utf-8"
            )
        ).hexdigest()
        return f"oauth-token:{digest}"

    def put(
        self,
        *,
        actor: ActorContext,
        provider_id: str,
        resource_url: str,
        audience: str,
        token: StoredToken,
    ) -> None:
        tenant, principal = self._require_verified(actor)
        key = self._storage_key(tenant, principal, provider_id, resource_url, audience)
        record = {
            "access_token": token.access_token,
            "refresh_token": token.refresh_token,
            "token_type": token.token_type,
            "expires_at": token.expires_at,
            "granted_scope": token.granted_scope,
            "key_version": token.key_version,
            "audience": token.audience,
            "revoked": False,
        }
        self._secrets.set(key, json.dumps(record, separators=(",", ":")))

    def get(
        self,
        *,
        actor: ActorContext,
        provider_id: str,
        resource_url: str,
        audience: str,
    ) -> StoredToken | None:
        """Return the caller's OWN token, or ``None``. Never another principal's."""
        tenant, principal = self._require_verified(actor)
        key = self._storage_key(tenant, principal, provider_id, resource_url, audience)
        raw = self._secrets.get(key)
        if raw is None:
            return None
        try:
            data = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict) or data.get("revoked"):
            # Fail closed: a revoked or malformed record reads as absent, never as a
            # usable token.
            return None
        try:
            return StoredToken(
                access_token=str(data["access_token"]),
                refresh_token=data.get("refresh_token"),
                token_type=str(data.get("token_type", "Bearer")),
                expires_at=float(data["expires_at"]),
                granted_scope=str(data.get("granted_scope", "")),
                key_version=int(data.get("key_version", self.CURRENT_KEY_VERSION)),
                audience=str(data.get("audience", audience)),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def refresh(
        self,
        *,
        actor: ActorContext,
        provider: ProviderDescriptor,
        resource_url: str,
        audience: str,
        http_client: httpx.Client,
    ) -> StoredToken:
        """Rotate the refresh token atomically under a per-token lock.

        Fails closed: an absent record, a revoked record, or a missing refresh
        token all raise rather than degrading to "nothing to do" — the caller must
        treat any of these as "re-authenticate", never as "proceed unauthenticated".
        """
        tenant, principal = self._require_verified(actor)
        key = self._storage_key(
            tenant, principal, provider.provider_id, resource_url, audience
        )
        lock = _lock_for(key)
        with lock:
            raw = self._secrets.get(key)
            if raw is None:
                raise OAuthTokenAbsentError(
                    "no stored token for this principal/provider"
                )
            try:
                current = json.loads(raw)
            except (TypeError, ValueError, json.JSONDecodeError):
                raise OAuthRevokedError(
                    "stored record is unreadable; ambiguous -> revoked"
                ) from None
            if not isinstance(current, dict) or current.get("revoked"):
                raise OAuthRevokedError("token is revoked")
            refresh_token = current.get("refresh_token")
            if not refresh_token:
                raise OAuthTokenAbsentError("no refresh token on this record")
            payload = _exchange_refresh_token(http_client, provider, refresh_token)
            new_record = {
                "access_token": payload["access_token"],
                "refresh_token": payload.get("refresh_token", refresh_token),
                "token_type": payload.get("token_type", "Bearer"),
                "expires_at": time.time() + float(payload.get("expires_in", 3600)),
                "granted_scope": payload.get("scope", current.get("granted_scope", "")),
                "key_version": self.CURRENT_KEY_VERSION,
                "audience": audience,
                "revoked": False,
            }
            rotated = self._secrets.compare_and_set(
                key, raw, json.dumps(new_record, separators=(",", ":"))
            )
            if not rotated:
                raise OAuthRefreshRaceError(
                    "token record changed during refresh; retry"
                )
            return StoredToken(
                access_token=new_record["access_token"],
                refresh_token=new_record["refresh_token"],
                token_type=new_record["token_type"],
                expires_at=new_record["expires_at"],
                granted_scope=new_record["granted_scope"],
                key_version=new_record["key_version"],
                audience=new_record["audience"],
            )

    def revoke(
        self,
        *,
        actor: ActorContext,
        provider_id: str,
        resource_url: str,
        audience: str,
    ) -> None:
        """Fail-closed revoke: always writes a tombstone record.

        An ambiguous outcome (record already gone, backend write partially failed,
        concurrent mutation) must never be readable as "still valid" — so this writes
        an explicit ``revoked: true`` record rather than merely attempting a delete.
        :meth:`get` treats ``revoked`` the same as absent.
        """
        tenant, principal = self._require_verified(actor)
        key = self._storage_key(tenant, principal, provider_id, resource_url, audience)
        self._secrets.set(
            key,
            json.dumps(
                {"revoked": True, "revoked_at": time.time()}, separators=(",", ":")
            ),
        )


# ---------------------------------------------------------------------------
# Token exchange (mock-server tested only in this lane; see module docstring)
# ---------------------------------------------------------------------------
def _bounded_token_post(
    client: httpx.Client, url: str, data: dict[str, str]
) -> dict[str, Any]:
    parsed = urlsplit(url)
    if parsed.scheme.casefold() != "https" or not parsed.hostname:
        raise OAuthDiscoveryError("token endpoint must be an exact HTTPS URL")
    try:
        with client.stream("POST", url, data=data) as resp:
            resp.raise_for_status()
            body = bytearray()
            for chunk in resp.iter_bytes(65536):
                body.extend(chunk)
                if len(body) > _MAX_TOKEN_RESPONSE_BYTES:
                    raise OAuthDiscoveryError("token response exceeded its bound")
    except httpx.HTTPError as exc:
        raise OAuthDiscoveryError("token exchange request failed") from exc
    try:
        payload = json.loads(bytes(body))
    except (ValueError, UnicodeDecodeError):
        raise OAuthDiscoveryError("token response was not valid JSON") from None
    if not isinstance(payload, dict) or "access_token" not in payload:
        raise OAuthDiscoveryError("token response is missing access_token")
    return payload


def _exchange_authorization_code(
    client: httpx.Client,
    as_metadata: AuthorizationServerMetadata,
    provider: ProviderDescriptor,
    *,
    code: str,
    verifier: str,
    redirect_uri: str,
) -> dict[str, Any]:
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": provider.client_id,
        "code_verifier": verifier,
    }
    return _bounded_token_post(client, as_metadata.token_endpoint, data)


def _exchange_refresh_token(
    client: httpx.Client, provider: ProviderDescriptor, refresh_token: str
) -> dict[str, Any]:
    # Refresh reuses the provider's already-discovered token endpoint via the caller
    # (OAuthTokenStore.refresh's http_client is built against it); the endpoint itself
    # is re-resolved by the caller each time rather than cached indefinitely.
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": provider.client_id,
    }
    return _bounded_token_post(client, provider.resource_url, data)


# ---------------------------------------------------------------------------
# Sanitized audit — never a code/token/state/verifier value.
# ---------------------------------------------------------------------------
def _pseudonymize(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _audit(
    event: str,
    *,
    provider: str,
    tenant: str,
    principal: str,
    outcome: str,
    latency_ms: float | None = None,
) -> None:
    _audit_logger.info(
        "oauth_broker event=%s provider=%s tenant=%s principal=%s outcome=%s latency_ms=%s",
        event,
        provider,
        _pseudonymize(tenant),
        _pseudonymize(principal),
        outcome,
        latency_ms,
    )


# ---------------------------------------------------------------------------
# The broker
# ---------------------------------------------------------------------------
def _default_broker_http_client() -> httpx.Client:
    from agent_utilities.core.http_client import create_http_client

    return create_http_client(
        timeout=_DISCOVERY_TIMEOUT_S,
        verify=True,
        follow_redirects=False,
        trust_env=False,
    )


class RemoteOAuthBroker:
    """Provider-agnostic remote browser-OAuth broker.

    Construction takes a :class:`ProviderRegistry` (administrator-populated) and an
    optional :class:`~agent_utilities.security.secrets_client.SecretsClient` (defaults
    to :func:`agent_utilities.security.secrets_client.create_secrets_client` — the
    same OpenBao/engine-backed encrypted store every other secret in this repo uses).
    """

    def __init__(
        self,
        *,
        registry: ProviderRegistry,
        secrets_client: Any | None = None,
        http_client_factory: Any | None = None,
    ) -> None:
        if secrets_client is None:
            from agent_utilities.security.secrets_client import create_secrets_client

            secrets_client = create_secrets_client()
        self.registry = registry
        self.transactions = TransactionStore(secrets_client)
        self.tokens = OAuthTokenStore(secrets_client)
        self._http_client_factory = http_client_factory or _default_broker_http_client

    def begin(
        self,
        *,
        provider_id: str,
        actor: ActorContext,
        browser_session_id: str,
    ) -> str:
        """Stage 1-4: discover, mint a transaction, return the authorization URL.

        This is the only method meant to be exercised against a REAL provider in
        this lane (the explicit "stop at the consent URL" gate); it is still only
        ever tested here against a mock authorization server.
        """
        if not isinstance(browser_session_id, str) or not browser_session_id:
            raise OAuthBindingError("a browser session id is required")
        tenant, principal = OAuthTokenStore._require_verified(actor)
        provider = self.registry.require_enabled(provider_id)
        client = self._http_client_factory()
        try:
            resource_meta = discover_protected_resource(client, provider.resource_url)
            as_meta = discover_authorization_server(
                client,
                resource_meta,
                authorization_server_url=provider.authorization_server_url,
            )
        finally:
            client.close()

        verifier, challenge = _generate_pkce()
        state = secrets.token_urlsafe(32)
        nonce = secrets.token_urlsafe(16)
        now = time.time()
        scope = " ".join(provider.scopes)
        tx = OAuthTransaction(
            state=state,
            nonce=nonce,
            verifier=verifier,
            provider_id=provider.provider_id,
            tenant_id=tenant,
            principal_id=principal,
            browser_session_id=browser_session_id,
            redirect_uri=provider.redirect_uri,
            scope=scope,
            created_at=now,
            expires_at=now + _TRANSACTION_TTL_S,
        )
        self.transactions.put(tx)
        _audit(
            "begin",
            provider=provider.provider_id,
            tenant=tenant,
            principal=principal,
            outcome="authorization_url_issued",
        )
        params = {
            "response_type": "code",
            "client_id": provider.client_id,
            "redirect_uri": provider.redirect_uri,
            "scope": scope,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        return f"{as_meta.authorization_endpoint}?{urlencode(params)}"

    def callback(
        self,
        *,
        code: str,
        state: str,
        actor: ActorContext,
        browser_session_id: str,
    ) -> StoredToken:
        """Stage 5-6: validate the callback, exchange the code, store the token.

        Mock-server tested only in this lane (see module docstring). Rejects
        missing/replayed state, principal/tenant/session mismatch, and scope
        widening, all before any network call is made against the token endpoint
        except the discovery + exchange calls themselves.
        """
        tx = self.transactions.consume_once(state)
        tenant, principal = OAuthTokenStore._require_verified(actor)
        if principal != tx.principal_id or tenant != tx.tenant_id:
            _audit(
                "callback",
                provider=tx.provider_id,
                tenant=tenant,
                principal=principal,
                outcome="rejected_principal_mismatch",
            )
            raise OAuthBindingError(
                "callback principal/tenant does not match the initiating transaction"
            )
        if browser_session_id != tx.browser_session_id:
            _audit(
                "callback",
                provider=tx.provider_id,
                tenant=tenant,
                principal=principal,
                outcome="rejected_session_mismatch",
            )
            raise OAuthBindingError(
                "callback session does not match the initiating browser session"
            )
        provider = self.registry.require_enabled(tx.provider_id)
        client = self._http_client_factory()
        try:
            resource_meta = discover_protected_resource(client, provider.resource_url)
            as_meta = discover_authorization_server(
                client,
                resource_meta,
                authorization_server_url=provider.authorization_server_url,
            )
            payload = _exchange_authorization_code(
                client,
                as_meta,
                provider,
                code=code,
                verifier=tx.verifier,
                redirect_uri=tx.redirect_uri,
            )
        finally:
            client.close()

        granted_scope = str(payload.get("scope", tx.scope))
        requested = set(tx.scope.split())
        granted = set(granted_scope.split())
        if not granted <= requested:
            _audit(
                "callback",
                provider=provider.provider_id,
                tenant=tenant,
                principal=principal,
                outcome="rejected_scope_widened",
            )
            raise OAuthScopeError("granted scope exceeds what was requested")

        access_token = payload["access_token"]
        if not isinstance(access_token, str) or not access_token:
            raise OAuthDiscoveryError(
                "token response contained an invalid access token"
            )
        token = StoredToken(
            access_token=access_token,
            refresh_token=payload.get("refresh_token"),
            token_type=str(payload.get("token_type", "Bearer")),
            expires_at=time.time() + float(payload.get("expires_in", 3600)),
            granted_scope=granted_scope,
            key_version=OAuthTokenStore.CURRENT_KEY_VERSION,
            audience=provider.resource_url,
        )
        self.tokens.put(
            actor=actor,
            provider_id=provider.provider_id,
            resource_url=provider.resource_url,
            audience=provider.resource_url,
            token=token,
        )
        _audit(
            "callback",
            provider=provider.provider_id,
            tenant=tenant,
            principal=principal,
            outcome="token_stored",
        )
        return token

    def bearer_headers_for(
        self, *, actor: ActorContext, provider_id: str, resource_url: str
    ) -> dict[str, str]:
        """Endpoint-bound bearer lookup: forwards ONLY to the exact registered resource.

        This is the seam a future per-principal multiplexer child (W06/W08, not built
        in this lane) would call per outbound request. ``resource_url`` must equal the
        provider's registered exact resource URL, or this raises rather than forwarding
        a token to a different endpoint (rejects cross-provider forwarding).
        """
        provider = self.registry.require_enabled(provider_id)
        if resource_url != provider.resource_url:
            raise OAuthProviderError(
                "token forwarding is bound to the provider's exact registered resource"
            )
        token = self.tokens.get(
            actor=actor,
            provider_id=provider.provider_id,
            resource_url=provider.resource_url,
            audience=provider.resource_url,
        )
        if token is None:
            raise OAuthTokenAbsentError("no stored token for this principal/provider")
        return {"Authorization": f"{token.token_type} {token.access_token}"}
