#!/usr/bin/python
from __future__ import annotations

"""Typed source-credential registry — the shapes a connector applies to a request.

CONCEPT:OS-5.39 — Typed Source-Credential Registry

The companion to :mod:`credential_provider` (CONCEPT:OS-5.38). Where the
:class:`~agent_utilities.security.secrets_client.SecretsBackend` stores raw secret
*strings* keyed by name, this module models the *kind* of credential an external
data source needs and how it is **applied to an outbound HTTP request** — a header,
a query parameter, a cookie jar, or basic auth — plus how it **refreshes** itself
when it expires.

Design:

  * Each credential implements :meth:`SourceCredential.materialize` returning an
    :class:`AuthMaterial` (headers / params / cookies the caller merges onto its
    request) and :meth:`SourceCredential.is_present` (does it actually carry usable
    secret material — this is what drives a connector's backend-ladder selection).
  * :class:`NoCredential` is the keyless sentinel: present, but contributes nothing.
  * :class:`OAuth2Credential` self-refreshes via the OAuth2 ``refresh_token`` grant,
    reusing the cache-and-skew pattern of
    :class:`~agent_utilities.mcp.client_credentials.ClientCredentialsTokenProvider`
    (OS-5.32) rather than introducing a second refresher.
  * Each type knows how to build itself from a declarative descriptor +
    a :class:`~agent_utilities.security.secrets_client.SecretsClient`, via
    :meth:`SourceCredential.from_descriptor`, so the provider stays generic.
"""

import abc
import base64
import json
import threading
import time
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from agent_utilities.security.secrets_client import SecretsClient

__all__ = [
    "AuthMaterial",
    "SourceCredential",
    "NoCredential",
    "ApiKeyCredential",
    "CookieSessionCredential",
    "BasicAuthCredential",
    "OAuth2Credential",
    "CREDENTIAL_TYPES",
    "build_credential",
]

# Refresh an OAuth token this many seconds before it actually expires.
_EXPIRY_SKEW_S = 30.0


class AuthMaterial(BaseModel):
    """The request pieces a credential contributes.

    A caller merges these onto its outbound HTTP request: ``headers`` and
    ``params`` are shallow-merged, ``cookies`` populate the cookie jar.
    """

    headers: dict[str, str] = Field(default_factory=dict)
    params: dict[str, str] = Field(default_factory=dict)
    cookies: dict[str, str] = Field(default_factory=dict)

    def merged_into(
        self,
        headers: dict[str, str] | None = None,
        params: dict[str, str] | None = None,
        cookies: dict[str, str] | None = None,
    ) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
        """Return ``(headers, params, cookies)`` with this material merged on top."""
        h = dict(headers or {})
        h.update(self.headers)
        p = dict(params or {})
        p.update(self.params)
        c = dict(cookies or {})
        c.update(self.cookies)
        return h, p, c


def _resolve(secrets: SecretsClient | None, ref: str | None) -> str | None:
    """Resolve a secret URI ref (``vault://``/``env://``/``sqlite://``/plain)."""
    if not ref:
        return None
    if secrets is None:
        return None
    return secrets.resolve_ref(ref)


class SourceCredential(abc.ABC):
    """Base class for a typed, applyable, refreshable source credential.

    Subclasses set :attr:`type_name` (the registry key) and implement
    :meth:`materialize`, :meth:`is_present`, and :meth:`from_descriptor`.
    """

    #: Registry key used by :data:`CREDENTIAL_TYPES` / :func:`build_credential`.
    type_name: ClassVar[str] = "base"

    @abc.abstractmethod
    def materialize(self) -> AuthMaterial:
        """Return the headers/params/cookies to apply to a request."""
        raise NotImplementedError  # ABSTRACT-OK

    @abc.abstractmethod
    def is_present(self) -> bool:
        """Whether this credential carries usable material (drives ladder choice)."""
        raise NotImplementedError  # ABSTRACT-OK

    def refresh(self) -> None:  # noqa: B027 — optional override, no-op for static creds
        """Refresh the credential in place. No-op for static credentials."""
        return None

    @classmethod
    @abc.abstractmethod
    def from_descriptor(
        cls, descriptor: dict[str, Any], secrets: SecretsClient | None
    ) -> SourceCredential:
        """Build the credential from a declarative descriptor + secrets client."""
        raise NotImplementedError  # ABSTRACT-OK


class NoCredential(SourceCredential):
    """Keyless sentinel — present (nothing to satisfy) but contributes nothing."""

    type_name: ClassVar[str] = "none"

    def materialize(self) -> AuthMaterial:
        return AuthMaterial()

    def is_present(self) -> bool:
        return True

    @classmethod
    def from_descriptor(
        cls, descriptor: dict[str, Any], secrets: SecretsClient | None
    ) -> NoCredential:
        return cls()


class ApiKeyCredential(SourceCredential):
    """An API key / bearer token applied as a header or query parameter.

    Descriptor fields:
        secret: URI ref resolving to the key value (required to be present).
        placement: ``"header"`` (default) or ``"query"``.
        name: header/param name (default ``"Authorization"``).
        prefix: value prefix (default ``"Bearer "`` for the Authorization header;
            set to ``""`` for a raw key).
    """

    type_name: ClassVar[str] = "api_key"

    def __init__(
        self,
        key: str | None,
        *,
        placement: str = "header",
        name: str = "Authorization",
        prefix: str = "Bearer ",
    ) -> None:
        self.key = key
        self.placement = placement
        self.name = name
        self.prefix = prefix

    def materialize(self) -> AuthMaterial:
        if not self.key:
            return AuthMaterial()
        value = f"{self.prefix}{self.key}"
        if self.placement == "query":
            return AuthMaterial(params={self.name: value})
        return AuthMaterial(headers={self.name: value})

    def is_present(self) -> bool:
        return bool(self.key)

    @classmethod
    def from_descriptor(
        cls, descriptor: dict[str, Any], secrets: SecretsClient | None
    ) -> ApiKeyCredential:
        return cls(
            key=_resolve(secrets, descriptor.get("secret")),
            placement=descriptor.get("placement", "header"),
            name=descriptor.get("name", "Authorization"),
            prefix=descriptor.get("prefix", "Bearer "),
        )


class CookieSessionCredential(SourceCredential):
    """A name→value cookie jar (e.g. X ``auth_token``/``ct0``, ``reddit_session``).

    Descriptor fields:
        secret: URI ref resolving to either a JSON object of cookies, or a raw
            ``"name=value; name2=value2"`` cookie string.
        cookies: an inline ``{name: value}`` mapping (alternative to ``secret``).
    """

    type_name: ClassVar[str] = "cookie_session"

    def __init__(self, cookies: dict[str, str] | None) -> None:
        self.cookies = dict(cookies or {})

    def materialize(self) -> AuthMaterial:
        if not self.cookies:
            return AuthMaterial()
        cookie_header = "; ".join(f"{k}={v}" for k, v in self.cookies.items())
        return AuthMaterial(
            headers={"Cookie": cookie_header}, cookies=dict(self.cookies)
        )

    def is_present(self) -> bool:
        return bool(self.cookies)

    @staticmethod
    def _parse(raw: str | None, inline: dict[str, str] | None) -> dict[str, str]:
        if inline:
            return dict(inline)
        if not raw:
            return {}
        raw = raw.strip()
        if raw.startswith("{"):
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    return {str(k): str(v) for k, v in obj.items()}
            except json.JSONDecodeError:
                pass
        # Fall back to a "k=v; k2=v2" cookie string.
        out: dict[str, str] = {}
        for part in raw.split(";"):
            part = part.strip()
            if "=" in part:
                k, v = part.split("=", 1)
                out[k.strip()] = v.strip()
        return out

    @classmethod
    def from_descriptor(
        cls, descriptor: dict[str, Any], secrets: SecretsClient | None
    ) -> CookieSessionCredential:
        raw = _resolve(secrets, descriptor.get("secret"))
        return cls(cls._parse(raw, descriptor.get("cookies")))


class BasicAuthCredential(SourceCredential):
    """HTTP Basic auth applied as an ``Authorization: Basic`` header.

    Descriptor fields:
        username: literal username, or a URI ref via ``username_secret``.
        password_secret: URI ref resolving to the password.
    """

    type_name: ClassVar[str] = "basic"

    def __init__(self, username: str | None, password: str | None) -> None:
        self.username = username
        self.password = password

    def materialize(self) -> AuthMaterial:
        if not self.username:
            return AuthMaterial()
        raw = f"{self.username}:{self.password or ''}".encode()
        token = base64.b64encode(raw).decode()
        return AuthMaterial(headers={"Authorization": f"Basic {token}"})

    def is_present(self) -> bool:
        return bool(self.username)

    @classmethod
    def from_descriptor(
        cls, descriptor: dict[str, Any], secrets: SecretsClient | None
    ) -> BasicAuthCredential:
        username = descriptor.get("username") or _resolve(
            secrets, descriptor.get("username_secret")
        )
        password = _resolve(secrets, descriptor.get("password_secret"))
        return cls(username, password)


class OAuth2Credential(SourceCredential):
    """An OAuth2 access token with self-refresh via the ``refresh_token`` grant.

    Reuses the cache-and-skew refresh discipline of
    :class:`~agent_utilities.mcp.client_credentials.ClientCredentialsTokenProvider`
    (OS-5.32). ``materialize`` lazily refreshes when the access token is missing or
    within :data:`_EXPIRY_SKEW_S` of expiry and a refresh token + token URL exist.

    Descriptor fields:
        secret: URI ref resolving to either a JSON object
            ``{access_token, refresh_token, expires_at?}`` or a raw access token.
        token_url, client_id: required for refresh.
        client_secret_secret: URI ref resolving to the client secret.
        scope: optional space-separated scopes.
    """

    type_name: ClassVar[str] = "oauth2"

    def __init__(
        self,
        access_token: str | None = None,
        refresh_token: str | None = None,
        *,
        token_url: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        scope: str | None = None,
        expires_at: float = 0.0,
        verify: bool = True,
        timeout: int = 15,
    ) -> None:
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.token_url = token_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.scope = scope
        # ``expires_at`` is a ``time.monotonic`` deadline; 0 means "unknown".
        self._expires_at = expires_at
        self.verify = verify
        self.timeout = timeout
        self._lock = threading.Lock()

    def _needs_refresh(self) -> bool:
        if not self.access_token:
            return True
        if self._expires_at <= 0:
            return False  # unknown TTL → assume valid (static-ish token)
        return time.monotonic() >= self._expires_at - _EXPIRY_SKEW_S

    def refresh(self) -> None:
        with self._lock:
            if not (self.refresh_token and self.token_url and self.client_id):
                return
            import requests

            data = {
                "grant_type": "refresh_token",
                "refresh_token": self.refresh_token,
                "client_id": self.client_id,
            }
            if self.scope:
                data["scope"] = self.scope
            auth = (
                (self.client_id, self.client_secret)
                if self.client_secret is not None
                else None
            )
            resp = requests.post(
                self.token_url,
                data=data,
                auth=auth,
                verify=self.verify,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
            self.access_token = payload["access_token"]
            if payload.get("refresh_token"):
                self.refresh_token = payload["refresh_token"]
            self._expires_at = time.monotonic() + float(payload.get("expires_in", 3600))

    def materialize(self) -> AuthMaterial:
        if self._needs_refresh():
            try:
                self.refresh()
            except Exception:  # noqa: BLE001 — a failed refresh degrades to no header
                pass
        if not self.access_token:
            return AuthMaterial()
        return AuthMaterial(headers={"Authorization": f"Bearer {self.access_token}"})

    def is_present(self) -> bool:
        return bool(self.access_token or self.refresh_token)

    @classmethod
    def from_descriptor(
        cls, descriptor: dict[str, Any], secrets: SecretsClient | None
    ) -> OAuth2Credential:
        raw = _resolve(secrets, descriptor.get("secret"))
        access_token: str | None = raw
        refresh_token: str | None = descriptor.get("refresh_token")
        expires_at = 0.0
        if raw and raw.strip().startswith("{"):
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    access_token = obj.get("access_token")
                    refresh_token = obj.get("refresh_token", refresh_token)
                    if obj.get("expires_in"):
                        expires_at = time.monotonic() + float(obj["expires_in"])
            except json.JSONDecodeError:
                pass
        if descriptor.get("refresh_token_secret"):
            refresh_token = _resolve(secrets, descriptor["refresh_token_secret"])
        verify = bool(descriptor.get("verify", True))
        return cls(
            access_token=access_token,
            refresh_token=refresh_token,
            token_url=descriptor.get("token_url"),
            client_id=descriptor.get("client_id"),
            client_secret=_resolve(secrets, descriptor.get("client_secret_secret")),
            scope=descriptor.get("scope"),
            expires_at=expires_at,
            verify=verify,
        )


#: Registry mapping a descriptor ``type`` string to its credential class.
CREDENTIAL_TYPES: dict[str, type[SourceCredential]] = {
    NoCredential.type_name: NoCredential,
    ApiKeyCredential.type_name: ApiKeyCredential,
    CookieSessionCredential.type_name: CookieSessionCredential,
    BasicAuthCredential.type_name: BasicAuthCredential,
    OAuth2Credential.type_name: OAuth2Credential,
}


def build_credential(
    descriptor: dict[str, Any], secrets: SecretsClient | None = None
) -> SourceCredential:
    """Build a typed credential from a descriptor's ``type`` field.

    Args:
        descriptor: ``{"type": <key>, ...type-specific fields...}``. An empty or
            missing ``type`` yields a :class:`NoCredential`.
        secrets: client used to resolve any ``*_secret`` / ``secret`` URI refs.

    Raises:
        KeyError: when ``type`` is not a registered credential kind.
    """
    type_name = (descriptor or {}).get("type", "none")
    cred_cls = CREDENTIAL_TYPES.get(type_name)
    if cred_cls is None:
        raise KeyError(
            f"Unknown credential type {type_name!r}. "
            f"Available: {', '.join(sorted(CREDENTIAL_TYPES))}"
        )
    return cred_cls.from_descriptor(descriptor, secrets)
