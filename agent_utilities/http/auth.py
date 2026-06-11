"""Auth header injection strategies for fleet HTTP clients.

CONCEPT:ECO-4.35 — Fleet HTTP Client Library

The fleet's connectors authenticate in four recurring ways; each is one
strategy here instead of a hand-rolled ``_build_headers`` per repo:

* :class:`TokenAuth` — a header token with a configurable scheme prefix
  (``Bearer`` for most APIs, ``SSWS`` for Okta, no prefix for raw API keys
  like Portainer's ``X-API-Key``). Accepts either a static ``token`` or a
  ``token_provider`` callable for OAuth/JWT manager delegation (dockerhub's
  ``TokenManager``, salesforce flows) — the provider is consulted on every
  request, so refresh logic stays in the manager.
* :class:`BasicAuth` — RFC 7617 ``Authorization: Basic`` credentials.
* :class:`QueryApiKeyAuth` — an API key carried as a query parameter.
* The :class:`AuthHeaderInjector` base — no credentials (anonymous), and the
  extension point for bespoke schemes.

Strategies expose :meth:`~AuthHeaderInjector.secrets` so the client can feed
its :class:`~agent_utilities.http.LogRedactor`, and may define an
``invalidate()`` method to opt in to one transparent retry after a 401
(the client duck-types for it).
"""

from __future__ import annotations

from base64 import b64encode
from collections.abc import Callable

__all__ = [
    "AuthHeaderInjector",
    "BasicAuth",
    "QueryApiKeyAuth",
    "TokenAuth",
]


class AuthHeaderInjector:
    """Base auth strategy: contributes headers / query params per request.

    The base class injects nothing (anonymous access). Subclasses override
    :meth:`headers` and/or :meth:`params`; both are re-evaluated on every
    request so live credentials (rotating tokens) stay current.
    """

    def headers(self) -> dict[str, str]:
        """Headers to merge into every request."""
        return {}

    def params(self) -> dict[str, str]:
        """Query parameters to merge into every request."""
        return {}

    def secrets(self) -> list[str]:
        """Literal secret values for log redaction."""
        return []


class TokenAuth(AuthHeaderInjector):
    """Header-token auth with a configurable header name and scheme prefix.

    Args:
        token: A static token value.
        token_provider: A zero-argument callable returning the current token
            (mutually exclusive with ``token``); called per request, so a
            delegated token manager controls caching and refresh.
        header: Header to carry the token (default ``Authorization``).
        prefix: Scheme prefix before the token — ``"Bearer"`` (default),
            ``"SSWS"`` (Okta), or ``None``/``""`` for a bare token value
            (e.g. ``X-API-Key`` style headers).
    """

    def __init__(
        self,
        token: str | None = None,
        *,
        token_provider: Callable[[], str] | None = None,
        header: str = "Authorization",
        prefix: str | None = "Bearer",
    ) -> None:
        if (token is None) == (token_provider is None):
            raise ValueError(
                "TokenAuth requires exactly one of 'token' or 'token_provider'"
            )
        self._token = token
        self._token_provider = token_provider
        self.header = header
        self.prefix = prefix or ""

    def _current_token(self) -> str:
        if self._token_provider is not None:
            return self._token_provider()
        return self._token or ""

    def headers(self) -> dict[str, str]:
        token = self._current_token()
        value = f"{self.prefix} {token}" if self.prefix else token
        return {self.header: value}

    def secrets(self) -> list[str]:
        # Provider-backed tokens are intentionally not minted here: secrets()
        # may run before any request, and the scheme-pattern redaction in
        # LogRedactor already covers "<prefix> <token>" strings.
        return [self._token] if self._token else []


class BasicAuth(AuthHeaderInjector):
    """RFC 7617 ``Authorization: Basic`` username/password credentials."""

    def __init__(self, username: str, password: str) -> None:
        self.username = username
        self._password = password

    def headers(self) -> dict[str, str]:
        credentials = b64encode(f"{self.username}:{self._password}".encode()).decode(
            "ascii"
        )
        return {"Authorization": f"Basic {credentials}"}

    def secrets(self) -> list[str]:
        return [self._password] if self._password else []


class QueryApiKeyAuth(AuthHeaderInjector):
    """API key carried as a query parameter on every request."""

    def __init__(self, param: str, key: str) -> None:
        if not param or not key:
            raise ValueError("QueryApiKeyAuth requires both 'param' and 'key'")
        self.param = param
        self._key = key

    def params(self) -> dict[str, str]:
        return {self.param: self._key}

    def secrets(self) -> list[str]:
        return [self._key]
