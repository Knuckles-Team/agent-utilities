#!/usr/bin/python
from __future__ import annotations

"""Universal outbound credential provider — "give me credentials for source X".

CONCEPT:OS-5.38 — Universal Outbound CredentialProvider

This is the **inverse** of the secrets store
(:mod:`agent_utilities.security.secrets_client`): callers ask for a credential **by
source name** ("x", "reddit", "github", …) rather than by secret key, and get back a
typed, ready-to-apply :class:`~agent_utilities.security.source_credentials.SourceCredential`
(OS-5.39) — an API key, a cookie session, an auto-refreshing OAuth2 token, etc.

It unifies what was previously ad-hoc, per-agent credential plumbing (e.g.
``agents/github-agent/github_agent/auth.py`` reading env vars directly) behind one
abstraction. Its first consumer is the PulseLink open-web/social source server, whose
backend ladders pick the highest-fidelity backend whose credential
:meth:`available` reports usable — keyless backends declare ``NoCredential`` and
always qualify; cookie/official backends light up only when their credential exists.

Storage is **not** reimplemented: descriptors point at secret URIs
(``vault://``/``env://``/``sqlite://``/plain) resolved through the existing
:class:`~agent_utilities.security.secrets_client.SecretsClient`, so Vault / SQLite /
in-memory all work unchanged.

Descriptor map (declarative, config.json-driven — never bare ``os.environ``)::

    SOURCE_CREDENTIALS = {
      "x":      {"type": "cookie_session", "secret": "vault://pulselink/x/session"},
      "reddit": {"type": "oauth2", "secret": "vault://pulselink/reddit/token",
                 "token_url": "https://www.reddit.com/api/v1/access_token",
                 "client_id": "abc", "client_secret_secret": "vault://pulselink/reddit/cs"},
      "github": {"type": "api_key", "secret": "env://GITHUB_TOKEN", "prefix": "token "}
    }

Read once from ``config.setting("SOURCE_CREDENTIALS", "{}")`` (a JSON object) and/or
registered programmatically via :meth:`CredentialProvider.register`.
"""

import json
import logging
import threading
from typing import Any

from agent_utilities.core._env import setting
from agent_utilities.security.secrets_client import SecretsClient, create_secrets_client
from agent_utilities.security.source_credentials import (
    NoCredential,
    SourceCredential,
    build_credential,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CredentialProvider",
    "get_credential_provider",
]


def _load_descriptors() -> dict[str, dict[str, Any]]:
    """Load the source→descriptor map from ``SOURCE_CREDENTIALS`` (JSON object)."""
    raw = setting("SOURCE_CREDENTIALS", "{}")
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("SOURCE_CREDENTIALS is not valid JSON — ignoring.")
        return {}
    if not isinstance(obj, dict):
        logger.warning("SOURCE_CREDENTIALS must be a JSON object — ignoring.")
        return {}
    return {str(k): dict(v) for k, v in obj.items() if isinstance(v, dict)}


class CredentialProvider:
    """Resolve typed, auto-refreshing credentials by source name.

    Thread-safe; credentials are built lazily on first request and cached. Stateful
    credentials (e.g. :class:`OAuth2Credential`) refresh themselves in place, so the
    cache holds a live object rather than a frozen token.
    """

    def __init__(
        self,
        descriptors: dict[str, dict[str, Any]] | None = None,
        secrets: SecretsClient | None = None,
    ) -> None:
        self._descriptors: dict[str, dict[str, Any]] = dict(
            descriptors if descriptors is not None else _load_descriptors()
        )
        self._secrets = secrets if secrets is not None else create_secrets_client()
        self._cache: dict[str, SourceCredential] = {}
        self._lock = threading.Lock()

    def register(self, source: str, descriptor: dict[str, Any]) -> None:
        """Register/replace the credential descriptor for ``source`` at runtime."""
        with self._lock:
            self._descriptors[source] = dict(descriptor)
            self._cache.pop(source, None)

    def get(self, source: str) -> SourceCredential:
        """Return the typed credential for ``source`` (``NoCredential`` if none).

        A source with no descriptor resolves to :class:`NoCredential` — keyless
        backends remain selectable with zero configuration.
        """
        with self._lock:
            cached = self._cache.get(source)
            if cached is not None:
                return cached
            descriptor = self._descriptors.get(source)
            if not descriptor:
                cred: SourceCredential = NoCredential()
            else:
                try:
                    cred = build_credential(descriptor, self._secrets)
                except Exception as exc:  # noqa: BLE001 — bad descriptor must not crash callers
                    logger.warning(
                        "Failed to build credential for %r (%s) — treating as keyless.",
                        source,
                        exc,
                    )
                    cred = NoCredential()
            self._cache[source] = cred
            return cred

    def available(self, source: str) -> bool:
        """Whether a *real, present* credential exists for ``source``.

        This is what a connector's cookie/official backend consults: it returns
        ``False`` for an unconfigured source (or an explicit keyless ``"none"``
        descriptor), so those higher-fidelity backends stay dark and the ladder
        falls back to its keyless backend. Keyless backends do **not** call this —
        they declare a :class:`NoCredential` requirement and are always eligible.
        """
        cred = self.get(source)
        return cred.type_name != NoCredential.type_name and cred.is_present()

    def status(self) -> dict[str, dict[str, Any]]:
        """Read-only per-source ``{type, available}`` map — **never** secret values.

        Safe to surface on the gateway/MCP: reports which sources have a usable
        credential and of what kind, without exposing the material itself.
        """
        out: dict[str, dict[str, Any]] = {}
        for source in sorted(self._descriptors):
            cred = self.get(source)
            out[source] = {
                "type": cred.type_name,
                "available": self.available(source),
            }
        return out


_provider: CredentialProvider | None = None
_provider_lock = threading.Lock()


def get_credential_provider() -> CredentialProvider:
    """Return the process-wide :class:`CredentialProvider` singleton."""
    global _provider
    if _provider is not None:
        return _provider
    with _provider_lock:
        if _provider is None:
            _provider = CredentialProvider()
        return _provider
