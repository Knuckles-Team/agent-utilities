"""Credentials-by-reference for harness accounts (RF-ADR-010 §6.2 ``start``).

A :class:`RunSpec` never carries a secret. It names an ``account_ref``
(``vault://…``, ``secret://…`` or ``env://…``) that is resolved here, at launch,
through AU's one secret authority (:class:`~agent_utilities.security.
secrets_client.SecretsClient`). A missing or unresolvable reference is a
typed :class:`HarnessNotConfigured`, never an empty credential.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent_utilities.layers.contracts import HarnessNotConfigured, RunSpec


@runtime_checkable
class CredentialResolver(Protocol):
    """Resolves a secret reference to its value, or ``None`` when absent."""

    def resolve(self, ref: str) -> str | None: ...


class _SecretRefClient(Protocol):
    def resolve_ref(self, ref: str) -> str | None: ...


class SecretsCredentialResolver:
    """:class:`CredentialResolver` over AU's configured secrets client."""

    def __init__(self, client: _SecretRefClient | None = None) -> None:
        self._client = client

    def resolve(self, ref: str) -> str | None:
        if self._client is None:
            from agent_utilities.security.secrets_client import create_secrets_client

            self._client = create_secrets_client()
        return self._client.resolve_ref(ref)


def api_key_for(spec: RunSpec, resolver: CredentialResolver, harness: str) -> str:
    """The resolved API key for an ``api_key``-mode run, or fail closed."""
    ref = spec.account_ref
    if not ref:
        raise HarnessNotConfigured(f"{harness} api_key mode needs an account_ref")
    try:
        value = resolver.resolve(ref)
    except ValueError as exc:
        raise HarnessNotConfigured(
            f"{harness} account_ref is not a valid secret reference"
        ) from exc
    if not value:
        raise HarnessNotConfigured(f"{harness} account_ref resolved to no credential")
    return value


__all__ = ["CredentialResolver", "SecretsCredentialResolver", "api_key_for"]
