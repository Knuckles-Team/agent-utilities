"""The ONE classifier for "may this configuration value cross a tool surface?".

CONCEPT:AU-OS.config.two-surfaces-by-default

Every surface that reports configuration — ``graph_config`` (get/describe/diff),
``graph_configure``'s ``get_config``/``list_config`` actions, and the durable MCP
server-definition validator — must agree on which keys are sensitive. When that
predicate exists in more than one place the surfaces drift, and a drift here is
a credential disclosure, not a cosmetic inconsistency.

It lives in its own small module (no ``agent_utilities`` imports beyond the
standard library) so both the heavy MCP tool modules and the lightweight config
admin core can share it without an import cycle.

Two distinct questions, deliberately kept apart:

* :func:`configuration_key_is_sensitive` — is the KEY one whose value must not
  be echoed? Answered from the key's own words, so an unknown/newly added
  setting is classified conservatively rather than defaulting to "safe".
* :func:`runtime_reference` — is the VALUE a non-literal reference
  (``vault://``, ``env://``, ``secret://``, ``${VAR}``)? A reference is the
  thing an operator is *supposed* to see; only literals need redacting.
"""

from __future__ import annotations

import re
from typing import Any

#: A value that names where a secret lives rather than being one. MCP clients
#: expand ``${VAR}`` themselves; secret-store URIs are resolved by the consumer.
#: Defaults inside a ``${VAR:-fallback}`` form are intentionally NOT matched —
#: they put the supposedly external value back into the durable document.
RUNTIME_REFERENCE_RE = re.compile(
    r"^(?:\$\{[A-Za-z_][A-Za-z0-9_]*\}|(?:vault|env|secret)://[A-Za-z0-9_./#-]+)$"
)

SENSITIVE_KEY_PARTS = frozenset(
    {
        "authorization",
        "credential",
        "credentials",
        "email",
        "identity",
        "password",
        "secret",
        "tenant",
        "token",
        "user",
        "username",
    }
)
ENDPOINT_KEY_PARTS = frozenset(
    {
        "address",
        "baseurl",
        "broker",
        "brokers",
        "endpoint",
        "endpoints",
        "host",
        "hostname",
        "hosts",
        "server",
        "servers",
        "uri",
        "uris",
        "url",
        "urls",
    }
)
PATH_KEY_PARTS = frozenset(
    {
        "bundle",
        "ca",
        "cert",
        "certificate",
        "cwd",
        "directory",
        "dir",
        "file",
        "keyfile",
        "path",
        "root",
        "workspace",
    }
)


def normalised_key_parts(key: str) -> set[str]:
    """Lower-cased word set of a config key, camelCase and snake_case alike."""
    normalised = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key).lower()
    parts = {part for part in re.split(r"[^a-z0-9]+", normalised) if part}
    if "base" in parts and "url" in parts:
        parts.add("baseurl")
    if "key" in parts and "file" in parts:
        parts.add("keyfile")
    return parts


def configuration_key_is_sensitive(
    env_key: str, metadata: dict[str, Any] | None = None
) -> bool:
    """Classify durable settings whose values must not cross the MCP surface."""

    parts = normalised_key_parts(env_key)
    if bool((metadata or {}).get("secret")):
        return True
    if parts & (SENSITIVE_KEY_PARTS | ENDPOINT_KEY_PARTS | PATH_KEY_PARTS):
        return True
    if env_key.upper() == "MCP_CONFIG":
        return True
    if "id" in parts and parts & {
        "actor",
        "agent",
        "client",
        "identity",
        "tenant",
        "user",
    }:
        return True
    return "key" in parts and bool(
        parts
        & {"api", "auth", "client", "encryption", "hmac", "private", "signing", "tls"}
    )


def runtime_reference(value: Any) -> bool:
    """Return whether ``value`` is a non-literal runtime reference."""

    return isinstance(value, str) and bool(RUNTIME_REFERENCE_RE.fullmatch(value))
