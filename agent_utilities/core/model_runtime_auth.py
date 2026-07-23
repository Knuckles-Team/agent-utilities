"""Resolve reference-only model credentials and header maps in memory.

Durable model declarations carry only ``env://``, ``vault://``, or
``secret://`` references. Resolved material is bounded, validated, returned to
the immediate client-construction call, and never written back to AgentConfig.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

_REFERENCE_RE = re.compile(
    r"^(?:"
    r"env://[A-Za-z_][A-Za-z0-9_]{0,127}"
    r"|(?:vault|secret)://[A-Za-z0-9][A-Za-z0-9_./#-]{0,511}"
    r")$"
)
_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]{1,128}$")
_FORBIDDEN_HEADERS = frozenset(
    {
        "connection",
        "content-length",
        "host",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


class ModelRuntimeAuthError(ValueError):
    """A model runtime reference or its resolved material is invalid.

    Messages intentionally omit references, header names, and values.
    """


def _resolve(reference: str) -> str:
    if not isinstance(reference, str) or not _REFERENCE_RE.fullmatch(reference.strip()):
        raise ModelRuntimeAuthError("model runtime reference is invalid")
    try:
        from agent_utilities.security.cli_secrets import (
            resolve_runtime_secret_reference,
        )

        return resolve_runtime_secret_reference(reference.strip())
    except Exception:
        raise ModelRuntimeAuthError("model runtime reference is unavailable") from None


def resolve_model_api_key(
    *, value: str | None = None, reference: str | None = None
) -> str | None:
    """Return one bounded API key from a direct runtime value or a reference."""
    if value and reference:
        raise ModelRuntimeAuthError("model authentication source is ambiguous")
    if reference:
        resolved = _resolve(reference)
    else:
        resolved = value
        if resolved and _REFERENCE_RE.fullmatch(resolved.strip()):
            resolved = _resolve(resolved.strip())
    if resolved is None:
        return None
    if not isinstance(resolved, str):
        raise ModelRuntimeAuthError("model credential is invalid")
    try:
        size = len(resolved.encode("utf-8"))
    except UnicodeError:
        raise ModelRuntimeAuthError("model credential is invalid") from None
    if (
        not resolved
        or size > 64 * 1024
        or any(character in resolved for character in "\x00\r\n")
    ):
        raise ModelRuntimeAuthError("model credential is invalid")
    return resolved


def validate_model_headers(value: Mapping[str, Any] | None) -> dict[str, str]:
    """Return a bounded, injection-safe copy of a model header mapping."""
    if value is None:
        return {}
    if not isinstance(value, Mapping) or len(value) > 64:
        raise ModelRuntimeAuthError("model headers are invalid")
    normalized: dict[str, str] = {}
    observed_names: set[str] = set()
    total_bytes = 0
    for raw_name, raw_value in value.items():
        if not isinstance(raw_name, str) or not isinstance(raw_value, str):
            raise ModelRuntimeAuthError("model headers are invalid")
        name = raw_name.strip()
        lowered = name.casefold()
        if (
            not _HEADER_NAME_RE.fullmatch(name)
            or lowered in _FORBIDDEN_HEADERS
            or any(
                ord(character) < 32 or ord(character) == 127 for character in raw_value
            )
        ):
            raise ModelRuntimeAuthError("model headers are invalid")
        try:
            item_bytes = len(name.encode("ascii")) + len(raw_value.encode("utf-8"))
        except UnicodeError:
            raise ModelRuntimeAuthError("model headers are invalid") from None
        if item_bytes > 16 * 1024:
            raise ModelRuntimeAuthError("model headers are invalid")
        total_bytes += item_bytes
        if total_bytes > 64 * 1024 or lowered in observed_names:
            raise ModelRuntimeAuthError("model headers are invalid")
        observed_names.add(lowered)
        normalized[name] = raw_value
    return normalized


def resolve_model_headers(
    *,
    value: Mapping[str, Any] | None = None,
    reference: str | None = None,
) -> dict[str, str]:
    """Resolve and validate one direct runtime or reference-backed header map."""
    if value and reference:
        raise ModelRuntimeAuthError("model header source is ambiguous")
    if reference:
        raw = _resolve(reference)
        try:
            raw_size = len(raw.encode("utf-8"))
        except UnicodeError:
            raise ModelRuntimeAuthError("model headers are invalid") from None
        if raw_size > 64 * 1024:
            raise ModelRuntimeAuthError("model headers are invalid")

        def reject_duplicate_keys(
            pairs: list[tuple[str, Any]],
        ) -> dict[str, Any]:
            result: dict[str, Any] = {}
            observed: set[str] = set()
            for key, item in pairs:
                normalized = str(key).strip().casefold()
                if normalized in observed:
                    raise ValueError("duplicate")
                observed.add(normalized)
                result[key] = item
            return result

        try:
            parsed = json.loads(raw, object_pairs_hook=reject_duplicate_keys)
        except Exception:
            raise ModelRuntimeAuthError("model headers are invalid") from None
        if not isinstance(parsed, dict):
            raise ModelRuntimeAuthError("model headers are invalid")
        return validate_model_headers(parsed)
    return validate_model_headers(value)
