"""Secret-reference-only command-line arguments.

Secret material must never be placed in a process argument: command lines are
commonly visible to other local users, process supervisors, crash reporters,
and deployment control planes.  CLI surfaces therefore accept only a bounded
``env://``, ``vault://``, or ``secret://`` reference and resolve it in memory at
parse time.  Existing Python call sites still receive the resolved runtime
value through the argument namespace; only the public CLI contract changes.
"""

from __future__ import annotations

import argparse
import re
from typing import Any

from agent_utilities.core.config import setting

__all__ = [
    "RuntimeSecretReferenceAction",
    "RuntimeSecretReferenceError",
    "resolve_runtime_secret_reference",
    "validate_runtime_secret_reference",
]

_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_STORE_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./#-]{0,511}$")
_MAX_REFERENCE_BYTES = 1_024
_MAX_SECRET_BYTES = 4 * 1024 * 1024


class RuntimeSecretReferenceError(ValueError):
    """A CLI secret reference is invalid or unavailable.

    Messages deliberately omit the reference and resolved value.
    """


def _validated_reference(reference: Any) -> tuple[str, str]:
    rendered = str(reference or "").strip()
    if (
        not rendered
        or len(rendered.encode("utf-8")) > _MAX_REFERENCE_BYTES
        or any(character.isspace() or ord(character) < 32 for character in rendered)
    ):
        raise RuntimeSecretReferenceError("runtime secret reference is invalid")
    scheme, separator, target = rendered.partition("://")
    if not separator or scheme not in {"env", "vault", "secret"}:
        raise RuntimeSecretReferenceError(
            "runtime secret reference must use env://, vault://, or secret://"
        )
    pattern = _ENV_NAME_RE if scheme == "env" else _STORE_KEY_RE
    if pattern.fullmatch(target) is None or ".." in target.split("/"):
        raise RuntimeSecretReferenceError("runtime secret reference is invalid")
    return scheme, rendered


def resolve_runtime_secret_reference(reference: Any) -> str:
    """Resolve one validated CLI reference without disclosing it on failure."""

    scheme, rendered = _validated_reference(reference)
    try:
        if scheme == "env":
            value = setting(rendered.removeprefix("env://"))
        else:
            from agent_utilities.security.secrets_client import create_secrets_client

            value = create_secrets_client().resolve_ref(rendered)
    except Exception:
        value = None
    if not isinstance(value, str) or not value:
        raise RuntimeSecretReferenceError("runtime secret reference is unavailable")
    if len(value.encode("utf-8")) > _MAX_SECRET_BYTES or "\x00" in value:
        raise RuntimeSecretReferenceError("resolved runtime secret is invalid")
    return value


def validate_runtime_secret_reference(reference: Any) -> str:
    """Return one canonical reference without resolving its runtime value."""

    _scheme, rendered = _validated_reference(reference)
    return rendered


class RuntimeSecretReferenceAction(argparse.Action):
    """Argparse action that resolves a reference into the destination field."""

    def __call__(
        self,
        _parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | list[str] | None,
        _option_string: str | None = None,
    ) -> None:
        if not isinstance(values, str):
            raise argparse.ArgumentError(self, "runtime secret reference is invalid")
        try:
            resolved = resolve_runtime_secret_reference(values)
        except RuntimeSecretReferenceError as exc:
            raise argparse.ArgumentError(self, str(exc)) from None
        setattr(namespace, self.dest, resolved)
