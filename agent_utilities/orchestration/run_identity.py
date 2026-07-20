"""Opaque current-contract identifiers for governed orchestration runs.

Run handles cross MCP, persistence, trace, and job-status boundaries. They use
128 bits from the operating system CSPRNG and are validated as one strict
current format; truncated and compatibility forms are intentionally rejected.
"""

from __future__ import annotations

import re
import secrets

RUN_ID_PATTERN = re.compile(r"run:[a-f0-9]{32}")


def new_run_id() -> str:
    """Return a collision-resistant opaque orchestration run handle."""

    return f"run:{secrets.token_hex(16)}"


def is_run_id(value: object) -> bool:
    """Return whether ``value`` exactly matches the current run contract."""

    return isinstance(value, str) and RUN_ID_PATTERN.fullmatch(value) is not None


def require_run_id(value: object) -> str:
    """Return a valid current run handle or fail closed."""

    if not is_run_id(value):
        raise ValueError("run_id_invalid")
    return str(value)


__all__ = ["RUN_ID_PATTERN", "is_run_id", "new_run_id", "require_run_id"]
