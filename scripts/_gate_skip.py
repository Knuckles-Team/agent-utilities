"""Exit contract for a gate whose tool or sibling checkout is not present.

A gate that cannot run has not found nothing, but a fresh clone without an
optional native scanner is not a finding either. Locally the gate prints
``SKIPPED (<gate>): <reason>`` and exits 0; under CI (``CI`` set), where the
tool must be provisioned, it prints CANNOT RUN and exits 2.

Use it only for an ABSENT tool or checkout. A tool that is present but fails,
or a malformed configuration, is still a hard failure.

The gates share one local-paths-only tool lookup (``find_local_tool``): a hook
that resolves its tool from a package index at hook time is how a previous
fleet sweep shipped a gate that could not pass anywhere (69 push failures
across 226 repos), so no gate here ever installs or downloads its tool.
"""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn


def unavailable(gate: str, reason: str) -> NoReturn:
    if os.environ.get("CI"):
        print(f"{gate}: CANNOT RUN: {reason}", file=sys.stderr)
        raise SystemExit(2)
    print(f"SKIPPED ({gate}): {reason}")
    raise SystemExit(0)


def find_local_tool(name: str) -> str | None:
    """Return ``~/.local/bin/<name>``, ``/usr/local/bin/<name>`` or the $PATH
    hit, in that order; ``None`` when the tool is absent from all three."""

    for candidate in (Path.home() / ".local/bin" / name, Path("/usr/local/bin") / name):
        if candidate.is_file():
            return str(candidate)
    return shutil.which(name)


def resolve_local_tool(
    name: str,
    *,
    env_name: str,
    configured: str,
    gate: str,
    install_hint: str,
    die: Callable[[str], NoReturn],
) -> str:
    """Resolve an explicitly configured binary, else ``find_local_tool``.

    A configured path that is not a file is a hard failure (``die``); a tool
    that is simply absent follows the ``unavailable`` contract above.
    """

    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_file():
            die(f"{env_name} points to a non-file path: {candidate}")
        return str(candidate)
    found = find_local_tool(name)
    if found is None:
        unavailable(
            gate,
            f"`{name}` not found. Looked at ${env_name}, ~/.local/bin/{name}, "
            f"/usr/local/bin/{name} and $PATH. {install_hint}",
        )
    return found


def repository_setting(name: str, default: str, die: Callable[[str], NoReturn]) -> str:
    """Read a live process override through the repository config boundary."""

    try:
        from agent_utilities.core.config import setting

        value = setting(name, default, cast=str)
    except (
        ImportError,
        ModuleNotFoundError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        die(f"could not read repository setting {name}: {exc}")
    return str(value or default).strip()
