"""Register the portable GraphOS stdio launcher with Codex.

Codex owns its MCP configuration in ``config.toml``.  This module deliberately
uses the public ``codex mcp`` command instead of writing that file or translating
an IDE-oriented ``mcp_config.json``.  The registered launcher contains no
environment variables, credentials, working directory, or installation path;
all deployment settings continue to resolve through :class:`AgentConfig`.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Sequence
from typing import Any

CODEX_GRAPHOS_SERVER = "graph-os"
CODEX_GRAPHOS_COMMAND: tuple[str, ...] = (
    "graph-os",
    "--transport",
    "stdio",
)
_COMMAND_TIMEOUT_SECONDS = 20


class CodexRegistrationError(RuntimeError):
    """A path- and credential-safe Codex registration failure."""


def graphos_stdio_spec() -> dict[str, Any]:
    """Return the complete portable launcher specification registered in Codex."""
    return {
        "command": CODEX_GRAPHOS_COMMAND[0],
        "args": list(CODEX_GRAPHOS_COMMAND[1:]),
    }


def _is_canonical_registration(payload: Any) -> bool:
    """Whether ``codex mcp get --json`` describes the exact supported launcher."""
    if not isinstance(payload, dict):
        return False
    transport = payload.get("transport")
    if not isinstance(transport, dict):
        return False

    return (
        payload.get("name") == CODEX_GRAPHOS_SERVER
        and payload.get("enabled", True) is True
        and not payload.get("disabled_reason")
        and not payload.get("startup_timeout_sec")
        and not payload.get("tool_timeout_sec")
        and not payload.get("enabled_tools")
        and not payload.get("disabled_tools")
        and transport.get("type") == "stdio"
        and transport.get("command") == CODEX_GRAPHOS_COMMAND[0]
        and transport.get("args") == list(CODEX_GRAPHOS_COMMAND[1:])
        and transport.get("cwd") is None
        and not transport.get("env")
        and not transport.get("env_vars")
    )


def _run_codex(
    args: Sequence[str],
    *,
    executable: str,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> subprocess.CompletedProcess[str]:
    try:
        return runner(
            [executable, *args],
            capture_output=True,
            text=True,
            timeout=_COMMAND_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        # Do not include command output, filesystem paths, or environment-derived
        # exception text in a durable installer/doctor result.
        raise CodexRegistrationError(
            f"Codex MCP registration failed ({type(exc).__name__})."
        ) from None


def register_codex_graphos(
    *,
    executable: str = "codex",
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, str]:
    """Idempotently register the exact portable GraphOS launcher with Codex.

    A stale registration is removed before it is recreated so legacy env, cwd,
    absolute-command, timeout, and tool-filter fields cannot survive.  The result
    is intentionally safe to print or retain in an installation report.
    """
    invoke = runner or subprocess.run
    current = _run_codex(
        ("mcp", "get", CODEX_GRAPHOS_SERVER, "--json"),
        executable=executable,
        runner=invoke,
    )

    if current.returncode == 0:
        try:
            payload = json.loads(current.stdout)
        except (json.JSONDecodeError, TypeError):
            payload = None
        if _is_canonical_registration(payload):
            return {
                "status": "unchanged",
                "server": CODEX_GRAPHOS_SERVER,
                "transport": "stdio",
            }

        removed = _run_codex(
            ("mcp", "remove", CODEX_GRAPHOS_SERVER),
            executable=executable,
            runner=invoke,
        )
        if removed.returncode != 0:
            raise CodexRegistrationError("Codex MCP registration could not be replaced.")
        status = "replaced"
    else:
        status = "registered"

    added = _run_codex(
        (
            "mcp",
            "add",
            CODEX_GRAPHOS_SERVER,
            "--",
            *CODEX_GRAPHOS_COMMAND,
        ),
        executable=executable,
        runner=invoke,
    )
    if added.returncode != 0:
        raise CodexRegistrationError("Codex MCP registration could not be written.")

    return {
        "status": status,
        "server": CODEX_GRAPHOS_SERVER,
        "transport": "stdio",
    }


__all__ = [
    "CODEX_GRAPHOS_COMMAND",
    "CODEX_GRAPHOS_SERVER",
    "CodexRegistrationError",
    "graphos_stdio_spec",
    "register_codex_graphos",
]
