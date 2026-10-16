"""GATE — ``MCP_TOOL_MODE`` is retired: no mode switch exists anywhere.

CONCEPT:AU-ECO.mcp.intent-surface-condensed-collapse

The operator retired ``MCP_TOOL_MODE`` and every surface mode (``hybrid``,
``intent``, ``condensed``, ``verbose``, ``both``): every MCP server built on
agent-utilities serves the one condensed intent contract. This gate fails if
the setting reappears in code, configuration, deployment manifests, scripts,
docs or skills — a mode switch that nothing reads is dead configuration, and
one something reads is a second contract.

No occurrence is allowed: the inline-env allowlist that once accepted the key
moved to graph-os with MCP serving (agent-utilities 3.0.0).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_PATTERN = re.compile(r"MCP_TOOL_MODE|mcp_tool_mode")
#: Trees and root files whose content ships or configures a deployment.
_SCANNED = (
    "agent_utilities",
    "scripts",
    "deploy",
    "docs",
    "docker",
    ".env.example",
    "mcp_config.example.json",
    "mcp_config.bus.json",
    "pyproject.toml",
    "README.md",
    "AGENTS.md",
)
_TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".example",
    ".sh",
    ".txt",
    ".cfg",
    ".ini",
    "",
}
_ALLOWED: set[Path] = set()


def _files() -> list[Path]:
    found: list[Path] = []
    for entry in _SCANNED:
        path = ROOT / entry
        if path.is_file():
            found.append(path)
        elif path.is_dir():
            found += [
                p
                for p in path.rglob("*")
                if p.is_file()
                and p.suffix in _TEXT_SUFFIXES
                and "__pycache__" not in p.parts
            ]
    return found


def test_mcp_tool_mode_is_read_and_declared_nowhere() -> None:
    offenders = []
    for path in _files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if _PATTERN.search(text) and path.relative_to(ROOT) not in _ALLOWED:
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, offenders


def test_no_mode_api_survives() -> None:
    from agent_utilities.core.config import AgentConfig

    fields = AgentConfig.model_fields
    assert "mcp_tool_mode" not in fields
