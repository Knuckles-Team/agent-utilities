#!/usr/bin/python
from __future__ import annotations

"""Agent Hook Installer.

CONCEPT:ECO-4.0 -- Cross-Agent Memory Hook Installer

Writes startup/checkpoint hooks into external agent configurations so they
call ``agent-utilities context`` at session start and ``agent-utilities observe``
at session end.  Supports 10 agent surfaces with platform-aware pathing.

Supported Agents:
    Claude Code, Codex, Grok Build, Devin, Antigravity, Windsurf,
    OpenCode, agent-terminal-ui, Cowork, Hermes
"""

import json
import logging
import os
import platform
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["HookInstaller", "install_hooks", "uninstall_hooks", "doctor"]

_IS_WINDOWS = platform.system() == "Windows"


def _home() -> Path:
    return Path.home()


def _appdata() -> Path:
    """Windows %APPDATA% or fallback to home."""
    return Path(os.environ.get("APPDATA", _home()))


def _localappdata() -> Path:
    """Windows %LOCALAPPDATA% or fallback."""
    return Path(os.environ.get("LOCALAPPDATA", _home() / "AppData" / "Local"))


# -- Hook Config Templates --

_CONTEXT_CMD = "agent-utilities context --for {agent} --cwd $PWD"
_OBSERVE_CMD = "agent-utilities observe --source {agent}"
_LINT_CMD = "agent-utilities lint-check --cwd $PWD"
_REFLECTOR_CMD = "agent-utilities reflect --source {agent} --cwd $PWD"
_STALENESS_CMD = "agent-utilities staleness-audit --cwd $PWD"

_CLAUDE_HOOKS = {
    "hooks": {
        "SessionStart": [
            {"type": "command", "command": _CONTEXT_CMD.format(agent="claude")}
        ],
        "SessionEnd": [
            {"type": "command", "command": _OBSERVE_CMD.format(agent="claude")},
            {"type": "command", "command": _REFLECTOR_CMD.format(agent="claude")},
        ],
        "PreCompact": [
            {"type": "command", "command": _OBSERVE_CMD.format(agent="claude")}
        ],
    }
}

_CODEX_HOOKS = {
    "hooks": {
        "on_session_start": _CONTEXT_CMD.format(agent="codex"),
        "on_session_end": _OBSERVE_CMD.format(agent="codex"),
    }
}

_GROK_HOOKS = {
    "name": "agent-utilities-memory",
    "version": "1.0.0",
    "hooks": {
        "SessionStart": {"command": _CONTEXT_CMD.format(agent="grok")},
        "SessionEnd": {"command": _OBSERVE_CMD.format(agent="grok")},
    },
    "compatibility": {"claude_hooks": True},
}


# -- Agent Surface Registry --

AGENT_SURFACES: dict[str, dict[str, Any]] = {
    "claude": {
        "name": "Claude Code",
        "config_path": lambda: _home() / ".claude" / "settings.json",
        "hook_data": _CLAUDE_HOOKS,
        "merge_key": "hooks",
    },
    "codex": {
        "name": "Codex",
        "config_path": lambda: _home() / ".codex" / "hooks.json",
        "hook_data": _CODEX_HOOKS,
        "merge_key": "hooks",
    },
    "grok": {
        "name": "Grok Build",
        "config_path": lambda: (
            _home() / ".grok" / "hooks" / "agent-utilities-memory.json"
        ),
        "hook_data": _GROK_HOOKS,
        "merge_key": None,
    },
    "devin": {
        "name": "Devin",
        "config_path": lambda: _home() / ".devin" / "hooks.json",
        "hook_data": _CODEX_HOOKS,
        "merge_key": "hooks",
    },
    "antigravity": {
        "name": "Antigravity IDE",
        "config_path": lambda: _home() / ".gemini" / "antigravity" / "hooks.json",
        "hook_data": _CLAUDE_HOOKS,
        "merge_key": "hooks",
    },
    "windsurf": {
        "name": "Windsurf",
        "config_path": lambda: (
            _localappdata() / "Windsurf" / "hooks.json"
            if _IS_WINDOWS
            else _home() / ".codeium" / "windsurf" / "hooks.json"
        ),
        "hook_data": _CLAUDE_HOOKS,
        "merge_key": "hooks",
    },
    "opencode": {
        "name": "OpenCode",
        "config_path": lambda: _home() / ".opencode" / "hooks.json",
        "hook_data": _CODEX_HOOKS,
        "merge_key": "hooks",
    },
    "agent-terminal-ui": {
        "name": "agent-terminal-ui",
        "config_path": lambda: None,
        "hook_data": {},
        "merge_key": None,
    },
    "cowork": {
        "name": "Claude Cowork",
        "config_path": lambda: (
            _home()
            / "Library"
            / "Application Support"
            / "Claude"
            / "local-agent-mode-plugins"
            / "agent-utilities-memory.json"
            if platform.system() == "Darwin"
            else None
        ),
        "hook_data": _CLAUDE_HOOKS,
        "merge_key": "hooks",
    },
    "hermes": {
        "name": "Hermes",
        "config_path": lambda: (
            Path(os.environ.get("HERMES_HOME", _home() / ".hermes"))
            / "plugins"
            / "agent-utilities-memory.json"
        ),
        "hook_data": _CLAUDE_HOOKS,
        "merge_key": "hooks",
    },
}


class HookInstaller:
    """Installs agent-utilities memory hooks into external agent configurations.

    CONCEPT:ECO-4.0 -- Cross-Agent Memory Hook Installer
    """

    def __init__(self) -> None:
        self.installed: list[str] = []
        self.skipped: list[str] = []
        self.errors: list[str] = []

    def install(self, agents: list[str] | None = None) -> dict[str, str]:
        """Install hooks for specified agents (or all).

        Args:
            agents: List of agent names. None = all supported agents.

        Returns:
            Dict mapping agent name to status ('installed'|'skipped'|'error').
        """
        targets = agents or list(AGENT_SURFACES.keys())
        results: dict[str, str] = {}

        for agent_key in targets:
            agent_key = agent_key.lower().strip()
            if agent_key not in AGENT_SURFACES:
                results[agent_key] = "unknown_agent"
                self.errors.append(f"Unknown agent: {agent_key}")
                continue

            surface = AGENT_SURFACES[agent_key]
            config_path = surface["config_path"]()

            if config_path is None:
                if agent_key == "agent-terminal-ui":
                    results[agent_key] = "integrated"
                    logger.info(
                        "[ECO-4.6] %s: already integrated via Python API",
                        surface["name"],
                    )
                else:
                    results[agent_key] = "unsupported_platform"
                    self.skipped.append(agent_key)
                continue

            try:
                self._write_hook(config_path, surface)
                results[agent_key] = "installed"
                self.installed.append(agent_key)
                logger.info(
                    "[ECO-4.6] Installed hooks for %s at %s",
                    surface["name"],
                    config_path,
                )
            except Exception as e:
                results[agent_key] = f"error: {e}"
                self.errors.append(f"{agent_key}: {e}")
                logger.warning(
                    "[ECO-4.6] Failed to install hooks for %s: %s", surface["name"], e
                )

        return results

    def uninstall(self, agents: list[str] | None = None) -> dict[str, str]:
        """Remove hooks for specified agents (or all)."""
        targets = agents or list(AGENT_SURFACES.keys())
        results: dict[str, str] = {}

        for agent_key in targets:
            agent_key = agent_key.lower().strip()
            if agent_key not in AGENT_SURFACES:
                continue
            surface = AGENT_SURFACES[agent_key]
            config_path = surface["config_path"]()
            if config_path is None or not config_path.exists():
                results[agent_key] = "not_installed"
                continue

            try:
                if surface.get("merge_key"):
                    self._remove_merged_hook(config_path, surface)
                else:
                    config_path.unlink()
                results[agent_key] = "uninstalled"
            except Exception as e:
                results[agent_key] = f"error: {e}"

        return results

    def doctor(self) -> dict[str, dict[str, Any]]:
        """Verify all hook installations and return health report."""
        report: dict[str, dict[str, Any]] = {}
        for agent_key, surface in AGENT_SURFACES.items():
            config_path = surface["config_path"]()
            entry: dict[str, Any] = {
                "name": surface["name"],
                "path": str(config_path) if config_path else None,
            }
            if config_path is None:
                entry["status"] = (
                    "integrated" if agent_key == "agent-terminal-ui" else "n/a"
                )
            elif not config_path.exists():
                entry["status"] = "not_installed"
            else:
                content = config_path.read_text(encoding="utf-8")
                entry["status"] = "healthy" if "agent-utilities" in content else "stale"
                entry["size_bytes"] = config_path.stat().st_size
            report[agent_key] = entry
        return report

    # -- Internal --

    def _write_hook(self, path: Path, surface: dict[str, Any]) -> None:
        """Write hook config, merging with existing if needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        merge_key = surface.get("merge_key")

        if merge_key and path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            hook_data = surface["hook_data"]
            if merge_key in existing and merge_key in hook_data:
                existing[merge_key].update(hook_data[merge_key])
            else:
                existing.update(hook_data)
            path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        else:
            path.write_text(
                json.dumps(surface["hook_data"], indent=2), encoding="utf-8"
            )

    def _remove_merged_hook(self, path: Path, surface: dict[str, Any]) -> None:
        """Remove agent-utilities hooks from a merged config file."""
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            merge_key = surface.get("merge_key")
            if merge_key and merge_key in existing:
                hooks = existing[merge_key]
                keys_to_remove = [
                    k
                    for k, v in hooks.items()
                    if isinstance(v, list | dict | str) and "agent-utilities" in str(v)
                ]
                for k in keys_to_remove:
                    del hooks[k]
                path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        except Exception:
            pass  # nosec B110


# -- Module-level convenience --


def install_hooks(agents: list[str] | None = None) -> dict[str, str]:
    """Install memory hooks into external agents."""
    return HookInstaller().install(agents)


def uninstall_hooks(agents: list[str] | None = None) -> dict[str, str]:
    """Remove memory hooks from external agents."""
    return HookInstaller().uninstall(agents)


def doctor() -> dict[str, dict[str, Any]]:
    """Verify all hook installations."""
    return HookInstaller().doctor()
