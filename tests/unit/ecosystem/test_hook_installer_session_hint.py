"""CONCEPT:AU-OS.deployment.workspace-venv-reconciler (D-VS-6).

Regression coverage for wiring the near-zero-cost venv-drift hint into every
agent surface's SessionStart hook. Structural only — these do not exercise
``install_hooks()`` against a real config file (a separate, larger surface);
they pin the shape of the hook templates themselves, so a future edit cannot
silently drop the hint or its safety suffix.
"""

from __future__ import annotations

from agent_utilities.ecosystem.hook_installer import (
    _CLAUDE_HOOKS,
    _CODEX_HOOKS,
    _GROK_HOOKS,
    _VENV_SESSION_HINT_CMD,
)


def test_venv_session_hint_command_is_silent_safe() -> None:
    """Never propagates a failure and never spams stderr on a host without it."""
    assert _VENV_SESSION_HINT_CMD.startswith("agent-utilities-venv session-hint")
    assert "2>/dev/null" in _VENV_SESSION_HINT_CMD
    assert _VENV_SESSION_HINT_CMD.rstrip().endswith("|| true")


def test_claude_shaped_session_start_includes_the_hint_after_context() -> None:
    """Claude/Antigravity/Windsurf/Cowork/Hermes all share this exact list."""
    session_start = _CLAUDE_HOOKS["hooks"]["SessionStart"]
    commands = [entry["command"] for entry in session_start]
    assert commands[0].startswith("agent-utilities context")
    assert _VENV_SESSION_HINT_CMD in commands


def test_codex_shaped_session_start_chains_the_hint() -> None:
    """Codex/Devin/OpenCode share this single-string hook shape."""
    on_start = _CODEX_HOOKS["hooks"]["on_session_start"]
    assert on_start.startswith("agent-utilities context")
    assert _VENV_SESSION_HINT_CMD in on_start


def test_grok_session_start_chains_the_hint() -> None:
    on_start = _GROK_HOOKS["hooks"]["SessionStart"]["command"]
    assert on_start.startswith("agent-utilities context")
    assert _VENV_SESSION_HINT_CMD in on_start
