"""Universal messaging command registry (CONCEPT:ECO-4.57).

ONE command spec, surfaced identically across every messaging service — registered on each
platform via its native mechanism (Telegram ``setMyCommands``, Slack/Mattermost slash
commands) and importable by agent-terminal-ui — so the user gets the **same** commands
everywhere instead of per-platform menus. Built-in commands are answered here; anything
else (``/claude``, ``/skill``, unknown) falls through to the messaging agent.

CONCEPT:ECO-4.57 — Universal cross-platform messaging command registry
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MessagingCommand:
    """One command in the universal registry.

    ``builtin`` True → handled by :func:`handle_command` (a deterministic reply); False →
    falls through to the agent/model (e.g. ``/claude`` is a model route, ``/skill`` is
    interpreted by the agent).
    """

    name: str
    description: str
    builtin: bool


# The single source of truth — shared by every platform and the terminal UI.
COMMANDS: tuple[MessagingCommand, ...] = (
    MessagingCommand("help", "Show what this assistant can do and list commands", True),
    MessagingCommand("status", "Show messaging + agent status", True),
    MessagingCommand(
        "tools", "Describe the tools, skills, and MCP fleet available", True
    ),
    MessagingCommand(
        "claude",
        "Reply using Claude instead of the local LLM (needs ANTHROPIC_API_KEY)",
        False,
    ),
    MessagingCommand("skill", "Run a skill by name: /skill <name> [args]", False),
)


def command_specs() -> list[dict[str, str]]:
    """Spec for per-platform registration and agent-terminal-ui rendering."""
    return [{"command": c.name, "description": c.description} for c in COMMANDS]


def _parse(content: str) -> tuple[str, str] | None:
    """Return (command_name, args) for a leading ``/cmd``, else None."""
    s = (content or "").strip()
    if not s.startswith("/"):
        return None
    head, _, rest = s[1:].partition(" ")
    # Telegram allows "/cmd@botname"; strip the bot suffix.
    name = head.split("@", 1)[0].lower()
    return name, rest.strip()


async def handle_command(content: str, *, service: Any) -> str | None:
    """Handle a built-in command and return its reply, or None to fall through to the agent.

    CONCEPT:ECO-4.57 — the single inbound command dispatcher used by every platform.
    """
    parsed = _parse(content)
    if parsed is None:
        return None
    name, _args = parsed
    cmd = next((c for c in COMMANDS if c.name == name), None)
    if cmd is None or not cmd.builtin:
        # Unknown command, or one the agent/model handles (/claude, /skill) → fall through.
        return None

    if name == "help":
        lines = ["I'm your agent-utilities assistant. Commands:"]
        lines += [f"/{c.name} — {c.description}" for c in COMMANDS]
        lines.append(
            "Or just message me normally — I use the Knowledge Graph and skills to help."
        )
        return "\n".join(lines)
    if name == "status":
        try:
            return "status: " + json.dumps(service.status())
        except Exception:  # noqa: BLE001
            return "status: unavailable"
    if name == "tools":
        return (
            "I'm a full agent-utilities agent: I have the universal tools (Knowledge Graph "
            "search, reach_user, and more), the agent skill library, and the MCP server "
            "fleet — loaded on demand. Just ask in plain language and I'll use what fits."
        )
    return None
