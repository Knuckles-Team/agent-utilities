"""Tests for the universal messaging command registry (CONCEPT:ECO-4.57)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from agent_utilities.messaging.commands import COMMANDS, command_specs, handle_command


class _Svc:
    def status(self) -> dict[str, Any]:
        return {"configured": ["telegram"], "connected": ["telegram"]}


def test_command_specs_shape() -> None:
    specs = command_specs()
    assert specs and all({"command", "description"} <= set(s) for s in specs)
    # The registry is the single source of truth shared by every platform + the TUI.
    assert {s["command"] for s in specs} == {c.name for c in COMMANDS}


@pytest.mark.asyncio
async def test_help_lists_commands() -> None:
    reply = await handle_command("/help", service=_Svc())
    assert reply is not None
    for c in COMMANDS:
        assert f"/{c.name}" in reply


@pytest.mark.asyncio
async def test_status_returns_service_status() -> None:
    reply = await handle_command("/status", service=_Svc())
    assert reply is not None and "telegram" in reply
    assert json.loads(reply.split("status: ", 1)[1])["connected"] == ["telegram"]


@pytest.mark.asyncio
async def test_tools_summarizes_from_kg() -> None:
    # /tools answers from the KG catalog (counts), not by loading everything (ECO-4.64).
    class _Eng:
        def query_cypher(self, q: str, _p: dict):
            if "count" in q and "Server" in q:
                return [{"c": 12}]
            if "count" in q and "Skill" in q:
                return [{"c": 40}]
            if "count" in q:
                return [{"c": 3}]
            if "Server" in q:
                return [{"name": "gitlab-mcp"}, {"name": "servicenow-api"}]
            return [{"name": "code-enhancer"}]

    class _SvcKG:
        def _resolve_engine(self):
            return _Eng()

    reply = await handle_command("/tools", service=_SvcKG())
    assert reply is not None
    assert "12 MCP servers" in reply and "40 skills" in reply
    assert "gitlab-mcp" in reply


@pytest.mark.asyncio
async def test_tools_fallback_without_engine() -> None:
    reply = await handle_command("/tools", service=_Svc())  # no _resolve_engine
    assert reply is not None and "on demand" in reply


@pytest.mark.asyncio
async def test_botname_suffix_is_stripped() -> None:
    # Telegram delivers "/help@MyBot" in groups.
    assert await handle_command("/help@Crabhammerbot", service=_Svc()) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text", ["hello there", "/claude do x", "/skill foo", "/unknown", ""]
)
async def test_non_builtin_falls_through(text: str) -> None:
    # Plain text, model-routed (/claude), agent-routed (/skill), and unknown all fall
    # through to the agent (return None) rather than being answered here.
    assert await handle_command(text, service=_Svc()) is None


def test_command_specs_surface_filtering() -> None:
    # CONCEPT:ECO-4.71 — one registry, per-surface views; model is terminal-only.
    from agent_utilities.messaging.commands import command_specs, describe

    msg = {s["command"] for s in command_specs("messaging")}
    term = {s["command"] for s in command_specs("terminal")}
    assert "model" in term and "model" not in msg
    assert {"help", "status", "tools", "goal", "sdd", "agents"} <= msg
    assert describe("/goal") == describe("goal") and describe("goal")
    assert describe("nonexistent") is None
