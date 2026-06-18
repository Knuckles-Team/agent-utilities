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
