"""AU planner reaction decisions call the host rendering port."""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.messaging.router import _decide_reaction, _react_in_background
from agent_utilities.orchestration.reactions import AgentReaction, EmoteRegistry


class _ReachHost:
    def __init__(self) -> None:
        self.rendered: list[tuple[str, str, str, str]] = []

    async def react(
        self, platform: str, channel_id: str, message_id: str, emoji: str
    ) -> bool:
        self.rendered.append((platform, channel_id, message_id, emoji))
        return True


@pytest.mark.asyncio
async def test_decide_reaction_disabled_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REACTIONS", "0")
    assert await _decide_reaction("great job!") == ""


@pytest.mark.asyncio
async def test_core_reaction_reaches_graphos_host_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = _ReachHost()
    EmoteRegistry._instance = None

    async def _fake_decide(content: str, **kwargs: Any) -> AgentReaction:
        return AgentReaction(
            emote="👀", target_message_id=kwargs.get("target_message_id")
        )

    monkeypatch.setattr(
        "agent_utilities.orchestration.reactions.decide_reaction", _fake_decide
    )
    await _react_in_background(host, "telegram", "42", "100", "look into this")
    assert host.rendered == [("telegram", "42", "100", "👀")]
