"""Tests for the messaging RENDERER of core reactions (CONCEPT:ECO-4.81).

Messaging no longer owns the reaction logic — it renders the core orchestrator's
``AgentReaction`` (CONCEPT:ECO-4.79). These tests prove the renderer contract: a core
reaction decision lands on a backend's ``send_reaction``, end-to-end.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.messaging.router import _decide_reaction, _react_in_background
from agent_utilities.messaging.service import MessagingService
from agent_utilities.orchestration.reactions import AgentReaction, EmoteRegistry


class _ReactBackend:
    def __init__(self, supported: bool = True) -> None:
        self.id = "telegram"
        self._connected = True
        self.supported = supported
        self.reactions: list[tuple[str, str, str]] = []

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def send_reaction(self, channel_id: str, message_id: str, emoji: str) -> None:
        if not self.supported:
            raise NotImplementedError
        self.reactions.append((channel_id, message_id, emoji))


@pytest.fixture()
def svc(monkeypatch: pytest.MonkeyPatch) -> tuple[MessagingService, _ReactBackend]:
    MessagingService._instance = None
    service = MessagingService.instance(object())
    backend = _ReactBackend()

    async def _get_backend(_platform: str):
        return backend

    monkeypatch.setattr(service, "get_backend", _get_backend)
    return service, backend


@pytest.mark.asyncio
async def test_react_sends_when_supported(
    svc: tuple[MessagingService, _ReactBackend],
) -> None:
    service, backend = svc
    assert await service.react("telegram", "42", "100", "👍") is True
    assert backend.reactions == [("42", "100", "👍")]


@pytest.mark.asyncio
async def test_react_false_when_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    MessagingService._instance = None
    service = MessagingService.instance(object())
    backend = _ReactBackend(supported=False)

    async def _get_backend(_platform: str):
        return backend

    monkeypatch.setattr(service, "get_backend", _get_backend)
    assert await service.react("telegram", "42", "100", "👍") is False


@pytest.mark.asyncio
async def test_react_requires_all_fields(
    svc: tuple[MessagingService, _ReactBackend],
) -> None:
    service, _ = svc
    assert await service.react("telegram", "42", "", "👍") is False


@pytest.mark.asyncio
async def test_decide_reaction_disabled_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MESSAGING_REACTIONS", "0")
    assert await _decide_reaction("great job!") == ""


# ── Renderer contract: core AgentReaction → backend send_reaction (ECO-4.81) ──
@pytest.mark.asyncio
async def test_render_reaction_paints_core_output(
    svc: tuple[MessagingService, _ReactBackend],
) -> None:
    """A core AgentReaction is rendered onto the backend (the renderer contract)."""
    service, backend = svc
    reaction = AgentReaction(emote="🎉", target_message_id="100")
    assert await service.render_reaction("telegram", "42", reaction) is True
    assert backend.reactions == [("42", "100", "🎉")]


@pytest.mark.asyncio
async def test_render_reaction_ignores_empty(
    svc: tuple[MessagingService, _ReactBackend],
) -> None:
    service, backend = svc
    assert (
        await service.render_reaction("telegram", "42", AgentReaction(emote=""))
        is False
    )
    assert await service.render_reaction("telegram", "42", None) is False
    assert backend.reactions == []


@pytest.mark.asyncio
async def test_background_render_end_to_end(
    svc: tuple[MessagingService, _ReactBackend],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: the CORE decision flows through messaging onto Telegram's send_reaction.

    This is the Wire-First proof — the live messaging path (``_react_in_background``) calls
    the core ``decide_reaction`` and the resulting reaction is rendered on the backend.
    """
    service, backend = svc
    EmoteRegistry._instance = None

    async def _fake_decide(content: str, **kwargs: Any) -> AgentReaction:
        # Core returns an AgentReaction carrying the inbound message id it was given.
        return AgentReaction(
            emote="👀", target_message_id=kwargs.get("target_message_id")
        )

    monkeypatch.setattr(
        "agent_utilities.orchestration.reactions.decide_reaction", _fake_decide
    )
    await _react_in_background(service, "telegram", "42", "100", "look into this")
    assert backend.reactions == [("42", "100", "👀")]
