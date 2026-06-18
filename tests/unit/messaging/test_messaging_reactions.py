"""Tests for universal message reactions (CONCEPT:ECO-4.60)."""

from __future__ import annotations

import pytest

from agent_utilities.messaging.router import _decide_reaction
from agent_utilities.messaging.service import MessagingService


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
