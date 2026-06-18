"""Tests for per-backend send_reaction (CONCEPT:ECO-4.60)."""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.messaging.backends.slack import SlackBackend


@pytest.mark.asyncio
async def test_slack_reaction_maps_unicode_to_name() -> None:
    calls: list[dict[str, Any]] = []

    class _Client:
        async def reactions_add(self, **kw: Any) -> None:
            calls.append(kw)

    b = SlackBackend()
    b._client = _Client()
    await b.send_reaction("C123", "1700000000.0001", "👍")
    assert calls == [
        {"channel": "C123", "timestamp": "1700000000.0001", "name": "thumbsup"}
    ]


@pytest.mark.asyncio
async def test_slack_reaction_passes_through_unknown() -> None:
    calls: list[dict[str, Any]] = []

    class _Client:
        async def reactions_add(self, **kw: Any) -> None:
            calls.append(kw)

    b = SlackBackend()
    b._client = _Client()
    await b.send_reaction("C1", "ts", ":custom:")
    assert calls[0]["name"] == "custom"
