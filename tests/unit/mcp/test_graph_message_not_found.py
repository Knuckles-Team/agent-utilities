"""BUG-8 (kg-exhaustive-smoke.md): ``graph_message(action="receive",
channel_id="unknown")`` returned ``{"messages": [], "cursor": 0}`` — silently
indistinguishable from "channel exists, nothing new" — even though the server
log for the same call showed ``Channel 'unknown' not found``. Exercises the
REAL ``graph_message`` MCP tool dispatch end to end.
"""

from __future__ import annotations

import asyncio
import json

from agent_utilities.mcp import kg_server


class _NotFoundChannels:
    def get_messages(self, channel_id, limit=None):
        raise RuntimeError(f"Channel '{channel_id}' not found")


class _FakeEngine:
    def __init__(self, channels):
        self.graph_compute = type(
            "C", (), {"_client": type("X", (), {"channels": channels})()}
        )()


def _get_tool():
    kg_server.ensure_tools_registered()
    return kg_server.REGISTERED_TOOLS["graph_message"]


def test_receive_on_unknown_channel_returns_clean_not_found(monkeypatch):
    engine = _FakeEngine(_NotFoundChannels())
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)

    tool = _get_tool()
    out = json.loads(
        asyncio.run(tool(action="receive", channel_id="smoke-test-channel"))
    )

    assert out == {"error": "channel not found", "channel_id": "smoke-test-channel"}
    assert "messages" not in out  # never a silently-empty success shape
