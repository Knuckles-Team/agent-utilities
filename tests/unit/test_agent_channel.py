"""CONCEPT:ORCH-1.39 Phase 3 — native invoker↔spawned-agent message channel wiring.

Covers the agent_channel wrapper (deterministic id, send/receive cursor, graceful no-engine
degradation) and the GraphState→AgentDeps threading that hands a spawned agent its channel id.
"""

from __future__ import annotations

import pytest

from agent_utilities.messaging import agent_channel
from agent_utilities.models.agent import AgentDeps


class _FakeChannels:
    """In-memory stand-in for engine.graph_compute._client.channels."""

    def __init__(self):
        self._msgs: dict[str, list[dict]] = {}
        self._open: set[str] = set()

    def create(self, channel_id, channel_type, creator, initial_members):
        self._open.add(channel_id)
        self._msgs.setdefault(channel_id, [])

    def send_message(self, channel_id, sender, payload):
        self._msgs.setdefault(channel_id, []).append({"sender": sender, "payload": payload})

    def get_messages(self, channel_id, limit=None):
        return list(self._msgs.get(channel_id, []))

    def close(self, channel_id):
        self._open.discard(channel_id)


class _FakeEngine:
    def __init__(self):
        ch = _FakeChannels()
        self.graph_compute = type("C", (), {"_client": type("X", (), {"channels": ch})()})()


@pytest.mark.concept("ORCH-1.39")
def test_channel_id_is_deterministic():
    assert agent_channel.channel_id_for("s1", "r1") == "orch:s1:r1"


@pytest.mark.concept("ORCH-1.39")
def test_open_send_receive_cursor_roundtrip():
    eng = _FakeEngine()
    cid = agent_channel.open_channel(eng, "s1", "r1")
    assert cid == "orch:s1:r1"
    assert agent_channel.send(eng, cid, "invoker", "hi")
    assert agent_channel.send(eng, cid, "agent:r1", "ack")
    msgs, cursor = agent_channel.receive(eng, cid, since=0)
    assert [m["payload"] for m in msgs] == ["hi", "ack"]
    assert cursor == 2
    # cursor advances — nothing new
    msgs2, cursor2 = agent_channel.receive(eng, cid, since=cursor)
    assert msgs2 == [] and cursor2 == 2
    assert agent_channel.close(eng, cid)


@pytest.mark.concept("ORCH-1.39")
def test_graceful_when_no_engine_channels():
    bad = type("E", (), {})()  # no graph_compute
    assert agent_channel.open_channel(bad, "s", "r") is None
    assert agent_channel.send(bad, "orch:s:r", "x", "y") is False
    assert agent_channel.receive(bad, "orch:s:r") == ([], 0)
    assert agent_channel.close(bad, "orch:s:r") is False


@pytest.mark.concept("ORCH-1.39")
def test_deps_carry_channel_id_from_state():
    from agent_utilities.graph.executor import agent_deps_from_graph

    deps = type(
        "D",
        (),
        {
            "project_root": "",
            "knowledge_engine": None,
            "mcp_toolsets": [],
            "ssl_verify": True,
            "provider": None,
            "base_url": None,
            "api_key": None,
            "request_id": "r",
            "approval_timeout": 0.0,
            "event_queue": None,
        },
    )()
    state = type("S", (), {"invoker_channel_id": "orch:s:r", "invoker_cred_ref": None})()
    ad = agent_deps_from_graph(deps, [], state)
    assert isinstance(ad, AgentDeps)
    assert ad.message_channel_id == "orch:s:r"
