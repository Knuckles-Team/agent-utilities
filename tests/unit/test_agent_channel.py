"""CONCEPT:AU-ORCH.session.invoker-agent-handoff Phase 3 — native invoker↔spawned-agent message channel wiring.

Covers the agent_channel wrapper (deterministic id, send/receive cursor, graceful no-engine
degradation) and the GraphState→AgentDeps threading that hands a spawned agent its channel id.
"""

from __future__ import annotations

import pytest

from agent_utilities.messaging import agent_channel
from agent_utilities.models.agent import AgentDeps


class _FakeChannels:
    """In-memory stand-in for engine.graph_compute._client.channels.

    Mirrors the live engine's membership rule: send_message rejects a non-member sender
    (RuntimeError), and join() admits one — so the auto-join in agent_channel.send is exercised.
    """

    def __init__(self):
        self._msgs: dict[str, list[dict]] = {}
        self._members: dict[str, set[str]] = {}

    def create(self, channel_id, channel_type, creator, initial_members):
        self._msgs.setdefault(channel_id, [])
        self._members[channel_id] = set(initial_members or [])

    def join(self, channel_id, member):
        self._members.setdefault(channel_id, set()).add(member)

    def send_message(self, channel_id, sender, payload):
        if sender not in self._members.get(channel_id, set()):
            raise RuntimeError(
                f"Agent '{sender}' is not a member of channel '{channel_id}'"
            )
        self._msgs.setdefault(channel_id, []).append(
            {"sender": sender, "payload": payload}
        )

    def get_messages(self, channel_id, limit=None):
        return list(self._msgs.get(channel_id, []))

    def close(self, channel_id):
        self._members.pop(channel_id, None)


class _FakeEngine:
    def __init__(self):
        ch = _FakeChannels()
        self.graph_compute = type(
            "C", (), {"_client": type("X", (), {"channels": ch})()}
        )()
        self._nodes: dict[str, dict] = {}
        self._edges: list[tuple] = []

    # Phase 4 durable-write surface
    def add_node(self, nid, ntype, properties=None):
        self._nodes[nid] = {"type": ntype, "properties": dict(properties or {})}

    def add_edge(self, src, dst, rel):
        self._edges.append((src, dst, rel))

    def query_cypher(self, _query, params=None):
        snode = (params or {}).get("snode")
        out = []
        for src, dst, rel in self._edges:
            if rel == "HAS_MESSAGE" and src == snode:
                node = self._nodes.get(dst)
                if node and node["type"] == "AgentMessage":
                    out.append({"m": {"properties": node["properties"]}})
        return out


@pytest.mark.concept("AU-ORCH.session.invoker-agent-handoff")
def test_channel_id_is_deterministic():
    assert agent_channel.channel_id_for("s1", "r1") == "orch:s1:r1"


@pytest.mark.concept("AU-ORCH.session.invoker-agent-handoff")
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


@pytest.mark.concept("AU-ORCH.session.invoker-agent-handoff")
def test_arbitrary_sender_auto_joins():
    """A sender label that is not an initial member must still be able to send (auto-join)."""
    eng = _FakeEngine()
    cid = agent_channel.open_channel(eng, "s1", "r1")  # members: invoker:s1, agent:r1
    # 'invoker' (bare) is NOT an initial member — send must auto-join, not silently drop.
    assert agent_channel.send(eng, cid, "invoker", "hi")
    msgs, _ = agent_channel.receive(eng, cid)
    assert [m["payload"] for m in msgs] == ["hi"]


@pytest.mark.concept("AU-ORCH.session.invoker-agent-handoff")
def test_graceful_when_no_engine_channels():
    bad = type("E", (), {})()  # no graph_compute
    assert agent_channel.open_channel(bad, "s", "r") is None
    assert agent_channel.send(bad, "orch:s:r", "x", "y") is False
    assert agent_channel.receive(bad, "orch:s:r") == ([], 0)
    assert agent_channel.close(bad, "orch:s:r") is False


@pytest.mark.concept("AU-ORCH.session.invoker-agent-handoff")
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
    state = type(
        "S", (), {"invoker_channel_id": "orch:s:r", "invoker_cred_ref": None}
    )()
    ad = agent_deps_from_graph(deps, [], state)
    assert isinstance(ad, AgentDeps)
    assert ad.message_channel_id == "orch:s:r"


@pytest.mark.concept("AU-ORCH.session.invoker-agent-handoff")
def test_durable_send_persists_and_history_replays():
    eng = _FakeEngine()
    cid = agent_channel.open_channel(eng, "s1", "r1")
    agent_channel.send(eng, cid, "invoker", "ephemeral")  # not durable
    agent_channel.send(eng, cid, "invoker", "durable-1", durable=True)
    agent_channel.send(eng, cid, "agent:r1", "durable-2", durable=True)
    # live channel has all three
    msgs, _ = agent_channel.receive(eng, cid)
    assert [m["payload"] for m in msgs] == ["ephemeral", "durable-1", "durable-2"]
    # durable history has only the two persisted ones, ordered, session-anchored
    hist = agent_channel.history(eng, cid)
    assert [h["payload"] for h in hist] == ["durable-1", "durable-2"]
    assert all(h["session_id"] == "s1" and h["channel_id"] == cid for h in hist)
    # Session anchor + HAS_MESSAGE edges exist
    assert "session:s1" in eng._nodes
    assert sum(1 for _, _, rel in eng._edges if rel == "HAS_MESSAGE") == 2


# ── BUG-8 (kg-exhaustive-smoke.md): receive() on an unknown channel used to
# silently collapse into the SAME empty ([], since) result as "channel exists,
# nothing new" — the engine's own error text ("Channel 'X' not found") was
# logged but never surfaced to the caller. ─────────────────────────────────


class _NotFoundChannels:
    """Mirrors the live engine: get_messages on an unknown channel raises with
    "not found" in the message text (not a plain empty return)."""

    def get_messages(self, channel_id, limit=None):
        raise RuntimeError(f"Channel '{channel_id}' not found")


class _EngineWithChannels:
    def __init__(self, channels):
        self.graph_compute = type(
            "C", (), {"_client": type("X", (), {"channels": channels})()}
        )()


@pytest.mark.concept("AU-ORCH.session.invoker-agent-handoff")
def test_receive_on_unknown_channel_raises_channel_not_found():
    eng = _EngineWithChannels(_NotFoundChannels())
    with pytest.raises(agent_channel.ChannelNotFoundError) as excinfo:
        agent_channel.receive(eng, "smoke-test-channel")
    assert excinfo.value.channel_id == "smoke-test-channel"


@pytest.mark.concept("AU-ORCH.session.invoker-agent-handoff")
def test_receive_on_existing_channel_with_no_new_messages_does_not_raise():
    # The counterpart case: an existing channel with nothing new must still
    # degrade to ([], since) exactly as before — only the not-found case is new.
    eng = _FakeEngine()
    cid = agent_channel.open_channel(eng, "s1", "r1")
    msgs, cursor = agent_channel.receive(eng, cid, since=0)
    assert msgs == [] and cursor == 0


@pytest.mark.concept("AU-ORCH.session.invoker-agent-handoff")
def test_elicitation_bridge_drains_to_queue():
    import asyncio

    eng = _FakeEngine()
    cid = agent_channel.open_channel(eng, "s1", "r1")
    agent_channel.send(eng, cid, "agent:r1", "just a normal message")
    agent_channel.send_elicitation(eng, cid, "May I delete /tmp/x?", durable=False)
    q: asyncio.Queue = asyncio.Queue()
    cursor = agent_channel.drain_to_elicitation_queue(eng, cid, q, since=0)
    assert cursor == 2  # both messages consumed
    assert q.qsize() == 1  # only the elicitation request forwarded
    assert q.get_nowait() == "May I delete /tmp/x?"
