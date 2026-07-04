"""Native AgentBus agent-tools + capability injection (CONCEPT:AU-ECO.bus.agent-bus-awareness).

Every spawned agent inherits bus awareness (the prompt blurb) and the in-process bus_* tools,
so the orchestrator and swarm sub-agents coordinate over the bus natively.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_utilities.messaging.bus import AgentBus, bus_capability_prompt, swarm_topic
from agent_utilities.tools import agent_tools as at
from tests.unit.messaging.test_bus import _FakeGraph


@pytest.fixture()
def bus():
    AgentBus._instance = AgentBus(engine=_FakeGraph())
    return AgentBus._instance


def _ctx(session_id="s1", provider="anthropic"):
    return SimpleNamespace(
        deps=SimpleNamespace(session_id=session_id, provider=provider)
    )


def test_capability_prompt_is_actionable():
    p = bus_capability_prompt()
    for token in (
        "AgentBus",
        "bus_join",
        "bus_peers",
        "bus_send",
        "bus_check",
        "swarm",
    ):
        assert token in p


def test_self_id_resolves_from_deps():
    assert at._bus_self_id(_ctx(session_id="abc")) == "abc"
    assert at._bus_self_id(_ctx(session_id=None), override="explicit") == "explicit"
    assert at._bus_self_id(SimpleNamespace(deps=None)) == "agent"


@pytest.mark.asyncio
async def test_native_tools_join_peer_send_check(bus):
    # two agents join via the native tool
    assert "joined the bus as 'alice'" in await at.bus_join(
        _ctx("alice"), capabilities="code"
    )
    await at.bus_join(_ctx("bob"), capabilities="research")

    # discovery excludes self, includes the peer
    peers_for_alice = await at.bus_peers(_ctx("alice"))
    assert "bob" in peers_for_alice and "alice" not in peers_for_alice

    # alice messages bob; bob reads it
    sent = await at.bus_send(_ctx("alice"), "need help with the parser", to="bob")
    assert "delivered to ['bob']" in sent
    inbox = await at.bus_check(_ctx("bob"))
    assert "need help with the parser" in inbox and "alice" in inbox


@pytest.mark.asyncio
async def test_native_topic_coordination(bus):
    await at.bus_join(_ctx("w1"))
    await at.bus_join(_ctx("w2"))
    t = swarm_topic("xyz")
    # w2 subscribes by sending on the topic (bus_send auto-subscribes the sender too),
    # then w1 broadcasts and w2 receives
    await at.bus_send(_ctx("w2"), "online", topic=t)
    await at.bus_send(_ctx("w1"), "taking the IO subtask", topic=t)
    inbox = await at.bus_check(_ctx("w2"))
    assert "taking the IO subtask" in inbox


def test_swarm_topic_is_stable_per_session():
    assert swarm_topic("s") == "swarm:s"
    assert swarm_topic(None) == "swarm:default"
