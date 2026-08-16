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
    from agent_utilities.messaging.bus_log import current_bus_tenant
    from agent_utilities.messaging.bus_privacy import bus_reference

    # two agents join via the native tool
    assert "joined the bus as 'alice'" in await at.bus_join(
        _ctx("alice"), capabilities="code"
    )
    await at.bus_join(_ctx("bob"), capabilities="research")

    # Durable roster identifiers are non-reversible privacy references
    # (bus_privacy.bus_reference / CONCEPT:AU-ECO.bus.agent-bus-awareness) — the
    # roster never exposes the raw "alice"/"bob" strings, so assert against the
    # same hashed form the product mints deterministically.
    tenant = current_bus_tenant()
    alice_ref = bus_reference("agent", "alice", tenant=tenant)
    bob_ref = bus_reference("agent", "bob", tenant=tenant)

    # discovery excludes self, includes the peer
    peers_for_alice = await at.bus_peers(_ctx("alice"))
    assert bob_ref in peers_for_alice and alice_ref not in peers_for_alice

    # alice messages bob; bob reads it
    sent = await at.bus_send(_ctx("alice"), "need help with the parser", to="bob")
    assert f"delivered to ['{bob_ref}']" in sent
    inbox = await at.bus_check(_ctx("bob"))
    assert "need help with the parser" in inbox and alice_ref in inbox


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
