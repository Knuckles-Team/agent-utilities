"""BusFederationRelay — cross-hub forwarding for the agent bus (CONCEPT:ECO-4.86).

Drives two in-process hubs (each its own engine) and asserts forward → apply, idempotent
dedup, loop-break, and marking-scope, plus the KG-native peer registry helpers.
"""

from __future__ import annotations

import pytest

from agent_utilities.messaging.bus import AgentBus
from agent_utilities.messaging.federation import BusFederationRelay
from tests.unit.messaging.test_bus import _FakeGraph


def _wire(relay_from: BusFederationRelay, relay_to: BusFederationRelay) -> None:
    """Point ``relay_from``'s HTTP forward at ``relay_to``.apply_inbound (in-process)."""

    def fake_post(url, body):  # noqa: ANN001 — test shim mirrors _post(url, body)
        return relay_to.apply_inbound(
            group=body["group"],
            sender=body["sender"],
            recipients=[r for r in body["to"].split(",") if r],
            payload=body["payload"],
            topic=body["topic"],
            origin=body["origin"],
        )

    relay_from._post = fake_post  # type: ignore[method-assign]
    relay_from.list_hubs = lambda: [{"name": "B", "url": "http://hub-b"}]  # type: ignore[method-assign]


@pytest.fixture()
def hubs():
    AgentBus._instance = None
    BusFederationRelay._instance = None
    g1, g2 = _FakeGraph(), _FakeGraph()
    busA, busB = AgentBus(g1), AgentBus(g2)
    relayA = BusFederationRelay(g1)
    relayB = BusFederationRelay(g2)
    _wire(relayA, relayB)
    return busA, busB, relayA, relayB


def test_forward_delivers_to_peer_hub(hubs):
    busA, busB, relayA, _ = hubs
    busA.register("pub")
    sent = busA.send(sender="pub", to="bob", payload="cross-hub hi")
    group = sent["msg_group"]

    out = relayA.forward(group)
    assert out["ok"] and out["results"]["B"]["applied"] == 1
    # bob, living on hub B, now sees the forwarded message
    assert [m["payload"] for m in busB.receive("bob")["messages"]] == ["cross-hub hi"]


def test_forward_is_idempotent(hubs):
    busA, busB, relayA, _ = hubs
    busA.register("pub")
    group = busA.send(sender="pub", to="bob", payload="once")["msg_group"]
    relayA.forward(group)
    again = relayA.forward(group)
    assert again["results"]["B"]["dedup"] is True
    # bob still has exactly one copy
    assert len(busB.receive("bob")["messages"]) == 1


def test_loop_break_no_reforward(hubs):
    busA, busB, relayA, relayB = hubs
    busA.register("pub")
    group = busA.send(sender="pub", to="bob", payload="x")["msg_group"]
    relayA.forward(group)
    # hub B received it (federated_from set) → B must not forward it back
    out = relayB.forward(group)
    assert out["skipped"] == "already_federated"


def test_marked_scope_stays_local(hubs):
    busA, _busB, relayA, _ = hubs
    busA.register("pub")
    group = busA.send(sender="pub", to="bob", payload="secret")["msg_group"]
    out = relayA.forward(group, scope="private")
    assert out["forwarded"] == 0 and "scope" in out["skipped"]


def test_register_and_list_hubs(monkeypatch):
    from agent_utilities.models import A2APeerModel, A2ARegistryModel

    BusFederationRelay._instance = None
    relay = BusFederationRelay(_FakeGraph())
    captured = {}

    def fake_register(name, url, description="", capabilities="", auth="none"):
        captured.update(name=name, url=url, capabilities=capabilities)
        return "ok"

    def fake_list():
        return A2ARegistryModel(
            peers=[
                A2APeerModel(
                    name="B", url="http://hub-b", capabilities="agent-bus-hub"
                ),
                A2APeerModel(name="svc", url="http://svc", capabilities="tickets"),
            ]
        )

    monkeypatch.setattr(
        "agent_utilities.protocols.a2a.register_a2a_peer", fake_register
    )
    monkeypatch.setattr("agent_utilities.protocols.a2a.list_a2a_peers", fake_list)

    relay.register_hub("B", "http://hub-b")
    assert captured["capabilities"] == "agent-bus-hub"
    hubs = relay.list_hubs()
    assert hubs == [
        {"name": "B", "url": "http://hub-b"}
    ]  # the non-hub peer is filtered out
