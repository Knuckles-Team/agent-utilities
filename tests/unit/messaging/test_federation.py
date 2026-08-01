"""BusFederationRelay — cross-hub forwarding for the agent bus (CONCEPT:AU-ECO.bus.federation-relay).

Drives two in-process hubs (each its own engine) and asserts forward → apply, idempotent
dedup, loop-break, and marking-scope, plus the KG-native peer registry helpers.
"""

from __future__ import annotations

import pytest

from agent_utilities.messaging.bus import AgentBus
from agent_utilities.messaging.federation import BusFederationRelay
from tests.unit.messaging.test_bus import _FakeGraph


class _SessionScoped:
    """Delegate every call on ``obj`` to run under ``session``.

    Two peer hubs are, in production, two separate processes each with their
    own engine and their own verified :class:`GraphSession`. Simulating both
    in one pytest process/session means every operation on hub A's objects
    must run under hub A's ``GraphSession`` (targeting hub A's graph) and
    every operation on hub B's objects under hub B's — the engine's fail-closed
    guard rejects a write whose ambient session doesn't match the graph a
    ``for_graph()`` view is fixed to (CONCEPT:AU-KG.compute.data-is-private-its). This thin proxy re-enters the right
    session around each attribute call so test bodies can call ``busA``/``busB``/
    ``relayA``/``relayB`` directly without threading ``use_session`` through
    every call site themselves.
    """

    def __init__(self, obj, session) -> None:
        self._obj = obj
        self._session = session

    def __getattr__(self, name):
        from agent_utilities.knowledge_graph.core.session import use_session

        attr = getattr(self._obj, name)
        if not callable(attr):
            return attr

        def _call(*args, **kwargs):
            with use_session(self._session):
                return attr(*args, **kwargs)

        return _call


def _wire(relay_from: BusFederationRelay, relay_to: BusFederationRelay, to_session) -> None:
    """Point ``relay_from``'s HTTP forward at ``relay_to``.apply_inbound (in-process).

    The simulated HTTP call crosses into hub B's own process/session boundary,
    so it must re-enter hub B's ``GraphSession`` for the duration of
    ``apply_inbound`` regardless of which session is ambient on the sending
    (hub A) side.
    """
    from agent_utilities.knowledge_graph.core.session import use_session

    def fake_post(url, body):  # noqa: ANN001 — test shim mirrors _post(url, body)
        with use_session(to_session):
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
    import uuid

    from agent_utilities.knowledge_graph.core.session import (
        current_session,
        use_session,
    )

    AgentBus._instance = None
    BusFederationRelay._instance = None

    # Two independent hubs must land on two independent graphs — a bare
    # ``_FakeGraph()`` resolves the ambient tenant's ONE shared default graph
    # (deterministic per-tenant, not per-instance), which would silently
    # collapse both hubs onto the same graph and defeat every dedup/loop-break
    # assertion below (hub B would see hub A's outbox as already-delivered).
    # Only the FIRST engine construction in a test becomes the process root;
    # every later distinct graph name becomes a ``for_graph()`` *view* over
    # that one transport (CONCEPT:AU-KG.compute.data-is-private-its — one
    # client per process), so hub A and hub B share one transport but are
    # routed to different named graphs, each requiring its own verified
    # ``GraphSession`` for every operation (see ``_SessionScoped``/``_wire``).
    base = current_session()
    session_a = base.with_graph(f"test_hub_a_{uuid.uuid4().hex[:8]}")
    session_b = base.with_graph(f"test_hub_b_{uuid.uuid4().hex[:8]}")

    with use_session(session_a):
        g1 = _FakeGraph(graph_name=session_a.graph)
    with use_session(session_b):
        g2 = _FakeGraph(graph_name=session_b.graph)
        # ``for_graph()`` views deliberately do NOT auto-provision their tenant
        # (transport construction never performs an implicit privileged
        # TenantsCreate) — hub B's graph is such a view over hub A's root
        # transport, so this fixture, not the autouse isolation fixture, owns
        # provisioning (and cleanup) of hub B's named graph.
        try:
            g1._graph._client.tenants.create(session_b.graph)
        except Exception:
            pass

    busA_raw, busB_raw = AgentBus(g1), AgentBus(g2)
    relayA_raw = BusFederationRelay(g1)
    relayB_raw = BusFederationRelay(g2)
    _wire(relayA_raw, relayB_raw, session_b)

    busA = _SessionScoped(busA_raw, session_a)
    busB = _SessionScoped(busB_raw, session_b)
    relayA = _SessionScoped(relayA_raw, session_a)
    relayB = _SessionScoped(relayB_raw, session_b)
    try:
        yield busA, busB, relayA, relayB
    finally:
        with use_session(session_b):
            try:
                g1._graph._client.tenants.delete(session_b.graph)
            except Exception:
                pass


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
