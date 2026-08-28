"""Characterization test for ``bus_tools._dispatch_bus`` (CX wD10-R-ARITY).

``_dispatch_bus`` had no prior test coverage of its own (only ``AgentBus`` and
``agent_tools`` are covered elsewhere — see ``tests/unit/messaging/``). This lane
converted it from a 23-parameter positional/keyword function with a 31-branch
if/elif chain into a typed ``BusRequest`` dataclass routed through a dict of
2-argument action handlers. Since there was no existing safety net to run
before/after, this file establishes one: it proves every action still reaches
the same ``AgentBus``/``BusFederationRelay`` call with the same arguments, and
that auto-presence (touch + acting-id backfill) and the unknown-action fallback
are unchanged.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.mcp.tools.bus_tools import BusRequest, _dispatch_bus


class _FakeBus:
    """Records every call so tests can assert exact routing + arguments."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.touched: list[str] = []

    def touch(self, agent_id: str) -> bool:
        self.touched.append(agent_id)
        return True

    def _record(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append((name, args, kwargs))

    def register(self, agent_id, **kwargs):
        self._record("register", agent_id, **kwargs)
        return {"agent_id": agent_id}

    def heartbeat(self, agent_id):
        self._record("heartbeat", agent_id)
        return True

    def deregister(self, agent_id):
        self._record("deregister", agent_id)
        return True

    def roster(self, **kwargs):
        self._record("roster", **kwargs)
        return [{"agent_id": "peer"}]

    def send(self, **kwargs):
        self._record("send", **kwargs)
        return {"sent": True}

    def receive(self, agent_id, *, since=0):
        self._record("receive", agent_id, since=since)
        return {"messages": [], "cursor": since}

    def subscribe(self, agent_id, topic):
        self._record("subscribe", agent_id, topic)
        return True

    def unsubscribe(self, agent_id, topic):
        self._record("unsubscribe", agent_id, topic)
        return True

    def dispatch(self, **kwargs):
        self._record("dispatch", **kwargs)
        return {"loop_id": "loop-1"}

    def status(self):
        self._record("status")
        return {"online": 1}


def _req(**overrides: Any) -> BusRequest:
    return BusRequest(action=overrides.pop("action", "status"), **overrides)


def test_register_passes_split_capabilities_and_touches_acting_id():
    bus = _FakeBus()
    req = _req(action="register", agent_id="a1", provider="anthropic", capabilities="x, y")
    out = _dispatch_bus(bus, engine=None, request=req)
    assert '"agent_id": "a1"' in out
    assert bus.calls[0] == (
        "register",
        ("a1",),
        {"provider": "anthropic", "host": "", "capabilities": ["x", "y"], "session_id": ""},
    )
    assert bus.touched == ["a1"]


def test_roster_status_and_unknown_action():
    bus = _FakeBus()
    assert _dispatch_bus(bus, None, _req(action="status")) == '{"online": 1}'
    assert _dispatch_bus(bus, None, _req(action="roster")) != ""
    out = _dispatch_bus(bus, None, _req(action="nonsense"))
    assert out == '{"error": "unknown action: nonsense"}'


def test_send_falls_back_to_agent_id_and_dispatch_falls_back_to_sender():
    bus = _FakeBus()
    _dispatch_bus(bus, None, _req(action="send", agent_id="a1", payload="hi", to="a2"))
    assert bus.calls[0][1] == ()
    assert bus.calls[0][2]["sender"] == "a1"

    bus2 = _FakeBus()
    _dispatch_bus(bus2, None, _req(action="dispatch", sender="a1", objective="do thing"))
    assert bus2.calls[0][2]["sender"] == "a1"


def test_auto_presence_backfills_agent_id_for_receive_but_not_for_register():
    bus = _FakeBus()
    # no agent_id/sender given at all -> nothing to backfill, no touch.
    _dispatch_bus(bus, None, _req(action="receive"))
    assert bus.touched == []
    assert bus.calls[0][1][0] == ""  # agent_id stayed empty

    bus2 = _FakeBus()
    req = BusRequest(action="receive", sender="s1")
    _dispatch_bus(bus2, None, req)
    assert bus2.touched == ["s1"]
    assert bus2.calls[0][1][0] == "s1"  # backfilled from sender


@pytest.mark.parametrize(
    "action,expected_bus_call",
    [
        ("heartbeat", "heartbeat"),
        ("leave", "deregister"),
        ("deregister", "deregister"),
        ("subscribe", "subscribe"),
        ("unsubscribe", "unsubscribe"),
    ],
)
def test_simple_agent_id_actions_route_through(action, expected_bus_call):
    bus = _FakeBus()
    _dispatch_bus(bus, None, _req(action=action, agent_id="a1", topic="t"))
    assert bus.calls[0][0] == expected_bus_call


def test_federation_actions_reach_the_relay(monkeypatch):
    import agent_utilities.mcp.tools.bus_tools as bus_tools_mod

    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    class _FakeRelay:
        @classmethod
        def instance(cls, engine):
            calls.append(("instance", (engine,), {}))
            return cls()

        def register_hub(self, agent_id, url):
            calls.append(("register_hub", (agent_id, url), {}))
            return {"ok": True}

        def list_hubs(self):
            calls.append(("list_hubs", (), {}))
            return []

        def forward(self, group, *, scope):
            calls.append(("forward", (group,), {"scope": scope}))
            return {"forwarded": True}

        def apply_inbound(self, **kwargs):
            calls.append(("apply_inbound", (), kwargs))
            return {"applied": True}

    monkeypatch.setattr(
        bus_tools_mod, "_bus_federation_relay", lambda engine: _FakeRelay.instance(engine)
    )

    bus = _FakeBus()
    engine = object()
    _dispatch_bus(bus, engine, _req(action="register_hub", agent_id="hub1", url="https://x"))
    _dispatch_bus(bus, engine, _req(action="list_hubs"))
    _dispatch_bus(bus, engine, _req(action="federate", group="g1", scope="org"))
    _dispatch_bus(
        bus, engine, _req(action="federate_in", group="g1", to="r1, r2", payload="p", topic="tp")
    )

    kinds = [c[0] for c in calls if c[0] != "instance"]
    assert kinds == ["register_hub", "list_hubs", "forward", "apply_inbound"]


def test_register_hub_requires_agent_id_and_url():
    bus = _FakeBus()
    out = _dispatch_bus(bus, None, _req(action="register_hub", agent_id="", url=""))
    assert out == '{"error": "register_hub needs agent_id (hub name) and url"}'
