"""AgentBus — federated agent-to-agent bus over the KG (CONCEPT:ECO-4.84 / KG-2.141).

Exercises the core (presence, durable mailbox + cursor, pub/sub, governed send, dispatch
bridge) against an in-memory graph fake that interprets the handful of Cypher patterns the bus
issues, plus a live-path test through the ``graph_bus`` MCP tool registration.
"""

from __future__ import annotations

import json

import pytest

from agent_utilities.messaging.bus import AgentBus


class _FakeGraph:
    """Minimal in-memory LPG that answers the AgentBus Cypher patterns."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}  # node_id -> {"type", **props}
        self.edges: set[tuple[str, str, str]] = set()  # (src_id, dst_id, rel)

    # write surface (engine-compatible signatures)
    def add_node(self, node_id, node_type, properties=None):
        self.nodes.setdefault(node_id, {"type": node_type}).update(properties or {})
        self.nodes[node_id]["type"] = node_type

    def add_edge(self, source, target, rel_type=""):
        self.edges.add((source, target, rel_type))

    def _agent_node(self, agent_id):
        return self.nodes.get(f"agent:{agent_id}")

    # read surface — branch on the query's shape
    def query_cypher(self, cypher, params=None):
        params = params or {}
        if "HAS_BUS_MESSAGE]->(m:BusMessage)" in cypher:
            src = f"agent:{params.get('aid')}"
            mids = [
                d for (s, d, r) in self.edges if s == src and r == "HAS_BUS_MESSAGE"
            ]
            return [{"m": self.nodes[m]} for m in mids if m in self.nodes]
        if "BusMessage {msg_group:" in cypher:
            g = params.get("g")
            return [
                {"m": n}
                for n in self.nodes.values()
                if n.get("type") == "BusMessage" and n.get("msg_group") == g
            ]
        if "BusMessage {id:" in cypher:
            node = self.nodes.get(params.get("mid"))
            if node and node.get("recipient") == params.get("aid"):
                return [{"m": node}]
            return []
        if "SUBSCRIBES_TO]->(:Topic {name:" in cypher:
            topic = f"topic:{params.get('t')}"
            return [
                {"aid": self.nodes[s].get("agent_id")}
                for (s, d, r) in self.edges
                if d == topic and r == "SUBSCRIBES_TO" and s in self.nodes
            ]
        if "UNSUBSCRIBED]->(:Topic {name:" in cypher:
            topic = f"topic:{params.get('t')}"
            return [
                {"aid": self.nodes[s].get("agent_id")}
                for (s, d, r) in self.edges
                if d == topic and r == "UNSUBSCRIBED" and s in self.nodes
            ]
        if "(a:Agent {agent_id:" in cypher:
            node = self._agent_node(params.get("aid"))
            return [{"a": node}] if node else []
        if "(a:Agent) RETURN a" in cypher:
            return [{"a": n} for n in self.nodes.values() if n.get("type") == "Agent"]
        if "(t:Topic) RETURN t.name" in cypher:
            return [
                {"name": n.get("name")}
                for n in self.nodes.values()
                if n.get("type") == "Topic"
            ]
        return []


@pytest.fixture()
def bus():
    AgentBus._instance = None  # isolate the singleton per test
    return AgentBus(engine=_FakeGraph())


def test_register_and_roster_presence(bus):
    bus.register(
        "claude-a", provider="anthropic", host="h1", capabilities=["code", "review"]
    )
    bus.register("gpt-b", provider="openai", host="h2", capabilities=["search"])
    roster = bus.roster()
    ids = {a["agent_id"]: a for a in roster}
    assert set(ids) == {"claude-a", "gpt-b"}
    assert ids["claude-a"]["presence"] == "online"
    assert ids["claude-a"]["capabilities"] == ["code", "review"]
    # filters
    assert [a["agent_id"] for a in bus.roster(provider="openai")] == ["gpt-b"]
    assert [a["agent_id"] for a in bus.roster(capability="review")] == ["claude-a"]


def test_presence_goes_stale_without_heartbeat(bus):
    bus.register("idle", provider="anthropic")
    # force last_seen into the past, then roster with a tight window
    bus._engine.nodes["agent:idle"]["last_seen"] = 0.0
    assert bus.roster()[0]["presence"] == "offline"
    assert bus.roster(online_only=True) == []
    # a heartbeat brings it back online
    assert bus.heartbeat("idle") is True
    assert bus.roster()[0]["presence"] == "online"


def test_heartbeat_preserves_capabilities(bus):
    bus.register("a", capabilities=["x", "y"])
    bus._engine.nodes["agent:a"]["last_seen"] = 0.0
    bus.heartbeat("a")
    assert bus.roster()[0]["capabilities"] == ["x", "y"]  # not wiped by the upsert


def test_direct_send_and_cursor_receive(bus):
    bus.register("a")
    bus.register("b")
    r = bus.send(sender="a", to="b", payload="hello")
    assert r["ok"] and r["delivered"] == ["b"]
    got = bus.receive("b")
    assert [m["payload"] for m in got["messages"]] == ["hello"]
    assert got["cursor"] == 1
    # cursor advances: nothing new
    assert bus.receive("b", since=got["cursor"])["messages"] == []
    # a second message is delivered after the cursor
    bus.send(sender="a", to="b", payload="again")
    nxt = bus.receive("b", since=got["cursor"])
    assert [m["payload"] for m in nxt["messages"]] == ["again"]


def test_topic_fanout_to_subscribers_only(bus):
    for a in ("pub", "sub1", "sub2", "lurker"):
        bus.register(a)
    bus.subscribe("sub1", "research")
    bus.subscribe("sub2", "research")
    r = bus.send(sender="pub", topic="research", payload="paper dropped")
    assert set(r["delivered"]) == {"sub1", "sub2"}
    assert [m["payload"] for m in bus.receive("sub1")["messages"]] == ["paper dropped"]
    assert bus.receive("lurker")["messages"] == []


def test_unsubscribe_stops_delivery(bus):
    bus.register("pub")
    bus.register("s")
    bus.subscribe("s", "t")
    bus.unsubscribe("s", "t")  # no native edge-delete on the fake → tombstone path
    assert bus.send(sender="pub", topic="t", payload="x")["delivered"] == []


def test_send_requires_target(bus):
    bus.register("a")
    assert bus.send(sender="a", payload="x")["ok"] is False


def test_ack_marks_message(bus):
    bus.register("a")
    bus.register("b")
    bus.send(sender="a", to="b", payload="hi")
    mid = bus.receive("b")["messages"][0]["id"]
    assert bus.ack("b", mid) is True
    assert bus._engine.nodes[mid]["status"] == "acked"


def test_send_blocked_by_policy(monkeypatch, bus):
    """A deny decision from the ActionPolicy gate refuses the send."""

    class _Deny:
        allowed = False
        decision = "deny"
        reason = "policy off"

    monkeypatch.setattr(AgentBus, "_gate", lambda *a, **k: _Deny())
    bus.register("a")
    bus.register("b")
    out = bus.send(sender="a", to="b", payload="x")
    assert out["ok"] is False and "deny" in out["error"]


def test_dispatch_submits_loop(monkeypatch, bus):
    """dispatch bridges a message to fleet work via submit_loop (CONCEPT:ORCH-1.80)."""
    captured = {}

    def fake_submit_loop(engine, objective, *, kind="research", prio_bucket=2, **kw):
        captured.update(objective=objective, kind=kind, prio_bucket=prio_bucket)
        return {"id": "loop:1", "objective": objective}

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.research.loops.submit_loop", fake_submit_loop
    )
    out = bus.dispatch(
        sender="a", objective="audit the repo", kind="develop", priority="high"
    )
    assert out["ok"] and out["loop"]["id"] == "loop:1"
    assert captured["objective"] == "audit the repo" and captured["kind"] == "develop"
    assert captured["prio_bucket"] == 1  # 'high' -> bucket 1


def test_status_counts(bus):
    bus.register("a")
    bus.register("b")
    bus.subscribe("a", "topicA")
    st = bus.status()
    assert st["agents"] == 2 and st["online"] == 2 and st["topics"] == ["topicA"]


@pytest.mark.asyncio
async def test_graph_bus_tool_live_path(monkeypatch):
    """Live path: register the graph_bus MCP tool and drive register→send→receive."""
    AgentBus._instance = None
    fake = _FakeGraph()
    monkeypatch.setattr("agent_utilities.mcp.kg_server._get_engine", lambda: fake)

    captured = {}

    def _register(mcp_obj):
        from agent_utilities.mcp.tools.bus_tools import register_bus_tools

        register_bus_tools(mcp_obj)

    class _MCP:
        def tool(self, **kw):
            def deco(fn):
                captured["fn"] = fn
                return fn

            return deco

    _register(_MCP())
    graph_bus = captured["fn"]

    # FastMCP resolves Field() defaults at call time; calling the raw function in-test means
    # passing the params each action reads explicitly.
    async def call(**kw):
        base = dict(
            agent_id="",
            sender="",
            to="",
            topic="",
            payload="",
            objective="",
            kind="develop",
            priority="normal",
            provider="",
            host="",
            capabilities="",
            session_id="",
            message_id="",
            since=0,
            online_only=False,
            reason="",
        )
        base.update(kw)
        return json.loads(await graph_bus(**base))

    await call(action="register", agent_id="a", provider="anthropic")
    await call(action="register", agent_id="b", provider="openai")
    roster = await call(action="roster")
    assert {a["agent_id"] for a in roster["roster"]} == {"a", "b"}

    await call(action="send", sender="a", to="b", payload="ping")
    received = await call(action="receive", agent_id="b")
    assert [m["payload"] for m in received["messages"]] == ["ping"]
