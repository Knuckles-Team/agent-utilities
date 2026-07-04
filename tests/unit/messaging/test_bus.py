"""AgentBus — federated agent-to-agent bus over the KG (CONCEPT:AU-ECO.bus.agentbus-federated-agent-agent / KG-2.141).

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

    def delete_node(self, node_id):
        self.nodes.pop(node_id, None)

    def _agent_node(self, agent_id):
        return self.nodes.get(f"busagent:{agent_id}")

    # read surface — branch on the query's shape (all 1-hop, matching the live-robust bus)
    def query_cypher(self, cypher, params=None):
        params = params or {}
        if "BusMessage {recipient:" in cypher:
            aid = params.get("aid")
            return [
                {"m": n}
                for n in self.nodes.values()
                if n.get("type") == "BusMessage" and n.get("recipient") == aid
            ]
        if "BusMessage {msg_group:" in cypher:
            g = params.get("g")
            return [
                {"m": n}
                for n in self.nodes.values()
                if n.get("type") == "BusMessage" and n.get("msg_group") == g
            ]
        if "BusMessage {topic: $t, kind: 'topic'}" in cypher:
            t = params.get("t")
            return [
                {"m": n}
                for n in self.nodes.values()
                if n.get("type") == "BusMessage"
                and n.get("kind") == "topic"
                and n.get("topic") == t
            ]
        if "BusMessage {kind: 'topic'}" in cypher:
            return [
                {"m": n}
                for n in self.nodes.values()
                if n.get("type") == "BusMessage" and n.get("kind") == "topic"
            ]
        if "BusTopicCursor {agent_id:" in cypher:
            aid, t = params.get("aid"), params.get("t")
            return [
                {"c": n}
                for n in self.nodes.values()
                if n.get("type") == "BusTopicCursor"
                and n.get("agent_id") == aid
                and n.get("topic") == t
            ]
        if "BusSubscription {agent_id:" in cypher:
            aid = params.get("aid")
            return [
                {"s": n}
                for n in self.nodes.values()
                if n.get("type") == "BusSubscription" and n.get("agent_id") == aid
            ]
        if "BusMessage {id:" in cypher:
            node = self.nodes.get(params.get("mid"))
            if node and node.get("recipient") == params.get("aid"):
                return [{"m": node}]
            return []
        if "BusSubscription {topic:" in cypher:
            t = params.get("t")
            return [
                {"s": n}
                for n in self.nodes.values()
                if n.get("type") == "BusSubscription" and n.get("topic") == t
            ]
        if "(a:BusAgent {agent_id:" in cypher:
            node = self._agent_node(params.get("aid"))
            return [{"a": node}] if node else []
        if "(a:BusAgent) RETURN a" in cypher:
            return [
                {"a": n} for n in self.nodes.values() if n.get("type") == "BusAgent"
            ]
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
    bus._engine.nodes["busagent:idle"]["last_seen"] = 0.0
    assert bus.roster()[0]["presence"] == "offline"
    assert bus.roster(online_only=True) == []
    # a heartbeat brings it back online
    assert bus.heartbeat("idle") is True
    assert bus.roster()[0]["presence"] == "online"


def test_heartbeat_preserves_capabilities(bus):
    bus.register("a", capabilities=["x", "y"])
    bus._engine.nodes["busagent:a"]["last_seen"] = 0.0
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


def test_topic_send_with_no_subscribers_is_stored_then_replayed(bus):
    """Store-and-forward (ECO-4.91): a topic message with zero current subscribers is LEFT and
    delivered to an agent that subscribes later."""
    bus.register("pub")
    out = bus.send(sender="pub", topic="news", payload="breaking")
    assert out["ok"] and out["delivered"] == [] and out["stored"] is True

    # A peer subscribes AFTER the send. With replay_recent it backfills the recent window.
    bus.register("late")
    bus.subscribe("late", "news", replay_recent=True)
    got = bus.receive("late")
    assert [m["payload"] for m in got["messages"]] == ["breaking"]
    # Read once only — a second receive yields nothing (cursor advanced).
    assert bus.receive("late")["messages"] == []


def test_late_subscriber_replay_no_double_delivery_to_existing(bus):
    """A current subscriber gets the message ONCE (per-recipient), not again via backlog; a
    later subscriber gets the stored backlog exactly once (ECO-4.91)."""
    for a in ("pub", "early"):
        bus.register(a)
    bus.subscribe("early", "t")
    r = bus.send(sender="pub", topic="t", payload="m1")
    assert r["delivered"] == ["early"] and r["stored"] is True
    # The existing subscriber sees m1 once (per-recipient) and no backlog duplicate — passing
    # the returned cursor back yields nothing more (the topic-log entry isn't re-delivered).
    first = bus.receive("early")
    assert [m["payload"] for m in first["messages"]] == ["m1"]
    assert bus.receive("early", since=first["cursor"])["messages"] == []

    # A NEW subscriber joining without replay_recent gets only FUTURE messages, not m1.
    bus.register("late")
    bus.subscribe("late", "t")
    assert bus.receive("late")["messages"] == []
    bus.send(sender="pub", topic="t", payload="m2")
    # late gets m2 once (it's a current subscriber now → per-recipient delivery).
    got_late = bus.receive("late")
    assert [m["payload"] for m in got_late["messages"]] == ["m2"]
    assert bus.receive("late", since=got_late["cursor"])["messages"] == []


def test_reaper_prunes_expired_topic_messages(bus):
    bus.register("pub")
    bus.send(sender="pub", topic="t", payload="old")
    # Force the stored topic message past its TTL.
    for n in bus._engine.nodes.values():
        if n.get("type") == "BusMessage" and n.get("kind") == "topic":
            n["expires_at"] = 1.0
    assert bus.prune_topic_log() == 1
    assert not any(n.get("kind") == "topic" for n in bus._engine.nodes.values())


def test_auto_register_on_first_touch(bus):
    """ECO-4.92: touching the bus with a fresh id auto-creates the :BusAgent and shows online."""
    assert bus.roster() == []
    assert bus.touch("fresh") is True
    roster = bus.roster()
    assert [a["agent_id"] for a in roster] == ["fresh"]
    assert roster[0]["presence"] == "online"
    # Touch preserves any registered capabilities (no blob clobber).
    bus.register("capped", capabilities=["x"])
    bus._engine.nodes["busagent:capped"]["last_seen"] = 0.0
    bus.touch("capped")
    assert bus.roster(capability="x")[0]["agent_id"] == "capped"


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
    """dispatch bridges a message to fleet work via submit_loop (CONCEPT:AU-ORCH.routing.resolve-body-single-canonical)."""
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
            replay_recent=False,
            ctx=None,
        )
        base.update(kw)
        return json.loads(await graph_bus(**base))

    # ECO-4.92: a fresh agent_id auto-registers on first touch — no explicit register call.
    await call(action="receive", agent_id="a")
    roster = await call(action="roster")
    assert "a" in {x["agent_id"] for x in roster["roster"]}

    await call(action="register", agent_id="b", provider="openai")
    roster = await call(action="roster")
    assert {a["agent_id"] for a in roster["roster"]} == {"a", "b"}

    await call(action="send", sender="a", to="b", payload="ping")
    received = await call(action="receive", agent_id="b")
    assert [m["payload"] for m in received["messages"]] == ["ping"]


# ── Served-profile auth: a failed write surfaces WHY; auth gates register (ECO-4.98) ──


class _DenyingGraph(_FakeGraph):
    """An engine whose ``add_node`` is denied — the served-profile/ACL failure mode.

    Mirrors what happens under ``KG_BRAIN_ENFORCE`` when a bus write is rejected by
    the engine: ``add_node`` raises, so ``_add_node`` returns False. The point of
    ECO-4.98 is that this must NOT be swallowed as a benign ``ok:false``.
    """

    def add_node(self, node_id, node_type, properties=None):  # noqa: D401
        raise PermissionError("write denied for graph '__commons__' (tenant ACL)")


def test_register_failure_surfaces_error_not_silent_false():
    """A denied :BusAgent write returns ok:false WITH an explanatory error (ECO-4.98)."""
    AgentBus._instance = None
    bus = AgentBus(engine=_DenyingGraph())
    res = bus.register("agent-x", capabilities=["code"])
    assert res["ok"] is False
    # The real denial reason is surfaced — never a silent benign false.
    assert "error" in res and res["error"]
    assert "PermissionError" in res["error"]
    assert "denied" in res["error"]
    # And the bus stashed the reason for any caller that introspects it.
    assert "PermissionError" in bus._last_write_error


def test_register_missing_engine_surfaces_error():
    """With no durable store, register still explains why it could not land (ECO-4.98)."""
    AgentBus._instance = None
    bus = AgentBus(engine=object())  # no add_node attribute
    res = bus.register("agent-y")
    assert res["ok"] is False
    assert "error" in res and "no active engine" in res["error"]


@pytest.mark.asyncio
async def test_graph_bus_authenticated_register_lands_under_served_profile(monkeypatch):
    """Served profile: an AUTHENTICATED MCP caller registers and the node lands (ECO-4.98)."""
    from agent_utilities.models.company_brain import ActorType
    from agent_utilities.security.brain_context import ActorContext

    AgentBus._instance = None
    fake = _FakeGraph()
    monkeypatch.setattr("agent_utilities.mcp.kg_server._get_engine", lambda: fake)
    # Enforced-auth posture.
    monkeypatch.setattr("agent_utilities.mcp.kg_server._kg_auth_required", lambda: True)
    monkeypatch.setattr("agent_utilities.mcp.kg_server._PROCESS_ACTOR", None)
    # A validated MCP Bearer token mints an authenticated service actor.
    authed = ActorContext(
        actor_id="svc-account",
        actor_type=ActorType.AUTOMATED_SERVICE,
        roles=(),
        tenant_id="",
        authenticated=True,
    )
    monkeypatch.setattr(
        "agent_utilities.mcp.kg_server._actor_from_mcp_token", lambda: authed
    )

    graph_bus = _make_graph_bus()
    res = await _bus_call(graph_bus, action="register", agent_id="auth-agent")
    assert res["ok"] is True and "error" not in res
    roster = await _bus_call(graph_bus, action="roster")
    assert "auth-agent" in {a["agent_id"] for a in roster["roster"]}


@pytest.mark.asyncio
async def test_graph_bus_unauthenticated_register_rejected_with_error(monkeypatch):
    """Served profile: an UNauthenticated MCP register is rejected WITH an error (ECO-4.98)."""
    AgentBus._instance = None
    fake = _FakeGraph()
    monkeypatch.setattr("agent_utilities.mcp.kg_server._get_engine", lambda: fake)
    monkeypatch.setattr("agent_utilities.mcp.kg_server._kg_auth_required", lambda: True)
    monkeypatch.setattr("agent_utilities.mcp.kg_server._PROCESS_ACTOR", None)
    monkeypatch.setattr(
        "agent_utilities.mcp.kg_server._actor_from_mcp_token", lambda: None
    )

    graph_bus = _make_graph_bus()
    res = await _bus_call(graph_bus, action="register", agent_id="anon-agent")
    assert res["ok"] is False
    assert "error" in res and "KG_AUTH_REQUIRED" in res["error"]
    # The denied register never landed the node.
    assert "busagent:anon-agent" not in fake.nodes
    # …but read-only presence stays reachable for an unauthenticated caller.
    roster = await _bus_call(graph_bus, action="roster")
    assert "roster" in roster


def _make_graph_bus():
    """Register the graph_bus MCP tool against a capturing fake server and return it."""
    captured: dict = {}

    class _MCP:
        def tool(self, **kw):
            def deco(fn):
                captured["fn"] = fn
                return fn

            return deco

    from agent_utilities.mcp.tools.bus_tools import register_bus_tools

    register_bus_tools(_MCP())
    return captured["fn"]


async def _bus_call(graph_bus, **kw):
    """Call the raw graph_bus fn with all params defaulted (Field defaults aren't applied)."""
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
        url="",
        group="",
        origin="",
        scope="commons",
        replay_recent=False,
        ctx=None,
    )
    base.update(kw)
    return json.loads(await graph_bus(**base))
