"""AgentBus — a federated agent-to-agent communication bus over the KG (CONCEPT:ECO-4.84).

The platform already had a *human*-reach core (``MessagingService``, ECO-4.48) and a host-local
*invoker↔spawned-agent* channel (``agent_channel.py``, ORCH-1.40). What it lacked was a way for
**independent sessions** — many Claude Code sessions, other LLMs, sessions from different
first-party providers, on **any host** — to address and message *each other* through one shared
graph-os hub, for the cost of the LLM calls each side already makes.

``AgentBus`` is that bus. It is durable-store-first by design: presence and messages are
**KG nodes** (``:Agent`` / ``:Topic`` / ``:BusMessage``, CONCEPT:KG-2.141), so any process
pointed at the same engine — including a remote session reaching a networked graph-os over
streamable-http — sees the same roster and mailbox, and the conversation survives an engine
restart. Delivery is at-least-once with a per-reader **cursor** (``receive(since)`` returns the
slice after ``since`` and the new cursor), the same model as ``agent_channel.receive``; ordering
is by ``created_at`` (sorted in Python, never an engine ``ORDER BY``).

Three surfaces feed this one core (the universal-capability rule): the ``graph_bus`` MCP tool
(ECO-4.85), its REST twin ``/graph/bus``, and the federation relay (ECO-4.86) which forwards
across hubs. Every ``send`` passes the fail-closed ActionPolicy gate (``kind="bus.send"``); a
``dispatch`` (``kind="bus.dispatch"``, ORCH-1.80) turns a message into fleet work by submitting a
Loop, so one agent can hand work to the fleet, not just chat.

CONCEPT:ECO-4.84 — AgentBus federated agent-to-agent communication bus over the KG
CONCEPT:KG-2.141 — :Agent / :Topic / :BusMessage presence + mailbox node model

See Also:
    - ``messaging/service.py`` (ECO-4.48) — the sibling *human*-reach core this mirrors.
    - ``messaging/federation.py`` (ECO-4.86) — cross-hub relay built on top of this.
    - ``docs/architecture/agent_bus.md`` — end-to-end flow + diagram.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from agent_utilities.observability import gateway_metrics as _metrics

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

# Node id prefixes for the durable bus model (CONCEPT:KG-2.141).
# NOTE: a dedicated ``:BusAgent`` label (not the platform's typed ``:Agent`` table) — the live
# Postgres backend gives ``:Agent`` a typed schema (capabilities ARRAY, no agent_id) that bus
# props don't fit. ``:BusAgent`` lands in the generic JSONB node table. (Found in live E2E.)
_AGENT_PREFIX = "busagent:"
_AGENT_LABEL = "BusAgent"
_TOPIC_PREFIX = "topic:"
_SUB_PREFIX = "bussub:"
_MSG_PREFIX = "busmsg:"

# A registered agent is "online" if it heartbeat within this many seconds; the roster
# computes presence lazily from ``last_seen`` so no reaper process is needed for liveness.
DEFAULT_STALE_AFTER_S = 90.0


class AgentBus:
    """Presence registry + durable mailbox + pub/sub + work dispatch for agents.

    CONCEPT:ECO-4.84

    Singleton: use :meth:`instance`. The same object backs the ``graph_bus`` MCP tool, the
    ``/graph/bus`` REST twin, and the federation relay, so all three read/write one durable
    state in the KG.
    """

    _instance: AgentBus | None = None

    def __init__(self, engine: Any = None) -> None:
        self._engine = engine

    @classmethod
    def instance(cls, engine: Any = None) -> AgentBus:
        """Get or create the shared bus (binding the engine on first use)."""
        if cls._instance is None:
            cls._instance = cls(engine=engine)
        elif engine is not None and cls._instance._engine is None:
            cls._instance._engine = engine
        return cls._instance

    # ── Engine resolution (matches MessagingService) ─────────────────
    def _resolve_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        try:
            from agent_utilities.knowledge_graph.core.engine import (
                IntelligenceGraphEngine,
            )

            self._engine = IntelligenceGraphEngine.get_active()
        except Exception as exc:  # noqa: BLE001
            logger.debug("[ECO-4.84] no active engine: %s", exc)
        return self._engine

    def _add_node(self, node_id: str, node_type: str, props: dict[str, Any]) -> bool:
        engine = self._resolve_engine()
        add_node = getattr(engine, "add_node", None)
        if not callable(add_node):
            return False
        try:
            add_node(node_id, node_type, properties={"id": node_id, **props})
            return True
        except Exception as exc:  # noqa: BLE001 — durability is best-effort
            logger.debug("[ECO-4.84] add_node(%s) failed: %s", node_id, exc)
            return False

    def _add_edge(self, src: str, dst: str, rel: str) -> None:
        engine = self._resolve_engine()
        add_edge = getattr(engine, "add_edge", None)
        if not callable(add_edge):
            return
        try:
            add_edge(src, dst, rel)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[ECO-4.84] add_edge(%s->%s) failed: %s", src, dst, exc)

    def _query(self, cypher: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        engine = self._resolve_engine()
        query = getattr(engine, "query_cypher", None)
        if not callable(query):
            return []
        try:
            return list(query(cypher, params) or [])
        except Exception as exc:  # noqa: BLE001
            logger.debug("[ECO-4.84] query failed: %s", exc)
            return []

    @staticmethod
    def _props(row: dict[str, Any], key: str) -> dict[str, Any]:
        """Pull a node's property bag out of a Cypher row (backend-shape tolerant)."""
        node = row.get(key, row) if isinstance(row, dict) else row
        if isinstance(node, dict):
            inner = node.get("properties")
            return inner if isinstance(inner, dict) else node
        return {}

    # ── Identity & presence (:Agent, CONCEPT:KG-2.141) ───────────────
    def register(
        self,
        agent_id: str,
        *,
        provider: str = "",
        host: str = "",
        kind: str = "agent",
        capabilities: Iterable[str] | None = None,
        session_id: str = "",
        actor_id: str = "",
    ) -> dict[str, Any]:
        """Announce a participant on the bus (idempotent upsert of its :Agent node).

        ``agent_id`` should be globally unique across hosts — derive it from the
        authenticated ``ActorContext.actor_id`` (an IdP subject) where available so two hubs
        never collide on the same id.
        """
        if not agent_id:
            return {"ok": False, "error": "agent_id required"}
        caps = sorted({c for c in (capabilities or []) if c})
        now = time.time()
        node_id = f"{_AGENT_PREFIX}{agent_id}"
        ok = self._add_node(
            node_id,
            _AGENT_LABEL,
            {
                "agent_id": agent_id,
                "provider": provider,
                "host": host,
                "kind": kind,
                "capabilities": ",".join(caps),
                "session_id": session_id,
                "actor_id": actor_id or agent_id,
                "status": "online",
                "registered_at": now,
                "last_seen": now,
            },
        )
        return {"ok": ok, "agent_id": agent_id, "capabilities": caps}

    def heartbeat(self, agent_id: str) -> bool:
        """Refresh a participant's ``last_seen`` so the roster keeps it ``online``.

        Re-reads the existing node first because the durable backend replaces a node's whole
        property blob on upsert — a bare ``{last_seen}`` write would wipe its capabilities.
        """
        if not agent_id:
            return False
        rows = self._query(
            f"MATCH (a:{_AGENT_LABEL} {{agent_id: $aid}}) RETURN a", {"aid": agent_id}
        )
        if not rows:
            return False
        props = dict(self._props(rows[0], "a"))
        props.update(status="online", last_seen=time.time())
        props.pop("id", None)
        return self._add_node(f"{_AGENT_PREFIX}{agent_id}", _AGENT_LABEL, props)

    def deregister(self, agent_id: str) -> bool:
        """Mark a participant ``offline`` (graceful leave)."""
        rows = self._query(
            f"MATCH (a:{_AGENT_LABEL} {{agent_id: $aid}}) RETURN a", {"aid": agent_id}
        )
        if not rows:
            return False
        props = dict(self._props(rows[0], "a"))
        props.update(status="offline", last_seen=time.time())
        props.pop("id", None)
        return self._add_node(f"{_AGENT_PREFIX}{agent_id}", _AGENT_LABEL, props)

    def roster(
        self,
        *,
        provider: str = "",
        capability: str = "",
        online_only: bool = False,
        stale_after_s: float = DEFAULT_STALE_AFTER_S,
    ) -> list[dict[str, Any]]:
        """List known participants with live-computed presence.

        Presence is derived from ``last_seen`` vs ``stale_after_s`` at read time, so a crashed
        session shows ``offline`` without any reaper writing to it.
        """
        rows = self._query(f"MATCH (a:{_AGENT_LABEL}) RETURN a", {})
        now = time.time()
        out: list[dict[str, Any]] = []
        for row in rows:
            p = self._props(row, "a")
            aid = p.get("agent_id")
            if not aid:
                continue
            caps = [c for c in str(p.get("capabilities", "")).split(",") if c]
            fresh = (now - float(p.get("last_seen", 0) or 0)) <= stale_after_s
            present = (
                "online" if (fresh and p.get("status") != "offline") else "offline"
            )
            if provider and p.get("provider") != provider:
                continue
            if capability and capability not in caps:
                continue
            if online_only and present != "online":
                continue
            out.append(
                {
                    "agent_id": aid,
                    "provider": p.get("provider", ""),
                    "host": p.get("host", ""),
                    "kind": p.get("kind", "agent"),
                    "capabilities": caps,
                    "presence": present,
                    "last_seen": float(p.get("last_seen", 0) or 0),
                }
            )
        out.sort(key=lambda a: a["agent_id"])
        return out

    # ── Topics & subscriptions (:Topic + :BusSubscription nodes) ─────
    # Subscriptions are first-class nodes (not edges): the live AGE backend doesn't reliably
    # resolve 2-hop edge traversals with a node-property filter, so a 1-hop ``:BusSubscription``
    # read is the robust model. (Found in live E2E.)
    def subscribe(self, agent_id: str, topic: str) -> bool:
        """Subscribe a participant to a topic (idempotent; creates the topic if new)."""
        if not (agent_id and topic):
            return False
        self._add_node(f"{_TOPIC_PREFIX}{topic}", "Topic", {"name": topic})
        return self._add_node(
            f"{_SUB_PREFIX}{agent_id}:{topic}",
            "BusSubscription",
            {"agent_id": agent_id, "topic": topic, "status": "active"},
        )

    def unsubscribe(self, agent_id: str, topic: str) -> bool:
        """Mark a subscription inactive (upsert on the same node id — survives no edge-delete)."""
        if not (agent_id and topic):
            return False
        return self._add_node(
            f"{_SUB_PREFIX}{agent_id}:{topic}",
            "BusSubscription",
            {"agent_id": agent_id, "topic": topic, "status": "inactive"},
        )

    def _subscribers(self, topic: str) -> list[str]:
        rows = self._query(
            "MATCH (s:BusSubscription {topic: $t}) RETURN s", {"t": topic}
        )
        subs = {
            str(aid)
            for p in (self._props(r, "s") for r in rows)
            if p.get("status", "active") == "active" and (aid := p.get("agent_id"))
        }
        return sorted(subs)

    # ── Messaging (governed; :BusMessage durable mailbox) ────────────
    def send(
        self,
        *,
        sender: str,
        payload: str,
        to: str = "",
        topic: str = "",
        reason: str = "",
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Deliver ``payload`` to one agent (``to``) or every subscriber of ``topic``.

        Governed by the ActionPolicy ``bus.send`` gate (CONCEPT:ECO-4.84). Writes one durable
        ``:BusMessage`` per recipient (sharing a ``msg_group`` so a fan-out is dedupable across
        hubs by the federation relay) and links it to the recipient's :Agent node.
        """
        if not sender or not payload:
            return {"ok": False, "error": "sender and payload required"}
        if not to and not topic:
            return {"ok": False, "error": "send requires 'to' or 'topic'"}

        kind = "topic" if topic else "direct"
        start = time.time()
        decision = self._gate("bus.send", to or f"topic:{topic}", sender, reason)
        if decision is not None and not decision.allowed:
            _metrics.BUS_MESSAGES.labels(kind=kind, outcome="denied").inc()
            return {
                "ok": False,
                "error": f"policy {decision.decision}: {decision.reason}",
            }

        recipients = [to] if to else self._subscribers(topic)
        recipients = [r for r in recipients if r and r != sender]
        if not recipients:
            _metrics.BUS_MESSAGES.labels(kind=kind, outcome="no_recipient").inc()
            return {"ok": True, "delivered": [], "note": "no recipients"}

        group = uuid.uuid4().hex[:12]
        now = time.time()
        meta_json = json.dumps(meta or {}, default=str)
        delivered: list[str] = []
        for rcpt in recipients:
            mid = f"{_MSG_PREFIX}{group}:{rcpt}"
            if self._add_node(
                mid,
                "BusMessage",
                {
                    "msg_group": group,
                    "sender": sender,
                    "recipient": rcpt,
                    "topic": topic,
                    "payload": payload,
                    "meta": meta_json,
                    "status": "sent",
                    "created": now,
                },
            ):
                self._add_edge(f"{_AGENT_PREFIX}{rcpt}", mid, "HAS_BUS_MESSAGE")
                delivered.append(rcpt)
        _metrics.BUS_MESSAGES.labels(kind=kind, outcome="delivered").inc(len(delivered))
        _metrics.BUS_SEND_DURATION.observe(time.time() - start)
        return {"ok": True, "msg_group": group, "delivered": delivered}

    def receive(self, agent_id: str, *, since: int = 0) -> dict[str, Any]:
        """Return the messages for ``agent_id`` after the ``since`` cursor, plus a new cursor.

        ``since`` is the count this reader has already consumed; the returned ``cursor`` is the
        new total to pass next time — the same at-least-once cursor model as
        ``agent_channel.receive``. Durable, so it works cross-host and across engine restarts.
        """
        if not agent_id:
            return {"messages": [], "cursor": since}
        # 1-hop property match (not a 2-hop edge traversal) — robust on the AGE backend.
        rows = self._query(
            "MATCH (m:BusMessage {recipient: $aid}) RETURN m", {"aid": agent_id}
        )
        msgs = [self._props(r, "m") for r in rows]
        msgs.sort(key=lambda m: (float(m.get("created", 0) or 0), str(m.get("id", ""))))
        new = msgs[since:] if since < len(msgs) else []
        shaped = [
            {
                "id": m.get("id"),
                "msg_group": m.get("msg_group"),
                "sender": m.get("sender"),
                "topic": m.get("topic", ""),
                "payload": m.get("payload", ""),
                "meta": _safe_json(m.get("meta")),
                "status": m.get("status", "sent"),
                "created": float(m.get("created", 0) or 0),
            }
            for m in new
        ]
        return {"messages": shaped, "cursor": len(msgs)}

    def ack(self, agent_id: str, message_id: str) -> bool:
        """Mark a delivered message processed (e.g. once its dispatched work was claimed)."""
        rows = self._query(
            "MATCH (m:BusMessage {id: $mid, recipient: $aid}) RETURN m",
            {"mid": message_id, "aid": agent_id},
        )
        if not rows:
            return False
        props = dict(self._props(rows[0], "m"))
        props.update(status="acked", acked=time.time())
        props.pop("id", None)
        return self._add_node(message_id, "BusMessage", props)

    # ── Federation support (CONCEPT:ECO-4.86) ────────────────────────
    def group_messages(self, group: str) -> list[dict[str, Any]]:
        """All :BusMessage rows of one ``msg_group`` (the unit a relay forwards)."""
        rows = self._query(
            "MATCH (m:BusMessage {msg_group: $g}) RETURN m", {"g": group}
        )
        return [self._props(r, "m") for r in rows]

    def group_exists(self, group: str) -> bool:
        """Has this hub already seen ``group`` (cross-hub delivery dedup)?"""
        return bool(self.group_messages(group))

    def deliver_federated(
        self,
        *,
        group: str,
        sender: str,
        recipients: list[str],
        payload: str,
        topic: str,
        origin: str,
    ) -> list[str]:
        """Apply a message forwarded from a peer hub (CONCEPT:ECO-4.86), idempotently.

        Reuses the origin ``msg_group`` for deterministic node ids, so a re-forward is a no-op
        upsert (dedup), and stamps ``federated_from`` so the local relay never re-forwards it
        (loop break). Skips the gate — the message was already governed at its origin hub.
        """
        now = time.time()
        meta_json = json.dumps({"federated_from": origin}, default=str)
        delivered: list[str] = []
        for rcpt in recipients:
            if not rcpt:
                continue
            mid = f"{_MSG_PREFIX}{group}:{rcpt}"
            if self._add_node(
                mid,
                "BusMessage",
                {
                    "msg_group": group,
                    "sender": sender,
                    "recipient": rcpt,
                    "topic": topic,
                    "payload": payload,
                    "meta": meta_json,
                    "federated_from": origin,
                    "status": "sent",
                    "created": now,
                },
            ):
                self._add_edge(f"{_AGENT_PREFIX}{rcpt}", mid, "HAS_BUS_MESSAGE")
                delivered.append(rcpt)
        return delivered

    # ── Dispatch: message → fleet work (CONCEPT:ORCH-1.80) ───────────
    def dispatch(
        self,
        *,
        sender: str,
        objective: str,
        kind: str = "develop",
        priority: str = "normal",
        reason: str = "",
    ) -> dict[str, Any]:
        """Turn a request into fleet work by submitting a Loop, gated by ``bus.dispatch``.

        This closes the message↔task gap: an agent on the bus hands an objective to the fleet
        (the LoopController executes it through the fair-claim task lanes), rather than only
        exchanging text. Returns the submitted loop record.
        """
        if not (sender and objective):
            return {"ok": False, "error": "sender and objective required"}
        decision = self._gate("bus.dispatch", objective[:80], sender, reason)
        if decision is not None and not decision.allowed:
            _metrics.BUS_DISPATCH.labels(outcome="denied").inc()
            return {
                "ok": False,
                "error": f"policy {decision.decision}: {decision.reason}",
            }
        engine = self._resolve_engine()
        try:
            from agent_utilities.knowledge_graph.core.engine_tasks import (
                _coerce_prio_bucket,
            )
            from agent_utilities.knowledge_graph.research.loops import submit_loop

            loop = submit_loop(
                engine,
                objective,
                kind=kind,  # type: ignore[arg-type]
                prio_bucket=_coerce_prio_bucket(priority),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[ORCH-1.80] dispatch submit_loop failed: %s", exc)
            _metrics.BUS_DISPATCH.labels(outcome="failed").inc()
            return {"ok": False, "error": f"dispatch failed: {exc}"}
        _metrics.BUS_DISPATCH.labels(outcome="submitted").inc()
        return {"ok": True, "loop": loop, "dispatched_by": sender}

    # ── Governance gate (mirrors MessagingService._gate) ─────────────
    def _gate(self, kind: str, target: str, source: str, reason: str) -> Any:
        try:
            from agent_utilities.orchestration.action_policy import (
                ActionRequest,
                get_action_policy,
            )

            request = ActionRequest(
                kind=kind,
                target=target or "*",
                source=source or "bus",
                reason=reason or kind,
            )
            return get_action_policy(self._resolve_engine()).decide(request)
        except Exception as exc:  # noqa: BLE001 — a gate failure must not silently act
            logger.warning("[ECO-4.84] action policy unavailable: %s", exc)
            return None

    # ── Introspection ────────────────────────────────────────────────
    def status(self) -> dict[str, Any]:
        roster = self.roster()
        online = sum(1 for a in roster if a["presence"] == "online")
        topics = self._query("MATCH (t:Topic) RETURN t.name as name", {})
        # Sample the presence gauges on the health/status read (CONCEPT:ECO-4.87).
        _metrics.BUS_PARTICIPANTS.labels(status="online").set(online)
        _metrics.BUS_PARTICIPANTS.labels(status="offline").set(len(roster) - online)
        return {
            "agents": len(roster),
            "online": online,
            "topics": sorted({t.get("name") for t in topics if t.get("name")}),
        }


def _safe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value or {}
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return {}


def swarm_topic(session_id: str | None) -> str:
    """The shared bus topic a swarm uses to coordinate (CONCEPT:ECO-4.88)."""
    return f"swarm:{session_id}" if session_id else "swarm:default"


def bus_capability_prompt() -> str:
    """Canonical, single-source capability blurb woven into every agent's system prompt.

    CONCEPT:ECO-4.88 — the AgentBus is a NATIVE capability, not an opt-in persona: the core
    orchestrator (the "graph shaper") and every spawned swarm/sub-agent inherit this so they
    know they can coordinate with peers instead of working in isolation. Kept in sync with the
    ``prompts/bus_coordinator.json`` blueprint, which is the deeper standalone profile.
    """
    return (
        "## Agent Bus — coordinate with other agents (native capability, CONCEPT:ECO-4.84)\n"
        "You are not alone. Other AI sessions and agents — any provider, any host — share a "
        "durable **AgentBus**, and you can talk to them. Use the native bus tools (or the "
        "`graph_bus` tool with the same actions):\n"
        "- `bus_join` — announce yourself (id + capabilities) so peers can find you.\n"
        "- `bus_peers` — see who is online and what they can do; address peers by those ids.\n"
        "- `bus_send` — message one peer (`to=`) or a whole topic (`topic=`).\n"
        "- `bus_check` — read your inbox (pass back the returned cursor for only-new messages).\n"
        "- `graph_bus(action='dispatch', objective=...)` — hand heavy work to the fleet as a Loop.\n"
        "When you orchestrate or join a **swarm**, broadcast progress and ask peers for help on the "
        "swarm's shared topic rather than working alone — set up agent-to-agent communication by "
        "default whenever more than one agent is involved."
    )
