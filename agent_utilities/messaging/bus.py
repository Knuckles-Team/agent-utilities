"""AgentBus — a federated agent-to-agent communication bus over the KG (CONCEPT:AU-ECO.bus.agentbus-federated-agent-agent).

The platform already had a *human*-reach core (``MessagingService``, ECO-4.48) and a host-local
*invoker↔spawned-agent* channel (``agent_channel.py``, ORCH-1.40). What it lacked was a way for
**independent sessions** — many Claude Code sessions, other LLMs, sessions from different
first-party providers, on **any host** — to address and message *each other* through one shared
graph-os hub, for the cost of the LLM calls each side already makes.

``AgentBus`` is that bus. It is durable-store-first by design: presence,
subscriptions, transactional inboxes, outboxes, and WorkItems are KG records, so any process
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

**Delivery/wakeup plane (AU-P1-2, CONCEPT:AU-ECO.bus.partitioned-log-delivery).** The registry above —
presence, topic membership, subscriptions — stays exactly as described: small, low-churn KG
nodes. What does NOT stay on the graph is the high-volume message BODIES: ``send``/``receive``
resolve a durable **partitioned log** (:mod:`messaging.bus_log`) — the engine's native
AMQP-style broker, or Kafka, in that preference order — as the hot delivery path, with real
offsets/consumer cursors instead of a graph ``MATCH``, a DLQ for poison messages, and
backpressure via queue depth. A missing log or native inbox transaction fails closed.

CONCEPT:AU-ECO.bus.agentbus-federated-agent-agent — AgentBus federated agent-to-agent communication bus over the KG
CONCEPT:AU-KG.compute.user-override-prompt-library — semantic presence/subscription registry
CONCEPT:AU-ECO.bus.store-and-forward-log — durable topic log materialized to tenant inboxes
CONCEPT:AU-ECO.bus.auto-register-online-presence — auto-register + online presence on any bus touch (no explicit register)
CONCEPT:AU-ECO.bus.bus-register-under-served — bus register under the served auth profile: run as the request's authenticated identity + surface a denied write (never a silent ok:false)
CONCEPT:AU-ECO.bus.partitioned-log-delivery — durable partitioned log (engine broker / Kafka) as the delivery/wakeup plane; the KG keeps only the semantic registry

See Also:
    - ``messaging/service.py`` (ECO-4.48) — the sibling *human*-reach core this mirrors.
    - ``messaging/federation.py`` (ECO-4.86) — cross-hub relay built on top of this.
    - ``messaging/bus_log.py`` (AU-P1-2) — the partitioned-log delivery/wakeup plane.
    - ``docs/architecture/agent_bus.md`` — end-to-end flow + diagram.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from agent_utilities.messaging.bus_privacy import (
    bus_reference,
    sanitize_bus_content,
)
from agent_utilities.observability import gateway_metrics as _metrics

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

#: Cap on messages one ``receive()`` call drains from the log backend
#: (CONCEPT:AU-ECO.bus.partitioned-log-delivery) — the backpressure bound on the hot delivery path.
BUS_LOG_MAX_MESSAGES_PER_RECEIVE = 500

#: Sentinel distinguishing an unresolved backend cache.
_UNRESOLVED = object()

# Node id prefixes for the durable bus model (CONCEPT:AU-KG.compute.user-override-prompt-library).
# NOTE: a dedicated ``:BusAgent`` label (not the platform's typed ``:Agent`` table) — the live
# Postgres backend gives ``:Agent`` a typed schema (capabilities ARRAY, no agent_id) that bus
# props don't fit. ``:BusAgent`` lands in the generic JSONB node table. (Found in live E2E.)
_AGENT_PREFIX = "busagent:"
_AGENT_LABEL = "BusAgent"
_TOPIC_PREFIX = "topic:"
_SUB_PREFIX = "bussub:"

# A registered agent is "online" if it heartbeat within this many seconds; the roster
# computes presence lazily from ``last_seen`` so no reaper process is needed for liveness.
DEFAULT_STALE_AFTER_S = 90.0

class AgentBus:
    """Presence registry + durable mailbox + pub/sub + work dispatch for agents.

    CONCEPT:AU-ECO.bus.agentbus-federated-agent-agent

    Singleton: use :meth:`instance`. The same object backs the ``graph_bus`` MCP tool, the
    ``/graph/bus`` REST twin, and the federation relay, so all three read/write one durable
    state in the KG.
    """

    _instance: AgentBus | None = None

    def __init__(self, engine: Any = None) -> None:
        self._engine = engine
        # The reason the most recent :meth:`_add_node` write failed (CONCEPT:AU-ECO.bus.bus-register-under-served).
        # ``_add_node`` is best-effort and returns a bool, but a write that does NOT
        # land must never be swallowed as a benign ``ok:false`` — the caller (e.g.
        # :meth:`register`) reads this to tell a real engine/ACL denial apart from a
        # missing-engine no-op and surface WHY, instead of a silent false.
        self._last_write_error: str = ""
        # Required delivery/wakeup backend, resolved lazily and cached.
        # Resolved once per process (an explicit misconfiguration — e.g.
        # ``AGENT_BUS_LOG_BACKEND=kafka`` unreachable — raises here, a hard
        # contract like the rest of this codebase's selectable backends).
        self._log_backend_cache: Any = _UNRESOLVED

    def _log_backend(self) -> Any:
        """Resolve and return the required durable bus-log backend."""
        if self._log_backend_cache is _UNRESOLVED:
            from agent_utilities.messaging.bus_log import resolve_bus_log_backend

            self._log_backend_cache = resolve_bus_log_backend(
                engine=self._resolve_engine()
            )
            logger.info(
                "[AU-P1-2] AgentBus delivery/wakeup plane: %s (partitioned log)",
                self._log_backend_cache.name,
            )
        return self._log_backend_cache

    @staticmethod
    def _depth_from_stats(value: Any) -> int:
        if isinstance(value, dict):
            queues = value.get("queues")
            if isinstance(queues, dict):
                return sum(AgentBus._sum_numeric(item) for item in queues.values())
            values = [
                AgentBus._depth_from_stats(item)
                for key, item in value.items()
                if key.lower() in {"depth", "queue_depth", "ready", "messages", "lag"}
                or isinstance(item, dict | list | tuple)
            ]
            return max(values, default=0)
        if isinstance(value, list | tuple):
            return max((AgentBus._depth_from_stats(item) for item in value), default=0)
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _sum_numeric(value: Any) -> int:
        if isinstance(value, dict):
            return sum(AgentBus._sum_numeric(item) for item in value.values())
        if isinstance(value, list | tuple):
            return sum(AgentBus._sum_numeric(item) for item in value)
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0

    def _log_has_capacity(self, backend: Any) -> bool:
        from agent_utilities.core.config import config

        limit = max(1, int(getattr(config, "agent_bus_max_depth", 100_000) or 100_000))
        query = getattr(self._resolve_engine(), "query_cypher", None)
        if not callable(query):
            return False
        from agent_utilities.messaging.bus_log import current_bus_tenant

        tenant_ref = bus_reference("tenant", current_bus_tenant())
        try:
            rows = list(
                query(
                    "MATCH (o:BusOutbox {status: 'published', tenant: $tenant}) "
                    "RETURN count(o) AS n",
                    {"tenant": tenant_ref},
                )
                or []
            )
        except Exception as exc:  # noqa: BLE001 - backpressure fails closed
            logger.warning(
                "AgentBus could not verify durable log depth (%s)",
                type(exc).__name__,
            )
            return False
        durable_depth = int(rows[0].get("n", 0)) if rows else 0
        return max(self._depth_from_stats(backend.stats()), durable_depth) < limit

    @classmethod
    def reset_log_backend_cache_for_tests(cls) -> None:
        """Force re-resolution of the log backend on the next call (test isolation seam)."""
        if cls._instance is not None:
            cls._instance._log_backend_cache = _UNRESOLVED

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
            logger.debug("[ECO-4.84] no active engine (%s)", type(exc).__name__)
        return self._engine

    def _add_node(self, node_id: str, node_type: str, props: dict[str, Any]) -> bool:
        engine = self._resolve_engine()
        add_node = getattr(engine, "add_node", None)
        if not callable(add_node):
            self._last_write_error = "no active engine (bus has no durable store)"
            return False
        try:
            add_node(node_id, node_type, properties={"id": node_id, **props})
            self._last_write_error = ""
            return True
        except Exception as exc:  # noqa: BLE001 — durability is best-effort
            # Record WHY so the caller can surface it (CONCEPT:AU-ECO.bus.bus-register-under-served). A write that
            # does not land — e.g. an engine/ACL denial under the served profile
            # Mandatory authorization failures must not be swallowed as benign false.
            self._last_write_error = (
                f"{type(exc).__name__}: durable bus write rejected"
            )
            logger.warning(
                "[ECO-4.84] add_node(%s) failed (%s)",
                node_id,
                type(exc).__name__,
            )
            return False

    def _add_edge(self, src: str, dst: str, rel: str) -> None:
        engine = self._resolve_engine()
        add_edge = getattr(engine, "add_edge", None)
        if not callable(add_edge):
            return
        try:
            add_edge(src, dst, rel)
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "[ECO-4.84] add_edge(%s->%s) failed (%s)",
                src,
                dst,
                type(exc).__name__,
            )

    def _query(self, cypher: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        engine = self._resolve_engine()
        query = getattr(engine, "query_cypher", None)
        if not callable(query):
            return []
        try:
            return list(query(cypher, params) or [])
        except Exception as exc:  # noqa: BLE001
            logger.debug("[ECO-4.84] query failed (%s)", type(exc).__name__)
            return []

    @staticmethod
    def _props(row: dict[str, Any], key: str) -> dict[str, Any]:
        """Pull a node's property bag out of a Cypher row (backend-shape tolerant)."""
        node = row.get(key, row) if isinstance(row, dict) else row
        if isinstance(node, dict):
            inner = node.get("properties")
            return inner if isinstance(inner, dict) else node
        return {}

    # ── Identity & presence (:Agent, CONCEPT:AU-KG.compute.user-override-prompt-library) ───────────────
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
        from agent_utilities.messaging.bus_log import current_bus_tenant

        tenant = current_bus_tenant()
        agent_id = bus_reference("agent", agent_id, tenant=tenant)
        host_ref = bus_reference("host", host, tenant=tenant)
        session_ref = bus_reference("session", session_id, tenant=tenant)
        actor_ref = bus_reference("actor", actor_id or agent_id, tenant=tenant)
        _unused, profile_json, _profile_report = sanitize_bus_content(
            "", {"provider": provider, "kind": kind, "capabilities": list(capabilities or [])}
        )
        # ``sanitize_bus_content`` intentionally returns only serialized clean
        # metadata. Decode it locally; raw profile inputs are never persisted.
        try:
            profile = json.loads(profile_json)
        except (TypeError, ValueError):
            profile = {}
        caps = sorted({str(c) for c in profile.get("capabilities", []) if c})
        now = time.time()
        node_id = f"{_AGENT_PREFIX}{agent_id}"
        ok = self._add_node(
            node_id,
            _AGENT_LABEL,
            {
                "agent_id": agent_id,
                "provider": str(profile.get("provider") or ""),
                "host_ref": host_ref,
                "kind": str(profile.get("kind") or "agent"),
                "capabilities": ",".join(caps),
                "session_ref": session_ref,
                "actor_ref": actor_ref,
                "status": "online",
                "registered_at": now,
                "last_seen": now,
            },
        )
        result: dict[str, Any] = {
            "ok": ok,
            "agent_id": agent_id,
            "capabilities": caps,
        }
        # A failed register must say WHY (CONCEPT:AU-ECO.bus.bus-register-under-served) — never a silent ok:false.
        # The most common served-profile cause is an unattributed write under
        # The bus is fleet-coordination infrastructure, so a
        # legitimate *authenticated* session must be able to register (its identity
        # is propagated from the MCP/REST surface), while an unauthenticated caller is
        # cleanly rejected with this error rather than a benign-looking false.
        if not ok:
            result["error"] = self._last_write_error or (
                f"register write for {node_id!r} did not land (the :BusAgent node "
                "was not persisted)"
            )
        return result

    def heartbeat(self, agent_id: str) -> bool:
        """Refresh a participant's ``last_seen`` so the roster keeps it ``online``.

        Re-reads the existing node first because the durable backend replaces a node's whole
        property blob on upsert — a bare ``{last_seen}`` write would wipe its capabilities.
        """
        if not agent_id:
            return False
        from agent_utilities.messaging.bus_log import current_bus_tenant

        agent_id = bus_reference("agent", agent_id, tenant=current_bus_tenant())
        rows = self._query(
            f"MATCH (a:{_AGENT_LABEL} {{agent_id: $aid}}) RETURN a", {"aid": agent_id}
        )
        if not rows:
            return False
        props = dict(self._props(rows[0], "a"))
        props.update(status="online", last_seen=time.time())
        props.pop("id", None)
        return self._add_node(f"{_AGENT_PREFIX}{agent_id}", _AGENT_LABEL, props)

    def touch(self, agent_id: str) -> bool:
        """Keep a participant online by merely *using* the bus (CONCEPT:AU-ECO.bus.auto-register-online-presence).

        Auto-registers ``agent_id`` on first reference (so a session that has the ``graph_bus``
        tool appears in the roster without an explicit ``register`` call) and refreshes
        ``last_seen`` on every subsequent action, so any bus touch counts as presence. Returns
        whether a node now exists. Idempotent and best-effort; never raises into the action.
        """
        if not agent_id:
            return False
        from agent_utilities.messaging.bus_log import current_bus_tenant

        agent_id = bus_reference("agent", agent_id, tenant=current_bus_tenant())
        rows = self._query(
            f"MATCH (a:{_AGENT_LABEL} {{agent_id: $aid}}) RETURN a", {"aid": agent_id}
        )
        if rows:
            # Existing node: preserve its blob (capabilities/provider) and only bump presence —
            # a bare {last_seen} write would clobber the rest of the blob on upsert.
            props = dict(self._props(rows[0], "a"))
            props.update(status="online", last_seen=time.time())
            props.pop("id", None)
            return self._add_node(f"{_AGENT_PREFIX}{agent_id}", _AGENT_LABEL, props)
        # No node yet → auto-register a minimal :BusAgent so the agent is immediately rosterable.
        return self.register(agent_id, kind="agent").get("ok", False)

    def deregister(self, agent_id: str) -> bool:
        """Mark a participant ``offline`` (graceful leave)."""
        from agent_utilities.messaging.bus_log import current_bus_tenant

        agent_id = bus_reference("agent", agent_id, tenant=current_bus_tenant())
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
                    "host_ref": p.get("host_ref", ""),
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
        """Subscribe a participant to a topic (idempotent; creates the topic if new).

        The semantic registry write (:Topic + :BusSubscription) always lands in the graph —
        subscriptions are low-churn metadata, not the high-volume delivery path
        (CONCEPT:AU-ECO.bus.partitioned-log-delivery). Fixed materializer partitions consume
        topic events and consult this registry at commit time; subscriptions never allocate a
        queue or consumer.
        """
        if not (agent_id and topic):
            return False
        from agent_utilities.messaging.bus_log import current_bus_tenant

        tenant = current_bus_tenant()
        agent_id = bus_reference("agent", agent_id, tenant=tenant)
        topic = bus_reference("topic", topic, tenant=tenant)
        from agent_utilities.core.config import config

        current = self._subscribers_at(topic, float("inf"))
        subscriber_limit = int(config.agent_bus_max_topic_subscribers)
        if agent_id not in current and len(current) >= subscriber_limit:
            logger.warning("AgentBus topic subscriber bound reached")
            return False
        self._add_node(f"{_TOPIC_PREFIX}{topic}", "Topic", {"name": topic})
        return self._add_node(
            f"{_SUB_PREFIX}{agent_id}:{topic}",
            "BusSubscription",
            {
                "agent_id": agent_id,
                "topic": topic,
                "status": "active",
                "subscribed_at": time.time(),
            },
        )

    def _bind_log_subscriber(
        self, agent_id: str, topic: str, *, from_ts: float | None
    ) -> None:
        """Bind this subscriber's queue/consumer on the log backend, if one is configured.

        Best-effort: a bind failure never blocks ``subscribe`` (``receive`` re-attempts the bind
        lazily too — see ``EngineBrokerBusLog.receive`` / ``KafkaBusLog.receive``).
        """
        backend = self._log_backend()
        if backend is None:
            return
        from agent_utilities.messaging.bus_log import current_bus_tenant

        try:
            backend.bind_subscriber(
                tenant=current_bus_tenant(),
                agent_id=agent_id,
                topic=topic,
                from_ts=from_ts,
            )
        except Exception as exc:  # noqa: BLE001 — bind is best-effort, never blocks subscribe
            logger.warning(
                "[AU-P1-2] log backend bind_subscriber(%s, %s) failed: %s",
                agent_id,
                topic,
                exc,
            )

    def unsubscribe(self, agent_id: str, topic: str) -> bool:
        """Mark a subscription inactive (upsert on the same node id — survives no edge-delete)."""
        if not (agent_id and topic):
            return False
        from agent_utilities.messaging.bus_log import current_bus_tenant

        tenant = current_bus_tenant()
        agent_id = bus_reference("agent", agent_id, tenant=tenant)
        topic = bus_reference("topic", topic, tenant=tenant)
        return self._add_node(
            f"{_SUB_PREFIX}{agent_id}:{topic}",
            "BusSubscription",
            {"agent_id": agent_id, "topic": topic, "status": "inactive"},
        )

    def _subscribers(self, topic: str) -> list[str]:
        rows = self._query(
            "MATCH (s:BusSubscription {topic: $t}) RETURN s", {"t": topic}
        )
        subs: set[str] = {
            str(p.get("agent_id"))
            for p in (self._props(r, "s") for r in rows)
            if p.get("status", "active") == "active" and p.get("agent_id")
        }
        return sorted(subs)

    def _subscribers_at(self, topic: str, created: float) -> list[str]:
        """Return active subscribers eligible when a topic event was created."""
        query = getattr(self._resolve_engine(), "query_cypher", None)
        if not callable(query):
            raise RuntimeError("AgentBus subscription registry is unavailable")
        rows = list(
            query(
                "MATCH (s:BusSubscription {topic: $t}) RETURN s", {"t": topic}
            )
            or []
        )
        return sorted(
            {
                str(props.get("agent_id"))
                for props in (self._props(row, "s") for row in rows)
                if props.get("status", "active") == "active"
                and props.get("agent_id")
                and props.get("subscribed_at") is not None
                and float(props["subscribed_at"]) <= created
            }
        )

    # ── Messaging ─────────────────────────────────────────────────────
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

        Governed by the ActionPolicy ``bus.send`` gate (CONCEPT:AU-ECO.bus.agentbus-federated-agent-agent). The
        message BODY rides the required partitioned-log delivery plane after a
        transactional send-outbox record is durable.
        """
        if not sender or not payload:
            return {"ok": False, "error": "sender and payload required"}
        if not to and not topic:
            return {"ok": False, "error": "send requires 'to' or 'topic'"}

        from agent_utilities.messaging.bus_log import current_bus_tenant

        tenant = current_bus_tenant()
        sender = bus_reference("agent", sender, tenant=tenant)
        to = bus_reference("agent", to, tenant=tenant)
        topic = bus_reference("topic", topic, tenant=tenant)
        payload, meta_json, _privacy_report = sanitize_bus_content(payload, meta)
        reason, _reason_meta, _reason_report = sanitize_bus_content(reason, {})

        kind = "topic" if topic else "direct"
        start = time.time()
        decision = self._gate("bus.send", to or f"topic:{topic}", sender, reason)
        if decision is not None and not decision.allowed:
            _metrics.BUS_MESSAGES.labels(kind=kind, outcome="denied").inc()
            return {
                "ok": False,
                "error": f"policy {decision.decision}: {decision.reason}",
            }

        group = bus_reference("message_group", uuid.uuid4().hex, tenant=tenant)
        now = time.time()
        backend = self._log_backend()
        if not self._log_has_capacity(backend):
            _metrics.BUS_MESSAGES.labels(kind=kind, outcome="backpressure").inc()
            return {"ok": False, "error": "AgentBus is at its configured depth bound"}
        wire_message = {
            "id": f"busmsg:{group}",
            "msg_group": group,
            "sender": sender,
            "recipient": to,
            "topic": topic,
            "payload": payload,
            "meta": meta_json,
            "created": now,
        }
        from agent_utilities.messaging.bus_inbox import (
            commit_message_outbox,
            mark_message_outbox_published,
        )

        outbox = commit_message_outbox(
            self._resolve_engine(), wire_message, tenant=tenant, now=now
        )
        out = self._send_via_log(
            backend,
            kind=kind,
            group=group,
            sender=sender,
            to=to,
            topic=topic,
            payload=payload,
            meta_json=meta_json,
            now=now,
        )
        out["outbox_id"] = outbox.outbox_id
        if out.get("ok"):
            out["published"] = True
            try:
                mark_message_outbox_published(
                    self._resolve_engine(), wire_message, tenant=tenant
                )
            except Exception as exc:  # noqa: BLE001 - pending outbox is replayable
                logger.warning(
                    "AgentBus published but outbox confirmation is pending (%s)",
                    type(exc).__name__,
                )
        else:
            out["published"] = False
            out["durable"] = True
            out["queued_for_replay"] = True
        _metrics.BUS_SEND_DURATION.observe(time.time() - start)
        return out

    def _send_via_log(
        self,
        backend: Any,
        *,
        kind: str,
        group: str,
        sender: str,
        to: str,
        topic: str,
        payload: str,
        meta_json: str,
        now: float,
    ) -> dict[str, Any]:
        """Hot delivery path (CONCEPT:AU-ECO.bus.partitioned-log-delivery): ONE ``publish`` call, no per-recipient write.

        The fixed engine partitions or keyed Kafka topic carry one event. The
        materializer resolves the authoritative subscription registry at commit time;
        ``_subscribers(topic)`` here is only a reporting read.
        """
        from agent_utilities.messaging.bus_log import current_bus_tenant

        tenant = current_bus_tenant()
        if to:
            ok = backend.publish_direct(
                tenant=tenant,
                group=group,
                sender=sender,
                to=to,
                payload=payload,
                meta_json=meta_json,
                created=now,
            )
            delivered = [to] if ok else []
            _metrics.BUS_MESSAGES.labels(
                kind=kind, outcome="delivered" if ok else "failed"
            ).inc(max(len(delivered), 1))
            return {"ok": ok, "msg_group": group, "delivered": delivered}

        ok = backend.publish_topic(
            tenant=tenant,
            group=group,
            sender=sender,
            topic=topic,
            payload=payload,
            meta_json=meta_json,
            created=now,
        )
        delivered = (
            [a for a in self._subscribers(topic) if a and a != sender] if ok else []
        )
        _metrics.BUS_MESSAGES.labels(
            kind=kind, outcome="delivered" if ok else "failed"
        ).inc(max(len(delivered), 1))
        return {"ok": ok, "msg_group": group, "delivered": delivered, "stored": ok}

    def receive(self, agent_id: str, *, since: int = 0) -> dict[str, Any]:
        """Materialize bounded log partitions, then read this agent's durable inbox."""
        if not agent_id:
            return {"messages": [], "cursor": since}
        from agent_utilities.messaging.bus_log import current_bus_tenant

        tenant = current_bus_tenant()
        agent_id = bus_reference("agent", agent_id, tenant=tenant)
        backend = self._log_backend()
        self._replay_pending_outbox(backend, tenant=tenant)
        pending = backend.receive(
            tenant=bus_reference("tenant", tenant),
            agent_id=agent_id,
            topics=[],
            max_messages=BUS_LOG_MAX_MESSAGES_PER_RECEIVE,
        )
        self._materialize_deliveries(pending, tenant=tenant, backend=backend)
        return self._read_committed_inbox(agent_id, since=since)

    def _replay_pending_outbox(
        self, backend: Any, *, tenant: str, limit: int = 100
    ) -> int:
        """Republish bounded pending send intents; duplicate publish is safe."""
        tenant_ref = bus_reference("tenant", tenant)
        rows = self._query(
            "MATCH (o:BusOutbox {status: 'pending', tenant: $tenant}) RETURN o "
            f"LIMIT {max(1, min(int(limit), 1000))}",
            {"tenant": tenant_ref},
        )
        if not rows:
            return 0
        from agent_utilities.messaging.bus_inbox import (
            mark_message_outbox_published,
        )

        replayed = 0
        engine = self._resolve_engine()
        for row in rows:
            props = self._props(row, "o")
            message = {
                "id": f"busmsg:{props.get('group_ref', '')}",
                "msg_group": str(props.get("group_ref") or ""),
                "sender": str(props.get("sender_ref") or ""),
                "recipient": str(props.get("recipient_ref") or ""),
                "topic": str(props.get("topic_ref") or ""),
                "payload": str(props.get("payload") or ""),
                "meta": str(props.get("metadata") or "{}"),
                "created": float(props.get("created_at") or time.time()),
            }
            if not message["msg_group"]:
                continue
            if message["topic"]:
                published = backend.publish_topic(
                    tenant=tenant,
                    group=message["msg_group"],
                    sender=message["sender"],
                    topic=message["topic"],
                    payload=message["payload"],
                    meta_json=message["meta"],
                    created=message["created"],
                )
            elif message["recipient"]:
                published = backend.publish_direct(
                    tenant=tenant,
                    group=message["msg_group"],
                    sender=message["sender"],
                    to=message["recipient"],
                    payload=message["payload"],
                    meta_json=message["meta"],
                    created=message["created"],
                )
            else:
                continue
            if not published:
                continue
            try:
                mark_message_outbox_published(engine, message, tenant=tenant)
                replayed += 1
            except Exception as exc:  # noqa: BLE001 - retry remains pending
                logger.warning(
                    "AgentBus replay publish confirmation remains pending (%s)",
                    type(exc).__name__,
                )
        return replayed

    def _materialize_deliveries(
        self,
        messages: list[dict[str, Any]],
        *,
        tenant: str,
        backend: Any,
    ) -> int:
        """Commit every delivery target before acknowledging its log receipt.

        Processing is deliberately sequential.  It preserves broker order and
        prevents a later Kafka offset from being committed ahead of an earlier
        failed delivery.  A failed commit is nacked/requeued and omitted from
        the caller-visible result; no success is fabricated.
        """

        from agent_utilities.messaging.bus_inbox import (
            commit_message_to_work_item,
            mark_message_outbox_delivered,
        )

        engine = self._resolve_engine()
        committed = 0
        tenant_ref = bus_reference("tenant", tenant)
        for message in messages:
            if message.get("tenant") != tenant_ref:
                backend.nack(message, requeue=False)
                continue
            topic = str(message.get("topic") or "")
            sender = str(message.get("sender") or "")
            try:
                recipients = (
                    [
                        recipient
                        for recipient in self._subscribers_at(
                            topic, float(message.get("created") or 0)
                        )
                        if recipient != sender
                    ]
                    if topic
                    else [str(message.get("recipient") or "")]
                )
            except Exception as exc:  # noqa: BLE001 - registry authority failed
                logger.warning(
                    "AgentBus subscription resolution failed; delivery retained (%s)",
                    type(exc).__name__,
                )
                backend.nack(message, requeue=True)
                break
            recipients = [recipient for recipient in recipients if recipient]
            audit_sink = not recipients
            if not recipients:
                if not topic:
                    backend.nack(message, requeue=False)
                    continue
                recipients = [bus_reference("topic_sink", topic, tenant=tenant)]
            from agent_utilities.core.config import config

            if len(recipients) > int(config.agent_bus_max_topic_subscribers):
                logger.warning(
                    "AgentBus topic delivery exceeds the configured subscriber bound"
                )
                backend.nack(message, requeue=True)
                break
            try:
                for recipient in recipients:
                    commit_message_to_work_item(
                        engine,
                        {**message, "_audit_sink": audit_sink},
                        tenant=tenant,
                        recipient=recipient,
                    )
            except Exception as exc:  # noqa: BLE001 - leave delivery unacked
                logger.warning(
                    "AgentBus inbox transaction failed; delivery will be retried (%s)",
                    type(exc).__name__,
                )
                try:
                    backend.nack(message, requeue=True)
                except Exception:
                    logger.exception("AgentBus could not nack failed delivery")
                break

            try:
                acknowledged = bool(backend.ack(message))
            except Exception as exc:  # noqa: BLE001 - durable replay is safe
                acknowledged = False
                logger.warning(
                    "AgentBus broker ack failed after durable inbox commit (%s)",
                    type(exc).__name__,
                )
            if not acknowledged:
                logger.warning(
                    "AgentBus receipt remains pending after durable commit; replay is idempotent"
                )
            else:
                try:
                    mark_message_outbox_delivered(
                        engine, message, tenant=tenant
                    )
                except Exception as exc:  # noqa: BLE001 - safe false-positive depth
                    logger.warning(
                        "AgentBus delivery committed but outbox completion is pending (%s)",
                        type(exc).__name__,
                    )
            committed += len(recipients)
        return committed

    def _read_committed_inbox(
        self, agent_id: str, *, since: int
    ) -> dict[str, Any]:
        """Read committed inbox rows; the cursor is a durable-row offset."""
        query = getattr(self._resolve_engine(), "query_cypher", None)
        if not callable(query):
            raise RuntimeError("AgentBus inbox authority is unavailable")
        rows = list(
            query(
                "MATCH (i:BusInbox {recipient_ref: $aid}) RETURN i",
                {"aid": agent_id},
            )
            or []
        )
        inbox = [self._props(row, "i") for row in rows]
        inbox.sort(
            key=lambda item: (
                float(item.get("committed_at", 0) or 0),
                str(item.get("id") or ""),
            )
        )
        selected = inbox[max(0, int(since)) :]
        messages = [
            {
                "id": item.get("id"),
                "msg_group": item.get("message_ref"),
                "sender": item.get("sender_ref"),
                "topic": item.get("topic_ref", ""),
                "payload": item.get("payload", ""),
                "meta": _safe_json(item.get("metadata")),
                "status": item.get("status", "committed"),
                "created": float(item.get("created_at", 0) or 0),
                "work_item_id": item.get("work_item_ref"),
            }
            for item in selected
        ]
        return {"messages": messages, "cursor": len(inbox)}

    # ── Federation support (CONCEPT:AU-ECO.bus.federation-relay) ────────────────────────
    def group_messages(self, group: str) -> list[dict[str, Any]]:
        """Read the durable send outbox entry for one message group."""
        from agent_utilities.messaging.bus_log import current_bus_tenant

        group = bus_reference(
            "message_group", group, tenant=current_bus_tenant()
        )
        rows = self._query(
            "MATCH (o:BusOutbox {group_ref: $g}) RETURN o", {"g": group}
        )
        return [
            {
                "msg_group": props.get("group_ref"),
                "sender": props.get("sender_ref"),
                "recipient": props.get("recipient_ref"),
                "topic": props.get("topic_ref"),
                "payload": props.get("payload"),
                "meta": props.get("metadata"),
                "federated_from": _metadata_flag(
                    props.get("metadata"), "federated_from_ref"
                ),
                "status": props.get("status"),
            }
            for props in (self._props(row, "o") for row in rows)
        ]

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
        """Publish a peer-hub message through the same durable outbox/log path."""
        from agent_utilities.messaging.bus_log import current_bus_tenant

        tenant = current_bus_tenant()
        group = bus_reference("message_group", group, tenant=tenant)
        sender = bus_reference("agent", sender, tenant=tenant)
        recipients = [
            bus_reference("agent", recipient, tenant=tenant) for recipient in recipients
        ]
        topic = bus_reference("topic", topic, tenant=tenant)
        origin = bus_reference("origin", origin, tenant=tenant)
        payload, meta_json, _privacy_report = sanitize_bus_content(
            payload, {"federated_from_ref": origin}
        )
        now = time.time()
        wire_message = {
            "id": f"busmsg:{group}",
            "msg_group": group,
            "sender": sender,
            "recipient": "",
            "topic": topic,
            "payload": payload,
            "meta": meta_json,
            "created": now,
        }
        from agent_utilities.messaging.bus_inbox import (
            commit_message_outbox,
            mark_message_outbox_published,
        )

        backend = self._log_backend()
        delivered: list[str] = []
        if topic:
            commit_message_outbox(
                self._resolve_engine(), wire_message, tenant=tenant, now=now
            )
            if backend.publish_topic(
                tenant=tenant,
                group=group,
                sender=sender,
                topic=topic,
                payload=payload,
                meta_json=meta_json,
                created=now,
            ):
                delivered = [agent for agent in self._subscribers(topic) if agent != sender]
                mark_message_outbox_published(
                    self._resolve_engine(), wire_message, tenant=tenant
                )
        else:
            for recipient in recipients:
                if not recipient:
                    continue
                recipient_message = {**wire_message, "recipient": recipient}
                commit_message_outbox(
                    self._resolve_engine(), recipient_message, tenant=tenant, now=now
                )
                if backend.publish_direct(
                    tenant=tenant,
                    group=group,
                    sender=sender,
                    to=recipient,
                    payload=payload,
                    meta_json=meta_json,
                    created=now,
                ):
                    delivered.append(recipient)
                    mark_message_outbox_published(
                        self._resolve_engine(), recipient_message, tenant=tenant
                    )
        return delivered

    # ── Dispatch: message → fleet work (CONCEPT:AU-ORCH.routing.resolve-body-single-canonical) ───────────
    def dispatch(
        self,
        *,
        sender: str,
        objective: str,
        kind: str = "develop",
        priority: str = "normal",
        reason: str = "",
    ) -> dict[str, Any]:
        """Turn a request into the sole authoritative WorkItem state machine.

        This closes the message↔task gap: an agent on the bus hands an objective to the fleet
        The objective is first treated as a tenant-qualified inbox payload and
        committed with its WorkItem/audit/outbox records. Legacy Loop rows are
        no longer a second writable authority.
        """
        if not (sender and objective):
            return {"ok": False, "error": "sender and objective required"}
        from agent_utilities.messaging.bus_log import current_bus_tenant

        tenant = current_bus_tenant()
        sender = bus_reference("agent", sender, tenant=tenant)
        objective, _objective_meta, _privacy_report = sanitize_bus_content(objective, {})
        reason, _reason_meta, _reason_report = sanitize_bus_content(reason, {})
        decision = self._gate("bus.dispatch", objective[:80], sender, reason)
        if decision is not None and not decision.allowed:
            _metrics.BUS_DISPATCH.labels(outcome="denied").inc()
            return {
                "ok": False,
                "error": f"policy {decision.decision}: {decision.reason}",
            }
        engine = self._resolve_engine()
        try:
            from agent_utilities.messaging.bus_inbox import commit_message_to_work_item
            from agent_utilities.messaging.bus_log import current_bus_tenant

            group = uuid.uuid4().hex
            committed = commit_message_to_work_item(
                engine,
                {
                    "id": f"busdispatch:{group}",
                    "msg_group": group,
                    "sender": sender,
                    "recipient": "fleet",
                    "topic": "dispatch",
                    "payload": objective,
                    "meta": {"kind": kind, "priority": priority},
                    "created": time.time(),
                },
                tenant=current_bus_tenant(),
                recipient="fleet",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[ORCH-1.80] dispatch WorkItem commit failed (%s)",
                type(exc).__name__,
            )
            _metrics.BUS_DISPATCH.labels(outcome="failed").inc()
            return {
                "ok": False,
                "error": f"dispatch failed ({type(exc).__name__})",
            }
        _metrics.BUS_DISPATCH.labels(outcome="submitted").inc()
        return {
            "ok": True,
            "work_item_id": committed.work_item_id,
            "inbox_id": committed.inbox_id,
            "replay": committed.replay,
        }

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
            logger.warning(
                "[ECO-4.84] action policy unavailable (%s)", type(exc).__name__
            )
            from types import SimpleNamespace

            return SimpleNamespace(
                allowed=False,
                decision="deny",
                reason="action policy unavailable",
            )

    # ── Introspection ────────────────────────────────────────────────
    def status(self) -> dict[str, Any]:
        roster = self.roster()
        online = sum(1 for a in roster if a["presence"] == "online")
        topics = self._query("MATCH (t:Topic) RETURN t.name as name", {})
        # Sample the presence gauges on the health/status read (CONCEPT:AU-ECO.bus.operator-view-agentbus).
        _metrics.BUS_PARTICIPANTS.labels(status="online").set(online)
        _metrics.BUS_PARTICIPANTS.labels(status="offline").set(len(roster) - online)
        backend = self._log_backend()
        return {
            "agents": len(roster),
            "online": online,
            "topics": sorted({t.get("name") for t in topics if t.get("name")}),
            "log_backend": backend.name,
        }


def _safe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value or {}
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return {}


def _metadata_flag(value: Any, key: str) -> bool:
    metadata = _safe_json(value)
    return bool(metadata.get(key)) if isinstance(metadata, dict) else False


def swarm_topic(session_id: str | None) -> str:
    """The shared bus topic a swarm uses to coordinate (CONCEPT:AU-ECO.bus.native-capability)."""
    return f"swarm:{session_id}" if session_id else "swarm:default"


def bus_capability_prompt() -> str:
    """Canonical, single-source capability blurb woven into every agent's system prompt.

    CONCEPT:AU-ECO.bus.native-capability — the AgentBus is a NATIVE capability, not an opt-in persona: the core
    orchestrator (the "graph shaper") and every spawned swarm/sub-agent inherit this so they
    know they can coordinate with peers instead of working in isolation. Kept in sync with the
    ``prompts/bus_coordinator.json`` blueprint, which is the deeper standalone profile.
    """
    return (
        "## Agent Bus — coordinate with other agents (native capability, CONCEPT:AU-ECO.bus.agentbus-federated-agent-agent)\n"
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
