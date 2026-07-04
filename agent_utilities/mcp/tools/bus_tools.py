"""graph_bus MCP tool — the agent-to-agent communication bus surface (CONCEPT:AU-ECO.bus.agent-to-agent-bus).

Thin wrapper over :class:`agent_utilities.messaging.bus.AgentBus` (the one core). This is the
surface any session — a Claude Code session, another LLM, a session from any provider, on any
host — uses to register on the shared hub, discover peers, exchange messages, and hand work to
the fleet. The REST twin is ``/graph/bus`` (``gateway/graph_api.py`` via the generic adapter);
both dispatch into the same :class:`AgentBus` so they never drift.

CONCEPT:AU-ECO.bus.agent-to-agent-bus — graph_bus MCP tool and REST twin for agent-to-agent messaging
"""

from __future__ import annotations

import contextlib
import json

from fastmcp import Context
from pydantic import Field

from agent_utilities.mcp import kg_server


def _bus_actor_scope(action: str) -> contextlib.AbstractContextManager:
    """Scope a ``graph_bus`` call to the request's server-minted identity (CONCEPT:AU-ECO.bus.bus-register-under-served).

    ``graph_bus`` is a standalone FastMCP tool, so — unlike every action routed
    through :func:`kg_server._execute_tool` — it does NOT otherwise apply the
    validated per-request actor. That left the bus writing/reading as the ambient
    unauthenticated ``SYSTEM_ACTOR`` under the served profile (``KG_BRAIN_ENFORCE``):
    a ``register`` write was unattributed and, depending on the engine's isolation
    rules, did not land — surfacing as ``ok:false``. This mirrors ``_execute_tool``'s
    identity handling so an *authenticated* session registers under its own
    ``ActorContext`` (the write lands + the roster shows it), while an
    *unauthenticated* MCP caller is correctly rejected under ``KG_AUTH_REQUIRED``
    for the mutating actions instead of silently writing as SYSTEM.

    The bus is fleet-coordination INFRASTRUCTURE, so the read-only presence actions
    (``roster``/``status``/``list_hubs``) stay reachable for an unauthenticated caller
    (parity with :data:`kg_server.ANONYMOUS_READ_TOOLS`); the mutating actions require
    a real identity when the server enforces auth.
    """
    from agent_utilities.security.brain_context import current_actor, use_actor

    _READ_ONLY = {"roster", "status", "list_hubs"}

    # If the gateway middleware already scoped this request (REST surface), keep it.
    if current_actor().authenticated:
        return contextlib.nullcontext()

    actor = kg_server._actor_from_mcp_token()
    if actor is None and kg_server._PROCESS_ACTOR is not None:
        actor = kg_server._PROCESS_ACTOR

    if actor is not None:
        return use_actor(actor)

    # No server-minted identity. Under enforced auth a mutating action may not run
    # as the ambient privileged SYSTEM actor — reject it with a clear reason so the
    # caller learns it must authenticate (rather than landing an unattributed write).
    if kg_server._kg_auth_required() and action not in _READ_ONLY:
        raise PermissionError(
            f"KG_AUTH_REQUIRED=1: graph_bus action {action!r} needs an authenticated "
            "identity (an OIDC client-credentials Bearer token on the MCP connection, "
            "or KG_AUTH_TOKEN for stdio). Read-only presence actions "
            f"({sorted(_READ_ONLY)}) are exempt. (CONCEPT:AU-OS.identity.authenticated-identity-enforcement / ECO-4.98)"
        )
    return contextlib.nullcontext()


def _session_identity(ctx: Context | None) -> str:
    """Derive a stable per-session id from the served FastMCP request (CONCEPT:AU-ECO.bus.auto-register-online-presence).

    A session that calls ``graph_bus`` without passing ``agent_id`` can still be
    auto-registered + presence-tracked: FastMCP injects a ``Context`` on served requests whose
    ``session_id`` is stable for the life of the MCP connection (the ``client_id`` is the
    fallback). Headless/in-process calls have no Context, so this returns "" and the caller
    supplies the id explicitly — we never fabricate an identity. Prefixed so an auto-derived id
    never collides with an operator-chosen ``agent_id``.
    """
    if ctx is None:
        return ""
    for attr in ("session_id", "client_id"):
        try:
            val = getattr(ctx, attr, None)
        except Exception:  # noqa: BLE001 — Context attrs can raise off a live request
            val = None
        if val:
            return f"session:{val}"
    return ""


def register_bus_tools(mcp):
    """Register the ``graph_bus`` tool onto the MCP server. CONCEPT:AU-ECO.bus.agent-to-agent-bus"""

    @mcp.tool(
        name="graph_bus",
        description=(
            "CONCEPT:AU-ECO.bus.agent-to-agent-bus — the agent-to-agent communication bus: let this session talk "
            "to other Claude/LLM sessions (any provider, any host) through the shared graph-os "
            "hub. Actions: 'register' (agent_id [+provider,host,capabilities,session_id] → join "
            "the bus), 'heartbeat' (agent_id → stay online), 'roster' ([provider|capability|"
            "online_only] → discover peers + presence), 'send' (sender + payload + to|topic → "
            "message a peer or a topic, governed by bus.send), 'receive' (agent_id [+since] → "
            "new messages + cursor), 'subscribe'/'unsubscribe' (agent_id + topic), 'ack' "
            "(agent_id + message_id), 'dispatch' (sender + objective [+kind,priority] → hand an "
            "objective to the fleet as a Loop, governed by bus.dispatch), 'leave' (agent_id), "
            "'status'. Mesh/federation (ECO-4.86): 'register_hub' (agent_id=name + url), "
            "'list_hubs', 'federate' (group [+scope] → forward a message group to peer hubs), "
            "'federate_in' (apply a forwarded group). Durable + cross-host: state lives in the KG. "
            "Store-and-forward (ECO-4.91): a 'send' to a topic is also LEFT for peers who "
            "subscribe later (replayed once via a per-(agent,topic) cursor; subscribe with "
            "replay_recent=true to backfill a recent window). Auto-presence (ECO-4.92): merely "
            "using any action keeps this session online + rosterable — no explicit 'register' needed."
        ),
        tags=["graph-os", "messaging", "bus", "a2a"],
    )
    async def graph_bus(
        action: str = Field(
            default="roster",
            description=(
                "register | heartbeat | roster | send | receive | subscribe | "
                "unsubscribe | ack | dispatch | leave | status"
            ),
        ),
        agent_id: str = Field(
            default="", description="This participant's id (most actions)."
        ),
        sender: str = Field(default="", description="Sender agent id (send/dispatch)."),
        to: str = Field(default="", description="Recipient agent id (send, direct)."),
        topic: str = Field(
            default="", description="Topic name (send/subscribe/unsubscribe)."
        ),
        payload: str = Field(default="", description="Message body (send)."),
        objective: str = Field(default="", description="Work objective (dispatch)."),
        kind: str = Field(
            default="develop",
            description="Loop kind for dispatch: develop|research|skill.",
        ),
        priority: str = Field(
            default="normal",
            description="Bucket 0-3 or critical|high|normal|background (dispatch).",
        ),
        provider: str = Field(
            default="",
            description="Provider label, e.g. anthropic|openai|google (register/roster).",
        ),
        host: str = Field(
            default="", description="Host this session runs on (register)."
        ),
        capabilities: str = Field(
            default="",
            description="Comma-separated capability tags (register); single tag filter (roster).",
        ),
        session_id: str = Field(
            default="", description="Originating session id (register)."
        ),
        message_id: str = Field(default="", description="Message id to ack."),
        since: int = Field(
            default=0, description="Cursor: messages already consumed (receive)."
        ),
        online_only: bool = Field(
            default=False, description="Roster: only online peers."
        ),
        reason: str = Field(default="", description="Audit reason (send/dispatch)."),
        url: str = Field(default="", description="Peer hub base URL (register_hub)."),
        group: str = Field(
            default="", description="Message group to forward (federate)."
        ),
        origin: str = Field(default="", description="Origin hub id (federate_in)."),
        scope: str = Field(
            default="commons",
            description="Marking scope for federation: commons|org|private (federate).",
        ),
        replay_recent: bool = Field(
            default=False,
            description="Subscribe: backfill a bounded recent topic window for a late joiner.",
        ),
        ctx: Context | None = None,
    ) -> str:
        from agent_utilities.messaging.bus import AgentBus

        engine = kg_server._get_engine()
        bus = AgentBus.instance(engine)

        # Scope the whole call to the request's server-minted identity so the bus
        # writes/reads as the authenticated caller under the served profile, and an
        # unauthenticated caller is cleanly rejected for mutating actions rather than
        # silently writing as the ambient SYSTEM actor (CONCEPT:AU-ECO.bus.bus-register-under-served / OS-5.14).
        try:
            _scope = _bus_actor_scope(action)
        except PermissionError as exc:
            return json.dumps({"ok": False, "error": str(exc)})
        with _scope:
            return _dispatch_bus(
                bus,
                engine,
                action=action,
                agent_id=agent_id,
                sender=sender,
                to=to,
                topic=topic,
                payload=payload,
                objective=objective,
                kind=kind,
                priority=priority,
                provider=provider,
                host=host,
                capabilities=capabilities,
                session_id=session_id,
                message_id=message_id,
                since=since,
                online_only=online_only,
                reason=reason,
                url=url,
                group=group,
                origin=origin,
                scope=scope,
                replay_recent=replay_recent,
                ctx=ctx,
            )

    def _dispatch_bus(
        bus,
        engine,
        *,
        action: str,
        agent_id: str,
        sender: str,
        to: str,
        topic: str,
        payload: str,
        objective: str,
        kind: str,
        priority: str,
        provider: str,
        host: str,
        capabilities: str,
        session_id: str,
        message_id: str,
        since: int,
        online_only: bool,
        reason: str,
        url: str,
        group: str,
        origin: str,
        scope: str,
        replay_recent: bool,
        ctx: Context | None,
    ) -> str:
        # CONCEPT:AU-ECO.bus.auto-register-online-presence — auto-register + presence: a session that has this tool appears
        # online to peers without an explicit ``register`` call. Resolve the acting id (the
        # explicit agent_id/sender, else the stable served-session identity) and TOUCH the bus
        # so any action keeps the caller rosterable + bumps last_seen. ``touch`` auto-creates the
        # :BusAgent on first reference and is idempotent + best-effort.
        acting_id = agent_id or sender or _session_identity(ctx)
        if acting_id:
            bus.touch(acting_id)
            # Back-fill a derived id so per-id actions below operate on the same participant.
            if not agent_id and action in (
                "receive",
                "subscribe",
                "unsubscribe",
                "heartbeat",
                "ack",
                "leave",
                "deregister",
            ):
                agent_id = acting_id
            if not sender and action in ("send", "dispatch"):
                sender = acting_id

        if action == "register":
            caps = [c.strip() for c in capabilities.split(",") if c.strip()]
            return json.dumps(
                bus.register(
                    agent_id,
                    provider=provider,
                    host=host,
                    capabilities=caps,
                    session_id=session_id,
                ),
                default=str,
            )
        if action == "heartbeat":
            return json.dumps({"ok": bus.heartbeat(agent_id)})
        if action in ("leave", "deregister"):
            return json.dumps({"ok": bus.deregister(agent_id)})
        if action == "roster":
            return json.dumps(
                {
                    "roster": bus.roster(
                        provider=provider,
                        capability=capabilities.strip(),
                        online_only=online_only,
                    )
                },
                default=str,
            )
        if action == "send":
            return json.dumps(
                bus.send(
                    sender=sender or agent_id,
                    payload=payload,
                    to=to,
                    topic=topic,
                    reason=reason,
                ),
                default=str,
            )
        if action == "receive":
            return json.dumps(bus.receive(agent_id, since=since), default=str)
        if action == "subscribe":
            return json.dumps(
                {"ok": bus.subscribe(agent_id, topic, replay_recent=replay_recent)}
            )
        if action == "unsubscribe":
            return json.dumps({"ok": bus.unsubscribe(agent_id, topic)})
        if action == "ack":
            return json.dumps({"ok": bus.ack(agent_id, message_id)})
        if action == "dispatch":
            return json.dumps(
                bus.dispatch(
                    sender=sender or agent_id,
                    objective=objective,
                    kind=kind,
                    priority=priority,
                    reason=reason,
                ),
                default=str,
            )
        if action == "status":
            return json.dumps(bus.status(), default=str)
        # ── Federation / mesh (CONCEPT:AU-ECO.bus.federation-relay) ──
        if action in ("register_hub", "list_hubs", "federate", "federate_in"):
            from agent_utilities.messaging.federation import BusFederationRelay

            relay = BusFederationRelay.instance(engine)
            if action == "register_hub":
                if not (agent_id and url):
                    return json.dumps(
                        {"error": "register_hub needs agent_id (hub name) and url"}
                    )
                return json.dumps({"result": relay.register_hub(agent_id, url)})
            if action == "list_hubs":
                return json.dumps({"hubs": relay.list_hubs()}, default=str)
            if action == "federate":
                return json.dumps(relay.forward(group, scope=scope), default=str)
            if action == "federate_in":
                recipients = [r.strip() for r in to.split(",") if r.strip()]
                return json.dumps(
                    relay.apply_inbound(
                        group=group,
                        sender=sender,
                        recipients=recipients,
                        payload=payload,
                        topic=topic,
                        origin=origin,
                    ),
                    default=str,
                )
        return json.dumps({"error": f"unknown action: {action}"})

    kg_server.REGISTERED_TOOLS["graph_bus"] = graph_bus
