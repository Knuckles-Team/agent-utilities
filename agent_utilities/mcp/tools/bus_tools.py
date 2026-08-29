"""graph_bus MCP tool — the agent-to-agent communication bus surface (CONCEPT:AU-ECO.bus.agent-to-agent-bus).

Thin wrapper over :class:`agent_utilities.messaging.bus.AgentBus` (the one core). This is the
surface any session — a Claude Code session, another LLM, a session from any provider, on any
host — uses to register on the shared hub, discover peers, exchange messages, and hand work to
the fleet. The REST twin is ``/graph/bus`` (``gateway/graph_api.py`` via the generic adapter);
both dispatch into the same :class:`AgentBus` so they never drift.

CONCEPT:AU-ECO.bus.agent-to-agent-bus — graph_bus MCP tool and REST twin for agent-to-agent messaging

Internal dispatch shape (CX wD10-R-ARITY): ``graph_bus``'s own signature is the MCP wire
contract (FastMCP derives the published tool schema from it) and must never change shape.
Everything *behind* that boundary is typed: the raw arguments are parsed once into a single
``BusRequest`` dataclass (defined in the dependency-free leaf ``agent_utilities.mcp.bus_types``,
re-exported here — see that module's docstring for why it is NOT defined in this file), and
every action handler takes ``(BusExecContext, BusRequest)`` — two parameters, never the raw
20-argument tuple. See the lane report (``plans/complex/lane-reports/WD10-R-ARITY.md``) for the
pattern write-up before doing the same to another action-dispatch tool.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable
from dataclasses import replace

from fastmcp import Context
from pydantic import Field

from agent_utilities.mcp import kg_server
from agent_utilities.mcp.bus_types import BusExecContext as _BusExecContext
from agent_utilities.mcp.bus_types import BusRequest
from agent_utilities.security.error_surface import public_error_json

__all__ = ["BusRequest", "register_bus_tools"]


def _bus_actor_scope(action: str) -> contextlib.AbstractContextManager:
    """Scope ``graph_bus`` to middleware/process-minted authority.

    ``graph_bus`` is a standalone FastMCP tool, so — unlike every action routed
    through :func:`kg_server._execute_tool` — it must enter the same verified
    session scope explicitly. Read and write actions both require tenant-bound
    authority; anonymous roster/status access would otherwise leak fleet state.
    """
    del action
    return kg_server.verified_tool_session_scope()


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


_BUS_AUTO_AGENT_ID_ACTIONS = frozenset(
    {"receive", "subscribe", "unsubscribe", "heartbeat", "leave", "deregister"}
)
_BUS_AUTO_SENDER_ACTIONS = frozenset({"send", "dispatch"})


def _apply_bus_auto_presence(bus, request: BusRequest) -> BusRequest:
    """Auto-register + presence (CONCEPT:AU-ECO.bus.auto-register-online-presence).

    A session that has this tool appears online to peers without an explicit ``register``
    call. Resolve the acting id (the explicit agent_id/sender, else the stable served-session
    identity) and TOUCH the bus so any action keeps the caller rosterable + bumps last_seen.
    ``touch`` auto-creates the :BusAgent on first reference and is idempotent + best-effort.
    """
    acting_id = request.agent_id or request.sender or _session_identity(request.ctx)
    if not acting_id:
        return request
    bus.touch(acting_id)
    agent_id = request.agent_id
    if not agent_id and request.action in _BUS_AUTO_AGENT_ID_ACTIONS:
        agent_id = acting_id
    sender = request.sender
    if not sender and request.action in _BUS_AUTO_SENDER_ACTIONS:
        sender = acting_id
    return replace(request, agent_id=agent_id, sender=sender)


def _bus_handle_register(ctx: _BusExecContext, request: BusRequest) -> str:
    caps = [c.strip() for c in request.capabilities.split(",") if c.strip()]
    return json.dumps(
        ctx.bus.register(
            request.agent_id,
            provider=request.provider,
            host=request.host,
            capabilities=caps,
            session_id=request.session_id,
        ),
        default=str,
    )


def _bus_handle_heartbeat(ctx: _BusExecContext, request: BusRequest) -> str:
    return json.dumps({"ok": ctx.bus.heartbeat(request.agent_id)})


def _bus_handle_leave(ctx: _BusExecContext, request: BusRequest) -> str:
    return json.dumps({"ok": ctx.bus.deregister(request.agent_id)})


def _bus_handle_roster(ctx: _BusExecContext, request: BusRequest) -> str:
    return json.dumps(
        {
            "roster": ctx.bus.roster(
                provider=request.provider,
                capability=request.capabilities.strip(),
                online_only=request.online_only,
            )
        },
        default=str,
    )


def _bus_handle_send(ctx: _BusExecContext, request: BusRequest) -> str:
    return json.dumps(
        ctx.bus.send(
            sender=request.sender or request.agent_id,
            payload=request.payload,
            to=request.to,
            topic=request.topic,
            reason=request.reason,
        ),
        default=str,
    )


def _bus_handle_receive(ctx: _BusExecContext, request: BusRequest) -> str:
    return json.dumps(
        ctx.bus.receive(request.agent_id, since=request.since), default=str
    )


def _bus_handle_subscribe(ctx: _BusExecContext, request: BusRequest) -> str:
    return json.dumps({"ok": ctx.bus.subscribe(request.agent_id, request.topic)})


def _bus_handle_unsubscribe(ctx: _BusExecContext, request: BusRequest) -> str:
    return json.dumps({"ok": ctx.bus.unsubscribe(request.agent_id, request.topic)})


def _bus_handle_dispatch(ctx: _BusExecContext, request: BusRequest) -> str:
    return json.dumps(
        ctx.bus.dispatch(
            sender=request.sender or request.agent_id,
            objective=request.objective,
            kind=request.kind,
            priority=request.priority,
            reason=request.reason,
        ),
        default=str,
    )


def _bus_handle_status(ctx: _BusExecContext, request: BusRequest) -> str:
    return json.dumps(ctx.bus.status(), default=str)


def _bus_federation_relay(engine):
    # CONCEPT:AU-ECO.bus.federation-relay — lazy import, same as pre-refactor: federation is
    # rarely exercised and the relay module pulls in extra deps not every caller needs.
    from agent_utilities.messaging.federation import BusFederationRelay

    return BusFederationRelay.instance(engine)


def _bus_handle_register_hub(ctx: _BusExecContext, request: BusRequest) -> str:
    if not (request.agent_id and request.url):
        return json.dumps({"error": "register_hub needs agent_id (hub name) and url"})
    relay = _bus_federation_relay(ctx.engine)
    return json.dumps({"result": relay.register_hub(request.agent_id, request.url)})


def _bus_handle_list_hubs(ctx: _BusExecContext, request: BusRequest) -> str:
    del request
    relay = _bus_federation_relay(ctx.engine)
    return json.dumps({"hubs": relay.list_hubs()}, default=str)


def _bus_handle_federate(ctx: _BusExecContext, request: BusRequest) -> str:
    relay = _bus_federation_relay(ctx.engine)
    return json.dumps(relay.forward(request.group, scope=request.scope), default=str)


def _bus_handle_federate_in(ctx: _BusExecContext, request: BusRequest) -> str:
    relay = _bus_federation_relay(ctx.engine)
    recipients = [r.strip() for r in request.to.split(",") if r.strip()]
    return json.dumps(
        relay.apply_inbound(
            group=request.group,
            sender=request.sender,
            recipients=recipients,
            payload=request.payload,
            topic=request.topic,
            origin=request.origin,
        ),
        default=str,
    )


_BUS_ACTION_HANDLERS: dict[str, Callable[[_BusExecContext, BusRequest], str]] = {
    "register": _bus_handle_register,
    "heartbeat": _bus_handle_heartbeat,
    "leave": _bus_handle_leave,
    "deregister": _bus_handle_leave,
    "roster": _bus_handle_roster,
    "send": _bus_handle_send,
    "receive": _bus_handle_receive,
    "subscribe": _bus_handle_subscribe,
    "unsubscribe": _bus_handle_unsubscribe,
    "dispatch": _bus_handle_dispatch,
    "status": _bus_handle_status,
    "register_hub": _bus_handle_register_hub,
    "list_hubs": _bus_handle_list_hubs,
    "federate": _bus_handle_federate,
    "federate_in": _bus_handle_federate_in,
}


def _dispatch_bus(bus, engine, request: BusRequest) -> str:
    """Route one ``graph_bus`` call to its typed handler. 3 params, dict dispatch — cyc/cog low."""
    request = _apply_bus_auto_presence(bus, request)
    handler = _BUS_ACTION_HANDLERS.get(request.action)
    if handler is None:
        return json.dumps({"error": f"unknown action: {request.action}"})
    return handler(_BusExecContext(bus=bus, engine=engine), request)


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
            "new messages + cursor), 'subscribe'/'unsubscribe' (agent_id + topic), "
            "'dispatch' (sender + objective [+kind,priority] → hand an "
            "objective to the fleet as a Loop, governed by bus.dispatch), 'leave' (agent_id), "
            "'status'. Mesh/federation (ECO-4.86): 'register_hub' (agent_id=name + url), "
            "'list_hubs', 'federate' (group [+scope] → forward a message group to peer hubs), "
            "'federate_in' (apply a forwarded group). Durable + cross-host: committed inboxes, "
            "outboxes, subscriptions, and WorkItems live in the KG; message transport uses a "
            "bounded partitioned log. Auto-presence (ECO-4.92): merely "
            "using any action keeps this session online + rosterable — no explicit 'register' needed."
        ),
        tags=["graph-os", "messaging", "bus", "a2a"],
    )
    async def graph_bus(
        action: str = Field(
            default="roster",
            description=(
                "register | heartbeat | roster | send | receive | subscribe | "
                "unsubscribe | dispatch | leave | status"
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
            await kg_server._ensure_process_authority_current()
            _scope = _bus_actor_scope(action)
        except PermissionError as exc:
            return public_error_json(
                exc, code="permission_denied", context={"ok": False}
            )
        request = BusRequest(
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
            since=since,
            online_only=online_only,
            reason=reason,
            url=url,
            group=group,
            origin=origin,
            scope=scope,
            ctx=ctx,
        )
        with _scope:
            return _dispatch_bus(bus, engine, request)

    kg_server.REGISTERED_TOOLS["graph_bus"] = graph_bus
