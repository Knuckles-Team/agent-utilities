"""graph_bus MCP tool — the agent-to-agent communication bus surface (CONCEPT:ECO-4.85).

Thin wrapper over :class:`agent_utilities.messaging.bus.AgentBus` (the one core). This is the
surface any session — a Claude Code session, another LLM, a session from any provider, on any
host — uses to register on the shared hub, discover peers, exchange messages, and hand work to
the fleet. The REST twin is ``/graph/bus`` (``gateway/graph_api.py`` via the generic adapter);
both dispatch into the same :class:`AgentBus` so they never drift.

CONCEPT:ECO-4.85 — graph_bus MCP tool and REST twin for agent-to-agent messaging
"""

from __future__ import annotations

import json

from pydantic import Field

from agent_utilities.mcp import kg_server


def register_bus_tools(mcp):
    """Register the ``graph_bus`` tool onto the MCP server. CONCEPT:ECO-4.85"""

    @mcp.tool(
        name="graph_bus",
        description=(
            "CONCEPT:ECO-4.85 — the agent-to-agent communication bus: let this session talk "
            "to other Claude/LLM sessions (any provider, any host) through the shared graph-os "
            "hub. Actions: 'register' (agent_id [+provider,host,capabilities,session_id] → join "
            "the bus), 'heartbeat' (agent_id → stay online), 'roster' ([provider|capability|"
            "online_only] → discover peers + presence), 'send' (sender + payload + to|topic → "
            "message a peer or a topic, governed by bus.send), 'receive' (agent_id [+since] → "
            "new messages + cursor), 'subscribe'/'unsubscribe' (agent_id + topic), 'ack' "
            "(agent_id + message_id), 'dispatch' (sender + objective [+kind,priority] → hand an "
            "objective to the fleet as a Loop, governed by bus.dispatch), 'leave' (agent_id), "
            "'status'. Durable + cross-host: state lives in the KG."
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
    ) -> str:
        from agent_utilities.messaging.bus import AgentBus

        engine = kg_server._get_engine()
        bus = AgentBus.instance(engine)

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
            return json.dumps({"ok": bus.subscribe(agent_id, topic)})
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
        return json.dumps({"error": f"unknown action: {action}"})

    kg_server.REGISTERED_TOOLS["graph_bus"] = graph_bus
