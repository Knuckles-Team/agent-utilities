"""graph_reach MCP tool — outbound messaging + last-active channel routing (CONCEPT:AU-ECO.mcp.graph-reach-mcp-tool).

Thin wrapper over :class:`agent_utilities.messaging.service.MessagingService` (the one core).
This is the surface Claude (and any MCP client) uses to message the operator over Telegram
(or any configured backend) and to inspect routing state. The REST twin is ``/graph/reach``
(``gateway/graph_api.py``); both dispatch into the same service so they never drift.

CONCEPT:AU-ECO.mcp.graph-reach-mcp-tool — graph_reach MCP tool and REST twin for outbound user messaging
"""

from __future__ import annotations

import json

from pydantic import Field

from agent_utilities.mcp import kg_server


def register_reach_tools(mcp):
    """Register the ``graph_reach`` tool onto the MCP server. CONCEPT:AU-ECO.mcp.graph-reach-mcp-tool"""

    @mcp.tool(
        name="graph_reach",
        description=(
            "CONCEPT:AU-ECO.mcp.graph-reach-mcp-tool — reach the user over a messaging backend (Telegram, "
            "Slack, Discord, ...). Actions: 'reach_user' (text [+user_id] → routed to the "
            "user's LAST-ACTIVE channel, else the configured default — OpenClaw-style), "
            "'send' (platform+channel_id+text for an explicit target), 'list_channels' "
            "(platform), 'last_channel' ([user_id] → resolved platform+channel), 'status' "
            "(configured/connected backends). Every send is governed by the ActionPolicy "
            "gate and mirrored into KG conversational memory."
        ),
        tags=["graph-os", "messaging", "reach"],
    )
    async def graph_reach(
        action: str = Field(
            default="reach_user",
            description="reach_user | send | list_channels | last_channel | status",
        ),
        text: str = Field(default="", description="Message text (reach_user/send)."),
        platform: str = Field(
            default="", description="Backend id, e.g. 'telegram' (send/list_channels)."
        ),
        channel_id: str = Field(
            default="", description="Target channel/chat id (send)."
        ),
        user_id: str = Field(
            default="", description="User id for routing (reach_user/last_channel)."
        ),
        thread_id: str = Field(default="", description="Optional thread id (send)."),
        reply_to_id: str = Field(
            default="", description="Optional message id to reply to (send)."
        ),
        reason: str = Field(
            default="", description="Why this message is being sent (audit trail)."
        ),
    ) -> str:
        from agent_utilities.messaging.service import MessagingService

        engine = kg_server._get_engine()
        svc = MessagingService.instance(engine)

        if action == "reach_user":
            result = await svc.reach_user(
                text,
                user_id=user_id or None,
                source="mcp",
                reason=reason,
            )
            return result.model_dump_json()
        if action == "send":
            if not platform or not channel_id:
                return json.dumps({"error": "send requires platform and channel_id"})
            result = await svc.send(
                platform,
                channel_id,
                text,
                thread_id=thread_id,
                reply_to_id=reply_to_id,
                source="mcp",
                reason=reason,
            )
            return result.model_dump_json()
        if action == "list_channels":
            if not platform:
                return json.dumps({"error": "list_channels requires platform"})
            return json.dumps(
                {"channels": await svc.list_channels(platform)}, default=str
            )
        if action == "last_channel":
            plat, chan = svc.resolve_channel(user_id or None)
            return json.dumps({"platform": plat, "channel_id": chan})
        if action == "status":
            return json.dumps(svc.status())
        return json.dumps({"error": f"unknown action: {action}"})

    kg_server.REGISTERED_TOOLS["graph_reach"] = graph_reach
