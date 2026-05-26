"""WebSocket manager for real-time dashboard updates.

CONCEPT:GW-1.0 — Gateway Service Dashboard

Mountable alongside the REST router for widgets that support live streaming
(containers, uptime monitors, download progress).

Usage in agent-webui::

    from agent_utilities.gateway.ws import dashboard_ws_router
    app.include_router(dashboard_ws_router)
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from agent_utilities.gateway.aggregator import Aggregator
from agent_utilities.gateway.config import ConfigManager

logger = logging.getLogger(__name__)

dashboard_ws_router = APIRouter()


class ConnectionManager:
    """Manages active WebSocket connections for dashboard streaming."""

    def __init__(self) -> None:
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)
        logger.info("WebSocket client connected (%d total)", len(self.active))

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)
        logger.info("WebSocket client disconnected (%d remaining)", len(self.active))

    async def broadcast(self, data: dict[str, Any]) -> None:
        """Send data to all connected clients."""
        dead: list[WebSocket] = []
        message = json.dumps(data, default=str)
        for ws in self.active:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


_manager = ConnectionManager()
_aggregator: Aggregator | None = None


def _get_aggregator() -> Aggregator:
    global _aggregator
    if _aggregator is None:
        _aggregator = Aggregator(config_manager=ConfigManager())
    return _aggregator


@dashboard_ws_router.websocket("/ws/dashboard")
async def dashboard_ws(ws: WebSocket) -> None:
    """WebSocket endpoint for real-time dashboard updates.

    Sends a full data snapshot on connect, then streams updates
    at the configured refresh interval.

    Client can send JSON messages::

        {"type": "subscribe", "services": ["portainer-1", "uptime-1"]}
        {"type": "refresh"}
        {"type": "interval", "seconds": 10}
    """
    await _manager.connect(ws)
    aggregator = _get_aggregator()

    interval = 15.0  # Default stream interval
    subscribed_services: set[str] | None = None  # None = all

    try:
        # Send initial data snapshot
        data = await aggregator.fetch_all()
        await ws.send_json(
            {
                "type": "snapshot",
                "data": {k: v.model_dump(mode="json") for k, v in data.items()},
            }
        )

        while True:
            # Wait for client message or timeout (send update)
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=interval)
                msg = json.loads(raw)

                msg_type = msg.get("type", "")

                if msg_type == "subscribe":
                    subscribed_services = set(msg.get("services", []))
                    logger.debug("Client subscribed to: %s", subscribed_services)

                elif msg_type == "refresh":
                    # Force immediate refresh
                    pass

                elif msg_type == "interval":
                    interval = max(5.0, min(120.0, float(msg.get("seconds", 15))))
                    logger.debug("Client set interval to: %ss", interval)

            except TimeoutError:
                pass

            # Fetch and send update
            all_data = await aggregator.fetch_all()

            if subscribed_services:
                filtered = {
                    k: v for k, v in all_data.items() if k in subscribed_services
                }
            else:
                filtered = all_data

            await ws.send_json(
                {
                    "type": "update",
                    "data": {k: v.model_dump(mode="json") for k, v in filtered.items()},
                }
            )

    except WebSocketDisconnect:
        _manager.disconnect(ws)
    except Exception as e:
        logger.error("WebSocket error: %s", e, exc_info=True)
        _manager.disconnect(ws)
