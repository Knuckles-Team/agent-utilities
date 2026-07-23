"""WebSocket manager for real-time dashboard updates.

CONCEPT:AU-OS.config.gateway-service-dashboard — Gateway Service Dashboard

Mountable alongside the REST router for widgets that support live streaming
(containers, uptime monitors, download progress).

Usage in agent-webui::

    from agent_utilities.gateway.ws import dashboard_ws_router
    app.include_router(dashboard_ws_router)
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
from typing import Any
from urllib.parse import urlsplit

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
        from agent_utilities.core.config import config

        if len(self.active) >= min(config.server_max_connections, 512):
            await ws.close(code=4429, reason="connection limit reached")
            raise RuntimeError("WebSocket connection limit reached")
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
    from agent_utilities.core.config import config
    from agent_utilities.security.auth import authenticate_header_values
    from agent_utilities.security.http_boundary import normalize_origins

    def header_values(name: bytes) -> list[bytes]:
        return [
            bytes(value)
            for key, value in ws.scope.get("headers", ())
            if bytes(key).lower() == name
        ]

    origins = [
        value.strip()
        for value in str(config.allowed_origins or "").split(",")
        if value.strip()
    ]
    supplied_origins = header_values(b"origin")
    try:
        if supplied_origins:
            if len(supplied_origins) != 1:
                raise PermissionError("ambiguous origin")
            supplied = normalize_origins([supplied_origins[0].decode("ascii")])
            configured = normalize_origins(origins)
            local_same_origin = False
            if not configured and supplied:
                host_values = header_values(b"host")
                parsed_origin = urlsplit(next(iter(supplied)))
                try:
                    local_host = (
                        parsed_origin.hostname == "localhost"
                        or ipaddress.ip_address(str(parsed_origin.hostname)).is_loopback
                    )
                except ValueError:
                    local_host = False
                local_same_origin = bool(
                    local_host
                    and len(host_values) == 1
                    and parsed_origin.netloc.casefold()
                    == host_values[0].decode("ascii").casefold()
                )
            if not supplied or not (supplied.issubset(configured) or local_same_origin):
                raise PermissionError("origin rejected")
        identity = await authenticate_header_values(
            authorization=header_values(b"authorization"),
        )
        if identity and identity.get("auth_type") == "jwt":
            from agent_utilities.security.identity import (
                base_capabilities,
                normalize_identity,
            )

            capabilities = set(
                base_capabilities(
                    normalize_identity(identity),
                    config.identity_group_capability_map,
                )
            )
            if not capabilities.intersection(
                {"gateway:read", "gateway:write", "gateway:admin", "admin"}
            ):
                raise PermissionError("dashboard capability required")
    except Exception:  # noqa: BLE001 - every boundary failure rejects the upgrade
        await ws.close(code=4401, reason="authentication or origin rejected")
        return

    try:
        await _manager.connect(ws)
    except RuntimeError:
        return
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
                if len(raw.encode("utf-8")) > 65_536:
                    await ws.close(code=4400, reason="message too large")
                    break
                msg = json.loads(raw)
                if not isinstance(msg, dict) or len(msg) > 16:
                    await ws.close(code=4400, reason="invalid message")
                    break

                msg_type = msg.get("type", "")

                if msg_type == "subscribe":
                    services = msg.get("services", [])
                    if (
                        not isinstance(services, list)
                        or len(services) > 128
                        or not all(
                            isinstance(service, str)
                            and 1 <= len(service) <= 128
                            and all(
                                character.isalnum() or character in "-_.:"
                                for character in service
                            )
                            for service in services
                        )
                    ):
                        await ws.close(code=4400, reason="invalid subscription")
                        break
                    subscribed_services = set(services)
                    logger.debug("WebSocket subscription updated")

                elif msg_type == "refresh":
                    # Force immediate refresh
                    pass

                elif msg_type == "interval":
                    interval = max(5.0, min(120.0, float(msg.get("seconds", 15))))
                    logger.debug("Client set interval to: %ss", interval)

            except TimeoutError:
                pass
            except (TypeError, ValueError):
                await ws.close(code=4400, reason="invalid message")
                break

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
        pass
    except Exception as exc:
        logger.error("WebSocket error (exception_type=%s)", type(exc).__name__)
    finally:
        _manager.disconnect(ws)
