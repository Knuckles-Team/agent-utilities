"""Nextcloud Talk Backend (CONCEPT:ECO-4.0). REST API via httpx.

Install: ``pip install agent-utilities[messaging-nextcloud]``
CONCEPT:ECO-4.0 — Native Messaging Backend Abstraction
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from agent_utilities.core.config import setting
from agent_utilities.messaging.base import MessagingBackend
from agent_utilities.messaging.capabilities import (
    CAPABILITY_MATRIX,
    MessagingCapabilities,
)
from agent_utilities.messaging.models import (
    Channel,
    EventType,
    InboundEvent,
    Message,
    MessageDirection,
    MessagingConfig,
    PlatformId,
    SendResult,
)

logger = logging.getLogger(__name__)


class NextcloudTalkBackend(MessagingBackend):
    """Nextcloud Talk backend via OCS/Talk REST API. CONCEPT:ECO-4.0"""

    def __init__(self, config: MessagingConfig | None = None) -> None:
        super().__init__(config)
        self._client: Any = None
        self._base_url = ""
        self._event_queue: asyncio.Queue[InboundEvent] = asyncio.Queue()

    @property
    def id(self) -> str:
        return "nextcloud"

    @property
    def capabilities(self) -> MessagingCapabilities:
        return CAPABILITY_MATRIX["nextcloud"]

    async def connect(self) -> None:
        """Connect to Nextcloud Talk API. CONCEPT:ECO-4.0"""
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "Install: pip install agent-utilities[messaging-nextcloud]"
            ) from None

        self._base_url = self.config.extra.get(
            "url", setting("NEXTCLOUD_URL", "")
        ).rstrip("/")
        user = self.config.app_id or setting("NEXTCLOUD_USER", "")
        token = self.config.token or setting("NEXTCLOUD_TOKEN", "")
        if not self._base_url or not token:
            raise ValueError("Set NEXTCLOUD_URL and NEXTCLOUD_TOKEN.")
        self._client = httpx.AsyncClient(
            base_url=f"{self._base_url}/ocs/v2.php/apps/spreed/api/v4",
            auth=(user, token) if user else None,
            headers={"OCS-APIRequest": "true", "Accept": "application/json"},
        )
        self._connected = True
        logger.info("[CONCEPT:ECO-4.0] Nextcloud Talk backend connected.")

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
        await super().disconnect()

    async def send_message(
        self,
        channel_id: str,
        text: str,
        *,
        thread_id: str = "",
        reply_to_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> SendResult:
        try:
            payload: dict[str, Any] = {"message": text}
            if reply_to_id:
                payload["replyTo"] = int(reply_to_id)
            resp = await self._client.post(f"/room/{channel_id}/message", json=payload)
            data = resp.json().get("ocs", {}).get("data", {})
            return SendResult(
                success=resp.status_code == 201,
                message_id=str(data.get("id", "")),
                platform=PlatformId.NEXTCLOUD,
                channel_id=channel_id,
            )
        except Exception as e:
            return SendResult(
                success=False, platform=PlatformId.NEXTCLOUD, error=str(e)
            )

    async def send_reaction(self, channel_id: str, message_id: str, emoji: str) -> None:
        await self._client.post(
            f"/room/{channel_id}/message/{message_id}/reaction/{emoji}"
        )

    async def list_channels(self) -> list[Channel]:
        resp = await self._client.get("/room")
        rooms = resp.json().get("ocs", {}).get("data", [])
        return [
            Channel(
                id=str(r.get("token", "")),
                name=r.get("displayName", ""),
                platform=PlatformId.NEXTCLOUD,
                is_group=r.get("type", 0) > 1,
                member_count=r.get("participantCount", 0),
            )
            for r in rooms
        ]

    async def listen(self) -> AsyncIterator[InboundEvent]:
        """Poll for new messages (Nextcloud doesn't have native WebSocket). CONCEPT:ECO-4.0"""
        last_known: dict[str, int] = {}
        while self._connected:
            try:
                rooms = await self.list_channels()
                for room in rooms:
                    resp = await self._client.get(
                        f"/room/{room.id}/message",
                        params={"lookIntoFuture": 0, "limit": 10},
                    )
                    messages = resp.json().get("ocs", {}).get("data", [])
                    for msg in messages:
                        msg_id = msg.get("id", 0)
                        if msg_id > last_known.get(room.id, 0):
                            last_known[room.id] = msg_id
                            ev = InboundEvent(
                                event_type=EventType.MESSAGE,
                                platform=PlatformId.NEXTCLOUD,
                                channel_id=room.id,
                                user_id=msg.get("actorId", ""),
                                user_name=msg.get("actorDisplayName", ""),
                                content=msg.get("message", ""),
                                message=Message(
                                    id=str(msg_id),
                                    content=msg.get("message", ""),
                                    channel_id=room.id,
                                    platform=PlatformId.NEXTCLOUD,
                                    direction=MessageDirection.INBOUND,
                                ),
                                raw=msg,
                            )
                            yield ev
                await asyncio.sleep(5)  # Poll interval
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("[CONCEPT:ECO-4.0] Nextcloud poll error: %s", e)
                await asyncio.sleep(10)
