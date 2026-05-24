"""Matrix Messaging Backend (CONCEPT:ECO-4.5).

Uses ``matrix-nio`` for E2E encrypted Matrix protocol messaging.

Install: ``pip install agent-utilities[messaging-matrix]``

CONCEPT:ECO-4.5 — Native Messaging Backend Abstraction
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from agent_utilities.messaging.base import MessagingBackend
from agent_utilities.messaging.capabilities import CAPABILITY_MATRIX as CAP_MATRIX
from agent_utilities.messaging.capabilities import MessagingCapabilities
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


class MatrixBackend(MessagingBackend):
    """Matrix protocol backend via matrix-nio. CONCEPT:ECO-4.5"""

    def __init__(self, config: MessagingConfig | None = None) -> None:
        super().__init__(config)
        self._client: Any = None
        self._event_queue: asyncio.Queue[InboundEvent] = asyncio.Queue()

    @property
    def id(self) -> str:
        return "matrix"

    @property
    def capabilities(self) -> MessagingCapabilities:
        return CAP_MATRIX["matrix"]

    async def connect(self) -> None:
        """Connect to Matrix homeserver. CONCEPT:ECO-4.5"""
        try:
            from nio import AsyncClient, RoomMessageText
        except ImportError:
            raise ImportError(
                "Install: pip install agent-utilities[messaging-matrix]"
            ) from None
        import os

        homeserver = self.config.extra.get(
            "homeserver", os.environ.get("MATRIX_HOMESERVER", "")
        )
        user_id = self.config.app_id or os.environ.get("MATRIX_USER_ID", "")
        token = self.config.token or os.environ.get("MATRIX_ACCESS_TOKEN", "")
        if not homeserver or not token:
            raise ValueError("Set MATRIX_HOMESERVER and MATRIX_ACCESS_TOKEN.")
        self._client = AsyncClient(homeserver, user_id)
        self._client.access_token = token

        async def on_message(room: Any, event: Any) -> None:
            if event.sender == self._client.user_id:
                return
            ev = InboundEvent(
                event_type=EventType.MESSAGE,
                platform=PlatformId.MATRIX,
                channel_id=room.room_id,
                user_id=event.sender,
                user_name=event.sender,
                content=event.body,
                message=Message(
                    id=event.event_id,
                    content=event.body,
                    channel_id=room.room_id,
                    author_id=event.sender,
                    platform=PlatformId.MATRIX,
                    direction=MessageDirection.INBOUND,
                ),
            )
            await self._event_queue.put(ev)

        self._client.add_event_callback(on_message, RoomMessageText)
        asyncio.create_task(self._client.sync_forever(timeout=30000))
        self._connected = True
        logger.info("[CONCEPT:ECO-4.5] Matrix backend connected.")

    async def disconnect(self) -> None:
        if self._client:
            await self._client.close()
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
            content: dict[str, Any] = {"msgtype": "m.text", "body": text}
            if metadata and "html" in metadata:
                content["format"] = "org.matrix.custom.html"
                content["formatted_body"] = metadata["html"]
            resp = await self._client.room_send(channel_id, "m.room.message", content)
            msg_id = resp.event_id if hasattr(resp, "event_id") else ""
            return SendResult(
                success=True,
                message_id=msg_id,
                platform=PlatformId.MATRIX,
                channel_id=channel_id,
            )
        except Exception as e:
            return SendResult(success=False, platform=PlatformId.MATRIX, error=str(e))

    async def send_reaction(self, channel_id: str, message_id: str, emoji: str) -> None:
        content = {
            "m.relates_to": {
                "rel_type": "m.annotation",
                "event_id": message_id,
                "key": emoji,
            }
        }
        await self._client.room_send(channel_id, "m.reaction", content)

    async def list_channels(self) -> list[Channel]:
        resp = await self._client.joined_rooms()
        rooms = resp.rooms if hasattr(resp, "rooms") else []
        return [Channel(id=r, platform=PlatformId.MATRIX) for r in rooms]

    async def listen(self) -> AsyncIterator[InboundEvent]:
        while self._connected:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                yield event
            except TimeoutError:
                continue
