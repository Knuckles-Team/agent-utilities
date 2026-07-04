"""Google Chat Messaging Backend (CONCEPT:AU-ECO.messaging.native-backend-abstraction).

Uses ``google-api-python-client`` for Google Chat spaces/messages.

Install: ``pip install agent-utilities[messaging-googlechat]``

CONCEPT:AU-ECO.messaging.native-backend-abstraction — Native Messaging Backend Abstraction
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
    Thread,
)

logger = logging.getLogger(__name__)


class GoogleChatBackend(MessagingBackend):
    """Google Chat backend via Workspace API. CONCEPT:AU-ECO.messaging.native-backend-abstraction"""

    def __init__(self, config: MessagingConfig | None = None) -> None:
        super().__init__(config)
        self._service: Any = None
        self._event_queue: asyncio.Queue[InboundEvent] = asyncio.Queue()

    @property
    def id(self) -> str:
        return "googlechat"

    @property
    def capabilities(self) -> MessagingCapabilities:
        return CAPABILITY_MATRIX["googlechat"]

    async def connect(self) -> None:
        """Connect via Google Workspace service account. CONCEPT:AU-ECO.messaging.native-backend-abstraction"""
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
        except ImportError:
            raise ImportError(
                "Install: pip install agent-utilities[messaging-googlechat]"
            ) from None

        creds_path = self.config.token or setting("GOOGLE_CHAT_SERVICE_ACCOUNT", "")
        if not creds_path:
            raise ValueError("Set GOOGLE_CHAT_SERVICE_ACCOUNT path.")
        creds = service_account.Credentials.from_service_account_file(
            creds_path, scopes=["https://www.googleapis.com/auth/chat.bot"]
        )
        self._service = build("chat", "v1", credentials=creds)
        self._connected = True
        logger.info("[CONCEPT:AU-ECO.messaging.native-backend-abstraction] Google Chat backend connected.")

    async def send_message(
        self,
        channel_id: str,
        text: str,
        *,
        thread_id: str = "",
        reply_to_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> SendResult:
        """Send to Google Chat space. CONCEPT:AU-ECO.messaging.native-backend-abstraction"""
        try:
            body: dict[str, Any] = {"text": text}
            if thread_id:
                body["thread"] = {"name": thread_id}
            if metadata and "cards" in metadata:
                body["cardsV2"] = metadata["cards"]
            result = await asyncio.to_thread(
                self._service.spaces()
                .messages()
                .create(parent=channel_id, body=body)
                .execute
            )
            return SendResult(
                success=True,
                message_id=result.get("name", ""),
                platform=PlatformId.GOOGLECHAT,
                channel_id=channel_id,
            )
        except Exception as e:
            return SendResult(
                success=False, platform=PlatformId.GOOGLECHAT, error=str(e)
            )

    async def create_thread(
        self, channel_id: str, message_id: str, title: str = ""
    ) -> Thread:
        """Google Chat threads. CONCEPT:AU-ECO.messaging.native-backend-abstraction"""
        return Thread(
            id=message_id,
            parent_message_id=message_id,
            channel_id=channel_id,
            title=title,
        )

    async def list_channels(self) -> list[Channel]:
        """List spaces. CONCEPT:AU-ECO.messaging.native-backend-abstraction"""
        result = await asyncio.to_thread(self._service.spaces().list().execute)
        return [
            Channel(
                id=s["name"],
                name=s.get("displayName", ""),
                platform=PlatformId.GOOGLECHAT,
            )
            for s in result.get("spaces", [])
        ]

    async def listen(self) -> AsyncIterator[InboundEvent]:
        """Yield events (webhook-populated). CONCEPT:AU-ECO.messaging.native-backend-abstraction"""
        while self._connected:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                yield event
            except TimeoutError:
                continue

    async def process_webhook(self, body: dict[str, Any]) -> None:
        """Process Google Chat webhook event. CONCEPT:AU-ECO.messaging.native-backend-abstraction"""
        msg = body.get("message", {})
        if msg:
            event = InboundEvent(
                event_type=EventType.MESSAGE,
                platform=PlatformId.GOOGLECHAT,
                channel_id=body.get("space", {}).get("name", ""),
                user_id=msg.get("sender", {}).get("name", ""),
                user_name=msg.get("sender", {}).get("displayName", ""),
                content=msg.get("text", ""),
                message=Message(
                    id=msg.get("name", ""),
                    content=msg.get("text", ""),
                    platform=PlatformId.GOOGLECHAT,
                    direction=MessageDirection.INBOUND,
                ),
                raw=body,
            )
            await self._event_queue.put(event)
