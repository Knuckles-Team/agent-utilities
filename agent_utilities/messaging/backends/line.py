"""LINE Messaging Backend (CONCEPT:ECO-4.0).

Uses ``line-bot-sdk`` for LINE Messaging API.

Install: ``pip install agent-utilities[messaging-line]``

CONCEPT:ECO-4.0 — Native Messaging Backend Abstraction
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from agent_utilities.messaging.base import MessagingBackend
from agent_utilities.messaging.capabilities import (
    CAPABILITY_MATRIX,
    MessagingCapabilities,
)
from agent_utilities.messaging.models import (
    EventType,
    InboundEvent,
    Message,
    MessageDirection,
    MessagingConfig,
    PlatformId,
    SendResult,
)

logger = logging.getLogger(__name__)


class LINEBackend(MessagingBackend):
    """LINE messaging backend via line-bot-sdk. CONCEPT:ECO-4.0"""

    def __init__(self, config: MessagingConfig | None = None) -> None:
        super().__init__(config)
        self._api: Any = None
        self._event_queue: asyncio.Queue[InboundEvent] = asyncio.Queue()

    @property
    def id(self) -> str:
        return "line"

    @property
    def capabilities(self) -> MessagingCapabilities:
        return CAPABILITY_MATRIX["line"]

    async def connect(self) -> None:
        """Connect to LINE Messaging API. CONCEPT:ECO-4.0"""
        try:
            from linebot.v3.messaging import ApiClient, Configuration, MessagingApi
        except ImportError:
            raise ImportError(
                "Install: pip install agent-utilities[messaging-line]"
            ) from None
        import os

        token = self.config.token or os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
        if not token:
            raise ValueError("Set LINE_CHANNEL_ACCESS_TOKEN.")
        config = Configuration(access_token=token)
        self._api = MessagingApi(ApiClient(config))
        self._connected = True
        logger.info("[CONCEPT:ECO-4.0] LINE backend connected.")

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
            from linebot.v3.messaging import PushMessageRequest, TextMessage

            request = PushMessageRequest(
                to=channel_id, messages=[TextMessage(text=text)]
            )
            await asyncio.to_thread(self._api.push_message, request)
            return SendResult(
                success=True, platform=PlatformId.LINE, channel_id=channel_id
            )
        except Exception as e:
            return SendResult(success=False, platform=PlatformId.LINE, error=str(e))

    async def listen(self) -> AsyncIterator[InboundEvent]:
        """Yield events (webhook-populated). CONCEPT:ECO-4.0"""
        while self._connected:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                yield event
            except TimeoutError:
                continue

    async def process_webhook(self, body: dict[str, Any]) -> None:
        """Process LINE webhook event. CONCEPT:ECO-4.0"""
        for event in body.get("events", []):
            if event.get("type") == "message":
                msg = event.get("message", {})
                ev = InboundEvent(
                    event_type=EventType.MESSAGE,
                    platform=PlatformId.LINE,
                    channel_id=event.get("source", {}).get("userId", ""),
                    user_id=event.get("source", {}).get("userId", ""),
                    content=msg.get("text", ""),
                    message=Message(
                        id=msg.get("id", ""),
                        content=msg.get("text", ""),
                        platform=PlatformId.LINE,
                        direction=MessageDirection.INBOUND,
                    ),
                    raw=event,
                )
                await self._event_queue.put(ev)
