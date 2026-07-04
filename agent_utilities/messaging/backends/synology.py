"""Synology Chat Backend (CONCEPT:AU-ECO.messaging.native-backend-abstraction). Webhook-based via httpx.

Install: ``pip install agent-utilities[messaging-synology]``
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
    InboundEvent,
    MessagingConfig,
    PlatformId,
    SendResult,
)

logger = logging.getLogger(__name__)


class SynologyChatBackend(MessagingBackend):
    """Synology Chat backend via incoming/outgoing webhooks. CONCEPT:AU-ECO.messaging.native-backend-abstraction"""

    def __init__(self, config: MessagingConfig | None = None) -> None:
        super().__init__(config)
        self._client: Any = None
        self._event_queue: asyncio.Queue[InboundEvent] = asyncio.Queue()

    @property
    def id(self) -> str:
        return "synology"

    @property
    def capabilities(self) -> MessagingCapabilities:
        return CAPABILITY_MATRIX["synology"]

    async def connect(self) -> None:
        """Connect to Synology Chat webhook. CONCEPT:AU-ECO.messaging.native-backend-abstraction"""
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "Install: pip install agent-utilities[messaging-synology]"
            ) from None

        webhook_url = self.config.webhook_url or setting(
            "SYNOLOGY_CHAT_WEBHOOK_URL", ""
        )
        if not webhook_url:
            raise ValueError("Set SYNOLOGY_CHAT_WEBHOOK_URL.")
        self._client = httpx.AsyncClient()
        self._connected = True
        logger.info("[CONCEPT:AU-ECO.messaging.native-backend-abstraction] Synology Chat backend connected.")

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
            webhook = self.config.webhook_url or setting(
                "SYNOLOGY_CHAT_WEBHOOK_URL", ""
            )
            payload = f'payload={{"text": "{text}"}}'
            resp = await self._client.post(
                webhook,
                content=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            return SendResult(
                success=resp.status_code == 200,
                platform=PlatformId.SYNOLOGY,
                channel_id=channel_id,
            )
        except Exception as e:
            return SendResult(success=False, platform=PlatformId.SYNOLOGY, error=str(e))

    async def listen(self) -> AsyncIterator[InboundEvent]:
        while self._connected:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                yield event
            except TimeoutError:
                continue
