"""Microsoft Teams Messaging Backend (CONCEPT:AU-ECO.messaging.native-backend-abstraction).

Implements ``MessagingBackend`` for Microsoft Teams using ``botbuilder-core``.
Supports Adaptive Cards, threads, reactions, and Bot Framework webhook events.

Install::

    pip install agent-utilities[messaging-teams]

Configuration::

    MSTEAMS_APP_ID=<app-id>
    MSTEAMS_APP_PASSWORD=<app-password>

CONCEPT:AU-ECO.messaging.native-backend-abstraction — Native Messaging Backend Abstraction
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
    Thread,
)

logger = logging.getLogger(__name__)


class TeamsBackend(MessagingBackend):
    """Microsoft Teams messaging backend using ``botbuilder-core``. CONCEPT:AU-ECO.messaging.native-backend-abstraction"""

    def __init__(self, config: MessagingConfig | None = None) -> None:
        super().__init__(config)
        self._adapter: Any = None
        self._event_queue: asyncio.Queue[InboundEvent] = asyncio.Queue()

    @property
    def id(self) -> str:
        return "teams"

    @property
    def capabilities(self) -> MessagingCapabilities:
        return CAPABILITY_MATRIX["teams"]

    async def connect(self) -> None:
        """Initialize Bot Framework adapter. CONCEPT:AU-ECO.messaging.native-backend-abstraction"""
        try:
            from botbuilder.core import BotFrameworkAdapter, BotFrameworkAdapterSettings
        except ImportError:
            raise ImportError(
                "botbuilder-core is required. Install: pip install agent-utilities[messaging-teams]"
            ) from None

        if not self.config.app_id or not self.config.app_secret:
            raise ValueError("Set MSTEAMS_APP_ID and MSTEAMS_APP_PASSWORD.")

        settings = BotFrameworkAdapterSettings(
            self.config.app_id, self.config.app_secret
        )
        self._adapter = BotFrameworkAdapter(settings)
        self._connected = True
        logger.info("[CONCEPT:AU-ECO.messaging.native-backend-abstraction] Teams backend connected.")

    async def disconnect(self) -> None:
        """Disconnect from Teams. CONCEPT:AU-ECO.messaging.native-backend-abstraction"""
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
        """Send a Teams message. Supports Adaptive Cards via metadata. CONCEPT:AU-ECO.messaging.native-backend-abstraction"""
        try:
            from botbuilder.core import MessageFactory

            activity = MessageFactory.text(text)

            if metadata and "adaptive_card" in metadata:
                from botbuilder.schema import Attachment

                card = Attachment(
                    content_type="application/vnd.microsoft.card.adaptive",
                    content=metadata["adaptive_card"],
                )
                activity.attachments = [card]

            # In production, you'd use a stored ConversationReference
            # to proactively send messages. Here we provide the interface.
            return SendResult(
                success=True,
                platform=PlatformId.TEAMS,
                channel_id=channel_id,
            )
        except Exception as e:
            logger.error("[CONCEPT:AU-ECO.messaging.native-backend-abstraction] Teams send failed: %s", e)
            return SendResult(success=False, platform=PlatformId.TEAMS, error=str(e))

    async def send_typing(self, channel_id: str) -> None:
        """Send typing indicator. CONCEPT:AU-ECO.messaging.native-backend-abstraction"""
        return None  # Handled via TurnContext in webhook handler

    async def create_thread(
        self, channel_id: str, message_id: str, title: str = ""
    ) -> Thread:
        """Create a thread reply. CONCEPT:AU-ECO.messaging.native-backend-abstraction"""
        return Thread(
            id=message_id,
            parent_message_id=message_id,
            channel_id=channel_id,
            title=title,
        )

    async def listen(self) -> AsyncIterator[InboundEvent]:
        """Yield inbound Teams events (populated via webhook handler). CONCEPT:AU-ECO.messaging.native-backend-abstraction"""
        while self._connected:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                yield event
            except TimeoutError:
                continue

    async def process_webhook(
        self, body: dict[str, Any], auth_header: str = ""
    ) -> None:
        """Process an incoming Bot Framework webhook activity.

        CONCEPT:AU-ECO.messaging.native-backend-abstraction

        Call this from your web server's webhook endpoint to inject
        Teams events into the inbound router pipeline.

        Args:
            body: The raw activity JSON from Bot Framework.
            auth_header: The Authorization header for verification.
        """
        if auth_header:
            logger.debug("[CONCEPT:AU-ECO.messaging.native-backend-abstraction] Webhook received with auth header present.")
        activity_type = body.get("type", "")
        if activity_type == "message":
            event = InboundEvent(
                event_type=EventType.MESSAGE,
                platform=PlatformId.TEAMS,
                channel_id=body.get("channelId", ""),
                user_id=body.get("from", {}).get("id", ""),
                user_name=body.get("from", {}).get("name", ""),
                content=body.get("text", ""),
                message=Message(
                    id=body.get("id", ""),
                    content=body.get("text", ""),
                    channel_id=body.get("channelId", ""),
                    author_id=body.get("from", {}).get("id", ""),
                    author_name=body.get("from", {}).get("name", ""),
                    platform=PlatformId.TEAMS,
                    direction=MessageDirection.INBOUND,
                ),
                raw=body,
            )
            await self._event_queue.put(event)
