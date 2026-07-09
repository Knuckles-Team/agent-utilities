"""Signal Messaging Backend (CONCEPT:AU-ECO.messaging.native-backend-abstraction).

Uses ``semaphore-bot`` (signal-cli bridge) for Signal messaging.

Install: ``pip install agent-utilities[messaging-signal]``

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
    EventType,
    InboundEvent,
    Message,
    MessageDirection,
    MessagingConfig,
    PlatformId,
    SendResult,
)

logger = logging.getLogger(__name__)


class SignalBackend(MessagingBackend):
    """Signal messaging backend via semaphore-bot. CONCEPT:AU-ECO.messaging.native-backend-abstraction"""

    def __init__(self, config: MessagingConfig | None = None) -> None:
        super().__init__(config)
        self._bot: Any = None
        self._event_queue: asyncio.Queue[InboundEvent] = asyncio.Queue()

    @property
    def id(self) -> str:
        return "signal"

    @property
    def capabilities(self) -> MessagingCapabilities:
        return CAPABILITY_MATRIX["signal"]

    async def connect(self) -> None:
        """Connect to Signal via signal-cli. CONCEPT:AU-ECO.messaging.native-backend-abstraction"""
        try:
            from semaphore import Bot
        except ImportError:
            raise ImportError(
                "Install: pip install agent-utilities[messaging-signal]"
            ) from None

        phone = self.config.token or setting("SIGNAL_PHONE_NUMBER", "")
        if not phone:
            raise ValueError("Set SIGNAL_PHONE_NUMBER.")
        self._bot = Bot(phone)

        @self._bot.handler("")
        async def handle(ctx: Any) -> None:
            ev = InboundEvent(
                event_type=EventType.MESSAGE,
                platform=PlatformId.SIGNAL,
                channel_id=str(ctx.message.get_group_id() or ctx.message.source),
                user_id=ctx.message.source,
                user_name=ctx.message.source,
                content=ctx.message.get_body() or "",
                message=Message(
                    content=ctx.message.get_body() or "",
                    author_id=ctx.message.source,
                    platform=PlatformId.SIGNAL,
                    direction=MessageDirection.INBOUND,
                ),
            )
            await self._event_queue.put(ev)

        asyncio.create_task(self._bot.start())
        self._connected = True
        logger.info(
            "[CONCEPT:AU-ECO.messaging.native-backend-abstraction] Signal backend connected."
        )

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
            await self._bot.send_message(channel_id, text)
            return SendResult(
                success=True, platform=PlatformId.SIGNAL, channel_id=channel_id
            )
        except Exception as e:
            return SendResult(success=False, platform=PlatformId.SIGNAL, error=str(e))

    async def listen(self) -> AsyncIterator[InboundEvent]:
        while self._connected:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                yield event
            except TimeoutError:
                continue
