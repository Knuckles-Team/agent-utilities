"""Twitch Messaging Backend (CONCEPT:ECO-4.5).

Uses ``twitchio`` for Twitch IRC chat and EventSub.

Install: ``pip install agent-utilities[messaging-twitch]``
CONCEPT:ECO-4.5 — Native Messaging Backend Abstraction
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


class TwitchBackend(MessagingBackend):
    """Twitch chat backend via twitchio. CONCEPT:ECO-4.5"""

    def __init__(self, config: MessagingConfig | None = None) -> None:
        super().__init__(config)
        self._bot: Any = None
        self._event_queue: asyncio.Queue[InboundEvent] = asyncio.Queue()

    @property
    def id(self) -> str:
        return "twitch"

    @property
    def capabilities(self) -> MessagingCapabilities:
        return CAPABILITY_MATRIX["twitch"]

    async def connect(self) -> None:
        """Connect to Twitch IRC. CONCEPT:ECO-4.5"""
        try:
            from twitchio.ext import commands
        except ImportError:
            raise ImportError(
                "Install: pip install agent-utilities[messaging-twitch]"
            ) from None
        import os

        token = self.config.token or os.environ.get("TWITCH_OAUTH_TOKEN", "")
        channels = self.config.extra.get(
            "channels", os.environ.get("TWITCH_CHANNELS", "").split(",")
        )
        if not token:
            raise ValueError("Set TWITCH_OAUTH_TOKEN.")

        class AgentBot(commands.Bot):
            def __init__(bot_self, **kwargs: Any) -> None:
                super().__init__(**kwargs)

            async def event_message(bot_self, message: Any) -> None:
                if message.echo:
                    return
                ev = InboundEvent(
                    event_type=EventType.MESSAGE,
                    platform=PlatformId.TWITCH,
                    channel_id=message.channel.name if message.channel else "",
                    user_id=message.author.name if message.author else "",
                    user_name=message.author.display_name if message.author else "",
                    content=message.content,
                    message=Message(
                        id=message.id or "",
                        content=message.content,
                        channel_id=message.channel.name if message.channel else "",
                        platform=PlatformId.TWITCH,
                        direction=MessageDirection.INBOUND,
                    ),
                )
                await self._event_queue.put(ev)

        self._bot = AgentBot(
            token=token,
            prefix="!",
            initial_channels=[c.strip() for c in channels if c.strip()],
        )
        asyncio.create_task(self._bot.start())
        await asyncio.sleep(2)
        self._connected = True
        logger.info("[CONCEPT:ECO-4.5] Twitch backend connected.")

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
            ch = self._bot.get_channel(channel_id)
            if ch:
                await ch.send(text)
            return SendResult(
                success=True, platform=PlatformId.TWITCH, channel_id=channel_id
            )
        except Exception as e:
            return SendResult(success=False, platform=PlatformId.TWITCH, error=str(e))

    async def listen(self) -> AsyncIterator[InboundEvent]:
        while self._connected:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                yield event
            except TimeoutError:
                continue
