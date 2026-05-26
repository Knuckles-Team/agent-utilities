"""Slack Messaging Backend (CONCEPT:ECO-4.0).

Implements ``MessagingBackend`` for Slack using ``slack-bolt`` with Socket Mode
for real-time inbound events. Supports Block Kit rich formatting, threads,
reactions, and typing indicators.

Install::

    pip install agent-utilities[messaging-slack]

Configuration::

    SLACK_BOT_TOKEN=xoxb-...
    SLACK_APP_TOKEN=xapp-...   # Required for Socket Mode

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


class SlackBackend(MessagingBackend):
    """Slack messaging backend using ``slack-bolt``.

    CONCEPT:ECO-4.0
    """

    def __init__(self, config: MessagingConfig | None = None) -> None:
        super().__init__(config)
        self._app: Any = None
        self._client: Any = None
        self._event_queue: asyncio.Queue[InboundEvent] = asyncio.Queue()

    @property
    def id(self) -> str:
        return "slack"

    @property
    def capabilities(self) -> MessagingCapabilities:
        return CAPABILITY_MATRIX["slack"]

    async def connect(self) -> None:
        """Connect to Slack via Socket Mode. CONCEPT:ECO-4.0"""
        try:
            from slack_bolt.adapter.socket_mode.async_handler import (
                AsyncSocketModeHandler,
            )
            from slack_bolt.async_app import AsyncApp
        except ImportError:
            raise ImportError(
                "slack-bolt is required. Install: pip install agent-utilities[messaging-slack]"
            ) from None

        if not self.config.token:
            raise ValueError("Set SLACK_BOT_TOKEN or MESSAGING_SLACK_TOKEN.")

        app_token = self.config.extra.get("app_token", "")
        import os

        if not app_token:
            app_token = os.environ.get("SLACK_APP_TOKEN", "")

        self._app = AsyncApp(token=self.config.token)
        self._client = self._app.client

        @self._app.message("")
        async def handle_message(message: dict[str, Any]) -> None:
            event = InboundEvent(
                event_type=EventType.MESSAGE,
                platform=PlatformId.SLACK,
                channel_id=message.get("channel", ""),
                thread_id=message.get("thread_ts", ""),
                user_id=message.get("user", ""),
                user_name=message.get("user", ""),
                content=message.get("text", ""),
                message=Message(
                    id=message.get("ts", ""),
                    content=message.get("text", ""),
                    channel_id=message.get("channel", ""),
                    author_id=message.get("user", ""),
                    platform=PlatformId.SLACK,
                    direction=MessageDirection.INBOUND,
                ),
                raw=message,
            )
            await self._event_queue.put(event)

        @self._app.event("reaction_added")
        async def handle_reaction(event: dict[str, Any], **_: Any) -> None:
            ev = InboundEvent(
                event_type=EventType.REACTION_ADD,
                platform=PlatformId.SLACK,
                channel_id=event.get("item", {}).get("channel", ""),
                user_id=event.get("user", ""),
                emoji=event.get("reaction", ""),
                target_message_id=event.get("item", {}).get("ts", ""),
                raw=event,
            )
            await self._event_queue.put(ev)

        if app_token:
            handler = AsyncSocketModeHandler(self._app, app_token)
            asyncio.create_task(handler.start_async())

        self._connected = True
        logger.info("[CONCEPT:ECO-4.0] Slack backend connected.")

    async def disconnect(self) -> None:
        """Disconnect from Slack. CONCEPT:ECO-4.0"""
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
        """Send a Slack message. Supports Block Kit via metadata['blocks']. CONCEPT:ECO-4.0"""
        try:
            kwargs: dict[str, Any] = {"channel": channel_id, "text": text}
            if thread_id:
                kwargs["thread_ts"] = thread_id
            if metadata and "blocks" in metadata:
                kwargs["blocks"] = metadata["blocks"]

            response = await self._client.chat_postMessage(**kwargs)
            return SendResult(
                success=True,
                message_id=response.get("ts", ""),
                platform=PlatformId.SLACK,
                channel_id=channel_id,
            )
        except Exception as e:
            logger.error("[CONCEPT:ECO-4.0] Slack send failed: %s", e)
            return SendResult(success=False, platform=PlatformId.SLACK, error=str(e))

    async def send_reaction(self, channel_id: str, message_id: str, emoji: str) -> None:
        """Add a reaction. CONCEPT:ECO-4.0"""
        await self._client.reactions_add(
            channel=channel_id, timestamp=message_id, name=emoji
        )

    async def send_typing(self, channel_id: str) -> None:
        """Indicate typing. CONCEPT:ECO-4.0"""
        # Slack doesn't have a direct typing API for bots
        pass

    async def create_thread(
        self, channel_id: str, message_id: str, title: str = ""
    ) -> Thread:
        """Reply in thread to create it. CONCEPT:ECO-4.0"""
        return Thread(
            id=message_id,
            parent_message_id=message_id,
            channel_id=channel_id,
            title=title,
        )

    async def listen(self) -> AsyncIterator[InboundEvent]:
        """Yield inbound Slack events. CONCEPT:ECO-4.0"""
        while self._connected:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                yield event
            except TimeoutError:
                continue

    async def list_channels(self) -> list[Channel]:
        """List Slack channels. CONCEPT:ECO-4.0"""
        result = await self._client.conversations_list(
            types="public_channel,private_channel"
        )
        return [
            Channel(
                id=ch["id"],
                name=ch.get("name", ""),
                platform=PlatformId.SLACK,
                is_dm=ch.get("is_im", False),
                topic=ch.get("topic", {}).get("value", ""),
                member_count=ch.get("num_members", 0),
            )
            for ch in result.get("channels", [])
        ]
