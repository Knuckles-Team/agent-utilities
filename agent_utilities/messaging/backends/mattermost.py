"""Mattermost Messaging Backend (CONCEPT:ECO-4.5).

Uses ``mattermostdriver`` for WebSocket-based real-time messaging.

Install: ``pip install agent-utilities[messaging-mattermost]``

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
    Channel,
    InboundEvent,
    MessagingConfig,
    PlatformId,
    SendResult,
    Thread,
)

logger = logging.getLogger(__name__)


class MattermostBackend(MessagingBackend):
    """Mattermost backend via mattermostdriver. CONCEPT:ECO-4.5"""

    def __init__(self, config: MessagingConfig | None = None) -> None:
        super().__init__(config)
        self._driver: Any = None
        self._event_queue: asyncio.Queue[InboundEvent] = asyncio.Queue()

    @property
    def id(self) -> str:
        return "mattermost"

    @property
    def capabilities(self) -> MessagingCapabilities:
        return CAPABILITY_MATRIX["mattermost"]

    async def connect(self) -> None:
        """Connect to Mattermost server. CONCEPT:ECO-4.5"""
        try:
            from mattermostdriver import Driver
        except ImportError:
            raise ImportError(
                "Install: pip install agent-utilities[messaging-mattermost]"
            ) from None
        import os

        url = self.config.extra.get("url", os.environ.get("MATTERMOST_URL", ""))
        if not url or not self.config.token:
            raise ValueError("Set MATTERMOST_URL and MATTERMOST_TOKEN.")
        self._driver = Driver(
            {"url": url, "token": self.config.token, "scheme": "https", "port": 443}
        )
        await asyncio.to_thread(self._driver.login)
        self._connected = True
        logger.info("[CONCEPT:ECO-4.5] Mattermost backend connected.")

    async def disconnect(self) -> None:
        if self._driver:
            await asyncio.to_thread(self._driver.logout)
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
        """Send to Mattermost channel. CONCEPT:ECO-4.5"""
        try:
            payload: dict[str, Any] = {"channel_id": channel_id, "message": text}
            if thread_id:
                payload["root_id"] = thread_id
            result = await asyncio.to_thread(self._driver.posts.create_post, payload)
            return SendResult(
                success=True,
                message_id=result.get("id", ""),
                platform=PlatformId.MATTERMOST,
                channel_id=channel_id,
            )
        except Exception as e:
            return SendResult(
                success=False, platform=PlatformId.MATTERMOST, error=str(e)
            )

    async def send_reaction(self, channel_id: str, message_id: str, emoji: str) -> None:
        """Add reaction. CONCEPT:ECO-4.5"""
        user = await asyncio.to_thread(self._driver.users.get_user, "me")
        await asyncio.to_thread(
            self._driver.reactions.create_reaction,
            {"user_id": user["id"], "post_id": message_id, "emoji_name": emoji},
        )

    async def create_thread(
        self, channel_id: str, message_id: str, title: str = ""
    ) -> Thread:
        return Thread(
            id=message_id,
            parent_message_id=message_id,
            channel_id=channel_id,
            title=title,
        )

    async def list_channels(self) -> list[Channel]:
        """List Mattermost channels. CONCEPT:ECO-4.5"""
        user = await asyncio.to_thread(self._driver.users.get_user, "me")
        teams = await asyncio.to_thread(self._driver.teams.get_user_teams, user["id"])
        channels = []
        for team in teams:
            chs = await asyncio.to_thread(
                self._driver.channels.get_channels_for_user, user["id"], team["id"]
            )
            for ch in chs:
                channels.append(
                    Channel(
                        id=ch["id"],
                        name=ch.get("display_name", ""),
                        platform=PlatformId.MATTERMOST,
                        member_count=ch.get("total_msg_count", 0),
                    )
                )
        return channels

    async def listen(self) -> AsyncIterator[InboundEvent]:
        while self._connected:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                yield event
            except TimeoutError:
                continue
