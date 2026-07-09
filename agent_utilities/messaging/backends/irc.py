"""IRC Messaging Backend (CONCEPT:AU-ECO.messaging.native-backend-abstraction).

Uses the ``irc`` library for IRC protocol messaging.

Install: ``pip install agent-utilities[messaging-irc]``

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


class IRCBackend(MessagingBackend):
    """IRC messaging backend. CONCEPT:AU-ECO.messaging.native-backend-abstraction"""

    def __init__(self, config: MessagingConfig | None = None) -> None:
        super().__init__(config)
        self._connection: Any = None
        self._reactor: Any = None
        self._event_queue: asyncio.Queue[InboundEvent] = asyncio.Queue()

    @property
    def id(self) -> str:
        return "irc"

    @property
    def capabilities(self) -> MessagingCapabilities:
        return CAPABILITY_MATRIX["irc"]

    async def connect(self) -> None:
        """Connect to IRC server. CONCEPT:AU-ECO.messaging.native-backend-abstraction"""
        try:
            import irc.client
        except ImportError:
            raise ImportError(
                "Install: pip install agent-utilities[messaging-irc]"
            ) from None

        server = self.config.extra.get("server", setting("IRC_SERVER", ""))
        port = int(self.config.extra.get("port", setting("IRC_PORT", "6667")))
        nick = self.config.extra.get("nickname", setting("IRC_NICKNAME", "agent_bot"))
        if not server:
            raise ValueError("Set IRC_SERVER.")
        self._reactor = irc.client.Reactor()
        self._connection = self._reactor.server().connect(server, port, nick)

        def on_pubmsg(connection: Any, event: Any) -> None:
            ev = InboundEvent(
                event_type=EventType.MESSAGE,
                platform=PlatformId.IRC,
                channel_id=event.target,
                user_id=event.source.nick,
                user_name=event.source.nick,
                content=event.arguments[0] if event.arguments else "",
                message=Message(
                    content=event.arguments[0] if event.arguments else "",
                    channel_id=event.target,
                    author_id=event.source.nick,
                    platform=PlatformId.IRC,
                    direction=MessageDirection.INBOUND,
                ),
            )
            asyncio.get_event_loop().call_soon_threadsafe(
                self._event_queue.put_nowait, ev
            )

        self._connection.add_global_handler("pubmsg", on_pubmsg)
        self._connection.add_global_handler("privmsg", on_pubmsg)

        # Auto-join channels from config
        channels = self.config.extra.get("channels", [])
        for ch in channels:
            self._connection.join(ch)

        asyncio.create_task(asyncio.to_thread(self._reactor.process_forever))
        self._connected = True
        logger.info(
            "[CONCEPT:AU-ECO.messaging.native-backend-abstraction] IRC backend connected to %s.",
            server,
        )

    async def disconnect(self) -> None:
        if self._connection:
            self._connection.disconnect("Agent shutting down")
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
            # IRC has a 512-byte limit per message, split if needed
            for line in text.split("\n"):
                while len(line) > 400:
                    self._connection.privmsg(channel_id, line[:400])
                    line = line[400:]
                if line:
                    self._connection.privmsg(channel_id, line)
            return SendResult(
                success=True, platform=PlatformId.IRC, channel_id=channel_id
            )
        except Exception as e:
            return SendResult(success=False, platform=PlatformId.IRC, error=str(e))

    async def listen(self) -> AsyncIterator[InboundEvent]:
        while self._connected:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                yield event
            except TimeoutError:
                continue
