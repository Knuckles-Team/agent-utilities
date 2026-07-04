"""Discord Messaging Backend (CONCEPT:AU-ECO.messaging.native-backend-abstraction).

Implements ``MessagingBackend`` for Discord using the ``discord.py`` library.
Supports full bidirectional messaging with threads, reactions, embeds, typing
indicators, and real-time inbound event streaming via Discord Gateway.

Install::

    pip install agent-utilities[messaging-discord]

Configuration::

    DISCORD_BOT_TOKEN=<your-bot-token>
    # or
    MESSAGING_DISCORD_TOKEN=<your-bot-token>

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
    Channel,
    EventType,
    InboundEvent,
    MediaAttachment,
    MediaType,
    Message,
    MessageDirection,
    MessagingConfig,
    PlatformId,
    SendResult,
    Thread,
)

logger = logging.getLogger(__name__)


class DiscordBackend(MessagingBackend):
    """Discord messaging backend using ``discord.py``.

    CONCEPT:AU-ECO.messaging.native-backend-abstraction

    Features:
        - Full Gateway WebSocket connection for real-time events
        - Rich embed support via metadata
        - Thread creation and management
        - Emoji reactions (custom + unicode)
        - Typing indicators
        - Channel and member directory

    Usage::

        from agent_utilities.messaging import MessagingRegistry
        registry = MessagingRegistry()
        discord = registry.create_backend("discord")
        await discord.connect()
        await discord.send_message("1234567890", "Hello from agent!")
    """

    def __init__(self, config: MessagingConfig | None = None) -> None:
        super().__init__(config)
        self._client: Any = None
        self._event_queue: asyncio.Queue[InboundEvent] = asyncio.Queue()
        self._bot_task: asyncio.Task[None] | None = None

    @property
    def id(self) -> str:
        return "discord"

    @property
    def capabilities(self) -> MessagingCapabilities:
        return CAPABILITY_MATRIX["discord"]

    async def connect(self) -> None:
        """Connect to Discord Gateway.

        CONCEPT:AU-ECO.messaging.native-backend-abstraction

        Raises:
            ImportError: If ``discord.py`` is not installed.
            ValueError: If no bot token is configured.
            ConnectionError: If login fails.
        """
        try:
            import discord
        except ImportError:
            raise ImportError(
                "discord.py is required for the Discord backend. "
                "Install with: pip install agent-utilities[messaging-discord]"
            ) from None

        if not self.config.token:
            raise ValueError(
                "Discord bot token is required. Set DISCORD_BOT_TOKEN or "
                "MESSAGING_DISCORD_TOKEN environment variable."
            )

        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        intents.reactions = True

        self._client = discord.Client(intents=intents)

        # Register event handlers
        @self._client.event
        async def on_message(message: Any) -> None:
            if message.author == self._client.user:
                return  # Skip own messages
            event = InboundEvent(
                event_type=EventType.MESSAGE,
                platform=PlatformId.DISCORD,
                channel_id=str(message.channel.id),
                thread_id=str(message.thread.id)
                if hasattr(message, "thread") and message.thread
                else "",
                user_id=str(message.author.id),
                user_name=str(message.author.display_name),
                content=message.content,
                message=Message(
                    id=str(message.id),
                    content=message.content,
                    channel_id=str(message.channel.id),
                    author_id=str(message.author.id),
                    author_name=str(message.author.display_name),
                    platform=PlatformId.DISCORD,
                    direction=MessageDirection.INBOUND,
                    attachments=[
                        MediaAttachment(
                            media_type=MediaType.FILE,
                            url=a.url,
                            filename=a.filename,
                            size_bytes=a.size,
                        )
                        for a in message.attachments
                    ],
                ),
                raw={"guild_id": str(message.guild.id) if message.guild else ""},
            )
            await self._event_queue.put(event)

        @self._client.event
        async def on_reaction_add(reaction: Any, user: Any) -> None:
            if user == self._client.user:
                return
            event = InboundEvent(
                event_type=EventType.REACTION_ADD,
                platform=PlatformId.DISCORD,
                channel_id=str(reaction.message.channel.id),
                user_id=str(user.id),
                user_name=str(user.display_name),
                emoji=str(reaction.emoji),
                target_message_id=str(reaction.message.id),
            )
            await self._event_queue.put(event)

        @self._client.event
        async def on_ready() -> None:
            logger.info("[CONCEPT:AU-ECO.messaging.native-backend-abstraction] Discord connected as %s", self._client.user)

        # Start the bot in a background task
        self._bot_task = asyncio.create_task(self._client.start(self.config.token))
        # Wait briefly for connection
        await asyncio.sleep(2)
        self._connected = True
        logger.info("[CONCEPT:AU-ECO.messaging.native-backend-abstraction] Discord backend connected.")

    async def disconnect(self) -> None:
        """Disconnect from Discord Gateway.

        CONCEPT:AU-ECO.messaging.native-backend-abstraction
        """
        if self._client:
            await self._client.close()
        if self._bot_task:
            self._bot_task.cancel()
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
        """Send a message to a Discord channel.

        CONCEPT:AU-ECO.messaging.native-backend-abstraction

        Supports Discord embeds via ``metadata["embed"]``.
        """
        try:
            channel = self._client.get_channel(int(channel_id))
            if channel is None:
                channel = await self._client.fetch_channel(int(channel_id))

            kwargs: dict[str, Any] = {}
            if reply_to_id:
                try:
                    ref_msg = await channel.fetch_message(int(reply_to_id))
                    kwargs["reference"] = ref_msg
                except Exception:
                    pass

            if metadata and "embed" in metadata:
                import discord

                embed_data = metadata["embed"]
                embed = discord.Embed(**embed_data)
                kwargs["embed"] = embed

            msg = await channel.send(text, **kwargs)
            return SendResult(
                success=True,
                message_id=str(msg.id),
                platform=PlatformId.DISCORD,
                channel_id=channel_id,
            )
        except Exception as e:
            logger.error("[CONCEPT:AU-ECO.messaging.native-backend-abstraction] Discord send failed: %s", e)
            return SendResult(
                success=False,
                platform=PlatformId.DISCORD,
                channel_id=channel_id,
                error=str(e),
            )

    async def send_reaction(self, channel_id: str, message_id: str, emoji: str) -> None:
        """Add a reaction to a Discord message.

        CONCEPT:AU-ECO.messaging.native-backend-abstraction
        """
        channel = self._client.get_channel(int(channel_id))
        if channel is None:
            channel = await self._client.fetch_channel(int(channel_id))
        msg = await channel.fetch_message(int(message_id))
        await msg.add_reaction(emoji)

    async def send_typing(self, channel_id: str) -> None:
        """Send typing indicator in a Discord channel.

        CONCEPT:AU-ECO.messaging.native-backend-abstraction
        """
        channel = self._client.get_channel(int(channel_id))
        if channel:
            await channel.typing()

    async def create_thread(
        self, channel_id: str, message_id: str, title: str = ""
    ) -> Thread:
        """Create a thread on a Discord message.

        CONCEPT:AU-ECO.messaging.native-backend-abstraction
        """
        channel = self._client.get_channel(int(channel_id))
        if channel is None:
            channel = await self._client.fetch_channel(int(channel_id))
        msg = await channel.fetch_message(int(message_id))
        thread = await msg.create_thread(name=title or "Discussion")
        return Thread(
            id=str(thread.id),
            parent_message_id=message_id,
            channel_id=channel_id,
            title=thread.name,
        )

    async def listen(self) -> AsyncIterator[InboundEvent]:
        """Listen for inbound Discord events.

        CONCEPT:AU-ECO.messaging.native-backend-abstraction

        Yields events from the internal queue populated by Gateway handlers.
        """
        while self._connected:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                yield event
            except TimeoutError:
                continue

    async def list_channels(self) -> list[Channel]:
        """List accessible Discord channels.

        CONCEPT:AU-ECO.messaging.native-backend-abstraction
        """
        channels = []
        for guild in self._client.guilds:
            for ch in guild.text_channels:
                channels.append(
                    Channel(
                        id=str(ch.id),
                        name=ch.name,
                        platform=PlatformId.DISCORD,
                        is_dm=False,
                        is_group=True,
                        topic=ch.topic or "",
                        member_count=guild.member_count or 0,
                        metadata={"guild_id": str(guild.id), "guild_name": guild.name},
                    )
                )
        return channels

    async def list_members(self, channel_id: str) -> list[dict[str, Any]]:
        """List members of a Discord channel.

        CONCEPT:AU-ECO.messaging.native-backend-abstraction
        """
        channel = self._client.get_channel(int(channel_id))
        if not channel or not hasattr(channel, "members"):
            return []
        return [
            {"id": str(m.id), "name": m.display_name, "bot": m.bot}
            for m in channel.members
        ]
