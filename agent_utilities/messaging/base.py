"""Abstract Messaging Backend Protocol (CONCEPT:ECO-4.5).

Defines the ``MessagingBackend`` ABC that all 17 platform backends must
implement. This is the Python equivalent of OpenClaw's ``ChannelPlugin<T>``
interface, adapted for async Python idioms.

Architecture follows the proven ``TraceBackend`` pattern from
``harness/trace_backend.py``:

- Abstract methods define the required contract
- Concrete methods provide sensible defaults
- A factory function auto-detects the best backend

CONCEPT:ECO-4.5 — Native Messaging Backend Abstraction

See Also:
    - OpenClaw ``src/channels/plugins/types.plugin.ts`` for the TypeScript
      equivalent that this design is modeled after.
    - ``harness/trace_backend.py`` for the Python ABC pattern we follow.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from agent_utilities.messaging.capabilities import MessagingCapabilities
from agent_utilities.messaging.models import (
    Channel,
    InboundEvent,
    MediaAttachment,
    MessagingConfig,
    SendResult,
    Thread,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class MessagingBackend(ABC):
    """Abstract base class for all messaging platform backends.

    CONCEPT:ECO-4.5 — Native Messaging Backend Abstraction

    Every messaging platform (Discord, Slack, Telegram, etc.) implements
    this interface to provide a unified messaging surface for agents.
    The design is inspired by OpenClaw's ``ChannelPlugin<T>`` contract
    but follows Python conventions and Pydantic data models.

    Subclasses must implement:
        - ``id`` property: Platform identifier (e.g., ``"discord"``)
        - ``capabilities`` property: What the platform supports
        - ``connect()``: Establish connection to the platform
        - ``send_message()``: Send a text message

    Optional overrides for richer functionality:
        - ``disconnect()``: Clean shutdown
        - ``listen()``: Inbound event stream
        - ``send_media()``: Send media attachments
        - ``send_reaction()``: Add emoji reactions
        - ``send_typing()``: Typing indicators
        - ``reply_to()``: Reply to a specific message
        - ``create_thread()``: Create a new thread
        - ``list_channels()``: Channel directory
        - ``list_members()``: Member directory
        - ``health_check()``: Backend health status

    Example::

        class DiscordBackend(MessagingBackend):
            @property
            def id(self) -> str:
                return "discord"

            @property
            def capabilities(self) -> MessagingCapabilities:
                return CAPABILITY_MATRIX["discord"]

            async def connect(self) -> None:
                self._client = discord.Client(...)
                await self._client.login(self.config.token)

            async def send_message(self, channel_id, text, **kw) -> SendResult:
                ch = self._client.get_channel(int(channel_id))
                msg = await ch.send(text)
                return SendResult(success=True, message_id=str(msg.id))
    """

    def __init__(self, config: MessagingConfig | None = None) -> None:
        """Initialize the backend with optional configuration.

        Args:
            config: Platform-specific configuration. If not provided,
                the backend should attempt to auto-configure from
                environment variables.
        """
        self.config = config or MessagingConfig()
        self._connected = False

    # ── Required Properties ──────────────────────────────────────────

    @property
    @abstractmethod
    def id(self) -> str:
        """Platform identifier (e.g., ``"discord"``, ``"slack"``).

        Must match the ``PlatformId`` enum value and the entry-point
        name in ``pyproject.toml``.
        """

    @property
    @abstractmethod
    def capabilities(self) -> MessagingCapabilities:
        """Declare what this platform supports.

        Returns a ``MessagingCapabilities`` dataclass indicating
        which features (threads, reactions, media, polls, etc.)
        are available on this platform.
        """

    # ── Lifecycle ────────────────────────────────────────────────────

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the messaging platform.

        This should authenticate with the platform API, start any
        gateway/WebSocket connections, and prepare the backend for
        sending and receiving messages.

        Raises:
            ConnectionError: If authentication or connection fails.
            ValueError: If required configuration is missing.
        """

    async def disconnect(self) -> None:
        """Gracefully disconnect from the messaging platform.

        Subclasses should override this to close WebSocket connections,
        flush pending messages, and release resources.
        """
        self._connected = False
        logger.info("[CONCEPT:ECO-4.5] %s backend disconnected.", self.id)

    @property
    def is_connected(self) -> bool:
        """Whether the backend is currently connected and operational."""
        return self._connected

    async def health_check(self) -> bool:
        """Check if the backend is available and properly configured.

        Returns:
            True if the backend can serve requests.
        """
        return self._connected

    # ── Outbound Messaging ───────────────────────────────────────────

    @abstractmethod
    async def send_message(
        self,
        channel_id: str,
        text: str,
        *,
        thread_id: str = "",
        reply_to_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> SendResult:
        """Send a text message to a channel.

        CONCEPT:ECO-4.5

        Args:
            channel_id: Platform-specific channel/conversation identifier.
            text: Message text content.
            thread_id: Optional thread to send within.
            reply_to_id: Optional message ID to reply to.
            metadata: Platform-specific options (e.g., Discord embeds,
                Slack blocks, Teams Adaptive Cards).

        Returns:
            ``SendResult`` with success status and platform message ID.
        """

    async def send_media(
        self,
        channel_id: str,
        attachment: MediaAttachment,
        *,
        caption: str = "",
        thread_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> SendResult:
        """Send a media attachment to a channel.

        CONCEPT:ECO-4.5

        Default implementation sends the caption as text with the media
        URL appended. Override for native media upload support.

        Args:
            channel_id: Target channel identifier.
            attachment: Media attachment to send.
            caption: Optional caption text.
            thread_id: Optional thread to send within.
            metadata: Platform-specific options.

        Returns:
            ``SendResult`` with success status.
        """
        text = f"{caption}\n{attachment.url}" if caption else attachment.url
        return await self.send_message(
            channel_id, text, thread_id=thread_id, metadata=metadata
        )

    async def send_reaction(
        self,
        channel_id: str,
        message_id: str,
        emoji: str,
    ) -> None:
        """Add a reaction to a message.

        CONCEPT:ECO-4.5

        Args:
            channel_id: Channel containing the message.
            message_id: Message to react to.
            emoji: Emoji string (unicode or platform-specific format).

        Raises:
            NotImplementedError: If the platform doesn't support reactions.
        """
        raise NotImplementedError(f"{self.id} backend does not support reactions.")

    async def send_typing(self, channel_id: str) -> None:
        """Send a typing indicator to a channel.

        CONCEPT:ECO-4.5

        Args:
            channel_id: Channel to show typing in.
        """
        logger.debug("[CONCEPT:ECO-4.5] %s: typing indicator not supported.", self.id)

    # ── Threading ────────────────────────────────────────────────────

    async def reply_to(
        self,
        channel_id: str,
        message_id: str,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> SendResult:
        """Reply to a specific message.

        CONCEPT:ECO-4.5

        Default implementation calls ``send_message`` with ``reply_to_id``.

        Args:
            channel_id: Channel containing the original message.
            message_id: Message to reply to.
            text: Reply text content.
            metadata: Platform-specific options.

        Returns:
            ``SendResult`` with success status.
        """
        return await self.send_message(
            channel_id, text, reply_to_id=message_id, metadata=metadata
        )

    async def create_thread(
        self,
        channel_id: str,
        message_id: str,
        title: str = "",
    ) -> Thread:
        """Create a new thread on a message.

        CONCEPT:ECO-4.5

        Args:
            channel_id: Channel containing the root message.
            message_id: Message to start the thread from.
            title: Thread title (if supported).

        Returns:
            ``Thread`` object representing the new thread.

        Raises:
            NotImplementedError: If the platform doesn't support threads.
        """
        raise NotImplementedError(
            f"{self.id} backend does not support thread creation."
        )

    # ── Inbound Event Stream ─────────────────────────────────────────

    async def listen(self) -> AsyncIterator[InboundEvent]:
        """Listen for inbound events from the messaging platform.

        CONCEPT:ECO-4.5

        This is the core of bidirectional messaging — it yields
        ``InboundEvent`` objects as they arrive from the platform.
        The ``InboundRouter`` (``messaging/router.py``) consumes
        this stream and routes events to the planner graph agent.

        Yields:
            ``InboundEvent`` for each platform event (message,
            reaction, typing, etc.).

        Raises:
            NotImplementedError: If the platform doesn't support
                inbound listening (outbound-only mode).
        """
        raise NotImplementedError(
            f"{self.id} backend does not support inbound listening. "
            "Override listen() to enable bidirectional messaging."
        )
        # Make this a proper async generator even though it raises.
        # The `yield` is unreachable but required for type checking.
        yield  # type: ignore[misc]  # pragma: no cover

    # ── Directory ────────────────────────────────────────────────────

    async def list_channels(self) -> list[Channel]:
        """List available channels/conversations.

        CONCEPT:ECO-4.5

        Returns:
            List of ``Channel`` objects accessible by the bot/agent.
        """
        return []

    async def list_members(self, channel_id: str) -> list[dict[str, Any]]:
        """List members of a channel.

        CONCEPT:ECO-4.5

        Args:
            channel_id: Channel to list members of.

        Returns:
            List of member info dicts with at minimum ``id`` and ``name``.
        """
        return []

    # ── Utility ──────────────────────────────────────────────────────

    def __repr__(self) -> str:
        status = "connected" if self._connected else "disconnected"
        return f"<{self.__class__.__name__} id={self.id!r} status={status}>"
