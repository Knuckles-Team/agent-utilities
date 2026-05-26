"""Pydantic models for the Native Messaging Backend (CONCEPT:ECO-4.0).

Provides transport-agnostic data models for cross-platform messaging.
All models are Pydantic BaseModel subclasses for runtime validation
and seamless serialization to/from the Knowledge Graph.

CONCEPT:ECO-4.0 — Native Messaging Backend Abstraction
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

# ── Enums ────────────────────────────────────────────────────────────


class MessageDirection(enum.StrEnum):
    """Direction of a message relative to the agent."""

    INBOUND = "inbound"
    OUTBOUND = "outbound"


class EventType(enum.StrEnum):
    """Types of inbound events from messaging platforms.

    CONCEPT:ECO-4.0 — Modeled after OpenClaw's ChannelGatewayEventType
    with Python-idiomatic naming.
    """

    MESSAGE = "message"
    REACTION_ADD = "reaction_add"
    REACTION_REMOVE = "reaction_remove"
    TYPING = "typing"
    PRESENCE = "presence"
    THREAD_CREATE = "thread_create"
    THREAD_UPDATE = "thread_update"
    MEMBER_JOIN = "member_join"
    MEMBER_LEAVE = "member_leave"
    CHANNEL_CREATE = "channel_create"
    CHANNEL_UPDATE = "channel_update"
    CALL_START = "call_start"
    CALL_END = "call_end"
    POLL_VOTE = "poll_vote"
    COMMAND = "command"
    UNKNOWN = "unknown"


class MediaType(enum.StrEnum):
    """Supported media attachment types."""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    FILE = "file"
    STICKER = "sticker"
    GIF = "gif"
    VOICE_NOTE = "voice_note"
    CONTACT = "contact"
    LOCATION = "location"


class PlatformId(enum.StrEnum):
    """Canonical identifiers for all supported messaging platforms.

    CONCEPT:ECO-4.0 — Equivalent to OpenClaw's ChannelId union type.
    """

    DISCORD = "discord"
    SLACK = "slack"
    TELEGRAM = "telegram"
    WHATSAPP = "whatsapp"
    TEAMS = "teams"
    GOOGLECHAT = "googlechat"
    GOOGLEMEET = "googlemeet"
    MATTERMOST = "mattermost"
    MATRIX = "matrix"
    IRC = "irc"
    SIGNAL = "signal"
    IMESSAGE = "imessage"
    LINE = "line"
    TWITCH = "twitch"
    SYNOLOGY = "synology"
    VOICECALL = "voicecall"
    NEXTCLOUD = "nextcloud"


# ── Core Models ──────────────────────────────────────────────────────


class MediaAttachment(BaseModel):
    """A media file attached to a message.

    CONCEPT:ECO-4.0

    Attributes:
        media_type: The type of media (image, video, audio, etc.).
        url: URL or local path to the media file.
        filename: Original filename if available.
        mime_type: MIME type string (e.g., ``image/png``).
        size_bytes: File size in bytes.
        alt_text: Accessibility description.
        thumbnail_url: Optional thumbnail for previews.
    """

    media_type: MediaType = MediaType.FILE
    url: str = ""
    filename: str = ""
    mime_type: str = ""
    size_bytes: int = 0
    alt_text: str = ""
    thumbnail_url: str = ""


class Channel(BaseModel):
    """A messaging channel or conversation target.

    CONCEPT:ECO-4.0

    Represents a channel, group, DM, or conversation on any platform.
    The ``platform_id`` field ties this to a specific backend.

    Attributes:
        id: Platform-specific channel identifier.
        name: Human-readable channel name.
        platform: Which messaging platform this channel belongs to.
        is_dm: Whether this is a direct/private message channel.
        is_group: Whether this is a multi-user group channel.
        topic: Channel topic or description.
        member_count: Number of members (if known).
        metadata: Platform-specific extra data.
    """

    id: str
    name: str = ""
    platform: PlatformId | str = ""
    is_dm: bool = False
    is_group: bool = False
    topic: str = ""
    member_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class Thread(BaseModel):
    """A message thread or reply chain.

    CONCEPT:ECO-4.0

    Attributes:
        id: Platform-specific thread identifier.
        parent_message_id: ID of the root message that started this thread.
        channel_id: Channel containing this thread.
        title: Thread title (if supported, e.g., Telegram forum topics).
        reply_count: Number of replies in the thread.
        is_locked: Whether the thread is locked.
    """

    id: str
    parent_message_id: str = ""
    channel_id: str = ""
    title: str = ""
    reply_count: int = 0
    is_locked: bool = False


class Message(BaseModel):
    """A message sent or received on any messaging platform.

    CONCEPT:ECO-4.0

    This is the canonical representation of a message across all 17
    supported platforms. Platform-specific features (e.g., Discord embeds,
    Slack blocks, Teams Adaptive Cards) are stored in ``metadata``.

    Attributes:
        id: Platform-specific message identifier.
        content: Plain text content of the message.
        channel_id: Channel where the message was sent.
        thread_id: Thread ID if this is a threaded reply.
        author_id: Platform-specific user identifier of the sender.
        author_name: Display name of the sender.
        platform: Which messaging platform this message belongs to.
        direction: Whether this message is inbound or outbound.
        timestamp: When the message was sent (UTC).
        attachments: List of media attachments.
        reply_to_id: Message ID this is replying to (non-threaded).
        reactions: List of reaction emoji on this message.
        is_edited: Whether the message has been edited.
        metadata: Platform-specific extra data (embeds, blocks, cards, etc.).
    """

    id: str = ""
    content: str = ""
    channel_id: str = ""
    thread_id: str = ""
    author_id: str = ""
    author_name: str = ""
    platform: PlatformId | str = ""
    direction: MessageDirection = MessageDirection.OUTBOUND
    timestamp: datetime | None = None
    attachments: list[MediaAttachment] = Field(default_factory=list)
    reply_to_id: str = ""
    reactions: list[str] = Field(default_factory=list)
    is_edited: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class InboundEvent(BaseModel):
    """An event received from a messaging platform.

    CONCEPT:ECO-4.0

    All inbound events (messages, reactions, typing indicators, presence
    updates, etc.) are normalized into this model. The ``event_type``
    discriminator determines how the event should be handled.

    Attributes:
        event_type: The type of event (message, reaction, typing, etc.).
        platform: Originating platform identifier.
        channel_id: Channel where the event occurred.
        thread_id: Thread context (if applicable).
        user_id: Platform user who triggered the event.
        user_name: Display name of the user.
        message: The full message object (for message events).
        content: Quick-access text content (for message events).
        emoji: Reaction emoji (for reaction events).
        target_message_id: Message targeted by the event (reactions, replies).
        timestamp: Event timestamp (UTC).
        raw: Raw platform-specific event payload.
    """

    event_type: EventType = EventType.MESSAGE
    platform: PlatformId | str = ""
    channel_id: str = ""
    thread_id: str = ""
    user_id: str = ""
    user_name: str = ""
    message: Message | None = None
    content: str = ""
    emoji: str = ""
    target_message_id: str = ""
    timestamp: datetime | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class SendResult(BaseModel):
    """Result of sending a message to a platform.

    CONCEPT:ECO-4.0

    Attributes:
        success: Whether the message was sent successfully.
        message_id: Platform-assigned message ID (for tracking).
        platform: Platform that handled the send.
        channel_id: Channel the message was sent to.
        error: Error message if sending failed.
        metadata: Platform-specific response data.
    """

    success: bool = True
    message_id: str = ""
    platform: PlatformId | str = ""
    channel_id: str = ""
    error: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class MessagingConfig(BaseModel):
    """Configuration for a messaging backend instance.

    CONCEPT:ECO-4.0

    Attributes:
        platform: Which platform this config is for.
        enabled: Whether this backend is active.
        token: Primary authentication token (bot token, API key, etc.).
        app_id: Application/client ID (for OAuth-based platforms).
        app_secret: Application secret (for OAuth-based platforms).
        webhook_url: Webhook URL for inbound events (if applicable).
        webhook_port: Local port for webhook server.
        use_business_api: For WhatsApp — use Business API vs unofficial bridge.
        extra: Platform-specific additional configuration.
    """

    platform: PlatformId | str = ""
    enabled: bool = True
    token: str = ""
    app_id: str = ""
    app_secret: str = ""
    webhook_url: str = ""
    webhook_port: int = 0
    use_business_api: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)
