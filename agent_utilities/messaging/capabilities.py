"""Messaging Platform Capability Matrix (CONCEPT:ECO-4.0).

Declares what each of the 17 messaging platforms supports, allowing the
agent to dynamically adapt its behavior based on the channel context.

This is the Python equivalent of OpenClaw's ``ChannelCapabilities`` system
from ``src/channels/plugins/types.core.ts``.

CONCEPT:ECO-4.0 — Native Messaging Backend Abstraction

Usage::

    from agent_utilities.messaging.capabilities import CAPABILITY_MATRIX

    caps = CAPABILITY_MATRIX["discord"]
    if caps.threads:
        await backend.create_thread(channel_id, msg_id, "Discussion")
    if caps.reactions:
        await backend.send_reaction(channel_id, msg_id, "👍")
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MessagingCapabilities:
    """Declares the feature set supported by a messaging platform.

    CONCEPT:ECO-4.0

    Attributes:
        send_text: Can send plain text messages.
        send_media: Can send images, videos, files natively.
        threads: Supports message threading / reply chains.
        reactions: Supports emoji reactions on messages.
        typing_indicator: Can show "user is typing..." status.
        rich_formatting: Supports rich text (embeds, blocks, cards, HTML).
        polls: Can create polls / voting.
        inbound_listen: Can receive inbound messages in real-time.
        user_directory: Can list users / members of a channel.
        slash_commands: Supports bot slash commands.
        voice_call: Supports voice/video calling.
        webhooks: Supports webhook-based event delivery.
        presence: Can report user online/offline status.
        edit_messages: Can edit previously sent messages.
        delete_messages: Can delete messages.
        mentions: Supports @mentions for users/roles/channels.
        read_receipts: Can track message read status.
        max_message_length: Maximum characters per message (0 = unlimited).
    """

    send_text: bool = True
    send_media: bool = False
    threads: bool = False
    reactions: bool = False
    typing_indicator: bool = False
    rich_formatting: bool = False
    polls: bool = False
    inbound_listen: bool = False
    user_directory: bool = False
    slash_commands: bool = False
    voice_call: bool = False
    webhooks: bool = False
    presence: bool = False
    edit_messages: bool = False
    delete_messages: bool = False
    mentions: bool = False
    read_receipts: bool = False
    max_message_length: int = 0


# ── Capability Matrix for All 17 Platforms ───────────────────────────
# CONCEPT:ECO-4.0
#
# Each entry maps a PlatformId string to its MessagingCapabilities.
# These are modeled after OpenClaw's per-extension capability declarations
# and the platform API documentation.

CAPABILITY_MATRIX: dict[str, MessagingCapabilities] = {
    "discord": MessagingCapabilities(
        send_text=True,
        send_media=True,
        threads=True,
        reactions=True,
        typing_indicator=True,
        rich_formatting=True,  # Embeds
        polls=True,
        inbound_listen=True,  # Gateway WebSocket
        user_directory=True,
        slash_commands=True,
        voice_call=False,  # Voice channels exist but are complex
        webhooks=True,
        presence=True,
        edit_messages=True,
        delete_messages=True,
        mentions=True,
        read_receipts=False,
        max_message_length=2000,
    ),
    "slack": MessagingCapabilities(
        send_text=True,
        send_media=True,
        threads=True,
        reactions=True,
        typing_indicator=True,
        rich_formatting=True,  # Blocks / Block Kit
        polls=False,  # No native polls
        inbound_listen=True,  # Socket Mode / Events API
        user_directory=True,
        slash_commands=True,
        voice_call=False,
        webhooks=True,
        presence=True,
        edit_messages=True,
        delete_messages=True,
        mentions=True,
        read_receipts=False,
        max_message_length=40000,
    ),
    "telegram": MessagingCapabilities(
        send_text=True,
        send_media=True,
        threads=True,  # Forum topics
        reactions=True,
        typing_indicator=True,
        rich_formatting=True,  # HTML / Markdown
        polls=True,
        inbound_listen=True,  # Polling or Webhooks
        user_directory=False,  # Limited
        slash_commands=True,  # Bot commands
        voice_call=False,
        webhooks=True,
        presence=False,
        edit_messages=True,
        delete_messages=True,
        mentions=True,
        read_receipts=True,
        max_message_length=4096,
    ),
    "whatsapp": MessagingCapabilities(
        send_text=True,
        send_media=True,
        threads=False,
        reactions=True,
        typing_indicator=True,
        rich_formatting=False,  # Limited formatting
        polls=True,
        inbound_listen=True,  # WebSocket (neonize) / Webhooks (Business)
        user_directory=True,  # Contact list
        slash_commands=False,
        voice_call=False,
        webhooks=True,  # Business API
        presence=True,
        edit_messages=True,
        delete_messages=True,
        mentions=True,
        read_receipts=True,
        max_message_length=65536,
    ),
    "teams": MessagingCapabilities(
        send_text=True,
        send_media=True,  # Adaptive Cards
        threads=True,
        reactions=True,  # Limited set
        typing_indicator=True,
        rich_formatting=True,  # Adaptive Cards
        polls=False,
        inbound_listen=True,  # Bot Framework webhook
        user_directory=True,  # via Graph API
        slash_commands=False,
        voice_call=False,
        webhooks=True,
        presence=True,  # via Graph API
        edit_messages=True,
        delete_messages=True,
        mentions=True,
        read_receipts=True,
        max_message_length=28000,
    ),
    "googlechat": MessagingCapabilities(
        send_text=True,
        send_media=True,
        threads=True,
        reactions=True,
        typing_indicator=False,
        rich_formatting=True,  # Cards
        polls=False,
        inbound_listen=True,  # Pub/Sub or webhook
        user_directory=True,
        slash_commands=True,
        voice_call=False,
        webhooks=True,
        presence=False,
        edit_messages=True,
        delete_messages=True,
        mentions=True,
        read_receipts=False,
        max_message_length=4096,
    ),
    "googlemeet": MessagingCapabilities(
        send_text=False,  # Meet is voice/video, not text
        send_media=False,
        threads=False,
        reactions=False,
        typing_indicator=False,
        rich_formatting=False,
        polls=False,
        inbound_listen=True,  # Meeting events
        user_directory=True,  # Participant list
        slash_commands=False,
        voice_call=True,
        webhooks=True,  # Calendar event hooks
        presence=True,  # In-call status
        edit_messages=False,
        delete_messages=False,
        mentions=False,
        read_receipts=False,
        max_message_length=0,
    ),
    "mattermost": MessagingCapabilities(
        send_text=True,
        send_media=True,
        threads=True,
        reactions=True,
        typing_indicator=True,
        rich_formatting=True,  # Markdown + attachments
        polls=False,
        inbound_listen=True,  # WebSocket
        user_directory=True,
        slash_commands=True,
        voice_call=False,
        webhooks=True,
        presence=True,
        edit_messages=True,
        delete_messages=True,
        mentions=True,
        read_receipts=False,
        max_message_length=16383,
    ),
    "matrix": MessagingCapabilities(
        send_text=True,
        send_media=True,
        threads=True,
        reactions=True,
        typing_indicator=True,
        rich_formatting=True,  # HTML
        polls=True,  # MSC3381
        inbound_listen=True,  # Sync API
        user_directory=True,
        slash_commands=False,
        voice_call=False,
        webhooks=False,  # Uses sync, not webhooks
        presence=True,
        edit_messages=True,
        delete_messages=True,
        mentions=True,
        read_receipts=True,
        max_message_length=65536,
    ),
    "irc": MessagingCapabilities(
        send_text=True,
        send_media=False,
        threads=False,
        reactions=False,
        typing_indicator=False,
        rich_formatting=False,  # IRC formatting codes only
        polls=False,
        inbound_listen=True,  # TCP socket
        user_directory=True,  # NAMES / WHO
        slash_commands=False,
        voice_call=False,
        webhooks=False,
        presence=True,  # JOIN/PART/QUIT
        edit_messages=False,
        delete_messages=False,
        mentions=True,  # Nick highlighting
        read_receipts=False,
        max_message_length=512,
    ),
    "signal": MessagingCapabilities(
        send_text=True,
        send_media=True,
        threads=False,
        reactions=True,
        typing_indicator=False,
        rich_formatting=False,
        polls=False,
        inbound_listen=True,  # signal-cli / semaphore-bot
        user_directory=False,
        slash_commands=False,
        voice_call=False,
        webhooks=False,
        presence=False,
        edit_messages=False,
        delete_messages=True,
        mentions=True,
        read_receipts=True,
        max_message_length=65536,
    ),
    "imessage": MessagingCapabilities(
        send_text=True,
        send_media=True,
        threads=False,
        reactions=True,  # Tapback
        typing_indicator=True,
        rich_formatting=False,
        polls=False,
        inbound_listen=True,  # AppleScript polling
        user_directory=False,
        slash_commands=False,
        voice_call=False,
        webhooks=False,
        presence=False,
        edit_messages=True,
        delete_messages=True,
        mentions=False,
        read_receipts=True,
        max_message_length=20000,
    ),
    "line": MessagingCapabilities(
        send_text=True,
        send_media=True,
        threads=False,
        reactions=False,
        typing_indicator=False,
        rich_formatting=True,  # Flex Messages
        polls=False,
        inbound_listen=True,  # Webhook
        user_directory=True,  # Group member list
        slash_commands=False,
        voice_call=False,
        webhooks=True,
        presence=False,
        edit_messages=False,
        delete_messages=False,
        mentions=True,
        read_receipts=True,
        max_message_length=5000,
    ),
    "twitch": MessagingCapabilities(
        send_text=True,
        send_media=False,
        threads=False,
        reactions=False,
        typing_indicator=False,
        rich_formatting=False,
        polls=True,  # Channel polls
        inbound_listen=True,  # IRC / EventSub
        user_directory=True,  # Chatters list
        slash_commands=True,  # Chat commands
        voice_call=False,
        webhooks=True,  # EventSub
        presence=True,  # Stream online/offline
        edit_messages=False,
        delete_messages=True,  # Mod delete
        mentions=True,
        read_receipts=False,
        max_message_length=500,
    ),
    "synology": MessagingCapabilities(
        send_text=True,
        send_media=False,
        threads=False,
        reactions=False,
        typing_indicator=False,
        rich_formatting=False,
        polls=False,
        inbound_listen=True,  # Incoming webhook
        user_directory=False,
        slash_commands=False,
        voice_call=False,
        webhooks=True,
        presence=False,
        edit_messages=False,
        delete_messages=False,
        mentions=False,
        read_receipts=False,
        max_message_length=4096,
    ),
    "voicecall": MessagingCapabilities(
        send_text=True,  # SMS/text via Twilio
        send_media=False,
        threads=False,
        reactions=False,
        typing_indicator=False,
        rich_formatting=False,
        polls=False,
        inbound_listen=True,  # Webhook for inbound calls/SMS
        user_directory=False,
        slash_commands=False,
        voice_call=True,
        webhooks=True,
        presence=False,
        edit_messages=False,
        delete_messages=False,
        mentions=False,
        read_receipts=False,
        max_message_length=1600,  # SMS limit
    ),
    "nextcloud": MessagingCapabilities(
        send_text=True,
        send_media=True,
        threads=True,
        reactions=True,
        typing_indicator=False,
        rich_formatting=True,  # Markdown
        polls=True,
        inbound_listen=True,  # Long-polling / SSE
        user_directory=True,
        slash_commands=False,
        voice_call=False,
        webhooks=False,
        presence=True,
        edit_messages=True,
        delete_messages=True,
        mentions=True,
        read_receipts=True,
        max_message_length=32000,
    ),
}


def get_capabilities(platform_id: str) -> MessagingCapabilities:
    """Get capabilities for a specific platform.

    CONCEPT:ECO-4.0

    Args:
        platform_id: Platform identifier (e.g., ``"discord"``).

    Returns:
        ``MessagingCapabilities`` for the platform.

    Raises:
        KeyError: If the platform is not in the capability matrix.
    """
    if platform_id not in CAPABILITY_MATRIX:
        raise KeyError(
            f"Unknown platform '{platform_id}'. "
            f"Known platforms: {', '.join(sorted(CAPABILITY_MATRIX))}"
        )
    return CAPABILITY_MATRIX[platform_id]
