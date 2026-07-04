"""Native Messaging Backend Abstraction (CONCEPT:AU-ECO.messaging.native-backend-abstraction).

Provides a pluggable, transport-agnostic messaging framework that enables
agents to send and receive messages across 17+ platforms (Discord, Slack,
Telegram, WhatsApp, MS Teams, Google Chat, Google Meet, Mattermost, Matrix,
IRC, Signal, iMessage, LINE, Twitch, Synology Chat, Voice Call, Nextcloud Talk).

Architecture follows the proven ``TraceBackend`` ABC pattern from
``harness/trace_backend.py`` and the ``PluginRegistry`` discovery from
``graph/plugin_registry.py``, adapted for bidirectional async messaging.

Usage::

    from agent_utilities.messaging import MessagingRegistry, MessagingBackend

    # Discover all installed backends
    registry = MessagingRegistry()
    available = registry.list_backends()

    # Create and connect a specific backend
    discord = registry.create_backend("discord")
    await discord.connect()
    await discord.send_message("#general", "Hello from agent!")

    # Listen for inbound messages
    async for event in discord.listen():
        print(f"Got: {event.content}")

Installation::

    pip install agent-utilities[messaging-discord]       # single backend
    pip install agent-utilities[messaging-discord,messaging-slack]  # multiple
    pip install agent-utilities[messaging]               # all 17 backends

See Also:
    - ``docs/pillars/4_ecosystem_peripherals/ECO-4.5-Native_Messaging_Backend.md``
    - ``docs/architecture/messaging_architecture.md``
"""

# CONCEPT:AU-ECO.messaging.native-backend-abstraction — Native Messaging Backend Abstraction

from agent_utilities.messaging.base import MessagingBackend
from agent_utilities.messaging.capabilities import (
    CAPABILITY_MATRIX,
    MessagingCapabilities,
)
from agent_utilities.messaging.models import (
    Channel,
    InboundEvent,
    MediaAttachment,
    Message,
    MessagingConfig,
    SendResult,
    Thread,
)
from agent_utilities.messaging.registry import MessagingRegistry

__all__ = [
    # Core ABC
    "MessagingBackend",
    # Registry
    "MessagingRegistry",
    # Models
    "Message",
    "Channel",
    "Thread",
    "InboundEvent",
    "SendResult",
    "MediaAttachment",
    "MessagingConfig",
    # Capabilities
    "MessagingCapabilities",
    "CAPABILITY_MATRIX",
]
"""
Description: Public API for the messaging framework (CONCEPT:AU-ECO.messaging.native-backend-abstraction).
"""
