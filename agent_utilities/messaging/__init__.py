"""AU messaging models, planner behavior, intake lease, and GraphOS reach port."""

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

__all__ = [
    # Core ABC
    "MessagingBackend",
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
