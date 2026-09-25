"""Public outbound notification port supplied by the GraphOS host."""

from __future__ import annotations

from collections.abc import Callable
from importlib import metadata
from typing import Any, cast

NotificationPort = Callable[..., Any]
_ENTRY_POINT_GROUP = "agent_utilities.messaging.reach"


def notification_port() -> NotificationPort:
    """Resolve exactly one installed host implementation, or fail closed."""
    providers = [
        entry
        for entry in metadata.entry_points(group=_ENTRY_POINT_GROUP)
        if entry.name == "notify_sync"
    ]
    if len(providers) != 1:
        raise LookupError("exactly one messaging notification host is required")
    provider = providers[0].load()
    if not callable(provider):
        raise TypeError("messaging notification host must be callable")
    return cast(NotificationPort, provider)
