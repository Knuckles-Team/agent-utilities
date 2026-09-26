"""Public outbound notification port supplied by the GraphOS host."""

from __future__ import annotations

from collections.abc import Callable
from importlib import metadata
from typing import Any, Protocol, cast

from agent_utilities.messaging.models import SendResult

NotificationPort = Callable[..., Any]
_ENTRY_POINT_GROUP = "agent_utilities.messaging.reach"


class ReachServicePort(Protocol):
    """Outbound reach behavior consumed by AU agents and elicitation."""

    def _resolve_engine(self) -> Any: ...

    def configured_platforms(self) -> list[str]: ...

    def identity_digest(self, platform: str) -> str: ...

    def backend_status(self) -> dict[str, list[str]]: ...

    async def reach_user(
        self, text: str, *, source: str = "manual", reason: str = ""
    ) -> SendResult: ...

    async def reach_user_and_wait(
        self, text: str, *, source: str = "loop", reason: str = ""
    ) -> str | None: ...

    async def send(self, *args: Any, **kwargs: Any) -> SendResult: ...

    async def list_channels(self, platform: str) -> list[dict[str, Any]]: ...

    async def react(self, *args: Any, **kwargs: Any) -> Any: ...

    def resolve_channel(self, user_id: str | None = None) -> tuple[str, str]: ...

    def deliver_reply(self, platform: str, channel_id: str, text: str) -> bool: ...

    def record_inbound(self, event: Any) -> None: ...

    def status(self) -> dict[str, Any]: ...


def _load_port(name: str) -> Any:
    """Resolve one installed host provider by name, or fail closed."""
    providers = [
        entry
        for entry in metadata.entry_points(group=_ENTRY_POINT_GROUP)
        if entry.name == name
    ]
    if len(providers) != 1:
        raise LookupError(f"exactly one messaging {name} host is required")
    provider = providers[0].load()
    if not callable(provider):
        raise TypeError(f"messaging {name} host must be callable")
    return provider


def notification_port() -> NotificationPort:
    """Resolve the synchronous notification host."""
    return cast(NotificationPort, _load_port("notify_sync"))


def reach_service_port(engine: Any = None) -> ReachServicePort:
    """Resolve the GraphOS reach service through the public entrypoint."""
    provider = _load_port("service")
    service = provider() if engine is None else provider(engine)
    if not all(
        callable(getattr(service, name, None))
        for name in (
            "_resolve_engine",
            "configured_platforms",
            "identity_digest",
            "backend_status",
            "reach_user",
            "reach_user_and_wait",
            "send",
            "list_channels",
            "react",
            "resolve_channel",
            "deliver_reply",
            "record_inbound",
            "status",
        )
    ):
        raise TypeError("messaging service host has an incomplete reach contract")
    return cast(ReachServicePort, service)
