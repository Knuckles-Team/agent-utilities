#!/usr/bin/python
from __future__ import annotations

"""CONCEPT:ORCH-1.28 — Asynchronous Behavioral Event Dispatcher.

Implements the decoupled, reactive listener loop using a decorator-based
subscription pattern. Listeners subscribe to event topics and execute
concurrently when matching events are appended to the ledger.

Supports both sync and async listener functions gracefully.
"""

import asyncio
import functools
import inspect
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...models.knowledge_graph import EventNode

logger = logging.getLogger(__name__)


class BehaviorDispatcher:
    """Registry and async dispatcher for reactive behaviors."""

    _INSTANCE: BehaviorDispatcher | None = None

    def __init__(self) -> None:
        # Maps event_type (topic) -> list of subscriber functions
        self._registry: dict[str, list[Callable[..., Any]]] = {}

    @classmethod
    def instance(cls) -> BehaviorDispatcher:
        """Access the thread-safe global dispatcher singleton."""
        if cls._INSTANCE is None:
            cls._INSTANCE = BehaviorDispatcher()
        return cls._INSTANCE

    def register_behavior(
        self, event_types: list[str], func: Callable[..., Any]
    ) -> None:
        """Register a subscriber callback for specific event types.

        Args:
            event_types: List of event type topic strings.
            func: Callback function to invoke.
        """
        for event_type in event_types:
            self._registry.setdefault(event_type, []).append(func)
            logger.debug(
                "[BehaviorDispatcher] Registered callback '%s' for event '%s'",
                func.__name__,
                event_type,
            )

    async def dispatch_event(self, event: EventNode, *args: Any, **kwargs: Any) -> None:
        """Route an event asynchronously to all registered listeners.

        Executes all matching subscriber behaviors concurrently using asyncio.gather.

        Args:
            event: The EventNode instance being dispatched.
            args: Positional arguments forwarded to listeners.
            kwargs: Keyword arguments forwarded to listeners.
        """
        event_type = event.event_type
        listeners = self._registry.get(event_type, [])
        # Also support wildcard '*' listeners subscribing to all events
        listeners = listeners + self._registry.get("*", [])

        if not listeners:
            return

        logger.debug(
            "[BehaviorDispatcher] Dispatching event '%s' (%s) to %d listeners",
            event_type,
            event.id,
            len(listeners),
        )

        tasks = []
        for func in listeners:
            if inspect.iscoroutinefunction(func):
                # Schedule coroutine
                tasks.append(func(event, *args, **kwargs))
            else:
                # Wrap synchronous function to execute in executor to avoid blocking the loop
                loop = asyncio.get_running_loop()
                wrapped = functools.partial(func, event, *args, **kwargs)
                tasks.append(loop.run_in_executor(None, wrapped))

        if tasks:
            from ..gather import gather_with_resilience

            results = await gather_with_resilience(
                tasks, label=f"behavior:{event_type}"
            )
            for res, func in zip(results, listeners, strict=False):
                if isinstance(res, BaseException):
                    logger.error(
                        "[BehaviorDispatcher] Error executing listener '%s' on event '%s': %s",
                        func.__name__,
                        event_type,
                        res,
                        exc_info=res,
                    )


def reactive_behavior(
    on: list[str] | str, dispatcher: BehaviorDispatcher | None = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to declare an asynchronous reactive behavior callback.

    Usage::

        @reactive_behavior(on=["task.proposed", "event.error"])
        async def on_proposed_task(event: EventNode):
            print(f"Proposed task: {event.payload}")

    Args:
        on: List of event type topic strings, or a single topic string.
        dispatcher: Optional dispatcher instance. Defaults to global instance.
    """
    event_topics = [on] if isinstance(on, str) else list(on)
    disp = dispatcher or BehaviorDispatcher.instance()

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        disp.register_behavior(event_topics, func)
        return func

    return decorator
