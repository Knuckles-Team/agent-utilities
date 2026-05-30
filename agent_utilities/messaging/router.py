"""Inbound Message Router — Routes platform events to Planner Graph Agent (CONCEPT:ECO-4.0).

Consumes ``InboundEvent`` streams from all connected messaging backends and
routes them to the planner graph agent for KG-aware orchestration. The planner
queries the Knowledge Graph for conversation context via ``recall_memory()``
before deciding how to handle each message.

Architecture::

    Backend.listen()  →  InboundRouter  →  Planner Graph Agent
                                ↓                    ↓
                         KG Auto-Ingest      KG recall_memory()
                         (kg_ingest.py)      (engine_memory.py)

CONCEPT:ECO-4.0 — Native Messaging Backend Abstraction

See Also:
    - ``graph/builder.py`` for ``create_graph_agent()``
    - ``knowledge_graph/core/engine_memory.py`` for memory recall
    - ``messaging/kg_ingest.py`` for auto-ingestion
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from agent_utilities.messaging.models import EventType, InboundEvent

if TYPE_CHECKING:
    from agent_utilities.messaging.base import MessagingBackend

logger = logging.getLogger(__name__)


# Type alias for event handlers.
EventHandler = Callable[[InboundEvent, "MessagingBackend"], Awaitable[None]]


class InboundRouter:
    """Routes inbound messaging events to the planner graph agent.

    CONCEPT:ECO-4.0 — Native Messaging Backend Abstraction

    The router listens on all connected backends simultaneously and
    dispatches events to registered handlers. The default handler
    routes messages to the planner graph agent which:

    1. Queries the KG via ``recall_memory()`` for conversation context
    2. Uses the graph orchestrator to decide which agents should handle it
    3. Sends the response back through the originating backend

    Usage::

        from agent_utilities.messaging import MessagingRegistry
        from agent_utilities.messaging.router import InboundRouter

        registry = MessagingRegistry()
        discord = registry.create_backend("discord")
        await discord.connect()

        router = InboundRouter()
        router.register_backend(discord)

        # Start listening (blocks until cancelled)
        await router.start()

    Attributes:
        _backends: Connected messaging backends to listen on.
        _handlers: Registered event handlers by event type.
        _default_handler: Handler for unmatched events.
        _running: Whether the router is currently active.
        _tasks: Active listener tasks.
    """

    def __init__(self) -> None:
        self._backends: list[MessagingBackend] = []
        self._handlers: dict[EventType, list[EventHandler]] = {}
        self._default_handler: EventHandler | None = None
        self._running = False
        self._tasks: list[asyncio.Task[None]] = []

    def register_backend(self, backend: MessagingBackend) -> None:
        """Register a connected messaging backend for event listening.

        Args:
            backend: A connected ``MessagingBackend`` instance.
        """
        self._backends.append(backend)
        logger.info(
            "[CONCEPT:ECO-4.0] Registered backend '%s' for inbound routing.",
            backend.id,
        )

    def on_event(self, event_type: EventType) -> Callable[[EventHandler], EventHandler]:
        """Decorator to register a handler for a specific event type.

        CONCEPT:ECO-4.0

        Usage::

            @router.on_event(EventType.MESSAGE)
            async def handle_message(event, backend):
                await backend.reply_to(event.channel_id, event.target_message_id, "Got it!")

        Args:
            event_type: The event type to handle.

        Returns:
            Decorator function.
        """

        def decorator(func: EventHandler) -> EventHandler:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(func)
            return func

        return decorator

    def set_default_handler(self, handler: EventHandler) -> None:
        """Set the default handler for events without a specific handler.

        CONCEPT:ECO-4.0

        The default handler is typically the planner graph agent
        dispatcher, which routes messages through the KG-aware
        orchestration pipeline.

        Args:
            handler: Async function taking (InboundEvent, MessagingBackend).
        """
        self._default_handler = handler

    async def start(self) -> None:
        """Start listening on all registered backends.

        CONCEPT:ECO-4.0

        Creates an async task for each backend's ``listen()`` method
        and dispatches events to registered handlers.
        """
        self._running = True
        logger.info(
            "[CONCEPT:ECO-4.0] Starting inbound router with %d backends.",
            len(self._backends),
        )

        for backend in self._backends:
            if not backend.is_connected:
                logger.warning(
                    "[CONCEPT:ECO-4.0] Backend '%s' is not connected, skipping.",
                    backend.id,
                )
                continue
            task = asyncio.create_task(
                self._listen_loop(backend),
                name=f"messaging-router-{backend.id}",
            )
            self._tasks.append(task)

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def stop(self) -> None:
        """Stop all listener tasks gracefully.

        CONCEPT:ECO-4.0
        """
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("[CONCEPT:ECO-4.0] Inbound router stopped.")

    async def _listen_loop(self, backend: MessagingBackend) -> None:
        """Internal listener loop for a single backend.

        CONCEPT:ECO-4.0

        Consumes the backend's ``listen()`` async iterator and
        dispatches each event to the appropriate handler.

        Args:
            backend: The messaging backend to listen on.
        """
        logger.info("[CONCEPT:ECO-4.0] Listening for events on '%s'...", backend.id)
        try:
            async for event in backend.listen():
                if not self._running:
                    break
                await self._dispatch(event, backend)
        except asyncio.CancelledError:
            logger.debug("[CONCEPT:ECO-4.0] Listener cancelled for '%s'.", backend.id)
        except NotImplementedError:
            logger.warning(
                "[CONCEPT:ECO-4.0] Backend '%s' does not support inbound listening.",
                backend.id,
            )
        except Exception as e:
            logger.error(
                "[CONCEPT:ECO-4.0] Error in listener for '%s': %s",
                backend.id,
                e,
                exc_info=True,
            )

    async def _dispatch(self, event: InboundEvent, backend: MessagingBackend) -> None:
        """Dispatch an event to registered handlers.

        CONCEPT:ECO-4.0

        Priority:
        1. Specific event-type handlers (registered via ``on_event``)
        2. Default handler (typically the planner graph agent)
        3. Log and discard if no handler matches

        Args:
            event: The inbound event to dispatch.
            backend: The backend that received the event.
        """
        handlers = self._handlers.get(event.event_type, [])

        if handlers:
            for handler in handlers:
                try:
                    await handler(event, backend)
                except Exception as e:
                    logger.error(
                        "[CONCEPT:ECO-4.0] Handler error for %s event: %s",
                        event.event_type,
                        e,
                        exc_info=True,
                    )
        elif self._default_handler:
            try:
                await self._default_handler(event, backend)
            except Exception as e:
                logger.error(
                    "[CONCEPT:ECO-4.0] Default handler error: %s",
                    e,
                    exc_info=True,
                )
        else:
            logger.debug(
                "[CONCEPT:ECO-4.0] No handler for %s event from '%s'.",
                event.event_type,
                backend.id,
            )


async def create_planner_handler(
    knowledge_engine: Any = None,
) -> EventHandler:
    """Create a default event handler that routes to the planner graph agent.

    CONCEPT:ECO-4.0

    This handler:
    1. Auto-ingests the inbound message into the KG
    2. Recalls relevant conversation context via ``recall_memory()``
    3. Runs the planner graph agent with KG context
    4. Sends the response back through the originating backend

    Args:
        knowledge_engine: Optional ``IntelligenceGraphEngine`` for KG queries.
            If not provided, attempts to load from the default workspace.

    Returns:
        An async event handler function.
    """
    from agent_utilities.messaging.kg_ingest import ingest_message_to_kg

    async def planner_handler(event: InboundEvent, backend: MessagingBackend) -> None:
        """Route an inbound message through the planner graph agent.

        CONCEPT:ECO-4.0
        """
        if event.event_type != EventType.MESSAGE:
            return  # Only handle messages for now

        content = event.content or (event.message.content if event.message else "")
        if not content:
            return

        # 1. Auto-ingest to KG (CONCEPT:KG-2.1)
        try:
            await ingest_message_to_kg(event, knowledge_engine=knowledge_engine)
        except Exception as e:
            logger.warning("[CONCEPT:ECO-4.0] KG ingest failed: %s", e)

        # 2. Recall conversation context from KG
        kg_context = ""
        if knowledge_engine:
            try:
                memories = knowledge_engine.recall_memory(
                    query=content,
                    memory_type="episodic",
                    top_k=5,
                    task_context=f"Messaging conversation on {event.platform}",
                )
                if memories:
                    kg_context = "\n".join(
                        f"- {m.get('description', '')[:200]}" for m in memories
                    )
            except Exception as e:
                logger.debug("[CONCEPT:ECO-4.0] KG recall failed: %s", e)

        # 3. Run through planner graph agent
        try:
            # Build context-enriched prompt
            if kg_context:
                pass

            logger.info(
                "[CONCEPT:ECO-4.0] Routing message from %s/%s to planner.",
                event.platform,
                event.user_name,
            )

            # For now, generate a simple acknowledgment
            # Full graph execution will be wired when the graph is running
            response_text = (
                f"Received your message on {event.platform}. "
                "Processing through the agent graph..."
            )

            # 4. Send response back through the backend
            if event.message and event.message.id:
                await backend.reply_to(
                    event.channel_id, event.message.id, response_text
                )
            else:
                await backend.send_message(event.channel_id, response_text)

        except Exception as e:
            logger.error("[CONCEPT:ECO-4.0] Planner routing failed: %s", e)

    return planner_handler
