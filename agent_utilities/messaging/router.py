"""Inbound Message Router — messaging is thin transport to the ONE universal agent.

Consumes ``InboundEvent`` streams from all connected messaging backends. A chat turn is
NOT handled by a bespoke messaging-only reply path: it IS a run of the universal
orchestration pipeline (``Orchestrator.execute_agent`` → ``run_agent``), session-scoped per
channel (CONCEPT:ECO-4.78). That single path natively provides what the router used to hand-
roll — conversation CONTINUITY (the core memory primes each run with the recent compressed
mementos for the channel's session and persists this turn back as one) and DYNAMIC CAPABILITY
selection (specialists / skills / A2A / swarms / fleet tools, ActionPolicy-governed). The
router stays a transport: receive → run the universal agent → send its text back, with a
hard reply-timeout plain-chat fallback so a slow/hung graph run still answers.

Architecture::

    Backend.listen() → InboundRouter → Orchestrator.execute_agent → run_agent (graph)
                              ↓                    ↑ mementos (session)      ↓
                       KG Auto-Ingest  ───────────┘  + per-turn memento  ───┘
                       (kg_ingest.py)         (knowledge_graph/memory/*)

CONCEPT:ECO-4.0 — Native Messaging Backend Abstraction
CONCEPT:ECO-4.78 — Messaging routes through the universal graph/orchestration path

See Also:
    - ``orchestration/manager.py`` / ``orchestration/agent_runner.py`` for the universal path
    - ``knowledge_graph/memory/memento_compressor.py`` for session-scoped continuity
    - ``messaging/kg_ingest.py`` for episodic auto-ingestion
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
    dispatches events to registered handlers. The default handler runs
    each chat turn through the universal graph agent (CONCEPT:ECO-4.78),
    which:

    1. Recalls prior turns of this channel from the core memory (session-scoped mementos)
    2. Dynamically resolves which agents / skills / tools should handle it
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

        # CONCEPT:ECO-4.83 — durable-inbox reaper: re-answers inbound turns that were recorded
        # pending but never got a reply (engine down / crashed mid-flight), so nothing is lost.
        if self._backends:
            self._tasks.append(
                asyncio.create_task(
                    self._inbox_reaper_loop(), name="messaging-inbox-reaper"
                )
            )

        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    def _backend_for(self, platform: str) -> MessagingBackend | None:
        """The connected backend for a platform id (CONCEPT:ECO-4.83), for reaper retries."""
        for b in self._backends:
            if platform and (
                str(getattr(b, "id", "")) == platform
                or str(getattr(b, "platform", "")) == platform
            ):
                return b
        return self._backends[0] if self._backends else None

    async def _inbox_reaper_loop(self) -> None:
        """Periodically re-attempt durably-recorded but still-unanswered inbound messages
        (CONCEPT:ECO-4.83). Uses this router's own backends + the universal reply path, so a
        turn that failed while the engine was down is answered once the system recovers."""
        from agent_utilities.core.config import setting
        from agent_utilities.messaging.inbox import retry_unanswered
        from agent_utilities.messaging.service import MessagingService

        interval = float(setting("MESSAGING_INBOX_RETRY_S", "120"))
        while self._running:
            await asyncio.sleep(interval)
            try:
                engine = MessagingService.instance()._resolve_engine()

                async def _reply_send(m: dict[str, Any], _eng: Any = engine) -> bool:
                    backend = self._backend_for(str(m.get("platform", "")))
                    if backend is None:
                        return False
                    reply = await _graph_agent_reply(
                        _eng, m.get("text", ""), session=m.get("session", "")
                    )
                    if not reply:
                        return False
                    await backend.send_message(m.get("channel_id", ""), reply)
                    return True

                await retry_unanswered(engine, _reply_send)
                # CONCEPT:ECO-4.91 — same housekeeping cadence prunes expired AgentBus topic-log
                # messages so the store-and-forward backlog can't grow unbounded.
                from agent_utilities.messaging.bus import AgentBus

                AgentBus.instance(engine).prune_topic_log()
            except Exception as e:  # noqa: BLE001 — the reaper must survive any single pass
                logger.debug("[CONCEPT:ECO-4.83] inbox reaper pass failed: %s", e)

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
    """Create the default inbound handler that drives the universal graph agent (ECO-4.78).

    For each inbound message the handler:
    1. Delivers the message to a waiting goal-loop if it is the answer to a question the
       loop asked (CONCEPT:ECO-4.52) — in which case it is NOT re-routed to the agent.
    2. Coalesces a burst of messages into ONE turn (CONCEPT:ECO-4.63) and runs that turn
       through the universal path (``Orchestrator.execute_agent`` → ``run_agent``), session-
       scoped per channel (CONCEPT:ECO-4.78), so continuity + dynamic capability come from
       the core — not a messaging-specific recall. The returned text is sent back through the
       originating backend, with a hard ``MESSAGING_REPLY_TIMEOUT`` plain-chat fallback.
    3. AFTER the reply, off the reply path: records last-active (CONCEPT:ECO-4.49), auto-
       ingests the turn as episodic memory (CONCEPT:KG-2.1), and persists a per-session
       conversation memento that gives the NEXT turn its continuity (CONCEPT:ECO-4.78).

    Args:
        knowledge_engine: Optional ``IntelligenceGraphEngine``; falls back to the active
            served engine.

    Returns:
        An async event handler function.
    """
    from agent_utilities.core.config import setting
    from agent_utilities.messaging.coalescer import BurstCoalescer
    from agent_utilities.messaging.service import MessagingService

    async def _reply_to_burst(_key: str, items: list[Any]) -> None:
        """Answer a coalesced burst with ONE agent turn + ONE reply (CONCEPT:ECO-4.63)."""
        svc = MessagingService.instance(knowledge_engine)
        engine = svc._resolve_engine()
        event, backend = items[-1]["event"], items[-1]["backend"]
        # Combine the burst into one holistic prompt (preserve order).
        contents = [it["content"] for it in items]
        combined = contents[0] if len(contents) == 1 else "\n".join(contents)

        # One instinctive reaction for the burst (CONCEPT:ECO-4.60). It is COSMETIC and
        # involves an LLM call, so it runs OFF the reply path in the background
        # (CONCEPT:ECO-4.74) — a slow/hung reaction call must never block the actual reply,
        # which was the root cause of "message received but no answer".
        if event.message and event.message.id:
            _spawn_bg(
                _react_in_background(
                    svc,
                    str(event.platform),
                    event.channel_id,
                    event.message.id,
                    combined,
                )
            )

        # Collect image attachments across the burst → vision input (CONCEPT:ECO-4.67).
        image_urls: list[str] = []
        for it in items:
            msg = getattr(it["event"], "message", None)
            for att in getattr(msg, "attachments", None) or []:
                if str(getattr(att, "media_type", "")) == "image" and att.url:
                    image_urls.append(att.url)
        image_parts = await _fetch_image_parts(image_urls)

        # CONCEPT:ECO-4.78 — the reply IS the universal graph agent, session-scoped per
        # channel. NO bespoke recall on the reply path: continuity comes from the core memory
        # the universal path already threads — ``run_agent`` primes the run with the recent
        # compressed mementos for this ``session`` source (``get_recent_mementos``), and after
        # the reply we persist this turn as a memento under the SAME source (background, off
        # the reply path). So message 2 sees message 1 via core memory, not a messaging query.
        session = _channel_session(str(event.platform), event.channel_id)
        logger.info(
            "[CONCEPT:ECO-4.78] Routing a burst of %d message(s) (%d image[s]) from %s/%s "
            "through the universal graph agent (session=%s).",
            len(items),
            len(image_parts),
            event.platform,
            event.user_name,
            session,
        )
        # CONCEPT:ORCH-1.72 — the per-job SHAPE decides BOTH how long this turn should take
        # (its dynamic ``reply_budget_s``) and whether to answer INLINE or
        # acknowledge-now / deliver-later. A direct or lean turn is answered within its short
        # budget inline; a full multi-agent tool turn would blow any reasonable inline wait, so
        # we ack immediately and send the result as a follow-up when it is ready. The transport
        # stays thin — the core shape makes the call; here we only render it for this medium.
        # CONCEPT:ECO-4.83 — record the inbound turn DURABLY as pending BEFORE we attempt the
        # reply, so a turn that fails mid-flight (engine down, crash before _send) is found +
        # retried by the reaper instead of being silently lost ("I saved your message" → real).
        from agent_utilities.messaging.inbox import mark_answered, record_inbound

        _inbox_id = record_inbound(
            engine,
            platform=str(event.platform),
            channel_id=event.channel_id,
            message_id=getattr(getattr(event, "message", None), "id", None),
            text=combined,
            session=session,
        )

        from agent_utilities.orchestration.execution_profile import plan_execution_shape

        shape = plan_execution_shape(combined, profile_hint="chat", engine=engine)

        async def _send(text: str, *, threaded: bool) -> None:
            try:
                if threaded and event.message and event.message.id:
                    await backend.reply_to(event.channel_id, event.message.id, text)
                else:
                    await backend.send_message(event.channel_id, text)
            except Exception as e:  # noqa: BLE001
                logger.error("[CONCEPT:ECO-4.51] Sending reply failed: %s", e)

        async def _run_and_deliver(*, deferred: bool) -> None:
            reply = await _graph_agent_reply(
                engine,
                combined,
                session=session,
                image_parts=image_parts,
                budget=shape.reply_budget_s,
            )
            # An interactive turn threads its reply to the user's message; a deferred turn
            # already acked there, so its result lands as a fresh follow-up message.
            await _send(reply, threaded=not deferred)
            # CONCEPT:ECO-4.83 — the real reply was delivered → close the durable inbox entry.
            # If _run_and_deliver crashed BEFORE this (engine down), it stays pending → retried.
            mark_answered(engine, _inbox_id)
            # Persist + enrich AFTER the reply is sent (CONCEPT:ECO-4.74). EVERY KG write and
            # local-model encode (last-active, message ingest, episodic memory, and the
            # per-session conversation memento that gives the NEXT turn its continuity) runs
            # here — NEVER concurrently with reply generation, which otherwise contends with it
            # on the GIL-bound local embedding model and stalls the answer.
            _spawn_bg(
                _persist_and_enrich(
                    svc, engine, items, combined, reply, session=session
                )
            )

        if shape.is_interactive:
            await _run_and_deliver(deferred=False)
        else:
            # CONCEPT:ORCH-1.74 — describe the actual altitude: a focused-tools turn runs the
            # named servers' tools (in parallel), not the full planning graph.
            _n = len(shape.tool_servers)
            _kind = (
                f"focused-tools turn ({_n} tool{'s' if _n != 1 else ''} in parallel)"
                if shape.tool_servers
                else "full multi-agent turn"
            )
            logger.info(
                "[CONCEPT:ORCH-1.72] burst shaped as a %s (~%.0fs budget) "
                "— acknowledging now, delivering the result as a follow-up.",
                _kind,
                shape.reply_budget_s,
            )
            await _send(await _varied_ack(combined, shape), threaded=True)
            _spawn_bg(_run_and_deliver(deferred=True))

    coalescer = BurstCoalescer(
        _reply_to_burst,
        window_s=float(setting("MESSAGING_BURST_WINDOW_S", "2.5")),
        max_wait_s=float(setting("MESSAGING_BURST_MAX_S", "12")),
    )

    async def planner_handler(event: InboundEvent, backend: MessagingBackend) -> None:
        """Drive an inbound message through the graph agent. CONCEPT:ECO-4.51/4.63"""
        if event.event_type != EventType.MESSAGE:
            return  # Only handle messages

        content = event.content or (event.message.content if event.message else "")
        if not content:
            # CONCEPT:ECO-4.68 — no text? transcribe a voice/audio attachment and use that.
            content = await _transcribe_attachments(event)
            if content and event.message is not None:
                event.message.content = content
                event.content = content
        if not content:
            return

        svc = MessagingService.instance(knowledge_engine)

        # NOTE: last-active + KG ingest are NOT done here. They run AFTER the reply is sent
        # (in _reply_to_burst → _persist_and_enrich, CONCEPT:ECO-4.74) so no KG write or
        # local-model encode ever runs concurrently with reply generation.

        # 2. If a goal-loop is awaiting this user's reply, deliver it and stop
        #    (CONCEPT:ECO-4.52) — the message is an answer, not a new request.
        if svc.deliver_reply(str(event.platform), event.channel_id, content):
            logger.info(
                "[CONCEPT:ECO-4.52] Delivered reply on %s to a waiting loop.",
                event.platform,
            )
            return

        # 3b. Built-in universal command? (CONCEPT:ECO-4.57) Answer immediately and stop;
        #     /claude, /skill, and unknowns fall through to the coalesced agent reply.
        from agent_utilities.messaging.commands import handle_command

        cmd_reply = await handle_command(content, service=svc)
        if cmd_reply is not None:
            try:
                await backend.send_message(event.channel_id, cmd_reply)
            except Exception as e:  # noqa: BLE001
                logger.error("[CONCEPT:ECO-4.57] command reply send failed: %s", e)
            return

        # 4. Coalesce normal messages into one agent turn per burst (CONCEPT:ECO-4.63):
        #    a rapid run of messages → ONE holistic reply + ONE LLM call, not N.
        await coalescer.submit(
            f"{event.platform}:{event.channel_id}",
            {"event": event, "backend": backend, "content": content},
        )

    return planner_handler


async def _decide_reaction(content: str) -> str:
    """Return a single emoji to react with, or "" — a thin shim over the CORE decision.

    CONCEPT:ECO-4.81 — reactions are no longer owned by messaging: the instinctive,
    model-agnostic decision lives in the core orchestrator
    (``orchestration.reactions.decide_reaction``), so EVERY entrypoint produces reactions
    from the same one heuristic. Messaging is now just a RENDERER. This shim preserves the
    string contract its callers/tests expect (``""`` for no reaction); disabled with
    ``REACTIONS=0`` / ``MESSAGING_REACTIONS=0``.
    """
    from agent_utilities.orchestration.reactions import decide_reaction

    reaction = await decide_reaction(content)
    return reaction.emote if reaction and reaction.is_valid() else ""


async def _react_in_background(
    svc: Any, platform: str, channel_id: str, message_id: str, content: str
) -> None:
    """Render the CORE instinctive reaction on this platform, OFF the reply path.

    CONCEPT:ECO-4.81 — the decision is the core orchestrator's
    (``orchestration.reactions.decide_reaction`` → an :class:`AgentReaction`); messaging
    only RENDERS it via ``svc.react`` → the backend's ``send_reaction`` /
    ``setMessageReaction``. Best-effort and bounded (CONCEPT:ECO-4.74): the reply is
    generated and sent independently, so a slow or failing reaction never delays (or
    blocks) the user getting an answer.
    """
    from agent_utilities.orchestration.reactions import decide_reaction

    try:
        reaction = await decide_reaction(
            content, target_message_id=message_id, context=platform
        )
        if reaction and reaction.is_valid():
            await svc.react(
                platform,
                channel_id,
                reaction.target_message_id or message_id,
                reaction.emote,
            )
    except Exception as e:  # noqa: BLE001
        logger.debug("[CONCEPT:ECO-4.81] reaction render skipped: %s", e)


# CONCEPT:ECO-4.72 — fire-and-forget background tasks (strong refs so they aren't GC'd).
_BG_TASKS: set[asyncio.Task[Any]] = set()


def _spawn_bg(coro: Any) -> None:
    """Schedule a coroutine off the reply path, keeping a strong ref until it finishes."""
    task = asyncio.create_task(coro)
    _BG_TASKS.add(task)
    task.add_done_callback(_BG_TASKS.discard)


def _channel_session(platform: str, channel_id: str) -> str:
    """The universal-agent session key for a chat channel (CONCEPT:ECO-4.78).

    One stable id per ``(platform, channel_id)`` so successive turns of the same conversation
    run as one session — the core memory (mementos under this source) carries continuity, and
    ``run_agent`` anchors each turn's RunTrace to the matching Session node.
    """
    return f"messaging:{platform}:{channel_id}"


async def _persist_and_enrich(
    svc: Any,
    engine: Any,
    items: list[Any],
    combined: str,
    reply: str,
    *,
    session: str,
) -> None:
    """Record last-active + ingest every message + enrich — AFTER the reply (CONCEPT:ECO-4.74).

    Runs strictly off the reply path so no KG write or local-model encode (add_node, ingest,
    concept extraction, memento compression) ever competes with reply generation. Best-effort;
    failures here never affect the reply that already went out.

    CONCEPT:ECO-4.78 — this is also where conversation CONTINUITY is established: the just-
    finished turn (user prompt + assistant reply) is compressed into a memento under the
    channel's ``session`` source via the CORE memory primitive (``compress_to_memento``), so
    the NEXT turn of this channel recalls it through ``run_agent``'s native memento priming —
    no messaging-specific recall query. The compression involves an LLM call, which is exactly
    why it runs here (background), never on the reply path.
    """
    from agent_utilities.messaging.kg_ingest import ingest_message_to_kg

    last = items[-1]["event"]
    try:
        await asyncio.to_thread(svc.record_inbound, last)
    except Exception as e:  # noqa: BLE001
        logger.debug("[CONCEPT:ECO-4.49] record_inbound failed: %s", e)
    for it in items:
        try:
            await ingest_message_to_kg(it["event"], knowledge_engine=engine)
        except Exception as e:  # noqa: BLE001
            logger.warning("[CONCEPT:ECO-4.0] KG ingest failed: %s", e)
    # Compress this turn into a per-session memento so the next turn inherits continuity
    # through the universal path's core memory (CONCEPT:ECO-4.78).
    try:
        from agent_utilities.knowledge_graph.memory.memento_compressor import (
            compress_to_memento,
        )

        turn = [
            {"role": "user", "content": combined},
            {"role": "assistant", "content": reply},
        ]
        await asyncio.to_thread(
            compress_to_memento, engine, turn, source=session, refine=False
        )
        # CONCEPT:KG-2.131 — refresh the per-session memento cache in this SAME background
        # pass so the NEXT turn reads the just-written memento from memory, never from a
        # blocking ``get_recent_mementos`` round-trip on the reply path.
        from agent_utilities.knowledge_graph.memory.session_memento_cache import (
            refresh_session_memento_cache,
        )

        await asyncio.to_thread(refresh_session_memento_cache, engine, session)
    except Exception as e:  # noqa: BLE001
        logger.debug("[CONCEPT:ECO-4.78] session memento skipped: %s", e)
    try:
        from agent_utilities.messaging.enrichment import enrich_conversation

        convo = f"{combined}\n\nAssistant: {reply}"
        await asyncio.to_thread(
            enrich_conversation,
            engine,
            convo,
            platform=str(last.platform),
            channel_id=last.channel_id,
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("[CONCEPT:ECO-4.65] enrichment skipped: %s", e)


async def _transcribe_attachments(event: Any) -> str:
    """Transcribe any voice/audio attachments to text (CONCEPT:ECO-4.68)."""
    msg = getattr(event, "message", None)
    urls = [
        att.url
        for att in getattr(msg, "attachments", None) or []
        if str(getattr(att, "media_type", "")) in ("voice_note", "audio") and att.url
    ]
    if not urls:
        return ""
    from agent_utilities.messaging.voice import transcribe_voice

    parts = [t for t in [await transcribe_voice(u) for u in urls] if t]
    return "\n".join(parts)


async def _fetch_image_parts(urls: list[str]) -> list[Any]:
    """Download image attachments → pydantic-ai BinaryContent for vision (CONCEPT:ECO-4.67).

    Downloaded + inlined (not passed as a URL) so the vision model never has to fetch an
    external/token-bearing URL itself. Best-effort; unreachable images are skipped.
    """
    if not urls:
        return []
    try:
        import httpx
        from pydantic_ai import BinaryContent
    except Exception:  # noqa: BLE001
        return []
    parts: list[Any] = []
    async with httpx.AsyncClient(timeout=20.0) as client:
        for url in urls[:6]:  # cap per turn
            try:
                resp = await client.get(url)
                resp.raise_for_status()
                media = resp.headers.get("content-type", "").split(";")[0].strip()
                if not media or media == "application/octet-stream":
                    # Telegram serves photos as application/octet-stream — sniff the real
                    # image type from magic bytes so the vision model accepts it (it rejects
                    # octet-stream). Falls back to JPEG (the Telegram photo default).
                    media = _sniff_image_media_type(resp.content) or "image/jpeg"
                parts.append(BinaryContent(data=resp.content, media_type=media))
            except Exception as e:  # noqa: BLE001
                logger.debug("[ECO-4.67] image fetch failed (%s): %s", url, e)
    return parts


def _sniff_image_media_type(data: bytes) -> str | None:
    """Detect an image MIME type from magic bytes (CONCEPT:ECO-4.67) — used when the
    transport gives a generic/absent content-type (Telegram serves octet-stream)."""
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"GIF8":
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _is_backend_timeout(failure_text: str) -> bool:
    """True when a failure string is a backend/LLM timeout (CONCEPT:ORCH-1.62).

    Such a failure means the endpoint is slow/degraded; a second full LLM call to the same
    endpoint (the double-LLM tax) is exactly what we must avoid, so the caller surfaces a
    graceful message instead of falling through to the plain-chat fallback.
    """
    low = failure_text.lower()
    return any(
        marker in low
        for marker in ("timed out", "timeout", "cancellederror", "deadline")
    )


async def _graph_agent_reply(
    engine: Any,
    content: str,
    *,
    session: str,
    image_parts: list[Any] | None = None,
    budget: float | None = None,
) -> str:
    """Draft a reply by running the UNIVERSAL graph agent (CONCEPT:ECO-4.78).

    Messaging is thin transport: an inbound chat turn IS a run of the one universal
    orchestration path (``Orchestrator.execute_agent`` → ``run_agent``), session-scoped per
    channel. That path natively provides everything the messaging layer used to hand-roll:

      * **Continuity** — ``run_agent`` primes the run with the recent compressed mementos for
        this ``session`` source (``memento_source``) and anchors the RunTrace to the Session,
        so turn 2 sees turn 1 via the CORE memory, not a messaging-specific recall query.
      * **Dynamic capabilities** — the graph dynamically resolves specialists / skills / A2A /
        swarms and fleet tools (e.g. a GitHub request reaches ``graph_orchestrate`` →
        ``execute_agent`` for the github specialist), all governed by the ActionPolicy gate.

    It is wrapped in a hard ``MESSAGING_REPLY_TIMEOUT`` (CONCEPT:ECO-4.74): a slow/hung graph
    run must still yield a reply, so on timeout/error we fall through to a plain-chat
    completion. CONCEPT:ECO-4.67 — image attachments are carried on the fallback (vision)
    path; the responder label / ``/claude`` routing applies there too.

    CONCEPT:ORCH-1.62 — the run uses the ``chat`` execution profile, so each LLM round is
    bounded to the chat budget (≈12 s) instead of 300 s: a degraded backend fails fast
    inside the reply budget. And to remove the measured double-LLM tax, a *backend timeout*
    (the run hit the reply-timeout wall) does NOT trigger a second full LLM call to the same
    degraded endpoint — it returns a graceful message. The plain-chat fallback fires only on
    a genuine graph *error* (delegation/structural), where a single short attempt is cheap.
    """
    from agent_utilities.core.config import setting

    # The named agent the universal path routes a chat turn to. Unresolved names still go
    # through the full multi-agent orchestration graph (dynamic delegation) — which is what we
    # want — so the default is the dedicated messaging assistant identity.
    agent_name = str(setting("MESSAGING_AGENT", "")).strip() or "messaging-assistant"
    # CONCEPT:ORCH-1.72 — the reply budget is DYNAMIC: the caller passes the per-job shape's
    # ``reply_budget_s`` (how long a turn of this shape should reasonably take). A fixed 45 s
    # wall both over-waits a trivial turn and prematurely cuts a legitimate multi-agent tool
    # turn. ``MESSAGING_REPLY_TIMEOUT`` remains the fallback when no shape budget is supplied.
    reply_timeout = (
        float(budget)
        if budget and budget > 0
        else float(setting("MESSAGING_REPLY_TIMEOUT", "45"))
    )

    # CONCEPT:ECO-4.67 — the universal graph path (execute_agent → run_agent) does NOT carry
    # image attachments to the model: it answers text-only, "succeeds", and so never reaches
    # the vision-capable fallback. Route image turns straight to the vision responder so
    # "what is this photo of?" actually sees the image. (Pure-vision Q&A; tool/graph turns
    # with images would need image plumbing through run_agent — tracked separately.)
    if image_parts:
        logger.info(
            "[CONCEPT:ECO-4.67] %d image(s) attached — routing to the vision responder.",
            len(image_parts),
        )
        return await _plain_chat_reply(content, image_parts=image_parts)

    try:
        from agent_utilities.orchestration.manager import Orchestrator

        out = await asyncio.wait_for(
            Orchestrator(engine).execute_agent(
                agent_name=agent_name,
                task=content,
                session_id=session,
                memento_source=session,
                execution_profile="chat",
            ),
            timeout=reply_timeout,
        )
        text = str(out).strip() if out else ""
        # CONCEPT:ORCH-1.40/1.37 — when the run opened a native message channel (or carries a
        # mermaid diagram), run_agent returns a JSON ENVELOPE string
        # ``{"output", "channel_id"?, "mermaid"?}`` rather than the bare reply. The chat reply
        # is the ``output`` field; unwrap it so the user sees the rendered text, not raw JSON.
        # The membership check is exact (keys ⊆ the envelope's) so a genuine JSON reply from the
        # agent is never mis-unwrapped.
        if text.startswith("{") and '"output"' in text:
            import json

            try:
                _env = json.loads(text)
            except (ValueError, TypeError):
                _env = None
            if (
                isinstance(_env, dict)
                and "output" in _env
                and set(_env) <= {"output", "channel_id", "mermaid"}
            ):
                text = str(_env["output"]).strip()
        if text and not text.startswith("Agent execution failed"):
            return text
        # The run completed but returned a failure string. If that failure was a backend
        # timeout (an inner node hit the chat-profile bound), do NOT re-call the same slow
        # endpoint (CONCEPT:ORCH-1.62) — surface the graceful message. Only a non-timeout
        # failure (delegation/structural) is worth a single cheap plain-chat attempt.
        if _is_backend_timeout(text):
            logger.warning(
                "[CONCEPT:ORCH-1.62] universal agent timed out on a degraded backend "
                "(%.80s); skipping the double-LLM plain-chat call.",
                text,
            )
            return (
                "I saved your message, but the assistant backend is responding slowly "
                "right now, so I couldn't finish a reply in time. Please try again shortly."
            )
        logger.warning(
            "[CONCEPT:ECO-4.78] universal agent returned no usable reply (%.60s); "
            "falling back to a single plain-chat reply.",
            text,
        )
    except TimeoutError:
        # The whole turn hit the reply-timeout wall — the backend is slow/degraded. Making a
        # SECOND full LLM call to the same endpoint is the double-LLM tax that pushed a single
        # turn past 90 s (CONCEPT:ORCH-1.62). Return a graceful message instead; the chat
        # profile already bounded each round, so this path is now a fast bound + one message.
        logger.warning(
            "[CONCEPT:ORCH-1.62] universal agent hit the %ss reply budget — skipping the "
            "plain-chat fallback to avoid a second call to a degraded backend.",
            reply_timeout,
        )
        return (
            "I saved your message, but the assistant backend is responding slowly right "
            "now, so I couldn't finish a reply in time. Please try again shortly."
        )
    except Exception as e:  # noqa: BLE001 — a genuine graph error → ONE short plain-chat reply
        logger.warning(
            "[CONCEPT:ECO-4.78] universal agent failed (%s); "
            "falling back to a single plain-chat reply.",
            e,
        )
    # Plain-chat fallback — a single short bounded attempt, fired ONLY on a genuine graph
    # error (not a backend timeout), so it never doubles a slow round (CONCEPT:ORCH-1.62/4.74).
    return await _plain_chat_reply(content, image_parts=image_parts)


async def _varied_ack(content: str, shape: Any) -> str:
    """A short, NATURALLY-VARIED 'on it' acknowledgement for a deferred turn (CONCEPT:ECO-4.67).

    The old single template ("On it — …") read as obviously canned. Generate a quick line
    with the LITE model (so it varies + can reference what it's doing), bounded by a short
    timeout, and fall back to a randomly-chosen varied static line on any error/slowness —
    the ack must NEVER delay the real work.
    """
    import random

    servers = getattr(shape, "tool_servers", ()) or ()
    if servers:
        names = ", ".join(
            s.replace("-mcp", "").replace("-agent", "").replace("-api", "")
            for s in servers
        )
        static_pool = [
            f"On it — pulling that from {names} now, back in a moment. ⏳",
            f"Got it — hitting {names} for you, one sec. ⏳",
            f"Sure, checking {names} — I'll follow up here shortly. ⏳",
        ]
        what = f"it needs the {names} tool(s)"
    else:
        static_pool = [
            "On it — this needs a bit of digging, I'll reply here shortly. ⏳",
            "Sure — let me work through that, back in a moment. ⏳",
            "Got it — give me a sec to put this together. ⏳",
        ]
        what = "it takes a few steps"
    fallback = random.choice(static_pool)  # noqa: S311 — cosmetic phrasing, not security
    try:
        from agent_utilities.knowledge_graph.enrichment.cards import make_lite_llm_fn

        llm = make_lite_llm_fn()
        if llm is None:
            return fallback
        prompt = (
            "Write ONE short (max ~12 words), natural, friendly line acknowledging you're on "
            f"the request and will reply shortly (because {what}). Vary the wording, stay casual, "
            "end with an hourglass emoji. The user said: "
            f'"{content[:200]}". Reply with ONLY the line, no quotes.'
        )
        line = await asyncio.wait_for(asyncio.to_thread(llm, prompt), timeout=4.0)
        line = (
            str(line or "").strip().strip('"').splitlines()[0].strip() if line else ""
        )
        return line if 0 < len(line) <= 160 else fallback
    except Exception:  # noqa: BLE001 — the ack is best-effort; never block the reply
        return fallback


# CONCEPT:ECO-4.55 — Model-routed inbound responder with local LLM default and Claude address
def _select_responder(content: str) -> tuple[str, str, str | None, str]:
    """Pick the responder for an inbound message (CONCEPT:ECO-4.55).

    Returns ``(label, provider, model_id, task)`` — ``task`` has the trigger stripped.
    Default is the local LLM; an explicit ``/claude`` (configurable) address routes to
    Claude, falling back to local with a note when no Anthropic key is configured.
    """
    from agent_utilities.core.config import config, setting

    trigger = str(setting("MESSAGING_CLAUDE_TRIGGER", "/claude")).strip().lower()
    stripped = content.lstrip()
    addressed_claude = trigger and stripped.lower().startswith(trigger)
    if addressed_claude:
        task = stripped[len(trigger) :].lstrip(" :,-").strip() or stripped
        if getattr(config, "anthropic_api_key", None):
            model_id = str(setting("MESSAGING_CLAUDE_MODEL", "claude-sonnet-4-6"))
            return "claude", "anthropic", model_id, task
        # Addressed Claude but no key — answer locally and say so.
        local_id = str(setting("MESSAGING_LOCAL_MODEL", "")) or None
        return (
            "local (no Anthropic key — set ANTHROPIC_API_KEY for Claude)",
            "",
            local_id,
            task,
        )
    local_id = str(setting("MESSAGING_LOCAL_MODEL", "")) or None
    return "local", "", local_id, content


def _messaging_system_prompt() -> str:
    """Load the dedicated messaging-assistant system prompt (CONCEPT:ECO-4.56)."""
    import json
    from pathlib import Path

    pfile = Path(__file__).resolve().parents[1] / "prompts" / "messaging_assistant.json"
    try:
        blueprint = json.loads(pfile.read_text(encoding="utf-8"))
        return str(blueprint.get("instructions", {}).get("core_directive", "")).strip()
    except Exception as e:  # noqa: BLE001
        logger.debug("[ECO-4.56] messaging prompt load failed: %s", e)
        return (
            "You are the Agent-Utilities Messaging Assistant. Be concise and helpful."
        )


def _agent_input(prompt: str, image_parts: list[Any] | None) -> Any:
    """A bare prompt, or a multimodal [prompt, image, …] list when images are present."""
    return [prompt, *image_parts] if image_parts else prompt


async def _plain_chat_reply(
    content: str, *, image_parts: list[Any] | None = None
) -> str:
    """Plain chat completion — the ALWAYS-yields-a-reply fallback (CONCEPT:ECO-4.78).

    This is the only responder still owned by the messaging layer: a bare, tool-free chat
    completion used when the universal graph agent times out / errors (so a slow or hung
    graph run never leaves a message unanswered). The full tool/skill/MCP/delegation
    capability that the dedicated messaging agent used to carry now lives on the universal
    path (``_graph_agent_reply`` → ``Orchestrator.execute_agent``), so it is not duplicated
    here. CONCEPT:ECO-4.55 — the local-default / ``/claude``-address responder selection and
    its label are preserved; CONCEPT:ECO-4.67 — image attachments are passed to the (vision)
    model. Works on local models without function-calling.
    """
    label, provider, model_id, task = _select_responder(content)
    try:
        from pydantic_ai import Agent

        from agent_utilities.core.model_factory import create_model

        bare = Agent(
            create_model(provider=provider or None, model_id=model_id),
            system_prompt=_messaging_system_prompt(),
        )
        result = await bare.run(_agent_input(task, image_parts))
        text = str(getattr(result, "output", result)).strip()
        return f"[{label}] {text}" if text else f"[{label}] (no output)"
    except Exception as e:  # noqa: BLE001
        logger.error("[CONCEPT:ECO-4.78] plain-chat fallback failed: %s", e)
        return f"I saved your message, but couldn't draft a reply right now ({e})."
