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
        reply = await _graph_agent_reply(
            engine, combined, session=session, image_parts=image_parts
        )
        try:
            if event.message and event.message.id:
                await backend.reply_to(event.channel_id, event.message.id, reply)
            else:
                await backend.send_message(event.channel_id, reply)
        except Exception as e:  # noqa: BLE001
            logger.error("[CONCEPT:ECO-4.51] Sending reply failed: %s", e)

        # Persist + enrich AFTER the reply is sent (CONCEPT:ECO-4.74). EVERY KG write and
        # local-model encode (last-active, message ingest, episodic memory, and the
        # per-session conversation memento that gives the NEXT turn its continuity) runs here
        # — NEVER concurrently with reply generation, which otherwise contends with it on the
        # GIL-bound local embedding model and stalls the answer (the root of "no reply").
        _spawn_bg(
            _persist_and_enrich(svc, engine, items, combined, reply, session=session)
        )

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


# CONCEPT:ECO-4.60 — instinctive, model-agnostic emoji reaction decision
_REACTION_SYSTEM = (
    "You decide whether to react to a user's chat message with a single emoji, the way a "
    "thoughtful assistant would. Reply with ONE emoji if a reaction fits (e.g. 👍 to "
    "acknowledge a request/command, ❤️ for thanks or praise, 🎉 for good news, 👀 when "
    "starting to look into something), or the exact word NONE if no reaction fits. Output "
    "only the emoji or NONE — nothing else."
)


async def _decide_reaction(content: str) -> str:
    """Return a single emoji to react with, or "" (model-agnostic; plain completion).

    CONCEPT:ECO-4.60 — a cheap, tool-free classification so reactions work on ANY model
    (including local serves that can't do tool calls). Disabled with MESSAGING_REACTIONS=0.
    """
    from agent_utilities.core.config import setting

    if str(setting("MESSAGING_REACTIONS", "1")).strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return ""
    try:
        from pydantic_ai import Agent

        from agent_utilities.core.model_factory import create_model

        agent = Agent(create_model(), system_prompt=_REACTION_SYSTEM)
        # CONCEPT:ECO-4.74 — bound the reaction LLM call so it can never hang forever
        # (TimeoutError is an Exception subclass, so the handler below catches it too).
        result = await asyncio.wait_for(agent.run(content), timeout=10.0)
        out = str(getattr(result, "output", result)).strip()
        if not out or out.upper().startswith("NONE") or len(out) > 8:
            return ""
        return out
    except Exception as e:  # noqa: BLE001
        logger.debug("[CONCEPT:ECO-4.60] reaction decision skipped: %s", e)
        return ""


async def _react_in_background(
    svc: Any, platform: str, channel_id: str, message_id: str, content: str
) -> None:
    """Decide + apply an instinctive emoji reaction, OFF the reply path (CONCEPT:ECO-4.74).

    Best-effort and bounded: the reply is generated and sent independently, so a slow or
    failing reaction never delays (or blocks) the user getting an answer.
    """
    try:
        emoji = await _decide_reaction(content)
        if emoji:
            await svc.react(platform, channel_id, message_id, emoji)
    except Exception as e:  # noqa: BLE001
        logger.debug("[CONCEPT:ECO-4.60] reaction step skipped: %s", e)


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
                media = resp.headers.get("content-type", "image/jpeg").split(";")[0]
                parts.append(BinaryContent(data=resp.content, media_type=media))
            except Exception as e:  # noqa: BLE001
                logger.debug("[ECO-4.67] image fetch failed (%s): %s", url, e)
    return parts


async def _graph_agent_reply(
    engine: Any, content: str, *, session: str, image_parts: list[Any] | None = None
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
    """
    from agent_utilities.core.config import setting

    # The named agent the universal path routes a chat turn to. Unresolved names still go
    # through the full multi-agent orchestration graph (dynamic delegation) — which is what we
    # want — so the default is the dedicated messaging assistant identity.
    agent_name = str(setting("MESSAGING_AGENT", "")).strip() or "messaging-assistant"
    reply_timeout = float(setting("MESSAGING_REPLY_TIMEOUT", "45"))
    try:
        from agent_utilities.orchestration.manager import Orchestrator

        out = await asyncio.wait_for(
            Orchestrator(engine).execute_agent(
                agent_name=agent_name,
                task=content,
                session_id=session,
                memento_source=session,
            ),
            timeout=reply_timeout,
        )
        text = str(out).strip() if out else ""
        if text and not text.startswith("Agent execution failed"):
            return text
        logger.warning(
            "[CONCEPT:ECO-4.78] universal agent returned no usable reply (%.60s); "
            "falling back to plain chat.",
            text,
        )
    except Exception as e:  # noqa: BLE001 — timeout/hang/delegation error → plain-chat reply
        logger.warning(
            "[CONCEPT:ECO-4.78] universal agent failed/timed out (%s); "
            "falling back to plain chat.",
            e,
        )
    # Plain-chat fallback — ALWAYS yields a reply even if the graph run stalls (CONCEPT:ECO-4.74).
    return await _plain_chat_reply(content, image_parts=image_parts)


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
