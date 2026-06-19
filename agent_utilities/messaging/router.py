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
    """Create the default inbound handler that drives the graph agent (CONCEPT:ECO-4.51).

    For each inbound message the handler:
    1. Records the originating channel as the user's last-active one (CONCEPT:ECO-4.49).
    2. Delivers the message to a waiting goal-loop if it is the answer to a question the
       loop asked (CONCEPT:ECO-4.52) — in which case it is NOT re-routed to the planner.
    3. Auto-ingests the message into the KG as conversational memory (CONCEPT:KG-2.1).
    4. Recalls relevant context via ``recall_memory()`` and runs the graph agent
       (``Orchestrator.execute_agent``) to draft a real reply, sent back through the
       originating backend.

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

        # NO blocking recall pre-fetch on the reply path (CONCEPT:ECO-4.74). recall_memory
        # runs a heavy LOCAL embed + cross-encoder rerank (tens of CPU-bound "Batches") in
        # this process, which grinds the messaging daemon and stalls the answer. The reply
        # is generated immediately; the agent pulls KG context ON DEMAND via its
        # auto-approved kg_search/kg_recall tools when a question actually needs it.
        kg_context = ""
        logger.info(
            "[CONCEPT:ECO-4.63] Routing a burst of %d message(s) (%d image[s]) from %s/%s.",
            len(items),
            len(image_parts),
            event.platform,
            event.user_name,
        )
        reply = await _graph_agent_reply(
            engine, combined, kg_context, image_parts=image_parts
        )
        try:
            if event.message and event.message.id:
                await backend.reply_to(event.channel_id, event.message.id, reply)
            else:
                await backend.send_message(event.channel_id, reply)
        except Exception as e:  # noqa: BLE001
            logger.error("[CONCEPT:ECO-4.51] Sending reply failed: %s", e)

        # Persist + enrich AFTER the reply is sent (CONCEPT:ECO-4.74). EVERY KG write and
        # local-model encode (last-active, message ingest, concept enrichment) runs here —
        # NEVER concurrently with reply generation, which otherwise contends with it on the
        # GIL-bound local embedding model and stalls the answer (the root of "no reply").
        _spawn_bg(_persist_and_enrich(svc, engine, items, combined, reply))

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
        engine = svc._resolve_engine()

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


async def _persist_and_enrich(
    svc: Any, engine: Any, items: list[Any], combined: str, reply: str
) -> None:
    """Record last-active + ingest every message + enrich — AFTER the reply (CONCEPT:ECO-4.74).

    Runs strictly off the reply path so no KG write or local-model encode (add_node, ingest,
    concept extraction) ever competes with reply generation. Best-effort; failures here never
    affect the reply that already went out.
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


async def _recall_context(engine: Any, content: str, platform: str) -> str:
    """Recall relevant episodic memory for the inbound message (best-effort).

    CONCEPT:ECO-4.72 — ``recall_memory`` is a BLOCKING call (vector + graph retrieval);
    calling it directly in the async handler froze the whole messaging loop (poller + reply)
    whenever retrieval stalled. Run it off the event loop with a hard timeout so a slow/hung
    recall degrades to empty context instead of freezing inbound handling.
    """
    recall = getattr(engine, "recall_memory", None)
    if not callable(recall):
        return ""
    from agent_utilities.core.config import setting

    timeout = float(setting("MESSAGING_RECALL_TIMEOUT", "8"))
    try:
        memories = await asyncio.wait_for(
            asyncio.to_thread(
                recall,
                query=content,
                memory_type="episodic",
                top_k=5,
                task_context=f"Messaging conversation on {platform}",
            ),
            timeout=timeout,
        )
    except TimeoutError:
        logger.warning(
            "[CONCEPT:ECO-4.72] KG recall exceeded %.0fs — replying without context.",
            timeout,
        )
        return ""
    except Exception as e:  # noqa: BLE001
        logger.debug("[CONCEPT:ECO-4.0] KG recall failed: %s", e)
        return ""
    return "\n".join(f"- {m.get('description', '')[:200]}" for m in (memories or []))


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
    engine: Any, content: str, kg_context: str, *, image_parts: list[Any] | None = None
) -> str:
    """Draft a reply to an inbound message, routing to the right responder.

    CONCEPT:ECO-4.51 / ECO-4.55 — two responders, default local:
      * ``MESSAGING_AGENT`` set → run that full named graph agent (Orchestrator override).
      * otherwise → a lightweight model-routed reply: the **local LLM by default**, or
        **Claude** when the message is addressed to it (``MESSAGING_CLAUDE_TRIGGER``,
        default ``/claude``). CONCEPT:ECO-4.67 — image attachments are passed to the
        (vision-capable) model. The reply is tagged with who answered.
    """
    from agent_utilities.core.config import setting

    agent_name = str(setting("MESSAGING_AGENT", "")).strip()
    if agent_name:
        try:
            from agent_utilities.orchestration.manager import Orchestrator

            out = await Orchestrator(engine).execute_agent(
                agent_name=agent_name, task=content, context=kg_context or None
            )
            return str(out) if out else "(the agent returned no output)"
        except Exception as e:  # noqa: BLE001
            logger.error("[CONCEPT:ECO-4.51] graph agent execution failed: %s", e)
            return f"I saved your message, but couldn't draft a reply right now ({e})."
    return await _model_routed_reply(content, kg_context, image_parts=image_parts)


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


# CONCEPT:ECO-4.56 — Dedicated messaging agent (own prompt + universal tools + skills + MCP fleet)
# Cache one fully-built agent per (provider, model_id) so the MCP/skills wiring is paid once,
# inside the single gateway daemon — never rebuilt per message, never a second daemon.
_MESSAGING_AGENTS: dict[tuple[str, str], Any] = {}


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


def _csv_setting(key: str) -> list[str] | None:
    """Parse a comma-separated config setting into a list (None if empty)."""
    from agent_utilities.core.config import setting

    raw = str(setting(key, "")).strip()
    return [p.strip() for p in raw.split(",") if p.strip()] or None


def _get_messaging_agent(provider: str, model_id: str | None) -> Any:
    """Build (once, cached) the dedicated messaging agent for a model (CONCEPT:ECO-4.56/4.58).

    Uses ``create_agent`` so the agent inherits the SAME universal tools (incl. reach_user
    + KG search) and MCP server fleet as the rest of agent-utilities, with its own system
    prompt. Cached per (provider, model_id) so the build is paid once in the gateway daemon.

    CONCEPT:ECO-4.58 — lean by default to avoid context burden: the full skill library is
    NOT pre-loaded (``MESSAGING_ENABLE_SKILLS=0``); fleet MCP tools load **on demand** via
    the mcp-multiplexer's dynamic mode (find_tools/load_tools). Operators opt into more:
    ``MESSAGING_ENABLE_SKILLS=1`` (or ``MESSAGING_SKILL_TYPES=a,b`` for a subset) and
    ``MESSAGING_TOOL_TAGS=x,y`` to scope the universal toolset.

    CONCEPT:ECO-4.59 — delegation by graph-os MCP. Point ``MESSAGING_MCP_URL`` at the
    served graph-os MCP (e.g. ``http://127.0.0.1:8100/sse``) or ``MESSAGING_MCP_CONFIG`` at
    the multiplexer config; the agent then keeps a lean local context and OFFLOADS heavy
    work through graph-os — ``graph_orchestrate(execute_agent)`` spawns a specialist with
    the needed skills/MCP tools and routes the result back, ``graph_search`` hits the KG,
    and ``find_tools``/``load_tools`` pull fleet tools on demand. No bespoke delegation code.
    """
    from agent_utilities.core.config import setting

    key = (provider or "", model_id or "")
    agent = _MESSAGING_AGENTS.get(key)
    if agent is not None:
        return agent
    from agent_utilities.agent.factory import create_agent

    enable_skills = str(setting("MESSAGING_ENABLE_SKILLS", "0")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    agent, _toolsets = create_agent(
        provider=provider or None,
        model_id=model_id,
        name="messaging-assistant",
        system_prompt=_messaging_system_prompt(),
        enable_universal_tools=True,
        enable_skills=enable_skills,
        skill_types=_csv_setting("MESSAGING_SKILL_TYPES"),
        tool_tags=_csv_setting("MESSAGING_TOOL_TAGS"),
        mcp_url=str(setting("MESSAGING_MCP_URL", "")) or None,
        mcp_config=str(setting("MESSAGING_MCP_CONFIG", "")) or None,
    )
    _MESSAGING_AGENTS[key] = agent
    return agent


# CONCEPT:ECO-4.62/4.75 — tools the messaging agent may auto-run from a chat message.
# Read-only KG tools PLUS the graph-os delegation/discovery surface: graph_orchestrate is
# how the agent offloads work to a spawned specialist (e.g. github-agent) — and that
# specialist's OWN fleet actions are still governed by the fail-closed ActionPolicy gate
# (OS-5.24), so auto-running the delegation entrypoint from chat is safe. Mutating tools
# (write/delete/update/...) stay denied — they are never a chat side effect.
_SAFE_AUTO_TOOLS = {
    "kg_search",
    "kg_recall",
    "kg_query",
    "graph_search",
    "graph_query",
    "graph_context",
    "graph_orchestrate",
    "find_tools",
    "load_tools",
    "list_catalog",
    "multiplexer_status",
}
_READONLY_HINTS = (
    "search",
    "list",
    "get",
    "query",
    "find",
    "recall",
    "status",
    "info",
    "describe",
    "read",
    "fetch",
    "show",
    "view",
)
_MUTATING_HINTS = (
    "write",
    "delete",
    "update",
    "create",
    "save",
    "remove",
    "post",
    "put",
    "patch",
    "send",
    "execute",
    "merge",
    "close",
    "cancel",
    "approve",
)


def _auto_approvable(tool_name: str) -> bool:
    """Whether a deferred tool may auto-run from chat (CONCEPT:ECO-4.75).

    The explicit delegation/KG surface always passes. Otherwise a fleet tool auto-runs only
    when it clearly READS (a read-only verb, no mutating verb) — so on-demand fetches like
    ``github_*`` list/get work, while writes default to deny (explicit approval required).
    """
    bare = tool_name.split("__")[-1].lower()
    if tool_name in _SAFE_AUTO_TOOLS or bare in _SAFE_AUTO_TOOLS:
        return True
    if any(h in bare for h in _MUTATING_HINTS):
        return False
    return any(h in bare for h in _READONLY_HINTS)


def _agent_input(prompt: str, image_parts: list[Any] | None) -> Any:
    """A bare prompt, or a multimodal [prompt, image, …] list when images are present."""
    return [prompt, *image_parts] if image_parts else prompt


async def _run_until_text(
    agent: Any,
    prompt: str,
    max_rounds: int = 4,
    *,
    image_parts: list[Any] | None = None,
) -> str:
    """Run the agent, auto-approving only safe read-only KG tools, until a text reply.

    CONCEPT:ECO-4.62 — the dedicated agent defers tool calls for approval
    (DeferredToolRequests). For chat we auto-approve the read-only KG tools so the agent can
    actually query the graph and answer, while DENYING any other (mutating) tool — those
    must be requested explicitly, not triggered as a chat side effect. CONCEPT:ECO-4.67 —
    ``image_parts`` are sent alongside the prompt for vision.
    """
    from pydantic_ai.tools import (
        DeferredToolRequests,
        DeferredToolResults,
        ToolApproved,
        ToolDenied,
    )

    result = await agent.run(_agent_input(prompt, image_parts))
    rounds = 0
    while isinstance(result.output, DeferredToolRequests) and rounds < max_rounds:
        rounds += 1
        approvals: dict[str, ToolApproved | ToolDenied] = {}
        for part in result.output.approvals:
            if _auto_approvable(part.tool_name):
                approvals[part.tool_call_id] = ToolApproved()
            else:
                approvals[part.tool_call_id] = ToolDenied(
                    message=f"'{part.tool_name}' isn't auto-run from chat; ask explicitly."
                )
        result = await agent.run(
            message_history=result.all_messages(),
            deferred_tool_results=DeferredToolResults(approvals=approvals),
        )
    if isinstance(result.output, DeferredToolRequests):
        return ""  # still pending after the cap — let the caller fall back
    return str(result.output).strip()


async def _model_routed_reply(
    content: str, kg_context: str, *, image_parts: list[Any] | None = None
) -> str:
    """Reply via the dedicated agent, degrading to plain chat if the model lacks tools.

    CONCEPT:ECO-4.56 — the tool/skill/MCP-bearing agent is tried first (full capability on
    tool-capable models like Claude). If the model rejects the tool-augmented request
    (e.g. a vllm-served local model without function-calling — ``System message must be at
    the beginning``), we fall back to a plain chat completion so the user always gets a
    reply. CONCEPT:ECO-4.62 — safe read-only KG tools auto-run so the agent answers FROM the
    graph. CONCEPT:ECO-4.67 — image attachments are passed to the vision model. Both paths
    are tagged with who answered.
    """
    label, provider, model_id, task = _select_responder(content)
    prompt = task if not kg_context else f"{task}\n\nRelevant context:\n{kg_context}"

    # 1) Full dedicated agent (KG tools + MCP fleet), auto-running safe read-only KG tools.
    try:
        agent = _get_messaging_agent(provider, model_id)
        text = await _run_until_text(agent, prompt, image_parts=image_parts)
        if text:
            return f"[{label}] {text}"
    except Exception as e:  # noqa: BLE001 — model may not support tool-augmented requests
        logger.warning(
            "[CONCEPT:ECO-4.56] dedicated agent failed (%s); falling back to plain chat.",
            e,
        )

    # 2) Plain chat completion — works on models without tool support.
    try:
        from pydantic_ai import Agent

        from agent_utilities.core.model_factory import create_model

        bare = Agent(
            create_model(provider=provider or None, model_id=model_id),
            system_prompt=_messaging_system_prompt(),
        )
        result = await bare.run(_agent_input(prompt, image_parts))
        text = str(getattr(result, "output", result)).strip()
        return f"[{label}] {text}" if text else f"[{label}] (no output)"
    except Exception as e:  # noqa: BLE001
        logger.error("[CONCEPT:ECO-4.56] messaging reply failed: %s", e)
        return f"I saved your message, but couldn't draft a reply right now ({e})."
