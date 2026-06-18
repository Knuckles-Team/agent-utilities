"""Messaging reach service — the one core for outbound + channel routing (CONCEPT:ECO-4.48).

This is the single source of truth that the MCP tool (``graph_reach``), the REST twin
(``/graph/reach``), the inbound router, the goal-loop, and the pydantic-ai agent toolset
all dispatch into. It owns:

- the set of connected messaging backends (auto-detected from configured tokens via
  :class:`~agent_utilities.messaging.registry.MessagingRegistry`),
- governed outbound sends (every send passes the fail-closed ActionPolicy gate,
  ``orchestration/action_policy.py`` — ``kind="message.send"``),
- the durable **last-active channel** routing state (CONCEPT:ECO-4.49) — like OpenClaw,
  ``reach_user`` delivers to whatever channel the user last interacted on, falling back to
  a configured default so a fresh system still works,
- auto-ingestion of every message into the KG as conversational memory (``kg_ingest``).

CONCEPT:ECO-4.48 — Messaging reach service for governed outbound and channel routing
CONCEPT:ECO-4.49 — Last-active channel routing state for the messaging reach service

See Also:
    - ``messaging/router.py`` for the inbound side that feeds ``record_inbound``.
    - ``docs/architecture/messaging_reach.md`` for the end-to-end flow + diagram.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from agent_utilities.core.config import setting
from agent_utilities.messaging.models import InboundEvent, MediaAttachment, SendResult

if TYPE_CHECKING:
    from agent_utilities.messaging.base import MessagingBackend

logger = logging.getLogger(__name__)

# Durable node id holding a user's most-recently-active channel (CONCEPT:ECO-4.49).
_PREF_NODE_PREFIX = "chanpref:"
# Sentinel user id for "the operator" when an event carries no user id.
_DEFAULT_USER = "operator"


class MessagingService:
    """Connected-backend manager + governed outbound sender + channel router.

    CONCEPT:ECO-4.48

    Singleton: use :meth:`instance`. The same service object is shared by the MCP
    tool, the REST handler, the inbound router, and the loop so they all read/write
    one set of backends and one routing state.
    """

    _instance: MessagingService | None = None

    def __init__(self, engine: Any = None) -> None:
        self._engine = engine
        self._backends: dict[str, MessagingBackend] = {}
        # CONCEPT:ECO-4.52 — elicitation and loop bridge so a blocked loop reaches the user and resumes on reply
        # Pending awaited replies keyed by "platform:channel_id":
        # reach_user_and_wait registers a future here; the inbound router resolves it
        # when the user replies on that channel. In-process (daemon) only.
        self._pending: dict[str, asyncio.Future[str]] = {}

    @classmethod
    def instance(cls, engine: Any = None) -> MessagingService:
        """Get or create the shared service (binding the engine on first use)."""
        if cls._instance is None:
            cls._instance = cls(engine=engine)
        elif engine is not None and cls._instance._engine is None:
            cls._instance._engine = engine
        return cls._instance

    # ── Engine resolution ────────────────────────────────────────────
    def _resolve_engine(self) -> Any:
        """Bind to the live served engine (matches gateway/daemon.py)."""
        if self._engine is not None:
            return self._engine
        try:
            from agent_utilities.knowledge_graph.core.engine import (
                IntelligenceGraphEngine,
            )

            self._engine = IntelligenceGraphEngine.get_active()
        except Exception as exc:  # noqa: BLE001
            logger.debug("[ECO-4.48] no active engine: %s", exc)
        return self._engine

    # ── Backend lifecycle ────────────────────────────────────────────
    async def get_backend(self, platform: str) -> MessagingBackend | None:
        """Return a connected, send-ready backend for ``platform``.

        ``connect()`` only prepares the backend to *send* — the inbound poller lives in
        ``listen()`` (started by the daemon's router), so a one-off send from the
        MCP/client process never competes with the daemon's inbound listener (e.g. the
        Telegram ``getUpdates`` 409 you would otherwise hit with two pollers).
        """
        existing = self._backends.get(platform)
        if existing is not None and existing.is_connected:
            return existing
        from agent_utilities.messaging.registry import MessagingRegistry

        registry = MessagingRegistry.instance()
        if not registry.is_installed(platform):
            logger.warning("[ECO-4.48] messaging backend '%s' not installed", platform)
            return None
        try:
            backend = registry.create_backend(platform)
            await backend.connect()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[ECO-4.48] connect '%s' failed: %s", platform, exc)
            return None
        self._backends[platform] = backend
        return backend

    def register_connected(self, backend: MessagingBackend) -> None:
        """Register a backend the daemon already connected (for shared reuse)."""
        self._backends[backend.id] = backend

    def configured_platforms(self) -> list[str]:
        """Platforms that are installed AND have a token/app id configured."""
        from agent_utilities.messaging.registry import MessagingRegistry

        registry = MessagingRegistry.instance()
        out: list[str] = []
        for pid in registry.list_backends():
            cfg = registry._auto_config(pid)
            if cfg.token or cfg.app_id:
                out.append(pid)
        return out

    # ── Outbound (governed) ──────────────────────────────────────────
    async def send(
        self,
        platform: str,
        channel_id: str,
        text: str,
        *,
        thread_id: str = "",
        reply_to_id: str = "",
        media: MediaAttachment | None = None,
        metadata: dict[str, Any] | None = None,
        source: str = "manual",
        reason: str = "",
    ) -> SendResult:
        """Send a message, gated by ActionPolicy and mirrored into KG memory.

        CONCEPT:ECO-4.48 — every outbound send is a governed ``message.send`` fleet action.
        """
        engine = self._resolve_engine()
        decision = self._gate(channel_id, platform, source=source, reason=reason)
        if decision is not None and not decision.allowed:
            return SendResult(
                success=False,
                platform=platform,
                channel_id=channel_id,
                error=f"policy {decision.decision}: {decision.reason}",
            )

        backend = await self.get_backend(platform)
        if backend is None:
            return SendResult(
                success=False,
                platform=platform,
                channel_id=channel_id,
                error=f"backend '{platform}' unavailable",
            )

        if media is not None:
            result = await backend.send_media(
                channel_id, media, caption=text, thread_id=thread_id, metadata=metadata
            )
        else:
            result = await backend.send_message(
                channel_id,
                text,
                thread_id=thread_id,
                reply_to_id=reply_to_id,
                metadata=metadata,
            )

        if result.success:
            await self._ingest_outbound(platform, channel_id, text, engine)
        return result

    def _gate(self, channel_id: str, platform: str, *, source: str, reason: str) -> Any:
        """Run the ActionPolicy gate for an outbound send (None if unavailable)."""
        try:
            from agent_utilities.orchestration.action_policy import (
                ActionRequest,
                get_action_policy,
            )

            request = ActionRequest(
                kind="message.send",
                target=f"{platform}:{channel_id}",
                source=source or "messaging",
                reason=reason or "outbound message to user",
            )
            return get_action_policy(self._resolve_engine()).decide(request)
        except Exception as exc:  # noqa: BLE001 — gate failure must not silently send
            logger.warning("[ECO-4.48] action policy unavailable: %s", exc)
            return None

    async def _ingest_outbound(
        self, platform: str, channel_id: str, text: str, engine: Any
    ) -> None:
        from agent_utilities.messaging.kg_ingest import ingest_outbound_to_kg
        from agent_utilities.messaging.models import Message

        try:
            await ingest_outbound_to_kg(
                Message(content=text, channel_id=channel_id, platform=platform),
                knowledge_engine=engine,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("[ECO-4.48] outbound ingest failed: %s", exc)

    # ── Reach the user (last-active routing) ─────────────────────────
    async def reach_user(
        self,
        text: str,
        *,
        user_id: str | None = None,
        media: MediaAttachment | None = None,
        metadata: dict[str, Any] | None = None,
        source: str = "manual",
        reason: str = "",
    ) -> SendResult:
        """Deliver ``text`` to the user on their last-active channel (CONCEPT:ECO-4.49).

        Resolution order: the user's recorded last-active channel → the configured default
        (``MESSAGING_DEFAULT_PLATFORM`` / ``MESSAGING_DEFAULT_CHANNEL``).
        """
        platform, channel_id = self.resolve_channel(user_id)
        if not platform or not channel_id:
            return SendResult(
                success=False,
                error=(
                    "no last-active channel and no MESSAGING_DEFAULT_CHANNEL configured"
                ),
            )
        return await self.send(
            platform,
            channel_id,
            text,
            media=media,
            metadata=metadata,
            source=source,
            reason=reason or "proactive outreach to user",
        )

    def resolve_channel(self, user_id: str | None = None) -> tuple[str, str]:
        """Return (platform, channel_id) for the user — last-active, else configured default."""
        pref = self._last_channel(user_id or _DEFAULT_USER)
        if pref:
            return pref
        platform = str(setting("MESSAGING_DEFAULT_PLATFORM", "telegram"))
        channel_id = str(setting("MESSAGING_DEFAULT_CHANNEL", ""))
        return (platform, channel_id) if channel_id else ("", "")

    # ── Last-active channel state (durable KG node, CONCEPT:ECO-4.49) ─
    def record_inbound(self, event: InboundEvent) -> None:
        """Record the channel an inbound event arrived on as the user's last-active one."""
        user = event.user_id or _DEFAULT_USER
        platform = str(event.platform)
        if not platform or not event.channel_id:
            return
        self._write_channel_pref(user, platform, event.channel_id)

    def _write_channel_pref(self, user_id: str, platform: str, channel_id: str) -> None:
        engine = self._resolve_engine()
        add_node = getattr(engine, "add_node", None)
        if not callable(add_node):
            return
        node_id = f"{_PREF_NODE_PREFIX}{user_id}"
        try:
            add_node(
                node_id,
                "UserChannelPreference",
                properties={
                    "id": node_id,
                    "user_id": user_id,
                    "platform": platform,
                    "channel_id": channel_id,
                    "updated_at": time.time(),
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("[ECO-4.49] channel pref write failed: %s", exc)

    def _last_channel(self, user_id: str) -> tuple[str, str] | None:
        engine = self._resolve_engine()
        query = getattr(engine, "query_cypher", None)
        if not callable(query):
            return None
        node_id = f"{_PREF_NODE_PREFIX}{user_id}"
        try:
            rows = query(
                "MATCH (p {id: $id}) RETURN p",
                {"id": node_id},
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("[ECO-4.49] channel pref read failed: %s", exc)
            return None
        for row in rows or []:
            node = row.get("p", row) if isinstance(row, dict) else row
            props = node.get("properties", node) if isinstance(node, dict) else {}
            platform = props.get("platform")
            channel_id = props.get("channel_id")
            if platform and channel_id:
                return str(platform), str(channel_id)
        return None

    # ── Awaited replies (blocked-loop bridge, CONCEPT:ECO-4.52) ──────
    async def reach_user_and_wait(
        self,
        text: str,
        *,
        user_id: str | None = None,
        timeout: float = 900.0,
        source: str = "loop",
        reason: str = "",
    ) -> str | None:
        """Send ``text`` to the user and block until they reply (or ``timeout``).

        Used by the goal-loop when a cycle needs human input: the question goes to the
        user's last-active channel and this coroutine resolves when the inbound router
        delivers the next message on that channel. Returns the reply text, or None on
        send failure / timeout.
        """
        platform, channel_id = self.resolve_channel(user_id)
        if not platform or not channel_id:
            return None
        key = f"{platform}:{channel_id}"
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._pending[key] = future
        result = await self.send(
            platform,
            channel_id,
            text,
            source=source,
            reason=reason or "loop needs input",
        )
        if not result.success:
            self._pending.pop(key, None)
            return None
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError:
            return None
        finally:
            self._pending.pop(key, None)

    def deliver_reply(self, platform: str, channel_id: str, text: str) -> bool:
        """Resolve a pending awaited reply for this channel. Returns True if consumed.

        The inbound router calls this first; when it returns True the message was an
        answer to a question the loop asked, so the router does NOT re-route it to the
        planner.
        """
        future = self._pending.get(f"{platform}:{channel_id}")
        if future is not None and not future.done():
            # The inbound router runs on its own loop/thread; resolve the future on the
            # loop that created it (the loop awaiting the reply) — thread-safe handoff.
            future.get_loop().call_soon_threadsafe(_safe_set, future, text)
            return True
        return False

    # ── Reactions (CONCEPT:ECO-4.60) ─────────────────────────────────
    async def react(
        self, platform: str, channel_id: str, message_id: str, emoji: str
    ) -> bool:
        """React to a message with an emoji where the platform supports it.

        Returns True if sent; False if the backend is unavailable or doesn't support it.
        """
        if not (platform and channel_id and message_id and emoji):
            return False
        backend = await self.get_backend(platform)
        if backend is None:
            return False
        try:
            await backend.send_reaction(channel_id, message_id, emoji)
            return True
        except NotImplementedError:
            return False
        except Exception as exc:  # noqa: BLE001
            logger.debug("[ECO-4.60] reaction failed on %s: %s", platform, exc)
            return False

    # ── Introspection ────────────────────────────────────────────────
    async def list_channels(self, platform: str) -> list[dict[str, Any]]:
        backend = await self.get_backend(platform)
        if backend is None:
            return []
        return [ch.model_dump() for ch in await backend.list_channels()]

    def status(self) -> dict[str, Any]:
        return {
            "configured": self.configured_platforms(),
            "connected": [pid for pid, b in self._backends.items() if b.is_connected],
            "default_platform": str(setting("MESSAGING_DEFAULT_PLATFORM", "telegram")),
            "default_channel_set": bool(setting("MESSAGING_DEFAULT_CHANNEL", "")),
        }


def _safe_set(future: asyncio.Future[str], text: str) -> None:
    """Set a future's result if it is still pending (runs on the future's own loop)."""
    if not future.done():
        future.set_result(text)


def reach_user_sync(text: str, **kwargs: Any) -> SendResult:
    """Synchronous wrapper around :meth:`MessagingService.reach_user` for sync callers.

    Used by the goal-loop (CONCEPT:ECO-4.52) which runs outside an event loop.
    """
    svc = MessagingService.instance()
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(svc.reach_user(text, **kwargs))
    # Already inside a loop — run in a private loop on a worker thread.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(svc.reach_user(text, **kwargs))).result()
