"""Mattermost Messaging Backend (CONCEPT:AU-ECO.messaging.native-backend-abstraction, CONCEPT:AU-ECO.messaging.mattermost-backend).

A first-class, bidirectional Mattermost backend modelled exactly on the Telegram
backend (CONCEPT:AU-ECO.messaging.messaging-reach-service-governed–4.54): the universal orchestrator is the ONE agent; this
backend is a thin transport that only (1) receives Mattermost posts and normalizes them
into the shared :class:`InboundEvent` the ``InboundRouter`` consumes, and (2) renders the
orchestrator's reply back to a Mattermost channel/DM via a bot token. All capability
(memory, dynamic delegation, reactions, last-active routing) lives in the core.

Uses ``mattermostdriver`` for the bot REST API (outbound) and its WebSocket event stream
(inbound, ``posted`` events). Like Telegram, the WebSocket consumer is started **lazily by
:meth:`listen`** — NOT by :meth:`connect` — so a send-only consumer (e.g. the ``graph_reach``
MCP tool in a client process) opens only the REST session and never a second WebSocket that
would duplicate the daemon's inbound stream.

Install: ``pip install agent-utilities[messaging-mattermost]``

Configuration (sanctioned ``config.setting()`` — set in ``config.json`` or the env)::

    MATTERMOST_URL=https://mattermost.example.com   # server base URL (scheme + port derived)
    MATTERMOST_TOKEN=<bot-token>                     # a Bot Account personal access token
    MATTERMOST_BOT_USER=<bot-username-or-id>         # optional; auto-resolved from the token

Operator provisioning (the bot account is created out-of-band):
    1. System Console → Integrations → Bot Accounts → enable, then "Add Bot Account".
    2. Copy the generated **token** → ``MATTERMOST_TOKEN`` (store in OpenBao ``apps/<svc>``).
    3. Add the bot to the teams/channels it should listen + post in (or DM it).

CONCEPT:AU-ECO.messaging.native-backend-abstraction — Native Messaging Backend Abstraction
CONCEPT:AU-ECO.messaging.mattermost-backend — Mattermost as a first-class bidirectional messaging platform
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlparse

from agent_utilities.core.config import setting
from agent_utilities.messaging.base import MessagingBackend
from agent_utilities.messaging.capabilities import (
    CAPABILITY_MATRIX,
    MessagingCapabilities,
)
from agent_utilities.messaging.models import (
    Channel,
    EventType,
    InboundEvent,
    Message,
    MessageDirection,
    MessagingConfig,
    PlatformId,
    SendResult,
    Thread,
)

logger = logging.getLogger(__name__)


class MattermostBackend(MessagingBackend):
    """Mattermost backend via mattermostdriver. CONCEPT:AU-ECO.messaging.native-backend-abstraction/4.90.

    Outbound uses the bot REST API; inbound consumes the server WebSocket event stream
    (``posted`` events), normalized into the shared :class:`InboundEvent` so the universal
    ``InboundRouter`` drives the reply through the one orchestration path.
    """

    def __init__(self, config: MessagingConfig | None = None) -> None:
        super().__init__(config)
        self._driver: Any = None
        self._event_queue: asyncio.Queue[InboundEvent] = asyncio.Queue()
        self._listening = False
        self._bot_user_id: str = ""

    @property
    def id(self) -> str:
        return "mattermost"

    @property
    def capabilities(self) -> MessagingCapabilities:
        return CAPABILITY_MATRIX["mattermost"]

    async def connect(self) -> None:
        """Connect to Mattermost — send-ready, no WebSocket. CONCEPT:AU-ECO.messaging.native-backend-abstraction/4.90.

        Logs in the bot for the REST API and resolves the bot's own user id (used to drop
        the bot's own posts from the inbound stream, the Mattermost analogue of avoiding an
        echo loop). The inbound WebSocket consumer is started lazily by :meth:`listen` so a
        send-only client process never opens a duplicate event stream.
        """
        try:
            from mattermostdriver import Driver
        except ImportError:
            raise ImportError(
                "mattermostdriver is required. "
                "Install: pip install agent-utilities[messaging-mattermost]"
            ) from None

        url = self.config.extra.get("url") or str(setting("MATTERMOST_URL", "")).strip()
        if not url or not self.config.token:
            raise ValueError("Set MATTERMOST_URL and MATTERMOST_TOKEN.")

        # Derive host/scheme/port from the configured URL (mattermostdriver wants them split).
        parsed = urlparse(url if "://" in url else f"https://{url}")
        scheme = parsed.scheme or "https"
        host = parsed.hostname or url
        port = parsed.port or (443 if scheme == "https" else 80)
        basepath = (
            parsed.path.rstrip("/") + "/api/v4" if parsed.path.strip("/") else "/api/v4"
        )

        self._driver = Driver(
            {
                "url": host,
                "token": self.config.token,
                "scheme": scheme,
                "port": port,
                "basepath": basepath,
                "verify": True,
            }
        )
        await asyncio.to_thread(self._driver.login)

        # Resolve the bot's own user id so the inbound stream can drop its own posts.
        configured_bot = str(setting("MATTERMOST_BOT_USER", "")).strip()
        try:
            me = await asyncio.to_thread(self._driver.users.get_user, "me")
            self._bot_user_id = str(me.get("id", ""))
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "[CONCEPT:AU-ECO.messaging.mattermost-backend] could not resolve bot user id: %s",
                exc,
            )
            self._bot_user_id = configured_bot

        self._connected = True
        logger.info(
            "[CONCEPT:AU-ECO.messaging.mattermost-backend] Mattermost backend connected (send-ready, bot=%s).",
            self._bot_user_id or configured_bot or "?",
        )

    async def disconnect(self) -> None:
        if self._driver:
            if self._listening:
                try:
                    self._driver.disconnect()  # closes the websocket loop
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "[CONCEPT:AU-ECO.messaging.mattermost-backend] websocket disconnect: %s",
                        exc,
                    )
                self._listening = False
            await asyncio.to_thread(self._driver.logout)
        await super().disconnect()

    async def send_message(
        self,
        channel_id: str,
        text: str,
        *,
        thread_id: str = "",
        reply_to_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> SendResult:
        """Send a Mattermost post (Markdown-native). CONCEPT:AU-ECO.messaging.native-backend-abstraction/4.90.

        The universal agent replies in Markdown, which Mattermost renders directly — no
        per-platform conversion needed (unlike Telegram's HTML subset). A threaded reply
        roots under ``thread_id``/``reply_to_id`` (Mattermost threads are rooted on a post id).
        """
        try:
            payload: dict[str, Any] = {"channel_id": channel_id, "message": text}
            # Mattermost threads are rooted on the originating post id; both a thread context
            # and a direct reply target collapse to the same root_id.
            root = thread_id or reply_to_id
            if root:
                payload["root_id"] = root
            result = await asyncio.to_thread(self._driver.posts.create_post, payload)
            return SendResult(
                success=True,
                message_id=result.get("id", ""),
                platform=PlatformId.MATTERMOST,
                channel_id=channel_id,
            )
        except Exception as e:
            logger.error(
                "[CONCEPT:AU-ECO.messaging.mattermost-backend] Mattermost send failed: %s",
                e,
            )
            return SendResult(
                success=False, platform=PlatformId.MATTERMOST, error=str(e)
            )

    async def send_reaction(self, channel_id: str, message_id: str, emoji: str) -> None:
        """Add a reaction (CONCEPT:AU-ECO.messaging.messaging-renderer-core-reaction/4.81 — the messaging renderer of a core reaction)."""
        user_id = self._bot_user_id
        if not user_id:
            user = await asyncio.to_thread(self._driver.users.get_user, "me")
            user_id = str(user["id"])
        await asyncio.to_thread(
            self._driver.reactions.create_reaction,
            {
                "user_id": user_id,
                "post_id": message_id,
                "emoji_name": emoji.strip(":"),
            },
        )

    async def create_thread(
        self, channel_id: str, message_id: str, title: str = ""
    ) -> Thread:
        return Thread(
            id=message_id,
            parent_message_id=message_id,
            channel_id=channel_id,
            title=title,
        )

    async def list_channels(self) -> list[Channel]:
        """List Mattermost channels the bot can reach. CONCEPT:AU-ECO.messaging.native-backend-abstraction"""
        user = await asyncio.to_thread(self._driver.users.get_user, "me")
        teams = await asyncio.to_thread(self._driver.teams.get_user_teams, user["id"])
        channels = []
        for team in teams:
            chs = await asyncio.to_thread(
                self._driver.channels.get_channels_for_user, user["id"], team["id"]
            )
            for ch in chs:
                channels.append(
                    Channel(
                        id=ch["id"],
                        name=ch.get("display_name", ""),
                        platform=PlatformId.MATTERMOST,
                        is_dm=ch.get("type") == "D",
                        is_group=ch.get("type") in ("O", "P", "G"),
                        member_count=ch.get("total_msg_count", 0),
                    )
                )
        return channels

    def _normalize_post_event(self, raw: dict[str, Any]) -> InboundEvent | None:
        """Turn a Mattermost ``posted`` WebSocket frame into an :class:`InboundEvent`.

        Returns ``None`` for frames that are not a user post we should route (non-``posted``
        events, or the bot's own posts — dropping the latter is what avoids an echo loop).
        """
        if raw.get("event") != "posted":
            return None
        data = raw.get("data", {}) or {}
        post_raw = data.get("post")
        if not post_raw:
            return None
        try:
            post = json.loads(post_raw) if isinstance(post_raw, str) else post_raw
        except (ValueError, TypeError):
            return None

        author_id = str(post.get("user_id", ""))
        # Drop our own posts (the bot echoing itself) — Mattermost's no-echo guard.
        if author_id and author_id == self._bot_user_id:
            return None
        # Defensive: also drop posts a bot/webhook generated.
        if (post.get("props") or {}).get("from_bot") in ("true", True):
            return None

        channel_id = str(post.get("channel_id", ""))
        root_id = str(post.get("root_id", ""))
        text = str(post.get("message", ""))
        sender = str(data.get("sender_name", "")).lstrip("@")

        return InboundEvent(
            event_type=EventType.MESSAGE,
            platform=PlatformId.MATTERMOST,
            channel_id=channel_id,
            thread_id=root_id,
            user_id=author_id,
            user_name=sender,
            content=text,
            message=Message(
                id=str(post.get("id", "")),
                content=text,
                channel_id=channel_id,
                thread_id=root_id,
                author_id=author_id,
                author_name=sender,
                platform=PlatformId.MATTERMOST,
                direction=MessageDirection.INBOUND,
            ),
            raw={"channel_type": data.get("channel_type", "")},
        )

    async def _start_intake(self) -> None:
        """Start the inbound WebSocket consumer once (CONCEPT:AU-ECO.messaging.mattermost-backend).

        ``mattermostdriver``'s ``init_websocket(handler)`` blocks on its own event loop, so
        we run it on a worker thread and hand each frame back to THIS loop's queue via
        ``call_soon_threadsafe`` (mirrors the WhatsApp bridge pattern).
        """
        if self._listening:
            return
        loop = asyncio.get_running_loop()

        async def _on_event(message: Any) -> None:
            try:
                raw = (
                    json.loads(message) if isinstance(message, str | bytes) else message
                )
            except (ValueError, TypeError):
                return
            event = self._normalize_post_event(raw or {})
            if event is not None:
                loop.call_soon_threadsafe(self._event_queue.put_nowait, event)

        def _run_ws() -> None:
            # init_websocket manages its own asyncio loop on this worker thread.
            try:
                self._driver.init_websocket(_on_event)
            except Exception as exc:  # noqa: BLE001 — surface, never crash the daemon
                logger.error(
                    "[CONCEPT:AU-ECO.messaging.mattermost-backend] Mattermost websocket stopped: %s",
                    exc,
                )

        asyncio.create_task(asyncio.to_thread(_run_ws))
        self._listening = True
        logger.info(
            "[CONCEPT:AU-ECO.messaging.mattermost-backend] Mattermost websocket inbound started."
        )

    async def listen(self) -> AsyncIterator[InboundEvent]:
        """Yield inbound Mattermost events from the WebSocket stream. CONCEPT:AU-ECO.messaging.native-backend-abstraction/4.90."""
        await self._start_intake()
        while self._connected:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                yield event
            except TimeoutError:
                continue
