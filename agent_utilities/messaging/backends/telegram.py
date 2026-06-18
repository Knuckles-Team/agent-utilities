"""Telegram Messaging Backend (CONCEPT:ECO-4.0).

Implements ``MessagingBackend`` for Telegram using ``python-telegram-bot``.
Supports forum topics (threads), reactions, inline keyboards, polls,
media groups, and bidirectional messaging via polling or webhooks.

Install::

    pip install agent-utilities[messaging-telegram]

Configuration::

    TELEGRAM_BOT_TOKEN=<your-bot-token>

CONCEPT:ECO-4.0 — Native Messaging Backend Abstraction
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from agent_utilities.messaging.base import MessagingBackend
from agent_utilities.messaging.capabilities import (
    CAPABILITY_MATRIX,
    MessagingCapabilities,
)
from agent_utilities.messaging.models import (
    EventType,
    InboundEvent,
    MediaAttachment,
    MediaType,
    Message,
    MessageDirection,
    MessagingConfig,
    PlatformId,
    SendResult,
)

logger = logging.getLogger(__name__)


class TelegramBackend(MessagingBackend):
    """Telegram messaging backend using ``python-telegram-bot``. CONCEPT:ECO-4.0"""

    def __init__(self, config: MessagingConfig | None = None) -> None:
        super().__init__(config)
        self._app: Any = None
        self._event_queue: asyncio.Queue[InboundEvent] = asyncio.Queue()
        self._polling = False

    @property
    def id(self) -> str:
        return "telegram"

    @property
    def capabilities(self) -> MessagingCapabilities:
        return CAPABILITY_MATRIX["telegram"]

    async def connect(self) -> None:
        """Connect to Telegram Bot API — send-ready, no poller. CONCEPT:ECO-4.0

        Polling for inbound updates is started lazily by :meth:`listen` (the inbound
        stream), NOT here, so a send-only consumer (e.g. the ``graph_reach`` MCP tool
        in a client process) never starts a second ``getUpdates`` poller that would
        409-conflict with the daemon's inbound listener.
        """
        try:
            from telegram.ext import ApplicationBuilder, MessageHandler, filters
        except ImportError:
            raise ImportError(
                "python-telegram-bot is required. "
                "Install: pip install agent-utilities[messaging-telegram]"
            ) from None

        if not self.config.token:
            raise ValueError("Set TELEGRAM_BOT_TOKEN or MESSAGING_TELEGRAM_TOKEN.")

        self._app = ApplicationBuilder().token(self.config.token).build()

        async def on_message(update: Any, context: Any) -> None:
            msg = update.message
            if not msg:
                return
            attachments = []
            if msg.photo:
                best = msg.photo[-1]
                file = await best.get_file()
                attachments.append(
                    MediaAttachment(
                        media_type=MediaType.IMAGE, url=file.file_path or ""
                    )
                )
            if msg.document:
                file = await msg.document.get_file()
                attachments.append(
                    MediaAttachment(
                        media_type=MediaType.FILE,
                        url=file.file_path or "",
                        filename=msg.document.file_name or "",
                    )
                )

            event = InboundEvent(
                event_type=EventType.MESSAGE,
                platform=PlatformId.TELEGRAM,
                channel_id=str(msg.chat_id),
                thread_id=str(msg.message_thread_id) if msg.message_thread_id else "",
                user_id=str(msg.from_user.id) if msg.from_user else "",
                user_name=msg.from_user.full_name if msg.from_user else "",
                content=msg.text or msg.caption or "",
                message=Message(
                    id=str(msg.message_id),
                    content=msg.text or msg.caption or "",
                    channel_id=str(msg.chat_id),
                    author_id=str(msg.from_user.id) if msg.from_user else "",
                    author_name=msg.from_user.full_name if msg.from_user else "",
                    platform=PlatformId.TELEGRAM,
                    direction=MessageDirection.INBOUND,
                    attachments=attachments,
                ),
                raw={"chat_type": msg.chat.type},
            )
            await self._event_queue.put(event)

        self._app.add_handler(MessageHandler(filters.ALL, on_message))
        await self._app.initialize()
        await self._app.start()
        self._connected = True
        logger.info("[CONCEPT:ECO-4.0] Telegram backend connected (send-ready).")

    async def disconnect(self) -> None:
        """Disconnect from Telegram. CONCEPT:ECO-4.0"""
        if self._app:
            if self._polling:
                await self._app.updater.stop()
                self._polling = False
            await self._app.stop()
            await self._app.shutdown()
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
        """Send a Telegram message. CONCEPT:ECO-4.0"""
        try:
            kwargs: dict[str, Any] = {"chat_id": int(channel_id), "text": text}
            if thread_id:
                kwargs["message_thread_id"] = int(thread_id)
            if reply_to_id:
                kwargs["reply_to_message_id"] = int(reply_to_id)
            parse_mode = (metadata or {}).get("parse_mode", "HTML")
            kwargs["parse_mode"] = parse_mode

            msg = await self._app.bot.send_message(**kwargs)
            return SendResult(
                success=True,
                message_id=str(msg.message_id),
                platform=PlatformId.TELEGRAM,
                channel_id=channel_id,
            )
        except Exception as e:
            logger.error("[CONCEPT:ECO-4.0] Telegram send failed: %s", e)
            return SendResult(success=False, platform=PlatformId.TELEGRAM, error=str(e))

    async def send_media(
        self,
        channel_id: str,
        attachment: MediaAttachment,
        *,
        caption: str = "",
        thread_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> SendResult:
        """Send media to Telegram. CONCEPT:ECO-4.0"""
        try:
            kwargs: dict[str, Any] = {"chat_id": int(channel_id)}
            if caption:
                kwargs["caption"] = caption
            if thread_id:
                kwargs["message_thread_id"] = int(thread_id)

            if attachment.media_type == MediaType.IMAGE:
                msg = await self._app.bot.send_photo(photo=attachment.url, **kwargs)
            elif attachment.media_type in (MediaType.VIDEO, MediaType.GIF):
                msg = await self._app.bot.send_video(video=attachment.url, **kwargs)
            elif attachment.media_type in (MediaType.AUDIO, MediaType.VOICE_NOTE):
                msg = await self._app.bot.send_audio(audio=attachment.url, **kwargs)
            else:
                msg = await self._app.bot.send_document(
                    document=attachment.url, **kwargs
                )

            return SendResult(
                success=True,
                message_id=str(msg.message_id),
                platform=PlatformId.TELEGRAM,
                channel_id=channel_id,
            )
        except Exception as e:
            return SendResult(success=False, platform=PlatformId.TELEGRAM, error=str(e))

    async def send_typing(self, channel_id: str) -> None:
        """Send typing action. CONCEPT:ECO-4.0"""
        await self._app.bot.send_chat_action(chat_id=int(channel_id), action="typing")

    async def listen(self) -> AsyncIterator[InboundEvent]:
        """Yield inbound Telegram events, starting the poller on first use. CONCEPT:ECO-4.0"""
        if not self._polling:
            await self._app.updater.start_polling()
            self._polling = True
            logger.info("[CONCEPT:ECO-4.0] Telegram polling started.")
        while self._connected:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                yield event
            except TimeoutError:
                continue
