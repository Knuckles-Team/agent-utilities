"""iMessage Backend (CONCEPT:ECO-4.0). macOS-only via AppleScript bridge.

Install: ``pip install agent-utilities[messaging-imessage]``
CONCEPT:ECO-4.0 — Native Messaging Backend Abstraction
"""

from __future__ import annotations

import asyncio
import logging
import platform as _platform
from collections.abc import AsyncIterator
from datetime import UTC
from typing import Any

from agent_utilities.messaging.base import MessagingBackend
from agent_utilities.messaging.capabilities import (
    CAPABILITY_MATRIX,
    MessagingCapabilities,
)
from agent_utilities.messaging.models import (
    InboundEvent,
    MessagingConfig,
    PlatformId,
    SendResult,
)

logger = logging.getLogger(__name__)


class IMessageBackend(MessagingBackend):
    """iMessage backend via AppleScript (macOS only). CONCEPT:ECO-4.0"""

    def __init__(self, config: MessagingConfig | None = None) -> None:
        super().__init__(config)

    @property
    def id(self) -> str:
        return "imessage"

    @property
    def capabilities(self) -> MessagingCapabilities:
        return CAPABILITY_MATRIX["imessage"]

    async def connect(self) -> None:
        if _platform.system() != "Darwin":
            raise ConnectionError("iMessage backend requires macOS.")
        self._connected = True
        logger.info("[CONCEPT:ECO-4.0] iMessage backend connected (macOS).")

    async def send_message(
        self,
        channel_id: str,
        text: str,
        *,
        thread_id: str = "",
        reply_to_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> SendResult:
        try:
            script = f'tell application "Messages" to send "{text}" to buddy "{channel_id}" of (service 1 whose service type is iMessage)'
            proc = await asyncio.create_subprocess_exec(
                "osascript",
                "-e",
                script,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                return SendResult(
                    success=False, platform=PlatformId.IMESSAGE, error=stderr.decode()
                )
            return SendResult(
                success=True, platform=PlatformId.IMESSAGE, channel_id=channel_id
            )
        except Exception as e:
            return SendResult(success=False, platform=PlatformId.IMESSAGE, error=str(e))

    async def listen(self) -> AsyncIterator[InboundEvent]:
        """Poll the macOS chat.db SQLite database for new inbound messages."""
        if _platform.system() != "Darwin":
            logger.warning(
                "iMessage listen called on a non-macOS system. Running in mock/fallback loop."
            )
            while True:
                await asyncio.sleep(3600)
                # Keep the generator alive, but yield nothing.
                if False:
                    yield InboundEvent(
                        event_id="mock",
                        channel_id="mock",
                        user_id="mock",
                        text="mock",
                        platform=PlatformId.IMESSAGE,
                        raw=None,
                    )

        import os
        import sqlite3
        from datetime import datetime

        db_path = os.path.expanduser("~/Library/Messages/chat.db")
        if not os.path.exists(db_path):
            logger.warning(
                f"iMessage database not found at {db_path}. Running in mock/fallback loop."
            )
            while True:
                await asyncio.sleep(3600)

        # Fetch current time in Cocoa epoch (seconds/nanoseconds since 2001-01-01)
        def get_cocoa_now() -> int:
            epoch = datetime(2001, 1, 1, tzinfo=UTC)
            # Modern macOS stores dates in nanoseconds
            return int((datetime.now(UTC) - epoch).total_seconds() * 1_000_000_000)

        last_timestamp = get_cocoa_now()

        while True:
            try:
                # Open read-only to avoid database lock issues
                conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
                cursor = conn.cursor()

                query = """
                SELECT
                    m.guid,
                    m.text,
                    h.id,
                    m.date
                FROM message m
                LEFT JOIN handle h ON m.handle_id = h.ROWID
                WHERE m.is_from_me = 0 AND m.date > ?
                ORDER BY m.date ASC
                """
                cursor.execute(query, (last_timestamp,))
                rows = cursor.fetchall()

                for guid, text, sender, date_val in rows:
                    if date_val > last_timestamp:
                        last_timestamp = date_val

                    yield InboundEvent(
                        event_id=guid or "",
                        channel_id=sender or "unknown",
                        user_id=sender or "unknown",
                        text=text or "",
                        platform=PlatformId.IMESSAGE,
                        raw={},
                    )

                conn.close()
            except Exception as e:
                logger.error(f"iMessage listen error: {e}")

            await asyncio.sleep(5)
