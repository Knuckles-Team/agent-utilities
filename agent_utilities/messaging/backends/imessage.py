"""iMessage Backend (CONCEPT:ECO-4.0). macOS-only via AppleScript bridge.

Install: ``pip install agent-utilities[messaging-imessage]``
CONCEPT:ECO-4.0 — Native Messaging Backend Abstraction
"""

from __future__ import annotations

import asyncio
import logging
import platform as _platform
from collections.abc import AsyncIterator
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
        if self.id == "__never__":
            yield None  # type: ignore[misc]
        raise NotImplementedError(
            "iMessage inbound requires polling AppleScript/chat.db."
        )
