"""Voice Call Backend (CONCEPT:ECO-4.0). Uses Twilio for voice/SMS.

Install: ``pip install agent-utilities[messaging-voicecall]``
CONCEPT:ECO-4.0 — Native Messaging Backend Abstraction
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from agent_utilities.core.config import setting
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


class VoiceCallBackend(MessagingBackend):
    """Voice/SMS backend via Twilio. CONCEPT:ECO-4.0"""

    def __init__(self, config: MessagingConfig | None = None) -> None:
        super().__init__(config)
        self._client: Any = None
        self._event_queue: asyncio.Queue[InboundEvent] = asyncio.Queue()

    @property
    def id(self) -> str:
        return "voicecall"

    @property
    def capabilities(self) -> MessagingCapabilities:
        return CAPABILITY_MATRIX["voicecall"]

    async def connect(self) -> None:
        """Initialize Twilio client. CONCEPT:ECO-4.0"""
        try:
            from twilio.rest import Client
        except ImportError:
            raise ImportError(
                "Install: pip install agent-utilities[messaging-voicecall]"
            ) from None

        sid = self.config.app_id or setting("TWILIO_ACCOUNT_SID", "")
        token = self.config.token or setting("TWILIO_AUTH_TOKEN", "")
        if not sid or not token:
            raise ValueError("Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN.")
        self._client = Client(sid, token)
        self._connected = True
        logger.info("[CONCEPT:ECO-4.0] Voice call backend connected (Twilio).")

    async def send_message(
        self,
        channel_id: str,
        text: str,
        *,
        thread_id: str = "",
        reply_to_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> SendResult:
        """Send SMS via Twilio. CONCEPT:ECO-4.0"""
        try:
            from_number = self.config.extra.get(
                "from_number", setting("TWILIO_FROM_NUMBER", "")
            )
            msg = await asyncio.to_thread(
                self._client.messages.create,
                body=text,
                from_=from_number,
                to=channel_id,
            )
            return SendResult(
                success=True,
                message_id=msg.sid,
                platform=PlatformId.VOICECALL,
                channel_id=channel_id,
            )
        except Exception as e:
            return SendResult(
                success=False, platform=PlatformId.VOICECALL, error=str(e)
            )

    async def make_call(
        self, to: str, twiml: str = "", url: str = ""
    ) -> dict[str, Any]:
        """Initiate a voice call. CONCEPT:ECO-4.0"""

        from_number = self.config.extra.get(
            "from_number", setting("TWILIO_FROM_NUMBER", "")
        )
        kwargs: dict[str, Any] = {"from_": from_number, "to": to}
        if twiml:
            kwargs["twiml"] = twiml
        elif url:
            kwargs["url"] = url
        else:
            kwargs["twiml"] = "<Response><Say>Hello from the agent.</Say></Response>"
        call = await asyncio.to_thread(self._client.calls.create, **kwargs)
        return {"call_sid": call.sid, "status": call.status}

    async def listen(self) -> AsyncIterator[InboundEvent]:
        """Yield events (webhook-populated for inbound calls/SMS). CONCEPT:ECO-4.0"""
        while self._connected:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                yield event
            except TimeoutError:
                continue
