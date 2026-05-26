"""WhatsApp Messaging Backend (CONCEPT:ECO-4.0).

Dual-mode WhatsApp backend supporting both:
1. **Official Business API** (``use_business_api=True``) — webhook-based via httpx
2. **Unofficial bridge** (default) — real-time via ``neonize`` WebSocket

Install::

    pip install agent-utilities[messaging-whatsapp]

Configuration::

    # Business API mode
    MESSAGING_WHATSAPP_USE_BUSINESS_API=true
    WHATSAPP_TOKEN=<access-token>
    WHATSAPP_PHONE_NUMBER_ID=<phone-number-id>

    # Bridge mode (default)
    # neonize auto-connects via QR code on first run

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
    Message,
    MessageDirection,
    MessagingConfig,
    PlatformId,
    SendResult,
)

logger = logging.getLogger(__name__)


class WhatsAppBackend(MessagingBackend):
    """WhatsApp messaging backend with dual Business API / bridge support.

    CONCEPT:ECO-4.0

    Mode is controlled by ``config.use_business_api``:
    - ``True``: Uses the official WhatsApp Business Cloud API via httpx
    - ``False``: Uses ``neonize`` for direct WhatsApp Web connection
    """

    def __init__(self, config: MessagingConfig | None = None) -> None:
        super().__init__(config)
        self._client: Any = None
        self._event_queue: asyncio.Queue[InboundEvent] = asyncio.Queue()
        self._use_business_api = config.use_business_api if config else False

    @property
    def id(self) -> str:
        return "whatsapp"

    @property
    def capabilities(self) -> MessagingCapabilities:
        return CAPABILITY_MATRIX["whatsapp"]

    async def connect(self) -> None:
        """Connect to WhatsApp (Business API or bridge). CONCEPT:ECO-4.0"""
        if self._use_business_api:
            await self._connect_business_api()
        else:
            await self._connect_bridge()
        self._connected = True
        logger.info(
            "[CONCEPT:ECO-4.0] WhatsApp backend connected (mode=%s).",
            "business_api" if self._use_business_api else "bridge",
        )

    async def _connect_business_api(self) -> None:
        """Initialize WhatsApp Business API client. CONCEPT:ECO-4.0"""
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required for WhatsApp Business API. "
                "Install: pip install agent-utilities[messaging-whatsapp]"
            ) from None

        if not self.config.token:
            raise ValueError("Set WHATSAPP_TOKEN for Business API mode.")
        if not self.config.app_id:
            raise ValueError("Set WHATSAPP_PHONE_NUMBER_ID for Business API mode.")

        self._client = httpx.AsyncClient(
            base_url="https://graph.facebook.com/v18.0",
            headers={"Authorization": f"Bearer {self.config.token}"},
        )

    async def _connect_bridge(self) -> None:
        """Initialize neonize bridge client. CONCEPT:ECO-4.0"""
        try:
            import neonize
        except ImportError:
            raise ImportError(
                "neonize is required for WhatsApp bridge mode. "
                "Install: pip install agent-utilities[messaging-whatsapp]"
            ) from None

        # neonize client initialization
        self._client = neonize.NewClient("agent_session")

        def on_message(client: Any, message: Any) -> None:
            event = InboundEvent(
                event_type=EventType.MESSAGE,
                platform=PlatformId.WHATSAPP,
                channel_id=str(getattr(message, "chat", {}).get("id", "")),
                user_id=str(getattr(message, "sender", "")),
                user_name=str(getattr(message, "pushName", "")),
                content=str(getattr(message, "text", "")),
                message=Message(
                    id=str(getattr(message, "id", "")),
                    content=str(getattr(message, "text", "")),
                    platform=PlatformId.WHATSAPP,
                    direction=MessageDirection.INBOUND,
                ),
                raw={"message_type": str(type(message).__name__)},
            )
            asyncio.get_event_loop().call_soon_threadsafe(
                self._event_queue.put_nowait, event
            )

        self._client.onMessage(on_message)
        asyncio.create_task(asyncio.to_thread(self._client.connect))

    async def disconnect(self) -> None:
        """Disconnect from WhatsApp. CONCEPT:ECO-4.0"""
        if self._use_business_api and hasattr(self._client, "aclose"):
            await self._client.aclose()
        elif hasattr(self._client, "disconnect"):
            self._client.disconnect()
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
        """Send a WhatsApp message. CONCEPT:ECO-4.0"""
        try:
            if self._use_business_api:
                return await self._send_business_api(channel_id, text, reply_to_id)
            else:
                return await self._send_bridge(channel_id, text, reply_to_id)
        except Exception as e:
            logger.error("[CONCEPT:ECO-4.0] WhatsApp send failed: %s", e)
            return SendResult(success=False, platform=PlatformId.WHATSAPP, error=str(e))

    async def _send_business_api(
        self, channel_id: str, text: str, reply_to_id: str = ""
    ) -> SendResult:
        """Send via Business API. CONCEPT:ECO-4.0"""
        payload: dict[str, Any] = {
            "messaging_product": "whatsapp",
            "to": channel_id,
            "text": {"body": text},
        }
        if reply_to_id:
            payload["context"] = {"message_id": reply_to_id}

        response = await self._client.post(
            f"/{self.config.app_id}/messages", json=payload
        )
        data = response.json()
        msg_id = data.get("messages", [{}])[0].get("id", "")
        return SendResult(
            success=response.status_code == 200,
            message_id=msg_id,
            platform=PlatformId.WHATSAPP,
            channel_id=channel_id,
        )

    async def _send_bridge(
        self, channel_id: str, text: str, reply_to_id: str = ""
    ) -> SendResult:
        """Send via neonize bridge. CONCEPT:ECO-4.0"""
        result = await asyncio.to_thread(self._client.sendMessage, channel_id, text)
        return SendResult(
            success=True,
            message_id=str(getattr(result, "id", "")),
            platform=PlatformId.WHATSAPP,
            channel_id=channel_id,
        )

    async def listen(self) -> AsyncIterator[InboundEvent]:
        """Yield inbound WhatsApp events. CONCEPT:ECO-4.0"""
        while self._connected:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                yield event
            except TimeoutError:
                continue
