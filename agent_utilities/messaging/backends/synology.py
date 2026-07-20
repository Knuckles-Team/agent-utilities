"""Synology Chat Backend (CONCEPT:AU-ECO.messaging.native-backend-abstraction). Webhook-based via httpx.

Install: ``pip install agent-utilities[messaging-synology]``
CONCEPT:AU-ECO.messaging.native-backend-abstraction — Native Messaging Backend Abstraction
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


def _configured_webhook(config: MessagingConfig) -> str:
    """Resolve the webhook from an injected profile or a secret reference."""
    if config.webhook_url:
        return str(config.webhook_url)
    reference = str(setting("SYNOLOGY_CHAT_WEBHOOK_URL_REF", "") or "").strip()
    if not reference:
        return ""
    from agent_utilities.security.secrets_client import create_secrets_client

    return str(create_secrets_client().resolve_ref(reference) or "").strip()


class SynologyChatBackend(MessagingBackend):
    """Synology Chat backend via incoming/outgoing webhooks. CONCEPT:AU-ECO.messaging.native-backend-abstraction"""

    def __init__(self, config: MessagingConfig | None = None) -> None:
        super().__init__(config)
        self._client: Any = None
        self._event_queue: asyncio.Queue[InboundEvent] = asyncio.Queue()

    @property
    def id(self) -> str:
        return "synology"

    @property
    def capabilities(self) -> MessagingCapabilities:
        return CAPABILITY_MATRIX["synology"]

    async def connect(self) -> None:
        """Connect to Synology Chat webhook. CONCEPT:AU-ECO.messaging.native-backend-abstraction"""
        from agent_utilities.core.http_client import create_async_http_client
        from agent_utilities.core.transport_security import (
            resolve_configured_tls_profile,
        )

        webhook_url = _configured_webhook(self.config)
        if not webhook_url:
            raise ValueError("Configure a Synology webhook reference.")
        trust = resolve_configured_tls_profile("synology-chat")
        try:
            self._client = create_async_http_client(**trust.httpx_kwargs())
        finally:
            trust.cleanup()
        self._connected = True
        logger.info(
            "[CONCEPT:AU-ECO.messaging.native-backend-abstraction] Synology Chat backend connected."
        )

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
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
        try:
            webhook = _configured_webhook(self.config)
            payload = f'payload={{"text": "{text}"}}'
            resp = await self._client.post(
                webhook,
                content=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            return SendResult(
                success=resp.status_code == 200,
                platform=PlatformId.SYNOLOGY,
                channel_id=channel_id,
            )
        except Exception as e:
            return SendResult(
                success=False,
                platform=PlatformId.SYNOLOGY,
                error=type(e).__name__,
            )

    async def listen(self) -> AsyncIterator[InboundEvent]:
        while self._connected:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                yield event
            except TimeoutError:
                continue
