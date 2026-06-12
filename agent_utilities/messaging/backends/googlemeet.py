"""Google Meet Messaging Backend (CONCEPT:ECO-4.0).

Manages Google Meet conference creation, participant tracking, and meeting events.
Voice/video platform — no text messaging, but supports call lifecycle events.

Install: ``pip install agent-utilities[messaging-googlemeet]``

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


class GoogleMeetBackend(MessagingBackend):
    """Google Meet backend via Calendar/Meet API. CONCEPT:ECO-4.0"""

    def __init__(self, config: MessagingConfig | None = None) -> None:
        super().__init__(config)
        self._service: Any = None
        self._event_queue: asyncio.Queue[InboundEvent] = asyncio.Queue()

    @property
    def id(self) -> str:
        return "googlemeet"

    @property
    def capabilities(self) -> MessagingCapabilities:
        return CAPABILITY_MATRIX["googlemeet"]

    async def connect(self) -> None:
        """Connect via Google service account. CONCEPT:ECO-4.0"""
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
        except ImportError:
            raise ImportError(
                "Install: pip install agent-utilities[messaging-googlemeet]"
            ) from None

        creds_path = self.config.token or setting("GOOGLE_MEET_SERVICE_ACCOUNT", "")
        if not creds_path:
            raise ValueError("Set GOOGLE_MEET_SERVICE_ACCOUNT path.")
        creds = service_account.Credentials.from_service_account_file(
            creds_path, scopes=["https://www.googleapis.com/auth/calendar"]
        )
        self._service = build("calendar", "v3", credentials=creds)
        self._connected = True
        logger.info("[CONCEPT:ECO-4.0] Google Meet backend connected.")

    async def send_message(
        self, channel_id: str, text: str, **kwargs: Any
    ) -> SendResult:
        """Not supported — Meet is voice/video only. CONCEPT:ECO-4.0"""
        return SendResult(
            success=False,
            platform=PlatformId.GOOGLEMEET,
            error="Google Meet does not support text messaging.",
        )

    async def create_meeting(
        self, title: str = "Agent Meeting", **kwargs: Any
    ) -> dict[str, Any]:
        """Create a Google Meet conference. CONCEPT:ECO-4.0"""
        import datetime

        event_body = {
            "summary": title,
            "start": {"dateTime": datetime.datetime.now(datetime.UTC).isoformat()},
            "end": {
                "dateTime": (
                    datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1)
                ).isoformat()
            },
            "conferenceData": {"createRequest": {"requestId": f"agent-{id(self)}"}},
        }
        result = await asyncio.to_thread(
            self._service.events()
            .insert(calendarId="primary", body=event_body, conferenceDataVersion=1)
            .execute
        )
        meet_link = result.get("hangoutLink", "")
        return {"event_id": result.get("id", ""), "meet_link": meet_link}

    async def listen(self) -> AsyncIterator[InboundEvent]:
        """Yield meeting lifecycle events. CONCEPT:ECO-4.0"""
        while self._connected:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=1.0)
                yield event
            except TimeoutError:
                continue
