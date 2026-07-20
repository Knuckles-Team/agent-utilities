"""Tests for the messaging media-attachment persist path (CONCEPT:AU-KG.identity.asset-occurrence).

AU-P1-4: ``_persist_media`` must create a DISTINCT occurrence per (message, attachment)
call — never collapse two different messages' attachments onto one asset — and pass
through per-message provenance (owner/event_time/platform/channel/thread/message id)
to :class:`MediaStore`. Offline: HTTP downloads are faked via a monkeypatched
``httpx.AsyncClient`` and :class:`MediaStore` itself is faked, so no engine or network
is touched.
"""

from __future__ import annotations

import httpx
import pytest

from agent_utilities.messaging import router
from agent_utilities.messaging.models import (
    EventType,
    InboundEvent,
    MediaAttachment,
    MediaType,
    Message,
)


class _FakeResp:
    def __init__(self, content: bytes = b"IMGBYTES") -> None:
        self.content = content
        self.headers = {"content-type": "image/png"}

    def raise_for_status(self) -> None:
        return None


class _FakeHttpxClient:
    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self) -> _FakeHttpxClient:
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def get(self, url: str) -> _FakeResp:
        return _FakeResp()


class _FakeStore:
    def __init__(self) -> None:
        self.calls: list[tuple[bytes, dict]] = []

    def store_media(self, data: bytes, **kwargs):
        self.calls.append((data, kwargs))
        return object()


def _event(*, channel_id: str, user_id: str, msg_id: str, url: str) -> InboundEvent:
    return InboundEvent(
        event_type=EventType.MESSAGE,
        platform="slack",
        channel_id=channel_id,
        thread_id="T1",
        user_id=user_id,
        message=Message(
            id=msg_id,
            author_id=user_id,
            channel_id=channel_id,
            attachments=[MediaAttachment(media_type=MediaType.IMAGE, url=url)],
        ),
    )


@pytest.mark.asyncio
async def test_persist_media_creates_distinct_call_per_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(httpx, "AsyncClient", _FakeHttpxClient)
    store = _FakeStore()
    monkeypatch.setattr(router, "_resolve_media_store", lambda engine: store)

    # Same image URL/bytes, TWO different messages/users/channels.
    ev1 = _event(channel_id="C1", user_id="u1", msg_id="m1", url="http://x/img.png")
    ev2 = _event(channel_id="C2", user_id="u2", msg_id="m2", url="http://x/img.png")

    await router._persist_media(object(), ev1, message_memory_id="mem:1")
    await router._persist_media(object(), ev2, message_memory_id="mem:2")

    # AU-P1-4: two attachments, two independent store_media calls — never
    # collapsed/skipped just because the bytes are identical.
    assert len(store.calls) == 2
    (data1, kw1), (data2, kw2) = store.calls
    assert data1 == data2 == b"IMGBYTES"

    # Each call carries its OWN message's provenance.
    assert kw1["message_id"] == "mem:1" and kw2["message_id"] == "mem:2"
    assert kw1["owner"] == "u1" and kw2["owner"] == "u2"
    assert kw1["source"] == "slack" and kw2["source"] == "slack"
    assert kw1["provenance"]["channel_id"] == "C1"
    assert kw2["provenance"]["channel_id"] == "C2"
    assert kw1["provenance"]["thread_id"] == "T1"
    assert kw1["provenance"]["message_id"] == "m1"
    assert kw2["provenance"]["message_id"] == "m2"


@pytest.mark.asyncio
async def test_persist_media_noop_without_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "AsyncClient", _FakeHttpxClient)
    monkeypatch.setattr(router, "_resolve_media_store", lambda engine: None)
    ev = _event(channel_id="C1", user_id="u1", msg_id="m1", url="http://x/img.png")
    # Must not raise even though no store is available.
    await router._persist_media(object(), ev, message_memory_id="mem:1")


@pytest.mark.asyncio
async def test_persist_media_noop_without_media(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _FakeStore()
    monkeypatch.setattr(router, "_resolve_media_store", lambda engine: store)
    ev = InboundEvent(
        event_type=EventType.MESSAGE,
        platform="slack",
        channel_id="C1",
        message=Message(id="m1", author_id="u1"),
    )
    await router._persist_media(object(), ev, message_memory_id="mem:1")
    assert store.calls == []
