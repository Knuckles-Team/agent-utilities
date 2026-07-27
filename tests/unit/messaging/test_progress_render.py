"""Tests for the messaging entrypoint's live progress render
(CONCEPT:AU-ORCH.execution.messaging-orchestration-transparency).

The messaging router renders the core ProgressEvent stream into ONE status message that evolves
in place. These tests cover the per-surface behaviors the feature requires — THROTTLED /
coalesced edits, edit-capability GATING, and graceful DEGRADE to a single final reply when a
message cannot be edited — none of which may ever break the reply path.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from agent_utilities.messaging.base import MessagingBackend
from agent_utilities.messaging.models import SendResult
from agent_utilities.messaging.router import (
    _backend_supports_edit,
    _progress_streaming_enabled,
    _ProgressChecklist,
)


def _event(stage: str, status: str = "ok", detail: str = "") -> Any:
    from agent_utilities.orchestration.agent_runner import ProgressEvent

    return ProgressEvent(
        run_id="run:x", stage=stage, status=status, detail=detail, ts=0.0
    )


class _EditStub(MessagingBackend):
    """A backend that supports in-place edit and records every send/edit."""

    def __init__(self, *, post_ok: bool = True) -> None:
        super().__init__()
        self._connected = True
        self.sent: list[str] = []
        self.edited: list[str] = []
        self._post_ok = post_ok

    @property
    def id(self) -> str:
        return "editstub"

    @property
    def capabilities(self) -> Any:  # never read by the checklist
        return None

    async def connect(self) -> None:
        return None

    async def send_message(
        self, channel_id, text, *, thread_id="", reply_to_id="", metadata=None
    ) -> SendResult:
        self.sent.append(text)
        return SendResult(
            success=self._post_ok, message_id="msg-1" if self._post_ok else ""
        )

    async def edit_message(
        self, channel_id, message_id, text, *, metadata=None
    ) -> SendResult:
        self.edited.append(text)
        return SendResult(success=True, message_id=message_id)


class _NoEditStub(MessagingBackend):
    """A backend that does NOT override ``edit_message`` (uses the safe base send-fallback)."""

    def __init__(self) -> None:
        super().__init__()
        self._connected = True
        self.sent: list[str] = []

    @property
    def id(self) -> str:
        return "noeditstub"

    @property
    def capabilities(self) -> Any:
        return None

    async def connect(self) -> None:
        return None

    async def send_message(
        self, channel_id, text, *, thread_id="", reply_to_id="", metadata=None
    ) -> SendResult:
        self.sent.append(text)
        return SendResult(success=True, message_id=f"m{len(self.sent)}")


def test_backend_supports_edit_detects_native_override() -> None:
    """Only a backend that OVERRIDES edit_message counts as edit-capable; the base default
    (a send-fallback) does not."""
    assert _backend_supports_edit(_EditStub()) is True
    assert _backend_supports_edit(_NoEditStub()) is False


def test_progress_streaming_flag_defaults_off_and_honors_opt_in() -> None:
    """The surface behavior is opt-in — off unless MESSAGING_PROGRESS_STREAMING is set."""
    with patch("agent_utilities.core.config.setting", return_value=False):
        assert _progress_streaming_enabled() is False
    with patch("agent_utilities.core.config.setting", return_value=True):
        assert _progress_streaming_enabled() is True


@pytest.mark.asyncio
async def test_checklist_throttles_and_coalesces_edits() -> None:
    """A burst of events within the throttle window posts ONE status and coalesces the rest
    (zero intermediate edits); finalize then writes the real answer with a single edit."""
    stub = _EditStub()
    # A huge interval means no event ever re-edits mid-run — the strongest coalescing case.
    checklist = _ProgressChecklist(stub, "chan", min_interval_s=1_000.0)

    for stage in ("start", "route", "tool_call", "tool_result", "tool_result", "done"):
        await checklist.sink(_event(stage))

    # Exactly ONE status message posted; every later event was coalesced (throttled) away.
    assert len(stub.sent) == 1
    assert stub.sent[0].startswith("🔎 Working on it")
    assert stub.edited == []

    delivered = await checklist.finalize("Here is the final answer.")
    assert delivered is True
    assert stub.edited == ["Here is the final answer."]


@pytest.mark.asyncio
async def test_checklist_edits_one_message_in_place_when_not_throttled() -> None:
    """With the throttle open, each event after the first edits the SAME message in place —
    one evolving message, never N separate posts."""
    stub = _EditStub()
    checklist = _ProgressChecklist(stub, "chan", min_interval_s=0.0)

    for stage in ("start", "route", "tool_call", "done"):
        await checklist.sink(_event(stage))

    assert len(stub.sent) == 1  # ONE status message, ever
    assert len(stub.edited) == 3  # the 3 events after the initial post each edited in place


@pytest.mark.asyncio
async def test_checklist_degrades_to_final_reply_when_status_not_postable() -> None:
    """When the status message cannot be posted (a failing/edit-less surface), finalize returns
    False so the caller sends the answer as a normal reply — the answer is never lost."""
    stub = _EditStub(post_ok=False)  # the initial status post fails → no message id captured
    checklist = _ProgressChecklist(stub, "chan", min_interval_s=0.0)

    await checklist.sink(_event("start"))
    await checklist.sink(_event("done"))

    delivered = await checklist.finalize("The answer.")
    assert delivered is False  # the caller must fall back to a normal send
    assert stub.edited == []  # nothing was edited (there was no message to edit)


@pytest.mark.asyncio
async def test_checklist_never_raises_when_backend_send_throws() -> None:
    """A throwing backend must never break the render — the sink swallows and moves on."""

    class _ThrowingBackend(_EditStub):
        async def send_message(self, *a, **k) -> SendResult:  # type: ignore[override]
            raise RuntimeError("platform down")

    checklist = _ProgressChecklist(_ThrowingBackend(), "chan", min_interval_s=0.0)
    # Must not raise despite the backend throwing on the initial post.
    await checklist.sink(_event("start"))
    await checklist.sink(_event("done"))
    assert await checklist.finalize("answer") is False
