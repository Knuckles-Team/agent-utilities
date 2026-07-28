"""FIX 2 (delegation resilience): the mandatory evidence compilation at the
model-transport boundary must run OFF the asyncio event loop and be BOUNDED, so a
slow/contended retrieval can never block the loop (the graph-os liveness crash) or
overrun the delegation's wall clock — it degrades to ungrounded passthrough fast.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from agent_utilities.core import contextual_model
from agent_utilities.core.contextual_model import (
    _compile_messages_bounded,
    _compiled_evidence_and_bundle_bounded,
)


class _Msg:
    """A stand-in message object (the helper only passes the list through)."""


@pytest.fixture
def messages() -> list[object]:
    return [_Msg(), _Msg()]


@pytest.fixture(autouse=True)
def reset_context_compile_breaker():
    """Keep the process-wide context-compile breaker hermetic per test."""
    contextual_model._ctx_compile_degradation_streak = 0
    contextual_model._ctx_compile_breaker_reopen_at = 0.0
    try:
        yield
    finally:
        contextual_model._ctx_compile_degradation_streak = 0
        contextual_model._ctx_compile_breaker_reopen_at = 0.0


async def test_bounded_degrades_to_passthrough_on_timeout(monkeypatch, messages):
    """A compile that blocks past the budget is abandoned; the request degrades to
    ungrounded passthrough ``(messages, None)`` in ~budget, NOT after the block."""
    monkeypatch.setattr(contextual_model, "_CONTEXT_COMPILE_TIMEOUT_S", 0.2)

    def _slow_blocking(_messages, _model_name):  # simulates future.result() stall
        time.sleep(5.0)
        raise AssertionError("compile should have been abandoned at the budget")

    monkeypatch.setattr(
        contextual_model, "_compiled_evidence_and_bundle", _slow_blocking
    )

    start = time.perf_counter()
    governed, bundle = await _compiled_evidence_and_bundle_bounded(messages, "m")
    elapsed = time.perf_counter() - start

    assert governed is messages  # unmodified passthrough (ungrounded)
    assert bundle is None
    assert elapsed < 2.0  # returned at the budget, not after the 5s block


async def test_bounded_degrades_to_passthrough_on_error(monkeypatch, messages):
    """Any compilation error degrades to passthrough rather than propagating."""

    def _boom(_messages, _model_name):
        raise RuntimeError("retrieval exploded")

    monkeypatch.setattr(contextual_model, "_compiled_evidence_and_bundle", _boom)

    governed, bundle = await _compiled_evidence_and_bundle_bounded(messages, "m")
    assert governed is messages
    assert bundle is None


async def test_bounded_fast_path_returns_full_context(monkeypatch, messages):
    """A quick retrieval still returns the FULL compiled result (fast path intact)."""
    sentinel_bundle = object()
    compiled = [_Msg(), *messages]

    def _quick(_messages, _model_name):
        return compiled, sentinel_bundle

    monkeypatch.setattr(contextual_model, "_compiled_evidence_and_bundle", _quick)

    governed, bundle = await _compiled_evidence_and_bundle_bounded(messages, "m")
    assert governed is compiled
    assert bundle is sentinel_bundle


async def test_bounded_does_not_block_the_event_loop(monkeypatch, messages):
    """While a slow compile runs, the event loop stays responsive: a concurrent
    ticker keeps advancing (it would not if the loop were blocked inline)."""
    monkeypatch.setattr(contextual_model, "_CONTEXT_COMPILE_TIMEOUT_S", 0.5)

    def _slow_blocking(_messages, _model_name):
        time.sleep(0.5)
        raise AssertionError("compile should have been abandoned at the budget")

    monkeypatch.setattr(
        contextual_model, "_compiled_evidence_and_bundle", _slow_blocking
    )

    ticks = 0

    async def _ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    ticker_task = asyncio.create_task(_ticker())
    try:
        governed, bundle = await _compiled_evidence_and_bundle_bounded(messages, "m")
    finally:
        ticker_task.cancel()

    assert governed is messages and bundle is None
    # A blocked loop would have advanced the ticker ~0 times during the compile;
    # off-loop, it advances many times across the ~0.5s window.
    assert ticks >= 5


async def test_compile_messages_bounded_degrades(monkeypatch, messages):
    """The messages-only bounded variant (count_tokens/compact_messages) degrades
    to the original messages on failure."""

    def _boom(_messages, _model_name):
        raise RuntimeError("nope")

    monkeypatch.setattr(contextual_model, "_compiled_evidence_and_bundle", _boom)

    governed = await _compile_messages_bounded(messages, "m")
    assert governed is messages
