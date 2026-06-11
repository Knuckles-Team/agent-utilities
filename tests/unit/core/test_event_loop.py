"""Tests for the scoped nest_asyncio shim (core/event_loop.py).

The hygiene contract under test: ``allow_nested_run_sync()`` must NEVER apply
the process-wide nest_asyncio patch unless the calling thread already has a
running event loop. An unconditional ``nest_asyncio.apply()`` permanently
mutates global asyncio internals and (on Python >= 3.14) breaks
``asyncio.current_task()`` for every subsequent task in the process — one
call site poisoned 100+ unrelated async tests in a full-suite run.
"""

from __future__ import annotations

import asyncio

import nest_asyncio
import pytest

from agent_utilities.core.event_loop import allow_nested_run_sync


def test_no_running_loop_does_not_patch(monkeypatch):
    """Outside a loop the shim is a strict no-op — apply() is never called."""
    calls: list[bool] = []
    monkeypatch.setattr(nest_asyncio, "apply", lambda *a, **k: calls.append(True))

    allow_nested_run_sync()

    assert calls == [], "nest_asyncio.apply must not run without a running loop"


async def test_running_loop_applies_patch(monkeypatch):
    """Inside a running loop the shim applies nest_asyncio (the one real need)."""
    calls: list[bool] = []
    monkeypatch.setattr(nest_asyncio, "apply", lambda *a, **k: calls.append(True))

    allow_nested_run_sync()

    assert calls == [True]


async def test_missing_or_failing_nest_asyncio_is_swallowed(monkeypatch):
    """A failing apply() degrades silently — callers keep their own fallback."""

    def _boom(*a, **k):
        raise RuntimeError("patch refused")

    monkeypatch.setattr(nest_asyncio, "apply", _boom)

    allow_nested_run_sync()  # must not raise


def test_memento_compression_does_not_patch_asyncio(monkeypatch):
    """Regression (cross-test contamination): a sync-context memento
    compression must not leave the global nest_asyncio patch behind."""
    from unittest.mock import MagicMock, patch

    from agent_utilities.knowledge_graph.memory import compress_to_memento

    calls: list[bool] = []
    monkeypatch.setattr(nest_asyncio, "apply", lambda *a, **k: calls.append(True))

    with patch("pydantic_ai.Agent.run_sync") as mock_run:
        mock_result = MagicMock()
        mock_result.data = "Memento: ok"
        mock_run.return_value = mock_result
        compress_to_memento(MagicMock(), [{"role": "user", "content": "hi"}], dry_run=True)

    assert calls == [], "compress_to_memento leaked the global nest_asyncio patch"


def test_event_loop_stays_usable_after_sync_call():
    """After a sync-context shim call, normal asyncio task semantics hold."""
    allow_nested_run_sync()

    async def probe() -> bool:
        # asyncio.timeout requires current_task() — the exact API that breaks
        # process-wide when nest_asyncio is applied on Python >= 3.14.
        async with asyncio.timeout(5):
            return asyncio.current_task() is not None

    assert asyncio.run(probe()) is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
