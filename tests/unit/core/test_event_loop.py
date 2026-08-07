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
import contextvars
import threading

import nest_asyncio
import pytest

from agent_utilities.core.event_loop import (
    allow_nested_run_sync,
    run_blocking_ordered,
    run_sync_isolated,
)


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
        compress_to_memento(
            MagicMock(), [{"role": "user", "content": "hi"}], dry_run=True
        )

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


def test_run_sync_isolated_runs_inline_with_no_running_loop():
    """No running loop on this thread -> ``fn`` runs directly, same thread."""
    this_thread = threading.current_thread()
    seen: dict[str, threading.Thread] = {}

    def fn() -> int:
        seen["thread"] = threading.current_thread()
        return 42

    assert run_sync_isolated(fn) == 42
    assert seen["thread"] is this_thread


async def test_run_sync_isolated_offloads_to_worker_thread_inside_a_running_loop():
    """BUG-2 (kg-exhaustive-smoke.md): called from inside a running event loop,
    ``fn`` must run on a different (worker) thread, not raise/deadlock."""
    this_thread = threading.current_thread()
    seen: dict[str, threading.Thread] = {}

    def fn() -> int:
        seen["thread"] = threading.current_thread()
        return 7

    result = run_sync_isolated(fn)

    assert result == 7
    assert seen["thread"] is not this_thread


async def test_run_sync_isolated_survives_a_real_run_sync_style_nested_loop():
    """The exact failure mode: a callable that itself spins ``asyncio.run()``
    (mirroring ``pydantic_ai.Agent.run_sync``) must not raise "This event loop
    is already running" when invoked from inside a running loop."""

    def fake_agent_run_sync() -> str:
        async def _inner() -> str:
            return "planned"

        return asyncio.run(_inner())

    assert run_sync_isolated(fake_agent_run_sync) == "planned"


async def test_run_sync_isolated_propagates_exceptions():
    def boom() -> None:
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        run_sync_isolated(boom)


async def test_run_blocking_ordered_keeps_loop_live_and_propagates_context():
    marker = contextvars.ContextVar("ordered_worker_marker", default="")
    entered = threading.Event()
    release = threading.Event()
    loop_thread = threading.current_thread()
    observed: dict[str, object] = {}

    def blocking() -> str:
        observed["thread"] = threading.current_thread()
        observed["marker"] = marker.get()
        entered.set()
        release.wait(0.5)
        return "done"

    token = marker.set("request-context")
    try:
        task = asyncio.create_task(run_blocking_ordered(blocking))
        assert await asyncio.to_thread(entered.wait, 0.2)

        # The coroutine can still be scheduled while the worker is blocked.
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        assert await asyncio.wait_for(task, timeout=1.0) == "done"
    finally:
        marker.reset(token)
        release.set()

    assert observed["marker"] == "request-context"
    assert observed["thread"] is not loop_thread


async def test_run_blocking_ordered_reports_cancellation_after_completion():
    entered = threading.Event()
    release = threading.Event()
    completed = threading.Event()

    def blocking_write() -> None:
        entered.set()
        release.wait(0.5)
        completed.set()

    task = asyncio.create_task(run_blocking_ordered(blocking_write))
    assert await asyncio.to_thread(entered.wait, 0.2)
    task.cancel()

    # Cancellation must not orphan an already-started graph write.
    await asyncio.sleep(0.02)
    assert not task.done()
    assert not completed.is_set()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1.0)
    assert completed.is_set()


async def test_run_blocking_ordered_consumes_worker_failure_after_cancellation():
    """A cancelled caller must not leak a later worker error to the loop."""
    entered = threading.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    loop_errors: list[dict[str, object]] = []
    previous_handler = loop.get_exception_handler()

    def failing_write() -> None:
        entered.set()
        release.wait(0.5)
        raise RuntimeError("late write failure")

    loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
    try:
        task = asyncio.create_task(run_blocking_ordered(failing_write))
        assert await asyncio.to_thread(entered.wait, 0.2)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
        # Give the loop a turn to deliver any unhandled-future callback.
        await asyncio.sleep(0)
        assert loop_errors == []
    finally:
        release.set()
        loop.set_exception_handler(previous_handler)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
