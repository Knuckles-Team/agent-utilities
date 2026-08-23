"""Regression: one platform's lease loss must not stop another platform's
inbound listener.

Previously ``messaging/daemon.py::_run_poll_loop`` ran EVERY backend's
listener under the ONE ``_serve`` asyncio task and cancelled that whole task
off the single shared ``stop_event`` — so
``messaging/intake_lease.py::run_with_intake_leases`` setting that event on
ANY platform's lease loss silently killed every OTHER healthy platform's
inbound polling too (this is what took Telegram inbound down in production
when only the Mattermost lease was lost).

This drives the REAL ``_run_poll_loop`` — its own event loop, its own
per-platform watcher threads, and the real ``_drop_platform`` task-name
lookup-and-cancel closure — and substitutes only ``_serve``'s backend/router
construction (irrelevant to this bug) with a fake that exposes named,
long-lived tasks through the same ``router_box`` handoff the real ``_serve``
populates. The seam under test — the per-platform ``threading.Event`` →
``loop.call_soon_threadsafe`` → single-task cancellation path — is never
mocked.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

from agent_utilities.messaging import daemon as messaging_daemon


class _FakeRouter:
    """Mimics just the sliver of ``InboundRouter`` ``_drop_platform`` reads."""

    def __init__(self, tasks: list[asyncio.Task[Any]]) -> None:
        self._tasks = tasks


def test_dropping_one_platform_leaves_the_other_listener_running(monkeypatch) -> None:
    ready = threading.Event()
    state: dict[str, dict[str, asyncio.Task[Any]]] = {}

    async def _fake_serve(
        engine: Any, platforms: list[str], router_box: dict[str, Any]
    ) -> None:
        tasks = {
            platform: asyncio.create_task(
                asyncio.sleep(3600), name=f"messaging-router-{platform}"
            )
            for platform in platforms
        }
        state["tasks"] = tasks
        router_box["router"] = _FakeRouter(list(tasks.values()))
        ready.set()
        await asyncio.gather(*tasks.values(), return_exceptions=True)

    monkeypatch.setattr(messaging_daemon, "_serve", _fake_serve)

    stop_event = threading.Event()
    platform_stop_events = {
        "mattermost": threading.Event(),
        "telegram": threading.Event(),
    }

    thread = threading.Thread(
        target=messaging_daemon._run_poll_loop,
        args=(
            object(),
            ["mattermost", "telegram"],
            stop_event,
            platform_stop_events,
        ),
        daemon=True,
        name="test-run-poll-loop",
    )
    thread.start()
    try:
        assert ready.wait(timeout=2.0)
        tasks = state["tasks"]

        # Only mattermost's lease is lost.
        platform_stop_events["mattermost"].set()

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not tasks["mattermost"].done():
            time.sleep(0.02)

        assert tasks["mattermost"].done(), (
            "mattermost's listener task must be cancelled once its "
            "platform_stop_event fires"
        )
        # Give a cancellation-that-shouldn't-happen a fair chance to show up.
        time.sleep(0.2)
        assert not tasks["telegram"].done(), (
            "telegram's listener task must NOT be touched by mattermost's "
            "lease loss"
        )
        assert not stop_event.is_set(), (
            "losing one of two leases must not request the full daemon stop"
        )
    finally:
        stop_event.set()
        thread.join(timeout=2.0)
        assert not thread.is_alive()
