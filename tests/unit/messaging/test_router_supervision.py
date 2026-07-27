"""Resilient supervised messaging listeners + observability (CONCEPT:AU-ECO.messaging.native-backend-abstraction).

Regression coverage for the Telegram-resilience fix:

FIX 1 — a per-backend listener must SELF-HEAL instead of dying on the first recoverable
error. ``InboundRouter._supervise_backend`` restarts ``listen()`` with a bounded
exponential backoff, so a transient Telegram ``getUpdates`` 409 restart race recovers
automatically. ``CancelledError`` still shuts down cleanly (no restart) and
``NotImplementedError`` still gives up (the backend can't listen). The Telegram backend
now surfaces ``telegram.error.Conflict`` (409) to that supervisor after stopping the
updater cleanly, instead of swallowing it into a dead generator.

FIX 2 — the bundled messaging co-service's lifecycle/error logs must reach stderr
(→ ``kubectl logs``) even though the graph-os root logger is pinned to WARNING at build
time. ``daemon._ensure_messaging_log_visibility`` attaches one dedicated INFO stderr
handler to the ``agent_utilities.messaging`` package logger and stops propagation.

All backends are mocked; no test touches the real Telegram API.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import types
from collections.abc import AsyncIterator
from typing import Any

import pytest

from agent_utilities.messaging import daemon as messaging_daemon
from agent_utilities.messaging.models import EventType
from agent_utilities.messaging.router import InboundRouter

# pytest is configured with ``asyncio_mode = auto`` (pytest.ini / pyproject), so ``async
# def`` tests run automatically and the two synchronous log-visibility tests stay sync.


class _ScriptedBackend:
    """A backend whose successive ``listen()`` calls follow a scripted sequence.

    Each script step is either a ``BaseException`` instance (raised before yielding) or a
    list of events to yield then complete. The last step repeats if ``listen()`` is called
    more times than the script length. ``listen_calls`` records how many attempts the
    supervisor made — the key signal for "did it restart / give up".
    """

    def __init__(self, script: list[Any]) -> None:
        self.id = "scripted"
        self._connected = True
        self.script = list(script)
        self.listen_calls = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def listen(self) -> AsyncIterator[Any]:
        idx = self.listen_calls
        self.listen_calls += 1
        step = self.script[idx] if idx < len(self.script) else self.script[-1]
        if isinstance(step, BaseException):
            raise step
        for event in step:
            yield event


def _msg_event() -> Any:
    """A minimal event that ``_dispatch`` can route to the default handler."""
    return types.SimpleNamespace(event_type=EventType.MESSAGE)


async def test_listener_restarts_after_one_failure_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FIX 1 (1): a backend that raises once is RESTARTED and then delivers — it must not
    die after a single failure."""
    monkeypatch.setenv("MESSAGING_LISTEN_BACKOFF_BASE_S", "0.01")
    monkeypatch.setenv("MESSAGING_LISTEN_BACKOFF_MAX_S", "0.02")

    received: list[Any] = []
    router = InboundRouter()
    router._running = True

    async def handler(event: Any, backend: Any) -> None:
        received.append(event)
        router._running = False  # stop once the restarted listener delivers

    router.set_default_handler(handler)

    event = _msg_event()
    backend = _ScriptedBackend([RuntimeError("transient 409-like boom"), [event]])

    await asyncio.wait_for(router._supervise_backend(backend), timeout=5)

    assert backend.listen_calls == 2  # failed once, supervisor restarted listen()
    assert received == [event]  # the restarted listener actually delivered an event


async def test_cancellederror_stops_cleanly_without_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FIX 1 (2): ``CancelledError`` is a clean shutdown — propagate, never restart,
    never back off."""
    slept: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

    router = InboundRouter()
    router._running = True
    backend = _ScriptedBackend([asyncio.CancelledError()])

    with pytest.raises(asyncio.CancelledError):
        await router._supervise_backend(backend)

    assert backend.listen_calls == 1  # gave up immediately, no restart attempt
    assert slept == []  # no backoff on a clean shutdown


async def test_notimplementederror_gives_up_without_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FIX 1 (3): ``NotImplementedError`` means the backend cannot listen — give up, do
    not restart, do not back off."""
    slept: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

    router = InboundRouter()
    router._running = True
    backend = _ScriptedBackend([NotImplementedError("outbound only")])

    await router._supervise_backend(backend)  # returns cleanly, no exception

    assert backend.listen_calls == 1  # single attempt, then gave up
    assert slept == []  # no backoff when giving up


async def test_backoff_is_bounded_and_never_busy_loops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FIX 1 (4): a persistently-failing backend backs off with a BOUNDED, capped,
    exponential delay — exactly one sleep per attempt (never a tight/busy loop)."""
    monkeypatch.setenv("MESSAGING_LISTEN_BACKOFF_BASE_S", "1")
    monkeypatch.setenv("MESSAGING_LISTEN_BACKOFF_MAX_S", "5")
    monkeypatch.setenv("MESSAGING_LISTEN_HEALTHY_RESET_S", "60")

    router = InboundRouter()
    router._running = True
    delays: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        delays.append(delay)
        if len(delays) >= 6:
            router._running = False  # end the supervision loop after enough samples

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

    backend = _ScriptedBackend([RuntimeError("hard down")])  # always fails
    await asyncio.wait_for(router._supervise_backend(backend), timeout=5)

    # One backoff sleep per failed attempt — never a zero-delay tight loop.
    assert len(delays) == 6
    assert all(d > 0 for d in delays)
    # Exponential from base=1, doubling, capped at 5: 1, 2, 4, 5, 5, 5.
    assert delays == [1.0, 2.0, 4.0, 5.0, 5.0, 5.0]
    assert max(delays) <= 5.0  # never exceeds the configured cap
    assert backend.listen_calls == len(delays)  # each sleep followed a real attempt


async def test_backoff_resets_after_a_sustained_healthy_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FIX 1: after a run that stayed healthy for at least the reset window, the backoff
    drops back to the base instead of continuing to grow."""
    monkeypatch.setenv("MESSAGING_LISTEN_BACKOFF_BASE_S", "1")
    monkeypatch.setenv("MESSAGING_LISTEN_BACKOFF_MAX_S", "60")
    # Any run that lasts >= 0s counts as "healthy" → the backoff always resets to base.
    monkeypatch.setenv("MESSAGING_LISTEN_HEALTHY_RESET_S", "0")

    router = InboundRouter()
    router._running = True
    delays: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        delays.append(delay)
        if len(delays) >= 3:
            router._running = False

    monkeypatch.setattr(asyncio, "sleep", _fake_sleep)

    backend = _ScriptedBackend([RuntimeError("blip")])
    await asyncio.wait_for(router._supervise_backend(backend), timeout=5)

    # ran_for (~0s) >= healthy_reset (0s) every time → delay stays pinned at the base.
    assert delays == [1.0, 1.0, 1.0]


# ── FIX 1 part B: Telegram getUpdates 409 (Conflict) handling ────────────────


@pytest.fixture()
def fake_telegram(monkeypatch: pytest.MonkeyPatch) -> type[Exception]:
    """Inject a minimal ``telegram.error`` module so the backend's lazy
    ``from telegram.error import Conflict`` resolves without python-telegram-bot
    installed. Returns the fake ``Conflict`` class."""
    tg = types.ModuleType("telegram")
    err = types.ModuleType("telegram.error")

    class Conflict(Exception):
        pass

    err.Conflict = Conflict  # type: ignore[attr-defined]
    tg.error = err  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "telegram", tg)
    monkeypatch.setitem(sys.modules, "telegram.error", err)
    return Conflict


async def test_telegram_conflict_propagates_and_stops_updater_cleanly(
    fake_telegram: type[Exception], monkeypatch: pytest.MonkeyPatch
) -> None:
    """FIX 1 part B: a 409 ``Conflict`` on ``start_polling`` must stop the updater cleanly
    (no leaked half-open poller) and PROPAGATE to the supervisor — not be swallowed."""
    from agent_utilities.messaging.backends.telegram import TelegramBackend
    from agent_utilities.messaging.models import MessagingConfig

    conflict_cls = fake_telegram
    monkeypatch.setenv("MESSAGING_WEBHOOK_BASE_URL", "")  # force the polling path

    stop_calls: list[bool] = []

    class _Updater:
        def __init__(self) -> None:
            self.running = True

        async def start_polling(self) -> None:
            raise conflict_cls("terminated by other getUpdates request")

        async def stop(self) -> None:
            stop_calls.append(True)
            self.running = False

    class _App:
        updater = _Updater()

    backend = TelegramBackend(MessagingConfig(token="123456789:SECRETPART"))
    backend._connected = True
    backend._app = _App()  # type: ignore[assignment]

    with pytest.raises(conflict_cls):
        await backend._start_intake()

    assert backend._polling is False  # not left half-open
    assert stop_calls == [True]  # updater.stop() ran on the way out (clean stop)


async def test_telegram_conflict_reaches_supervisor_and_retries(
    fake_telegram: type[Exception], monkeypatch: pytest.MonkeyPatch
) -> None:
    """FIX 1 end-to-end (mocked): a Telegram ``listen()`` that raises ``Conflict`` once is
    restarted by the supervisor and then serves — proving the 409 no longer permanently
    kills the listener."""
    from agent_utilities.messaging.backends.telegram import TelegramBackend
    from agent_utilities.messaging.models import MessagingConfig

    conflict_cls = fake_telegram
    monkeypatch.setenv("MESSAGING_LISTEN_BACKOFF_BASE_S", "0.01")
    monkeypatch.setenv("MESSAGING_LISTEN_BACKOFF_MAX_S", "0.02")

    backend = TelegramBackend(MessagingConfig(token="1:x"))
    backend._connected = True

    calls: list[int] = []
    event = _msg_event()

    async def _listen() -> AsyncIterator[Any]:
        calls.append(1)
        if len(calls) == 1:
            raise conflict_cls("409 restart race")
        yield event

    # Replace listen() with the scripted async generator (no real Telegram app needed).
    monkeypatch.setattr(backend, "listen", _listen)

    router = InboundRouter()
    router._running = True

    received: list[Any] = []

    async def handler(ev: Any, _b: Any) -> None:
        received.append(ev)
        router._running = False

    router.set_default_handler(handler)

    await asyncio.wait_for(router._supervise_backend(backend), timeout=5)

    assert len(calls) == 2  # Conflict on attempt 1, restarted, served on attempt 2
    assert received == [event]


# ── FIX 2: messaging-log visibility ──────────────────────────────────────────


@pytest.fixture()
def clean_messaging_logger() -> Any:
    """Snapshot and restore the shared ``agent_utilities.messaging`` logger so this file's
    logging manipulation never leaks into other tests."""
    pkg = logging.getLogger("agent_utilities.messaging")
    saved = (list(pkg.handlers), pkg.level, pkg.propagate, pkg.disabled)
    # Start from a known state: drop any handler our helper previously attached.
    for handler in list(pkg.handlers):
        if getattr(handler, messaging_daemon._MESSAGING_LOG_HANDLER_MARK, False):
            pkg.removeHandler(handler)
    pkg.propagate = True
    pkg.setLevel(logging.WARNING)
    pkg.disabled = False
    yield pkg
    pkg.handlers[:] = saved[0]
    pkg.setLevel(saved[1])
    pkg.propagate = saved[2]
    pkg.disabled = saved[3]


def test_ensure_messaging_log_visibility_emits_info_to_stderr(
    clean_messaging_logger: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """FIX 2: after the helper runs, a messaging INFO lifecycle line is emitted to stderr
    even with the package logger having inherited a WARNING/no-INFO posture."""
    pkg = clean_messaging_logger

    messaging_daemon._ensure_messaging_log_visibility()

    assert pkg.level == logging.INFO
    assert pkg.propagate is False  # decoupled from root → no double-log, always visible
    marked = [
        h
        for h in pkg.handlers
        if getattr(h, messaging_daemon._MESSAGING_LOG_HANDLER_MARK, False)
    ]
    assert len(marked) == 1
    assert marked[0].stream is sys.stderr  # stderr, never stdout (stdio JSON-RPC safety)

    # An INFO record from a child messaging logger now reaches stderr.
    logging.getLogger("agent_utilities.messaging.router").info(
        "Listening for events on 'telegram'..."
    )
    captured = capsys.readouterr()
    assert "Listening for events on 'telegram'" in captured.err
    assert captured.out == ""  # nothing leaked to stdout


def test_ensure_messaging_log_visibility_is_idempotent(
    clean_messaging_logger: Any,
) -> None:
    """FIX 2: repeated calls (co-service restarts) must not stack duplicate handlers."""
    pkg = clean_messaging_logger

    messaging_daemon._ensure_messaging_log_visibility()
    messaging_daemon._ensure_messaging_log_visibility()
    messaging_daemon._ensure_messaging_log_visibility()

    marked = [
        h
        for h in pkg.handlers
        if getattr(h, messaging_daemon._MESSAGING_LOG_HANDLER_MARK, False)
    ]
    assert len(marked) == 1  # exactly one dedicated handler, no matter how many calls
