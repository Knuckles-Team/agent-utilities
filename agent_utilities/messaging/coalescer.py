"""Burst-mode message coalescing (CONCEPT:ECO-4.63).

When the user fires several messages in quick succession, collapse them into a SINGLE
agent turn — one holistic reply, one LLM call — instead of answering each individually.
A per-conversation debounce window accumulates messages; when the user pauses (``window_s``)
or a hard cap (``max_wait_s``) elapses, the batch is flushed to one handler.

This is a shared core primitive: agent-terminal-ui imports the same ``BurstCoalescer`` so
burst behavior is identical across every chat surface.

CONCEPT:ECO-4.63 — Burst-mode message coalescing (one holistic reply per burst)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

FlushHandler = Callable[[str, list[Any]], Awaitable[None]]


class BurstCoalescer:
    """Debounce per-key items into batches flushed to ``on_flush(key, items)``.

    CONCEPT:ECO-4.63 — keyed by conversation (e.g. ``"telegram:<chat>"``). Each ``submit``
    restarts the quiet window; the batch flushes when the user pauses for ``window_s`` or
    ``max_wait_s`` elapses since the first message (so a continuous typer still gets a
    reply). ``window_s=0`` disables coalescing (flush each item immediately).
    """

    def __init__(
        self,
        on_flush: FlushHandler,
        *,
        window_s: float = 2.5,
        max_wait_s: float = 12.0,
    ) -> None:
        self._on_flush = on_flush
        self._window = max(0.0, float(window_s))
        self._max = max(self._window, float(max_wait_s))
        self._buffers: dict[str, list[Any]] = {}
        self._timers: dict[str, asyncio.Task[None]] = {}
        self._first: dict[str, float] = {}

    async def submit(self, key: str, item: Any) -> None:
        """Add ``item`` to ``key``'s batch and (re)arm the debounce timer."""
        loop = asyncio.get_running_loop()
        self._buffers.setdefault(key, []).append(item)
        self._first.setdefault(key, loop.time())
        # Hard cap: a user who never pauses still gets a reply.
        if loop.time() - self._first[key] >= self._max:
            await self._flush(key)
            return
        old = self._timers.get(key)
        if old is not None and not old.done():
            old.cancel()
        self._timers[key] = asyncio.create_task(self._wait_and_flush(key))

    async def _wait_and_flush(self, key: str) -> None:
        try:
            await asyncio.sleep(self._window)
        except asyncio.CancelledError:
            return
        await self._flush(key)

    async def _flush(self, key: str) -> None:
        items = self._buffers.pop(key, [])
        self._first.pop(key, None)
        timer = self._timers.pop(key, None)
        if timer is not None and not timer.done():
            timer.cancel()
        if not items:
            return
        try:
            await self._on_flush(key, items)
        except Exception as e:  # noqa: BLE001 — one bad batch must not kill the loop
            logger.error("[CONCEPT:ECO-4.63] burst flush failed for %s: %s", key, e)
