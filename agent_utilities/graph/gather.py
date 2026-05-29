"""Shared concurrent execution utility — CONCEPT:ORCH-1.25.

Centralizes the ``asyncio.gather + return_exceptions=True`` pattern
that was copy-pasted across 5+ files.  Provides a single function
with consistent logging, exception handling, and type-safe results.

Consumers:
    - ``heavy_thinking.py`` — parallel thinker spawning
    - ``hsm.py`` — orthogonal region execution
    - ``reactive/dispatcher.py`` — behavior event dispatch

Note: ``ParallelEngine._execute_wave`` keeps its own ``asyncio.gather``
because it has deeply integrated circuit breaker + wave result
construction that doesn't fit this generic utility.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


async def gather_with_resilience(
    tasks: list[Coroutine[Any, Any, T]],
    *,
    label: str = "gather",
) -> list[T | BaseException]:
    """Run awaitables concurrently, returning results and exceptions inline.

    Replaces the repeated pattern::

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                logger.warning(...)

    with a single call that handles logging automatically.

    Args:
        tasks: Awaitable tasks to execute concurrently.
        label: Logging label for diagnostic messages.

    Returns:
        List of results or ``BaseException`` instances, preserving
        the input order.  Callers should check ``isinstance(r, BaseException)``
        to identify failures.
    """
    if not tasks:
        return []

    raw: list[T | BaseException] = await asyncio.gather(*tasks, return_exceptions=True)

    for i, r in enumerate(raw):
        if isinstance(r, BaseException):
            logger.warning(
                "[%s] Task %d/%d failed: %s",
                label,
                i + 1,
                len(raw),
                r,
            )

    return raw
