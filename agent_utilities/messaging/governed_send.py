"""Bounded helpers for the governed outbound messaging boundary."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from agent_utilities.messaging.models import SendResult

logger = logging.getLogger(__name__)

_ACTION_POLICY_DEADLINE_S = 10.5
PolicyRunner = Callable[[Callable[[], Any]], Awaitable[Any]]


async def run_action_policy(
    operation: Callable[[], Any], *, policy_runner: PolicyRunner | None
) -> Any:
    """Run synchronous policy reads and audit writes off the event loop."""
    if policy_runner is None:
        from agent_utilities.mcp.concurrency import run_blocking

        policy_runner = run_blocking
    try:
        return await asyncio.wait_for(
            policy_runner(operation), timeout=_ACTION_POLICY_DEADLINE_S
        )
    except Exception as exc:  # noqa: BLE001 — timeout/capacity/policy-runner failure is authorization unavailable and must fail closed before provider invocation
        logger.warning(
            "[ECO-4.48] action policy runner unavailable; outbound send refused (%s)",
            type(exc).__name__,
        )
        return None


async def mirror_outbound_if_enabled(
    result: SendResult,
    ingest: Callable[[], Awaitable[None]],
    *,
    enabled: bool,
) -> None:
    """Mirror a confirmed send unless its caller disabled persistence."""
    if result.success and enabled:
        await ingest()
