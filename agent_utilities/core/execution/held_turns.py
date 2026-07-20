"""CONCEPT:AU-ORCH.execution.held-turn-registry-mid — Held-turn registry for mid-turn tool-result injection.

Assimilated from open-design's keep-stdin-open loop (``/api/runs/:id/tool-result``): when a step pauses
on a ``tool_use``, the run is registered as *waiting* for a host answer; a later POST resolves it and
the run resumes the same turn. Here the wait is an :class:`asyncio.Future` keyed by ``(run_id,
tool_use_id)`` — agent-agnostic, so a human (``/api/human``), another agent (A2A), or a tool can
answer. A timeout auto-fails a stuck turn.

This is the framework-native, transport-independent core; an adapter that keeps a subprocess stdin
open writes the resolved result back as a JSONL line.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _Pending:
    future: asyncio.Future
    tool_use_id: str
    created_at: float


@dataclass(slots=True)
class HeldTurnRegistry:
    """Tracks runs paused mid-turn awaiting a tool result."""

    _pending: dict[str, _Pending] = field(default_factory=dict)

    @staticmethod
    def _key(run_id: str, tool_use_id: str) -> str:
        return f"{run_id}::{tool_use_id}"

    def is_waiting(self, run_id: str, tool_use_id: str | None = None) -> bool:
        if tool_use_id is not None:
            return self._key(run_id, tool_use_id) in self._pending
        return any(k.startswith(f"{run_id}::") for k in self._pending)

    async def wait_for_result(
        self, run_id: str, tool_use_id: str, *, timeout: float = 300.0
    ) -> dict[str, Any]:
        """Register the run as waiting and block until a result is posted or ``timeout`` elapses.

        Raises :class:`asyncio.TimeoutError` on timeout (caller decides how to fail the turn).
        """
        loop = asyncio.get_event_loop()
        fut: asyncio.Future = loop.create_future()
        key = self._key(run_id, tool_use_id)
        self._pending[key] = _Pending(
            future=fut, tool_use_id=tool_use_id, created_at=loop.time()
        )
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        finally:
            self._pending.pop(key, None)

    def resolve(self, run_id: str, tool_use_id: str, result: dict[str, Any]) -> bool:
        """Resolve a waiting run with ``result``. Returns False if no matching waiter exists."""
        pending = self._pending.get(self._key(run_id, tool_use_id))
        if pending is None or pending.future.done():
            return False
        pending.future.set_result(result)
        return True

    def resolve_any(self, run_id: str, result: dict[str, Any]) -> bool:
        """Resolve the first pending tool_use for ``run_id`` (when the caller omits a tool_use_id)."""
        for key, pending in self._pending.items():
            if key.startswith(f"{run_id}::") and not pending.future.done():
                pending.future.set_result(result)
                return True
        return False


_default_registry: HeldTurnRegistry | None = None


def get_held_turn_registry() -> HeldTurnRegistry:
    """Process-wide held-turn registry (shared by the engine and the tool-result route)."""
    global _default_registry
    if _default_registry is None:
        _default_registry = HeldTurnRegistry()
    return _default_registry
