"""How AU reaches EG's ``Decide`` and ``DecisionLog`` methods.

:class:`DecideTransport` is the port; :class:`GeneratedTransport` is the one
adapter, over EG's generated senders (``send_decide``,
``send_decision_log``). Async callers await it directly; sync call sites run
the same coroutine on the engine client's own loop, the pattern
``GraphCompute`` already uses for every other engine call, bounded by a
timeout so a slow engine can only ever cost the fallback.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from agent_utilities.layers.clients import LayerUnavailable, generated

#: The longest a sync call site waits for EG before it falls back.
DEFAULT_SYNC_TIMEOUT_S = 2.0


class DecideTransport(Protocol):
    """Send one ``Decide`` request or ``DecisionLog`` op; return the payload."""

    async def decide(self, request: Mapping[str, Any]) -> Any: ...

    async def log(self, op: Mapping[str, Any]) -> Any: ...

    def run(self, call: Any) -> Any:
        """Drive one of the coroutines above from a sync call site."""
        ...


def _on_loop(loop: asyncio.AbstractEventLoop) -> bool:
    """True on ``loop``'s own thread, where blocking on it would deadlock."""
    try:
        return asyncio.get_running_loop() is loop
    except RuntimeError:
        return False


def _payload(result: Any) -> Any:
    return getattr(result, "payload", result)


@dataclass(frozen=True, slots=True)
class GeneratedTransport:
    """EG's generated contract senders, bound to one client and graph."""

    client: Any
    graph: str | None = None
    loop: asyncio.AbstractEventLoop | None = None
    sync_timeout_s: float = DEFAULT_SYNC_TIMEOUT_S

    async def decide(self, request: Mapping[str, Any]) -> Any:
        send = generated("query", "send_decide")
        return _payload(await send(self.client, {"request": dict(request)}, self.graph))

    async def log(self, op: Mapping[str, Any]) -> Any:
        send = generated("coordination", "send_decision_log")
        return _payload(await send(self.client, {"op": dict(op)}, self.graph))

    def run(self, call: Any) -> Any:
        if self.loop is None or self.loop.is_closed() or _on_loop(self.loop):
            call.close()
            raise LayerUnavailable("no engine loop a sync decision may block on")
        future = asyncio.run_coroutine_threadsafe(call, self.loop)
        return future.result(timeout=self.sync_timeout_s)


__all__ = ["DEFAULT_SYNC_TIMEOUT_S", "DecideTransport", "GeneratedTransport"]
