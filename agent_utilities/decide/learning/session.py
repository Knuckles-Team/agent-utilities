"""The EG session retrieval learning speaks through: the installed decision
runner's transport and tenant (:mod:`agent_utilities.decide`). With no runner
installed there is no session, and every learning call is a no-op -- the same
contract as a decision point with no engine: AU keeps working, EG learns
nothing.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar, cast

from agent_utilities import decide
from agent_utilities.decide.learning.ops import learn_op, recorded, rows_of

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class LearningSession:
    """One tenant's retrieval-learning channel: ``DecisionLog`` writes and
    SQL reads of the decision views."""

    transport: Any
    tenant: str

    async def asend(self, op: Mapping[str, Any]) -> Any:
        return await self.transport.log(op)

    async def learn(self, write: str, kind: str, **fields: Any) -> Mapping[str, Any]:
        """One ``DecisionLog.learn`` write; its ``kind`` recorded body."""
        return recorded(await self.asend(learn_op(self.tenant, write, **fields)), kind)

    async def aquery(self, sql: str) -> list[dict[str, Any]]:
        """One read-only SQL statement over the caller's decision views."""
        return rows_of(await self.transport.sql(sql))

    def query(self, sql: str) -> list[dict[str, Any]]:
        """Drive :meth:`aquery` from a sync call site on the engine loop."""
        return self.transport.run(self.aquery(sql))

    def send(self, op: Mapping[str, Any]) -> Any:
        """Drive :meth:`asend` from a sync call site on the engine loop."""
        return self.transport.run(self.asend(op))

    def drive(self, call: Coroutine[Any, Any, T]) -> T:
        """Drive a LONG learning job (a generation swap) on the transport's
        engine loop from a background thread, without the sync call-site
        timeout; a transport with no loop of its own drives it itself."""
        loop = getattr(self.transport, "loop", None)
        if loop is None:
            return cast(T, self.transport.run(call))
        return asyncio.run_coroutine_threadsafe(call, loop).result()


def current_session() -> LearningSession | None:
    """The session of the installed runner, if any."""
    runner = decide.current_runner()
    if runner is None:
        return None
    return LearningSession(runner.transport, runner.tenant)


__all__ = ["LearningSession", "current_session"]
