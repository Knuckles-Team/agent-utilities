#!/usr/bin/python
from __future__ import annotations

"""Ambient WorkItem-execution context — attribution aid only (BUG-070).

BUG-064: 45,478 of 47,465 live nodes were stamped ``_visibility='public'`` by a
raw Cypher payload (``MATCH (n) WHERE n._visibility IS NULL SET n._visibility
= 'public'``) that reached ``lifecycle.batch_update`` — almost certainly via a
queued :class:`~agent_utilities.orchestration.work_item.WorkItem`'s claimed,
pluggable ``executor(...)`` body. Three separate investigations (``git log
-S``/``-G`` across both repos, all branches) found nothing, because nothing
was ever committed: the caller/session identity that triggered the mutation
was never recorded at any layer.

This module closes the gap for the WorkItem-claimed path specifically (the
one ``POST /mcp`` never served in that incident window, so the MCP-layer
session/scope enforcement alone does not explain "who"): it is a small,
best-effort ``contextvars.ContextVar`` that
:func:`~agent_utilities.orchestration.agent_dispatch_worker.execute_work_item_turn`
and
:func:`~agent_utilities.orchestration.agent_dispatch_worker.execute_agent_task_turn`
bind for the exact duration of the pluggable ``executor(...)`` call. Any
engine mutation the executor triggers — most importantly a generic
``lifecycle.batch_update`` — can then be logged with "which WorkItem, which
lease, which agent, which capability" alongside the RPC
(:mod:`agent_utilities.knowledge_graph.core.engine_breaker` reads it).

This is attribution, NOT authority: nothing here grants or checks a
permission — the real fail-closed governance gate is the ambient
``GraphSession`` enforced in ``_SessionRoutedAsyncClient._send``
(CONCEPT:AU-KG BUG-033/058/062). This context only makes the WorkItem/lease/
capability that a mutation ran under visible from logs alone, instead of
requiring engine-side archaeology after the fact.
"""

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

__all__ = [
    "bind_work_item_context",
    "current_work_item_context",
]

_current: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "work_item_execution_context", default=None
)


def current_work_item_context() -> dict[str, Any] | None:
    """Return the ambient WorkItem-execution context, if any is bound.

    ``None`` outside a bound :func:`bind_work_item_context` block (e.g. a
    direct/interactive call never claimed through a WorkItem) — callers must
    treat that as "no WorkItem context available", never synthesize one.
    """
    return _current.get()


@contextmanager
def bind_work_item_context(
    *,
    work_item_id: str,
    agent_id: str = "",
    lease_id: str = "",
    capability: str = "",
    task_id: str = "",
) -> Iterator[None]:
    """Scope a block of work to a WorkItem's identity for attribution logging.

    Wrap the pluggable ``executor(...)`` invocation in
    ``execute_work_item_turn``/``execute_agent_task_turn`` with this so any
    engine RPC the executor body triggers can attribute itself to the claimed
    WorkItem — without threading these fields through every intermediate
    call signature. Restores the previous (or empty) context on exit, so
    nesting and reentry are both safe.
    """
    token = _current.set(
        {
            "work_item_id": str(work_item_id or ""),
            "agent_id": str(agent_id or ""),
            "lease_id": str(lease_id or ""),
            "capability": str(capability or ""),
            "task_id": str(task_id or ""),
        }
    )
    try:
        yield
    finally:
        _current.reset(token)
