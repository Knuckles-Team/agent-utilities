"""Cooperative cancellation for queue-owned background work.

The durable task worker is the lifetime owner of every claimed task.  A soft
timeout may request cancellation, but the owner must not detach an unfinished
thread and start a duplicate attempt while the original can still mutate state.
This context-local signal lets bounded loops stop promptly without making
cancellation caller-controlled or global.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_cancellation_event: ContextVar[threading.Event | None] = ContextVar(
    "agent_utilities_task_cancellation_event",
    default=None,
)


class TaskCancellationRequested(BaseException):
    """The owning task worker requested cooperative cancellation.

    This deliberately derives from :class:`BaseException` so best-effort
    ``except Exception`` blocks inside maintenance scans cannot accidentally
    swallow the ownership signal and continue mutating after the soft timeout.
    """


@contextmanager
def use_task_cancellation(event: threading.Event) -> Iterator[None]:
    """Bind ``event`` as the cancellation authority for the current task body."""

    token = _cancellation_event.set(event)
    try:
        yield
    finally:
        _cancellation_event.reset(token)


def task_cancellation_requested() -> bool:
    """Return whether the current queue-owned task has timed out."""

    event = _cancellation_event.get()
    return event is not None and event.is_set()


def raise_if_task_cancelled() -> None:
    """Stop a cooperative task body after its owner requests cancellation."""

    if task_cancellation_requested():
        raise TaskCancellationRequested
