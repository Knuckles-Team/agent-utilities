"""Select the single durable trace writer for a fenced WorkItem harness run."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_L5_TERMINAL_OWNER: ContextVar[bool] = ContextVar("l5_terminal_owner", default=False)


def l5_terminal_owns_trace() -> bool:
    """True only within a WorkItem run whose terminal commit owns L5 receipts."""

    return _L5_TERMINAL_OWNER.get()


@contextmanager
def l5_terminal_trace_owner() -> Iterator[None]:
    """Defer legacy trace writes until the caller's fenced terminal commit."""

    token = _L5_TERMINAL_OWNER.set(True)
    try:
        yield
    finally:
        _L5_TERMINAL_OWNER.reset(token)
