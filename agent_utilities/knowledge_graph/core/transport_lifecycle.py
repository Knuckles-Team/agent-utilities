"""Bounded GraphOS transport lifecycle gates.

The native client is process-owned, while MCP/GraphOS workers may be drained
by a pod or host supervisor.  A close alone cannot distinguish "all requests
finished" from "the socket was torn down while work was in flight".  This
small gate makes admission and drain state explicit, bounded, and shared by
all graph-scoped views over one process transport.

The gate contains no durable session or credential material.  A restarted
process starts in ``accepting`` and must establish a fresh verified graph
session and ClusterMembers snapshot; it cannot report continuity from this
in-memory state.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

__all__ = [
    "EngineDrainingError",
    "TransportDrainGate",
    "TransportDrainStatus",
]


class EngineDrainingError(ConnectionError):
    """Raised when a new operation arrives after transport drain begins."""


@dataclass(frozen=True)
class TransportDrainStatus:
    state: str
    active_requests: int
    generation: int
    started_monotonic: float | None
    completed_monotonic: float | None
    timed_out: bool


class TransportDrainGate:
    """Thread-safe admission gate shared by one process-owned engine client."""

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._state = "accepting"
        self._active_requests = 0
        self._generation = 0
        self._started_monotonic: float | None = None
        self._completed_monotonic: float | None = None
        self._timed_out = False

    @contextmanager
    def admit(self) -> Iterator[None]:
        with self._condition:
            if self._state != "accepting":
                raise EngineDrainingError("graph transport is draining")
            self._active_requests += 1
        try:
            yield None
        finally:
            with self._condition:
                self._active_requests -= 1
                if self._active_requests <= 0:
                    self._active_requests = 0
                    self._condition.notify_all()

    def begin(self, timeout_s: float) -> TransportDrainStatus:
        """Stop admission and wait at most ``timeout_s`` for active calls."""
        try:
            timeout = float(timeout_s)
        except (TypeError, ValueError) as exc:
            raise ValueError("transport drain timeout must be finite") from exc
        if not 0.0 <= timeout <= 300.0:
            raise ValueError("transport drain timeout is out of bounds")
        deadline = time.monotonic() + timeout
        with self._condition:
            if self._state == "drained":
                return self.status()
            if self._state == "accepting":
                self._state = "draining"
                self._generation += 1
                self._started_monotonic = time.monotonic()
                self._timed_out = False
            while self._active_requests:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._timed_out = True
                    break
                self._condition.wait(timeout=remaining)
            if not self._active_requests:
                self._state = "drained"
                self._completed_monotonic = time.monotonic()
            return self.status()

    def status(self) -> TransportDrainStatus:
        with self._condition:
            return TransportDrainStatus(
                state=self._state,
                active_requests=self._active_requests,
                generation=self._generation,
                started_monotonic=self._started_monotonic,
                completed_monotonic=self._completed_monotonic,
                timed_out=self._timed_out,
            )
