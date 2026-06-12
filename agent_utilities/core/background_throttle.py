"""Global background-work throttle + foreground pause (CONCEPT:KG-2.7).

The KG runs several background daemons (evolution, analysis, compaction, ingest)
that issue LLM/embedding/GPU work. On a single-GPU box they contend with
interactive agent runs and bottleneck everything. This primitive gives one shared
control point:

* a **bounded semaphore** caps concurrent background jobs, and
* a **foreground flag** — set while an interactive agent/synthesis runs — makes
  background jobs yield until it clears.

Daemons wrap heavy work in ``with get_throttle().background_slot():`` and the
interactive runner brackets execution with ``set_foreground(True/False)``. This is
the consolidation seam: a future unified scheduler enqueues through the same gate.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time

logger = logging.getLogger(__name__)


class BackgroundThrottle:
    def __init__(self, max_concurrent: int = 2) -> None:
        self._sem = threading.BoundedSemaphore(max(1, max_concurrent))
        self._foreground = threading.Event()  # set => pause background
        self._fg_depth = 0
        self._ingest = threading.Event()  # set => a bulk ingest is in flight
        self._ingest_depth = 0
        self._lock = threading.Lock()
        self.max_concurrent = max(1, max_concurrent)

    # ── foreground (interactive) signalling ──────────────────────────────
    def set_foreground(self, active: bool) -> None:
        """Mark interactive work active (reentrant via depth counter)."""
        with self._lock:
            if active:
                self._fg_depth += 1
                self._foreground.set()
            else:
                self._fg_depth = max(0, self._fg_depth - 1)
                if self._fg_depth == 0:
                    self._foreground.clear()

    @property
    def foreground_active(self) -> bool:
        return self._foreground.is_set()

    @contextlib.contextmanager
    def foreground(self):
        """Context manager: foreground active for its duration."""
        self.set_foreground(True)
        try:
            yield
        finally:
            self.set_foreground(False)

    # ── bulk-ingest signalling ───────────────────────────────────────────
    # A bulk codebase ingest is a single in-flight task, so the durable
    # submission-queue depth drops to 0 the moment it is claimed — the
    # queue-depth defer that maintenance uses then stops firing even though the
    # ingest is hammering the single-writer engine. This explicit gate is held
    # for the WHOLE ingest task lifecycle so every background drain yields to it,
    # independent of queue depth or the (interactive) foreground flag.
    def set_bulk_ingest(self, active: bool) -> None:
        """Mark a bulk ingest in flight (reentrant via depth counter)."""
        with self._lock:
            if active:
                self._ingest_depth += 1
                self._ingest.set()
            else:
                self._ingest_depth = max(0, self._ingest_depth - 1)
                if self._ingest_depth == 0:
                    self._ingest.clear()

    @property
    def bulk_ingest_active(self) -> bool:
        return self._ingest.is_set()

    @contextlib.contextmanager
    def bulk_ingest(self):
        """Context manager: a bulk ingest is active for its duration."""
        self.set_bulk_ingest(True)
        try:
            yield
        finally:
            self.set_bulk_ingest(False)

    @property
    def should_yield_background(self) -> bool:
        """True when background work should stand down — interactive foreground
        work OR a bulk ingest is in flight. The single check every background
        drain (maintenance, embedding backfill, relevance sweep) consults."""
        return self._foreground.is_set() or self._ingest.is_set()

    def wait_while_busy(
        self, poll: float = 0.5, max_wait: float | None = 120.0
    ) -> bool:
        """Cooperatively pause the CALLER while foreground/ingest is active.

        For use BETWEEN chunks of a long background batch so it yields mid-work
        instead of only at the top of its loop. Returns True if it returned
        because the gate cleared, False if it gave up after ``max_wait`` (so a
        permanently-busy foreground can never starve background work forever).
        """
        if not self.should_yield_background:
            return True
        waited = 0.0
        while self.should_yield_background:
            if max_wait is not None and waited >= max_wait:
                return False
            time.sleep(poll)
            waited += poll
        return True

    # ── background slot acquisition ──────────────────────────────────────
    @contextlib.contextmanager
    def background_slot(
        self,
        wait_foreground: bool = True,
        fg_poll: float = 0.5,
        acquire_timeout: float | None = 30.0,
    ):
        """Acquire a background work slot, yielding to foreground work.

        Yields True if a slot was acquired (proceed), False if the caller should
        skip this cycle (foreground active and ``wait_foreground=False``, or the
        semaphore couldn't be acquired in time).
        """
        if self.foreground_active and not wait_foreground:
            yield False
            return
        # Yield to interactive work first.
        while self.foreground_active:
            time.sleep(fg_poll)
        acquired = self._sem.acquire(timeout=acquire_timeout)
        try:
            yield acquired
        finally:
            if acquired:
                self._sem.release()


_throttle: BackgroundThrottle | None = None
_init_lock = threading.Lock()

# Background work runs at a deliberately low fixed concurrency so it never
# contends with foreground request/ingest work (config discipline: one correct
# value, not a per-deploy env knob — replaces KG_BACKGROUND_MAX_CONCURRENT).
_BACKGROUND_MAX_CONCURRENT = 2


def get_throttle() -> BackgroundThrottle:
    """Process-wide throttle singleton (concurrency from config)."""
    global _throttle
    if _throttle is None:
        with _init_lock:
            if _throttle is None:
                # Deliberately low fixed background concurrency (config
                # discipline): one correct value, not a per-deploy env knob.
                _throttle = BackgroundThrottle(
                    max_concurrent=_BACKGROUND_MAX_CONCURRENT
                )
    return _throttle


def set_foreground(active: bool) -> None:
    get_throttle().set_foreground(active)


def set_bulk_ingest(active: bool) -> None:
    get_throttle().set_bulk_ingest(active)
