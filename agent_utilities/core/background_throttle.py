"""Global background-work throttle + foreground pause (CONCEPT:AU-KG.query.vendor-agnostic-traversal).

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
import json
import logging
import math
import os
import stat
import threading
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


# A foreground run normally lasts far less than this lease.  The heartbeat keeps
# longer streams alive and the short expiry makes an unclean client exit harmless.
_FOREGROUND_LEASE_TTL = 3.0
_FOREGROUND_LEASE_HEARTBEAT = 1.0
_FOREGROUND_LEASE_SCAN_INTERVAL = 0.25
_MAX_FOREGROUND_LEASES = 128
_MAX_FOREGROUND_LEASE_BYTES = 256


class BackgroundThrottle:
    def __init__(
        self,
        max_concurrent: int = 2,
        *,
        lease_ttl: float = _FOREGROUND_LEASE_TTL,
        lease_heartbeat: float = _FOREGROUND_LEASE_HEARTBEAT,
        lease_scan_interval: float = _FOREGROUND_LEASE_SCAN_INTERVAL,
    ) -> None:
        self._sem = threading.BoundedSemaphore(max(1, max_concurrent))
        self._foreground = threading.Event()  # set => pause background
        self._fg_depth = 0
        self._ingest = threading.Event()  # set => a bulk ingest is in flight
        self._ingest_depth = 0
        self._lock = threading.Lock()
        self.max_concurrent = max(1, max_concurrent)
        self._lease_ttl = max(0.05, lease_ttl)
        self._lease_heartbeat = max(0.01, min(lease_heartbeat, self._lease_ttl / 2))
        self._lease_scan_interval = max(0.0, lease_scan_interval)
        self._lease_id = uuid.uuid4().hex
        self._lease_stop: threading.Event | None = None
        self._lease_thread: threading.Thread | None = None
        self._lease_cache_active = False
        self._lease_cache_at = 0.0

    def _lease_dir(self) -> Path | None:
        """Return the private shared lease directory, or disable sharing safely.

        ``runtime_dir()`` is already the cross-container root used by the host
        lock.  A separate 0700 child avoids exposing request, host, or identity
        data on the shared volume.  Refusing a symlink/non-directory is safer
        than following an operator- or attacker-controlled redirection.
        """
        from agent_utilities.core.paths import runtime_dir

        directory = runtime_dir() / "foreground-leases"
        try:
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            info = directory.lstat()
            if (
                directory.is_symlink()
                or not stat.S_ISDIR(info.st_mode)
                or info.st_uid != os.getuid()
            ):
                raise OSError("unsafe foreground lease directory")
            os.chmod(directory, 0o700)
            return directory
        except OSError:
            logger.debug("Cross-process foreground lease is unavailable")
            return None

    def _lease_path(self) -> Path | None:
        directory = self._lease_dir()
        return directory / f"{self._lease_id}.json" if directory else None

    def _write_lease(self) -> None:
        """Atomically publish this process's expiry-only foreground lease."""
        target = self._lease_path()
        if target is None:
            return
        payload = json.dumps(
            {"version": 1, "expires_at": time.time() + self._lease_ttl},
            separators=(",", ":"),
        ).encode("utf-8")
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        descriptor: int | None = None
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("foreground lease write did not make progress")
                view = view[written:]
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            os.replace(temporary, target)
            try:
                directory_fd = os.open(
                    target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                )
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:  # pragma: no cover - directory fsync is platform-specific
                pass
        except OSError:
            logger.debug("Cross-process foreground lease write failed")
        finally:
            if descriptor is not None:
                os.close(descriptor)
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def _remove_lease(self) -> None:
        target = self._lease_path()
        if target is None:
            return
        try:
            info = target.lstat()
            if stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid():
                target.unlink()
        except OSError:
            pass

    def _lease_is_valid(self, path: Path, now: float) -> bool:
        """Validate one untrusted lease without following symlinks."""
        try:
            info = path.lstat()
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_uid != os.getuid()
                or info.st_mode & 0o077
                or info.st_size > _MAX_FOREGROUND_LEASE_BYTES
            ):
                return False
            descriptor = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                opened = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_uid != os.getuid()
                    or opened.st_mode & 0o077
                    or opened.st_size > _MAX_FOREGROUND_LEASE_BYTES
                ):
                    return False
                payload = os.read(descriptor, _MAX_FOREGROUND_LEASE_BYTES + 1)
            finally:
                os.close(descriptor)
            record = json.loads(payload)
            expires_at = record.get("expires_at")
            return (
                record.get("version") == 1
                and isinstance(expires_at, (int, float))
                and not isinstance(expires_at, bool)
                and math.isfinite(expires_at)
                and now < expires_at <= now + (self._lease_ttl * 2)
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False

    def _external_foreground_active(self) -> bool:
        now = time.monotonic()
        with self._lock:
            if now - self._lease_cache_at < self._lease_scan_interval:
                return self._lease_cache_active
            directory = self._lease_dir()
            active = False
            if directory is not None:
                try:
                    wall_now = time.time()
                    with os.scandir(directory) as entries:
                        for index, entry in enumerate(entries):
                            if index >= _MAX_FOREGROUND_LEASES:
                                break
                            if entry.name.endswith(".json") and self._lease_is_valid(
                                Path(entry.path), wall_now
                            ):
                                active = True
                                break
                except OSError:
                    pass
            self._lease_cache_active = active
            self._lease_cache_at = now
            return active

    def _heartbeat_lease(self, stop: threading.Event) -> None:
        while not stop.wait(self._lease_heartbeat):
            with self._lock:
                if not self._foreground.is_set() or self._lease_stop is not stop:
                    return
                self._write_lease()

    # ── foreground (interactive) signalling ──────────────────────────────
    def set_foreground(self, active: bool) -> None:
        """Mark interactive work active (reentrant via depth counter)."""
        with self._lock:
            if active:
                self._fg_depth += 1
                if self._fg_depth == 1:
                    self._foreground.set()
                    stop = threading.Event()
                    self._lease_stop = stop
                    self._write_lease()
                    self._lease_thread = threading.Thread(
                        target=self._heartbeat_lease,
                        args=(stop,),
                        name="foreground-lease-heartbeat",
                        daemon=True,
                    )
                    self._lease_thread.start()
            else:
                self._fg_depth = max(0, self._fg_depth - 1)
                if self._fg_depth == 0:
                    self._foreground.clear()
                    if self._lease_stop is not None:
                        self._lease_stop.set()
                        self._lease_stop = None
                    self._remove_lease()
                    self._lease_cache_at = 0.0

    @property
    def foreground_active(self) -> bool:
        # The overwhelmingly common nested/local case remains a lock-free Event
        # check.  Only background-only processes perform the bounded lease scan.
        return self._foreground.is_set() or self._external_foreground_active()

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
        return self.foreground_active or self._ingest.is_set()

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
