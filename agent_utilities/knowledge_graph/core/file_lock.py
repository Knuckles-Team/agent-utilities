"""Cross-platform advisory file-lock primitives (CONCEPT:OS-5.64).

The KG's single-instance guards (:mod:`host_lock`, :mod:`engine_lock`) and the
liveness probe rely on advisory ``flock``. ``fcntl.flock`` is **POSIX-only**, so
this module routes the operations they need through ONE helper that has a Windows
fallback built on stdlib ``msvcrt.locking`` (no new third-party dep):

* :func:`lock_exclusive_nb` — non-blocking exclusive lock (the spawn/host guard).
* :func:`lock_shared_nb`    — non-blocking shared lock (the liveness probe).
* :func:`unlock`            — release.

Semantics are matched as closely as the two OSes allow:

* **POSIX:** ``fcntl.flock`` with ``LOCK_EX`` / ``LOCK_SH`` / ``LOCK_UN`` and
  ``LOCK_NB``. Auto-released by the kernel when the holder dies (the no-stale-PID
  property the guards depend on).
* **Windows:** ``msvcrt.locking`` with ``LK_NBLCK`` (non-blocking exclusive) /
  ``LK_UNLCK``. msvcrt has no shared-lock mode, so a "shared" acquire is emulated
  as a non-blocking exclusive lock that is *immediately released* — enough for the
  liveness probe's only question ("is anyone holding it?"): the probe acquires,
  sees it free, and unlocks; if a real holder has it, the acquire fails. Windows
  byte-range locks are also released when the owning handle/process closes, so the
  same crash-safety holds. The lock is taken on byte 0 of the file.

A non-blocking acquire that LOSES raises :class:`LockUnavailable` (a subclass of
``OSError``), so callers can keep their existing ``except (BlockingIOError,
OSError)`` arms unchanged across both platforms.

``sys.platform == "win32"`` is used as the branch guard precisely because mypy
special-cases that literal for platform narrowing: the ``msvcrt`` branch is only
type-checked on Windows and the ``fcntl`` branch only on POSIX, so neither set of
platform-specific attributes trips the type-checker on the other OS.
"""

from __future__ import annotations

import sys

_IS_WINDOWS = sys.platform == "win32"


class LockUnavailable(OSError):
    """Raised when a non-blocking lock acquire fails because a peer holds it."""


if sys.platform != "win32":
    import fcntl

    def lock_exclusive_nb(fd: int) -> None:
        """Non-blocking exclusive lock. Raises :class:`LockUnavailable` if held."""
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:  # already locked by a peer
            raise LockUnavailable(str(exc)) from exc

    def lock_shared_nb(fd: int) -> None:
        """Non-blocking shared lock. Raises :class:`LockUnavailable` if EX-held."""
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            raise LockUnavailable(str(exc)) from exc

    def unlock(fd: int) -> None:
        """Release any lock held on ``fd`` (idempotent / best-effort)."""
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass

else:  # pragma: no cover - exercised only on Windows
    import msvcrt
    import os

    def _seek0(fd: int) -> None:
        # msvcrt.locking locks a byte RANGE from the current file position, so
        # seek to 0 first and lock exactly 1 byte.
        try:
            os.lseek(fd, 0, os.SEEK_SET)
        except OSError:
            pass

    def lock_exclusive_nb(fd: int) -> None:
        """Non-blocking exclusive lock via msvcrt. Raises :class:`LockUnavailable`."""
        _seek0(fd)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError as exc:  # byte already locked by a peer handle/process
            raise LockUnavailable(str(exc)) from exc

    def lock_shared_nb(fd: int) -> None:
        """Emulated shared lock: probe-acquire-then-release.

        msvcrt has no shared mode. The only caller (the liveness probe) just needs
        to know whether a holder exists, so acquire exclusively non-blocking; if it
        succeeds, immediately release (nobody held it) so the caller treats that as
        "free"; if it fails, a real holder exists.
        """
        _seek0(fd)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise LockUnavailable(str(exc)) from exc
        _seek0(fd)
        try:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass

    def unlock(fd: int) -> None:
        """Release the byte-0 lock held on ``fd`` (idempotent / best-effort)."""
        _seek0(fd)
        try:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
