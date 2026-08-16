"""Cross-platform advisory file-lock primitives (R-07).

``agent_utilities.knowledge_graph.core.file_lock`` is THE chokepoint every
lock/lease site in this ecosystem is meant to route through instead of calling
``fcntl``/``msvcrt`` directly (17 sites across ``agent-utilities`` and
``repository-manager`` did the latter before R-07). This module verifies:

* The real POSIX branch (this test process's native platform) behaves
  identically to the bare ``fcntl.flock`` calls it replaces -- non-blocking
  exclusive/shared, blocking exclusive/shared, unlock, and the
  crash-releases-the-lock property the lane lease / ``canonical_guard``
  machinery depends on.
* The Windows branch -- which cannot run natively here -- is exercised by
  directly loading ``file_lock.py`` under a simulated Windows interpreter
  (``sys.platform == "win32"``, ``fcntl`` unavailable, ``msvcrt`` stubbed,
  and the Windows-only ``ctypes.WinDLL``/``get_last_error``/``WinError``
  names -- absent entirely from this Linux-built CPython's ``ctypes``
  module, confirmed by direct probe, not assumption -- monkey-patched onto
  the real ``ctypes`` module for the duration of the test), bypassing the
  ``agent_utilities`` package ``__init__`` chain so unrelated stdlib modules
  that make their own real ``import _winapi`` calls under a spoofed platform
  never enter the picture. This directly reproduces the R-07 defect ("17
  files raise ImportError at import time, not at lock time") for the one
  module that legitimately branches on platform, and additionally proves the
  Win32 ``LockFileEx``-based locking achieves TRUE parity with POSIX, not an
  approximation: real concurrent shared readers, a writer genuinely blocked
  by an active reader and vice versa, and blocking-mode genuinely waiting
  (not a silent no-op) -- exercised against a fake kernel32 that replicates
  the exact contention/blocking contract Microsoft's own documentation
  describes for ``LockFileEx``/``UnlockFileEx`` (cited in file_lock.py).
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import sys
import threading
import time
import types

import pytest

from agent_utilities.knowledge_graph.core import file_lock

FILE_LOCK_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "agent_utilities",
    "knowledge_graph",
    "core",
    "file_lock.py",
)


# ─────────────────────────────────────────────────────────────────────────
# POSIX branch: this test process's native platform (must not regress).
# ─────────────────────────────────────────────────────────────────────────


def test_lock_exclusive_nb_contention(tmp_path):
    path = tmp_path / "lock"
    fd_a = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    fd_b = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        file_lock.lock_exclusive_nb(fd_a)
        with pytest.raises(file_lock.LockUnavailable):
            file_lock.lock_exclusive_nb(fd_b)
        file_lock.unlock(fd_a)
        # free again
        file_lock.lock_exclusive_nb(fd_b)
        file_lock.unlock(fd_b)
    finally:
        os.close(fd_a)
        os.close(fd_b)


def test_lock_shared_nb_probe_semantics(tmp_path):
    """host_lock's liveness probe: acquire, observe free, then the CALLER
    unlocks (mirroring host_lock.py's own ``lock_shared_nb(fd); ...;
    unlock(fd)`` pattern -- see its module docstring). A real held shared
    lock until unlocked, identically on both platforms (Win32 LockFileEx
    shared locks are TRUE concurrent-reader locks, not an emulation -- see
    the Windows-simulation tests below).
    """
    path = tmp_path / "lock"
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        file_lock.lock_shared_nb(fd)  # must not raise
        file_lock.unlock(fd)
        fd2 = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            file_lock.lock_exclusive_nb(fd2)  # would raise if still held
            file_lock.unlock(fd2)
        finally:
            os.close(fd2)
    finally:
        os.close(fd)


def test_lock_exclusive_blocking_waits_then_acquires(tmp_path):
    path = tmp_path / "lock"
    fd_holder = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    fd_waiter = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        file_lock.lock_exclusive_nb(fd_holder)
        acquired = threading.Event()

        def _waiter():
            file_lock.lock_exclusive(fd_waiter)  # blocking=True default
            acquired.set()

        t = threading.Thread(target=_waiter)
        t.start()
        assert not acquired.wait(timeout=0.3), (
            "blocking acquire returned while still held"
        )
        file_lock.unlock(fd_holder)
        assert acquired.wait(timeout=5), "blocking acquire never acquired after release"
        t.join(timeout=5)
        file_lock.unlock(fd_waiter)
    finally:
        os.close(fd_holder)
        os.close(fd_waiter)


def test_lock_exclusive_nonblocking_returns_false_on_contention(tmp_path):
    path = tmp_path / "lock"
    fd_a = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    fd_b = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        assert file_lock.lock_exclusive(fd_a, blocking=False) is True
        assert file_lock.lock_exclusive(fd_b, blocking=False) is False
        file_lock.unlock(fd_a)
    finally:
        os.close(fd_a)
        os.close(fd_b)


def test_lock_shared_true_concurrent_readers(tmp_path):
    """POSIX lock_shared (held, not probe) allows two readers at once."""
    path = tmp_path / "lock"
    fd_a = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    fd_b = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        assert file_lock.lock_shared(fd_a) is True
        assert file_lock.lock_shared(fd_b) is True  # both readers hold it
        file_lock.unlock(fd_a)
        file_lock.unlock(fd_b)
    finally:
        os.close(fd_a)
        os.close(fd_b)


def test_unlock_is_idempotent_and_best_effort(tmp_path):
    path = tmp_path / "lock"
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        file_lock.unlock(fd)  # never locked -- must not raise
        file_lock.lock_exclusive_nb(fd)
        file_lock.unlock(fd)
        file_lock.unlock(fd)  # second release -- must not raise
    finally:
        os.close(fd)


def test_crash_releases_the_lock(tmp_path):
    """The lane lease / canonical_guard property: a dead holder's lock disappears.

    Advisory POSIX locks are released by the kernel when the holding process
    exits for any reason, including a crash -- simulated here with a
    subprocess that acquires and is killed without ever releasing.
    """
    import subprocess

    path = tmp_path / "lock"
    path.touch()
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import os, time, sys; "
                "sys.path.insert(0, sys.argv[1]); "
                "from agent_utilities.knowledge_graph.core import file_lock; "
                f"fd = os.open({str(path)!r}, os.O_CREAT | os.O_RDWR, 0o644); "
                "file_lock.lock_exclusive_nb(fd); "
                "time.sleep(60)"
            ),
            str(_repo_root()),
        ],
    )
    try:
        # Give the holder time to acquire.
        fd_probe = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
        deadline = time.monotonic() + 5
        held = False
        while time.monotonic() < deadline:
            try:
                file_lock.lock_exclusive_nb(fd_probe)
                file_lock.unlock(fd_probe)
                time.sleep(0.05)
            except file_lock.LockUnavailable:
                held = True
                break
        assert held, "holder subprocess never acquired the lock"

        holder.kill()
        holder.wait(timeout=5)

        deadline = time.monotonic() + 5
        released = False
        while time.monotonic() < deadline:
            try:
                file_lock.lock_exclusive_nb(fd_probe)
                file_lock.unlock(fd_probe)
                released = True
                break
            except file_lock.LockUnavailable:
                time.sleep(0.05)
        assert released, "lock was not released after the holder was killed"
        os.close(fd_probe)
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait(timeout=5)


def _repo_root() -> str:
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    )


# ─────────────────────────────────────────────────────────────────────────
# Windows branch: simulated (this suite never runs on real Windows).
#
# file_lock.py's Windows branch uses ctypes against kernel32.dll
# (LockFileEx/UnlockFileEx), NOT the limited msvcrt.locking CRT wrapper --
# see the module docstring for why (no shared-lock mode, ~10s blocking cap).
# msvcrt is still used for exactly one thing: get_osfhandle(), to convert a
# CRT fd into the real Win32 HANDLE these APIs need.
#
# This Linux-built CPython's ctypes module does not define WinDLL,
# get_last_error, or WinError at all -- confirmed directly with
# `hasattr(ctypes, "WinDLL")` (False here), not assumed. Those are
# conditionally compiled into CPython's own ctypes/__init__.py only on
# Windows, unlike ctypes.wintypes (HANDLE/DWORD/BOOL), which IS available
# cross-platform. So the fixture below monkey-patches the three missing
# names onto the real `ctypes` module for the duration of the test, backed
# by a small in-Python fake kernel32 whose LockFileEx/UnlockFileEx replicate
# the exact contention/blocking contract Microsoft's own documentation
# describes (cited in file_lock.py's module docstring):
#   - a shared lock allows concurrent readers (denies writers only)
#   - an exclusive lock denies everyone else
#   - blocking mode (no LOCKFILE_FAIL_IMMEDIATELY) waits until granted
#   - LOCKFILE_FAIL_IMMEDIATELY returns immediately on contention
# ─────────────────────────────────────────────────────────────────────────

_LOCKFILE_FAIL_IMMEDIATELY = 0x00000001
_LOCKFILE_EXCLUSIVE_LOCK = 0x00000002
_ERROR_LOCK_VIOLATION = 33


class _FakeKernel32:
    """Byte-range lock state keyed by (st_dev, st_ino) -- like real Win32
    locks, contention is per underlying FILE, not per fd/HANDLE integer, so
    two different fds open on the same file correctly contend.

    LockFileEx/UnlockFileEx are plain function objects assigned as instance
    attributes (not class methods): file_lock.py does
    ``_kernel32.LockFileEx.argtypes = [...]`` at import time, and a bound
    method has no ``__dict__`` to hold that; an instance-attribute function
    does.
    """

    def __init__(self):
        self._state: dict[tuple[int, int], object] = {}
        self._fd_by_handle: dict[int, int] = {}

        def _key(handle_value: int) -> tuple[int, int]:
            st = os.fstat(self._fd_by_handle[handle_value])
            return (st.st_dev, st.st_ino)

        def _LockFileEx(handle, flags, _reserved, _low, _high, _overlapped_ptr):
            hv = handle.value if hasattr(handle, "value") else handle
            key = _key(hv)
            exclusive = bool(flags & _LOCKFILE_EXCLUSIVE_LOCK)
            fail_immediately = bool(flags & _LOCKFILE_FAIL_IMMEDIATELY)

            def _try_acquire() -> bool:
                current = self._state.get(key)
                if current is None:
                    self._state[key] = "exclusive" if exclusive else {hv}
                    return True
                if exclusive or current == "exclusive":
                    return False
                current.add(hv)  # another concurrent reader: real parity
                return True

            if _try_acquire():
                return 1
            if fail_immediately:
                _set_last_error(_ERROR_LOCK_VIOLATION)
                return 0
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if _try_acquire():
                    return 1
                time.sleep(0.01)
            return 0

        def _UnlockFileEx(handle, _reserved, _low, _high, _overlapped_ptr):
            hv = handle.value if hasattr(handle, "value") else handle
            key = _key(hv)
            current = self._state.get(key)
            if current == "exclusive":
                self._state.pop(key, None)
            elif isinstance(current, set):
                current.discard(hv)
                if not current:
                    self._state.pop(key, None)
            return 1

        self.LockFileEx = _LockFileEx
        self.UnlockFileEx = _UnlockFileEx

    def register_handle(self, handle_value: int, fd: int) -> None:
        self._fd_by_handle[handle_value] = fd


_last_error_box = [0]


def _get_last_error() -> int:
    return _last_error_box[0]


def _set_last_error(v: int) -> None:
    _last_error_box[0] = v


class _FakeWinError(OSError):
    def __init__(self, code):
        super().__init__(code, f"fake WinError {code}")
        self.strerror = f"fake WinError {code}"


def _load_file_lock_under_simulated_windows():
    """Import file_lock.py directly by path with sys.platform spoofed to
    'win32', fcntl blocked, msvcrt stubbed, and the Windows-only ctypes
    names patched in -- reproduces the exact R-07 failure mode (ImportError
    at import time) if the guard regresses, and lets the real LockFileEx
    control-flow logic actually run against the fake kernel32 above.

    Loaded by file path (not via the ``agent_utilities.knowledge_graph.core
    .file_lock`` package path) so the parent package's ``__init__`` chain --
    which pulls in unrelated stdlib modules doing their OWN real
    ``import _winapi`` under a spoofed platform, unavailable on this
    Linux-built CPython -- is never touched. That is an interpreter-build
    limitation, not something this fix can (or needs to) work around: it is
    orthogonal to file_lock.py, which has zero non-stdlib, zero
    ``agent_utilities``-internal imports.
    """
    fake_kernel32 = _FakeKernel32()
    ctypes.WinDLL = lambda name, use_last_error=False: fake_kernel32  # noqa: ARG005
    ctypes.get_last_error = _get_last_error
    ctypes.WinError = _FakeWinError

    fake_msvcrt = types.ModuleType("msvcrt")
    fake_msvcrt.get_osfhandle = lambda fd: fd  # identity: fine for the sim
    sys.modules["msvcrt"] = fake_msvcrt

    real_platform = sys.platform
    real_import = __import__
    import builtins as _builtins

    def _fake_import(name, *a, **kw):
        if name == "fcntl":
            raise ImportError("No module named 'fcntl' (simulated Windows)")
        return real_import(name, *a, **kw)

    sys.platform = "win32"
    _builtins.__import__ = _fake_import
    try:
        spec = importlib.util.spec_from_file_location(
            "file_lock_win_sim", FILE_LOCK_PATH
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        _builtins.__import__ = real_import
        sys.platform = real_platform
        sys.modules.pop("msvcrt", None)

    # Wrap _handle() so the fake kernel32 can resolve a HANDLE back to the
    # underlying fd for (st_dev, st_ino) keying (a real Win32 HANDLE is
    # opaque; only our fake needs this extra bookkeeping for the test).
    orig_handle = mod._handle

    def _patched_handle(fd):
        h = orig_handle(fd)
        fake_kernel32.register_handle(h.value, fd)
        return h

    mod._handle = _patched_handle
    return mod


@pytest.fixture(scope="module")
def win_file_lock():
    mod = _load_file_lock_under_simulated_windows()
    yield mod
    for name in ("WinDLL", "get_last_error", "WinError"):
        if hasattr(ctypes, name):
            delattr(ctypes, name)


def test_windows_simulation_imports_cleanly(win_file_lock):
    """The R-07 headline defect, reproduced and proven fixed: importing this
    module when fcntl is unavailable on a win32-reporting interpreter must
    not raise ImportError."""
    assert win_file_lock.lock_exclusive_nb is not None


def test_windows_lock_exclusive_nb_contention(win_file_lock, tmp_path):
    path = tmp_path / "lock"
    fd_a = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    fd_b = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        win_file_lock.lock_exclusive_nb(fd_a)
        with pytest.raises(win_file_lock.LockUnavailable):
            win_file_lock.lock_exclusive_nb(fd_b)
        win_file_lock.unlock(fd_a)
        win_file_lock.lock_exclusive_nb(fd_b)
        win_file_lock.unlock(fd_b)
    finally:
        os.close(fd_a)
        os.close(fd_b)


def test_windows_lock_exclusive_nonblocking_returns_false(win_file_lock, tmp_path):
    path = tmp_path / "lock"
    fd_a = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    fd_b = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        assert win_file_lock.lock_exclusive(fd_a, blocking=False) is True
        assert win_file_lock.lock_exclusive(fd_b, blocking=False) is False
        win_file_lock.unlock(fd_a)
    finally:
        os.close(fd_a)
        os.close(fd_b)


def test_windows_lock_shared_true_concurrent_readers(win_file_lock, tmp_path):
    """The headline parity fix: Win32 shared byte-range locks genuinely allow
    concurrent readers (per Microsoft's own docs), so this is a REAL
    concurrent acquire, not a probe-and-release emulation."""
    path = tmp_path / "lock"
    fd_r1 = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    fd_r2 = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        assert win_file_lock.lock_shared(fd_r1) is True
        assert win_file_lock.lock_shared(fd_r2) is True  # both readers hold it
        win_file_lock.unlock(fd_r1)
        win_file_lock.unlock(fd_r2)
    finally:
        os.close(fd_r1)
        os.close(fd_r2)


def test_windows_shared_and_exclusive_mutually_exclude(win_file_lock, tmp_path):
    path = tmp_path / "lock"
    fd_reader = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    fd_writer = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        win_file_lock.lock_shared(fd_reader)
        assert win_file_lock.lock_exclusive(fd_writer, blocking=False) is False
        win_file_lock.unlock(fd_reader)
        assert win_file_lock.lock_exclusive(fd_writer, blocking=False) is True
        win_file_lock.unlock(fd_writer)
    finally:
        os.close(fd_reader)
        os.close(fd_writer)


def test_windows_lock_exclusive_blocking_waits_then_acquires(win_file_lock, tmp_path):
    """Proves the Windows blocking path genuinely blocks at the (simulated)
    kernel level -- no retry/poll loop in file_lock.py's own code, unlike an
    msvcrt.locking-based design would need (LK_LOCK gives up after ~10s)."""
    path = tmp_path / "lock"
    fd_holder = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    fd_waiter = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        win_file_lock.lock_exclusive_nb(fd_holder)
        acquired = threading.Event()

        def _waiter():
            win_file_lock.lock_exclusive(fd_waiter)
            acquired.set()

        t = threading.Thread(target=_waiter)
        t.start()
        assert not acquired.wait(timeout=0.3), (
            "blocking acquire returned while still held"
        )
        win_file_lock.unlock(fd_holder)
        assert acquired.wait(timeout=5), "blocking acquire never acquired after release"
        t.join(timeout=5)
        win_file_lock.unlock(fd_waiter)
    finally:
        os.close(fd_holder)
        os.close(fd_waiter)
