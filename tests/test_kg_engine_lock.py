"""Single-instance engine spawn guard (KG-2.8 / OS-5.9).

Validates that ``engine_lock.engine_spawn_guard`` is a per-socket advisory flock
that serializes engine spawners: a concurrent holder on the SAME socket blocks a
second acquirer (the split-brain prevention), while DIFFERENT sockets never
contend, and the lock auto-releases when the holder dies.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import time

import agent_utilities.knowledge_graph.core.engine_lock as el

PY = sys.executable


def test_lock_path_is_per_socket():
    a = el.engine_lock_path("/tmp/engine-a.sock")
    b = el.engine_lock_path("/tmp/engine-b.sock")
    assert a != b
    # Same socket → same lock path (stable hash), and None uses the default.
    assert el.engine_lock_path("/tmp/engine-a.sock") == a
    assert el.engine_lock_path(None) == el.engine_lock_path(None)


def test_guard_acquires_when_free():
    sock = f"/tmp/eg-test-{os.getpid()}.sock"
    with el.engine_spawn_guard(sock, timeout=1.0) as acquired:
        assert acquired is True
        holder = el.engine_lock_holder(sock)
        assert holder and holder.get("pid") == os.getpid()


def test_concurrent_holder_blocks_second_spawner():
    """A held guard on the same socket → a second acquirer times out to False.

    This is the split-brain prevention: the second would-be spawner cannot
    proceed to spawn while the first holds the guard.
    """
    sock = f"/tmp/eg-test-concurrent-{os.getpid()}.sock"
    holder_src = textwrap.dedent(
        f"""
        import time
        import agent_utilities.knowledge_graph.core.engine_lock as el
        with el.engine_spawn_guard({sock!r}, timeout=2.0) as ok:
            assert ok is True
            print("HELD", flush=True)
            time.sleep(3.0)
        """
    )
    proc = subprocess.Popen(
        [PY, "-c", holder_src],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        # Wait until the child confirms it holds the lock.
        deadline = time.monotonic() + 10
        line = ""
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if "HELD" in line:
                break
        assert "HELD" in line, f"child never acquired the guard (got {line!r})"

        # The child holds it; we must NOT be able to acquire within the timeout.
        t0 = time.monotonic()
        with el.engine_spawn_guard(sock, timeout=1.0) as acquired:
            elapsed = time.monotonic() - t0
            assert acquired is False, "second spawner acquired a held guard (split-brain!)"
            assert elapsed >= 0.9, "guard returned False before waiting out the timeout"
    finally:
        proc.wait(timeout=10)

    # After the holder exits the lock auto-releases — we can acquire now.
    with el.engine_spawn_guard(sock, timeout=2.0) as acquired:
        assert acquired is True


def test_different_sockets_never_contend():
    sock_a = f"/tmp/eg-test-a-{os.getpid()}.sock"
    sock_b = f"/tmp/eg-test-b-{os.getpid()}.sock"
    with el.engine_spawn_guard(sock_a, timeout=1.0) as a:
        assert a is True
        # A different socket's guard is independent — acquires immediately.
        with el.engine_spawn_guard(sock_b, timeout=1.0) as b:
            assert b is True
