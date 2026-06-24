"""Singleton enforcement for the consolidated KG host daemon (KG-2.8 / OS-5.9).

Validates that ``host_lock.resolve_daemon_role`` elects exactly one host via an
advisory flock: clients never lock, an explicit second host gets a descriptive
``KGHostAlreadyRunning``, ``auto`` self-heals to client, and the lock auto-releases
when the holder process dies.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import textwrap
import time

import pytest

PY = sys.executable


@pytest.fixture(autouse=True)
def _isolated_runtime_dir(tmp_path, monkeypatch):
    """Point the host lock at a per-test tmp runtime dir so it never collides with a
    LIVE ``graph-os`` daemon's real lock on the box.

    ``host_lock._lock_path`` honours ``AGENT_UTILITIES_RUNTIME_DIR`` (the package's
    standing runtime-root override); setting it via the environment isolates BOTH the
    in-process tests and the subprocesses they spawn (children inherit ``os.environ``),
    so the acquire/release, the LOCK_SH liveness probe, and the second-host-blocks
    assertions all run against an isolated lock file regardless of any real daemon.
    """
    monkeypatch.setenv("AGENT_UTILITIES_RUNTIME_DIR", str(tmp_path))


def _fresh_module():
    """Reload host_lock so per-process cached state (_effective_role/_lock_fd) resets.

    Clears ``AGENT_UTILITIES_TESTING`` for the reloaded module's lifetime so the
    REAL flock path runs — under that flag the lock is intentionally bypassed (so
    pytest-xdist workers don't fight over the singleton), which would make these
    lock-behaviour assertions meaningless.
    """
    os.environ.pop("AGENT_UTILITIES_TESTING", None)
    import agent_utilities.knowledge_graph.core.host_lock as hl

    try:
        hl.release_host_lock()
    except Exception:
        pass
    return importlib.reload(hl)


def test_client_role_takes_no_lock():
    os.environ["KG_DAEMON_ROLE"] = "client"
    try:
        hl = _fresh_module()
        assert hl.resolve_daemon_role() == "client"
        assert hl.effective_daemon_role() == "client"
        assert hl.is_host() is False
    finally:
        os.environ.pop("KG_DAEMON_ROLE", None)
        _fresh_module()


def test_host_acquires_then_releases():
    hl = _fresh_module()
    try:
        assert hl.resolve_daemon_role("host") == "host"
        assert hl.is_host() is True
        holder = hl.host_lock_holder()
        assert holder and holder.get("pid") == os.getpid()
    finally:
        hl.release_host_lock()


def test_host_daemon_running_is_a_readonly_liveness_probe():
    """The container HEALTHCHECK uses host_daemon_running(): it PROBES the flock (a held
    exclusive lock ⇒ a live host) without trying to acquire it — the crash-loop bug was
    acquiring → raising KGHostAlreadyRunning. False when no host holds the lock; True while
    one does; False again after release (immune to a released-but-undeleted lockfile)."""
    hl = _fresh_module()
    try:
        # no exclusive holder yet → not running
        assert hl.host_daemon_running() is False
        # become host (holds the exclusive flock) → running
        assert hl.resolve_daemon_role("host") == "host"
        assert hl.host_daemon_running() is True
        # release → the exclusive lock is gone → not running (even though the file remains)
        hl.release_host_lock()
        assert hl.host_daemon_running() is False
    finally:
        hl.release_host_lock()


def test_second_host_blocks_and_auto_becomes_client():
    """A held lock makes an explicit host raise and an auto fall back to client."""
    # Hold the lock in a separate process for a few seconds.
    holder = subprocess.Popen(
        [
            PY,
            "-c",
            textwrap.dedent(
                """
                from agent_utilities.knowledge_graph.core.host_lock import resolve_daemon_role
                assert resolve_daemon_role('auto') == 'host', 'holder should win'
                import time; time.sleep(5)
                """
            ),
        ],
        env={**os.environ, "AGENT_UTILITIES_TESTING": ""},
    )
    try:
        time.sleep(1.5)  # let the holder acquire
        check = subprocess.run(
            [
                PY,
                "-c",
                textwrap.dedent(
                    """
                    import agent_utilities.knowledge_graph.core.host_lock as hl
                    assert hl.resolve_daemon_role('auto') == 'client', 'auto must self-heal to client'
                    hl._effective_role = None
                    try:
                        hl.resolve_daemon_role('host'); print('NO_ERROR')
                    except hl.KGHostAlreadyRunning:
                        print('BLOCKED')
                    """
                ),
            ],
            env={**os.environ, "AGENT_UTILITIES_TESTING": ""},
            capture_output=True,
            text=True,
        )
        assert "BLOCKED" in check.stdout, check.stdout + check.stderr
    finally:
        holder.wait()

    # After the holder exits the flock is released — a new host can be elected.
    hl = _fresh_module()
    assert hl.resolve_daemon_role("host") == "host"
    hl.release_host_lock()
