"""Engine-resolution matrix (CONCEPT:OS-5.63).

One resolver, every entrypoint inherits it. The resolver decides — by ONE
precedence, remote → share-running-local → autostart-shared-supervised — how a
process reaches the ONE engine authority. These tests assert:

* (a) **remote configured** → ``mode="remote"``, autostart NEVER permitted (no
  spawn);
* (b) **a running engine** → ``mode="shared"`` (reuse it, single PID, no 2nd
  process);
* (c) **nothing reachable** → ``mode="autostart"``; a 2nd ``resolve_engine``
  shares the just-started engine (no redb-lock collision across concurrent
  resolves), and the supervised idle-shutdown grace is plumbed through;
* the **persistent** lifecycle (idle_shutdown_secs<=0 / engine_lifecycle=persistent)
  is long-living (no ``--idle-shutdown-secs`` flag).

The autostart legs use a fake ``epistemic-graph-server`` shell binary so the
matrix runs with no real engine and asserts *spawn count* (the single-PID /
no-collision contract) without depending on the Rust wheel.
"""

from __future__ import annotations

import contextlib
import os
import socket
import threading
import time
from pathlib import Path

import pytest

from agent_utilities.core.config import AgentConfig
from agent_utilities.knowledge_graph.core import engine_resolver as er


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _free_unix_sock(tmp_path: Path) -> str:
    return f"unix://{tmp_path / 'eg-test.sock'}"


class _UDSServer:
    """A trivial UDS listener so ``probe_endpoint`` sees an engine 'running'."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(path)
        self._sock.listen(8)
        self._stop = False
        self._t = threading.Thread(target=self._serve, daemon=True)
        self._t.start()

    def _serve(self) -> None:
        self._sock.settimeout(0.2)
        while not self._stop:
            try:
                conn, _ = self._sock.accept()
                conn.close()
            except OSError:
                continue

    def close(self) -> None:
        self._stop = True
        try:
            self._sock.close()
        finally:
            with contextlib.suppress(OSError):
                os.unlink(self.path)


# ---------------------------------------------------------------------------
# (a) remote configured → mode=remote, NEVER autostart
# ---------------------------------------------------------------------------


def test_remote_endpoint_resolves_remote_no_autostart(monkeypatch):
    """ENGINE_ENDPOINT set → remote; autostart must be impossible (no spawn)."""
    monkeypatch.setenv("ENGINE_MODE", "remote")
    monkeypatch.setenv("ENGINE_ENDPOINT", "tcp://engine.internal:9100")
    cfg = AgentConfig()

    resolved = er.resolve_engine(cfg, "__commons__")

    assert resolved.mode == "remote"
    assert resolved.endpoint == "tcp://engine.internal:9100"
    assert resolved.autostart_allowed is False
    # Remote is inherently persistent — the resolver never passes idle-shutdown.
    assert resolved.idle_shutdown_secs == 0


def test_sharded_endpoints_resolve_remote_no_autostart(monkeypatch):
    """2+ endpoints (sharding) → remote contract, fail-loud, never autostart."""
    monkeypatch.setenv("GRAPH_SERVICE_ENDPOINTS", "tcp://a:1,tcp://b:2")
    cfg = AgentConfig()

    resolved = er.resolve_engine(cfg, "__commons__")

    assert resolved.mode == "remote"
    assert resolved.autostart_allowed is False
    assert resolved.endpoint in {"tcp://a:1", "tcp://b:2"}


def test_remote_mode_disables_autostart_helper(monkeypatch):
    monkeypatch.setenv("ENGINE_MODE", "remote")
    monkeypatch.setenv("ENGINE_ENDPOINT", "tcp://x:1")
    assert er.setting_autostart(AgentConfig()) is False


# ---------------------------------------------------------------------------
# (b) a running engine → mode=shared, no spawn
# ---------------------------------------------------------------------------


def test_running_local_engine_is_shared(monkeypatch, tmp_path):
    """A local endpoint already serving → mode=shared (reuse; spawn nothing)."""
    sock_path = str(tmp_path / "eg-running.sock")
    server = _UDSServer(sock_path)
    try:
        monkeypatch.setenv("ENGINE_MODE", "embedded")
        monkeypatch.setenv("GRAPH_SERVICE_SOCKET", sock_path)
        monkeypatch.setenv("EPISTEMIC_GRAPH_AUTOSTART", "1")
        cfg = AgentConfig()

        resolved = er.resolve_engine(cfg, "__commons__")

        assert resolved.mode == "shared", "an already-running engine must be reused"
        assert resolved.endpoint == f"unix://{sock_path}"
    finally:
        server.close()


def test_no_running_engine_resolves_autostart(monkeypatch, tmp_path):
    """Nothing serving on a local endpoint → mode=autostart (caller will spawn)."""
    sock_path = str(tmp_path / "eg-absent.sock")
    monkeypatch.setenv("ENGINE_MODE", "embedded")
    monkeypatch.setenv("GRAPH_SERVICE_SOCKET", sock_path)
    monkeypatch.setenv("EPISTEMIC_GRAPH_AUTOSTART", "1")
    cfg = AgentConfig()

    resolved = er.resolve_engine(cfg, "__commons__")

    assert resolved.mode == "autostart"
    assert resolved.autostart_allowed is True
    assert resolved.idle_shutdown_secs == 60  # default refcounted grace


# ---------------------------------------------------------------------------
# idle-shutdown / lifecycle resolution
# ---------------------------------------------------------------------------


def test_refcounted_lifecycle_passes_grace(monkeypatch):
    monkeypatch.setenv("ENGINE_LIFECYCLE", "refcounted")
    monkeypatch.setenv("ENGINE_IDLE_SHUTDOWN_SECS", "45")
    assert er.engine_idle_shutdown_secs(AgentConfig()) == 45


def test_persistent_lifecycle_is_long_living(monkeypatch):
    """engine_lifecycle=persistent → 0 (never auto-stop), even with a grace set."""
    monkeypatch.setenv("ENGINE_LIFECYCLE", "persistent")
    monkeypatch.setenv("ENGINE_IDLE_SHUTDOWN_SECS", "60")
    assert er.engine_idle_shutdown_secs(AgentConfig()) == 0


def test_nonpositive_grace_is_persistent(monkeypatch):
    monkeypatch.setenv("ENGINE_IDLE_SHUTDOWN_SECS", "0")
    assert er.engine_idle_shutdown_secs(AgentConfig()) == 0
    monkeypatch.setenv("ENGINE_IDLE_SHUTDOWN_SECS", "-5")
    assert er.engine_idle_shutdown_secs(AgentConfig()) == 0


# ---------------------------------------------------------------------------
# (c) autostart spawn contract — single PID, no redb-lock collision across
#     concurrent resolves (uses a fake engine binary, no real engine needed)
# ---------------------------------------------------------------------------


_FAKE_ENGINE = """#!/usr/bin/env bash
# Fake epistemic-graph-server: --help advertises the flag; otherwise bind the
# socket and idle so a probe sees it 'running' and a spawn-count file proves how
# many were started.
if [ "$1" = "--help" ]; then
  echo "usage: epistemic-graph-server [--socket-path P] [--idle-shutdown-secs N]"
  exit 0
fi
sock=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --socket-path) sock="$2"; shift 2;;
    *) shift;;
  esac
done
echo "$$" >> "$SPAWN_COUNT_FILE"
python3 -c "
import socket, sys, time, os
p = sys.argv[1]
try: os.unlink(p)
except OSError: pass
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.bind(p); s.listen(8); s.settimeout(0.2)
end = time.time() + 8
while time.time() < end:
    try:
        c,_ = s.accept(); c.close()
    except OSError:
        pass
" "$sock"
"""


def _install_fake_engine(tmp_path: Path, monkeypatch) -> tuple[str, Path]:
    """Install a fake engine binary. Returns (binary_path, spawn_count_file)."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    binary = bindir / "epistemic-graph-server"
    binary.write_text(_FAKE_ENGINE)
    binary.chmod(0o755)
    count_file = tmp_path / "spawn_count"
    count_file.write_text("")
    monkeypatch.setenv("SPAWN_COUNT_FILE", str(count_file))
    return str(binary), count_file


@pytest.mark.skipif(
    not Path("/bin/bash").exists() and not Path("/usr/bin/bash").exists(),
    reason="spawn matrix needs bash + a writable runtime",
)
def test_concurrent_autostart_shares_one_engine(monkeypatch, tmp_path):
    """Two concurrent spawners on the same socket spawn exactly ONE engine.

    Exercises the per-socket ``engine_spawn_guard`` (double-checked) the resolver's
    autostart leg uses: the second spawner shares the first's engine instead of
    starting a second on the same --persist-dir (the redb-lock collision this
    prevents). Uses a fake engine binary so no real engine is required.
    """
    import subprocess

    from agent_utilities.knowledge_graph.core.engine_lock import engine_spawn_guard

    binary, count_file = _install_fake_engine(tmp_path, monkeypatch)
    sock = str(tmp_path / "eg-spawn.sock")

    def _resolve_and_spawn() -> None:
        # Mirror the resolver's autostart leg: probe, then guard + double-check.
        if er.probe_endpoint(f"unix://{sock}", timeout=0.3):
            return
        with engine_spawn_guard(sock):
            if er.probe_endpoint(f"unix://{sock}", timeout=0.3):
                return
            subprocess.Popen(  # noqa: S603
                [binary, "--socket-path", sock],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                env=dict(os.environ),
            )
            # Wait until the fake engine is actually listening before releasing
            # the guard, so the peer's double-check sees it up.
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                if er.probe_endpoint(f"unix://{sock}", timeout=0.2):
                    break
                time.sleep(0.1)

    t1 = threading.Thread(target=_resolve_and_spawn)
    t2 = threading.Thread(target=_resolve_and_spawn)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    spawned = [ln for ln in count_file.read_text().splitlines() if ln.strip()]
    assert len(spawned) == 1, f"expected ONE engine spawned, got {len(spawned)}"


def test_engine_supports_idle_shutdown_probe(monkeypatch, tmp_path):
    """The graceful-degradation probe detects the flag in the binary's --help."""
    from agent_utilities.knowledge_graph.core import graph_compute as gc

    binary, _ = _install_fake_engine(tmp_path, monkeypatch)
    # Fresh cache for this binary path.
    gc._idle_shutdown_support.pop(binary, None)
    assert gc._engine_supports_idle_shutdown(binary) is True

    # A lean binary whose --help lacks the flag → not supported (flag omitted).
    lean = tmp_path / "lean-server"
    lean.write_text("#!/usr/bin/env bash\necho 'usage: server [--socket-path P]'\n")
    lean.chmod(0o755)
    gc._idle_shutdown_support.pop(str(lean), None)
    assert gc._engine_supports_idle_shutdown(str(lean)) is False
