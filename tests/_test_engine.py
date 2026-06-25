#!/usr/bin/python
"""Ephemeral REAL epistemic-graph engine for the test suite (CONCEPT:KG-2.238).

USER DIRECTIVE: tests validate against the ACTUAL database we ship — NOT SQLite,
NOT mocks. This module owns the lifecycle of ONE real ``epistemic-graph-server``
process per test session, deployed ephemerally and destroyed/cleaned up after.

The shape:

* :func:`resolve_engine_binary` finds the engine binary, preferring (in order)
  the prebuilt **wheel** binary (next to ``sys.executable``), then a sibling
  ``epistemic-graph`` checkout's ``target/release`` / ``target/debug``. If none
  exists it BUILDS the lean **``pi``-tier** binary ONCE (pure-Rust, fast) and
  caches it under ``target/release`` so subsequent runs are instant.
* :class:`EphemeralEngine` starts that binary on an ISOLATED ephemeral UDS socket
  under a unique temp dir, with an isolated temp ``--persist-dir``, a test auth
  secret, and ``--idle-shutdown-secs`` (self-cleans if the suite dies). Teardown
  is a graceful SIGTERM (the engine checkpoints + exits cleanly), then the temp
  persist dir + socket are removed. Fully ephemeral, no residue.
* :class:`EngineUnavailable` is raised when the engine genuinely cannot be
  obtained (no binary AND no Rust toolchain) so the session fixture can xfail/skip
  with a clear message — but on any normal dev/CI box a real engine runs.

The ``conftest`` ``tiny_engine`` (session) + ``engine_graph`` (function) fixtures
wrap this; nothing here knows about pytest so it stays unit-testable on its own.
"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

#: Shared HMAC secret the test engine runs under. The engine REFUSES to start
#: without one (CONCEPT:OS-5.14); every client authenticates with this exact
#: value, exported as ``GRAPH_SERVICE_AUTH_SECRET`` so the resolver/EngineResolver
#: pick it up. Test-only; never a real credential.
TEST_AUTH_SECRET = "agent-utilities-test-engine-secret"  # nosec B105 — test-only

#: Reference-counted idle-shutdown grace (seconds). If the pytest session dies
#: without running teardown (a crash / SIGKILL), the engine self-terminates this
#: many seconds after its last client disconnects — so a dead suite leaves no
#: orphan process. Short enough to reap promptly, long enough to span a slow
#: session of back-to-back tests sharing the one engine.
IDLE_SHUTDOWN_SECS = 120

#: The lean tier we build when no binary exists. Pure-Rust + small → fast to
#: build and run, and a durable source of truth (redb-authoritative).
BUILD_TIER = "pi"

#: How long to wait for the engine's socket to appear after spawn, and for the
#: process to exit on graceful SIGTERM.
_SOCKET_WAIT_SECS = 30.0
_SHUTDOWN_WAIT_SECS = 15.0

_BINARY_NAME = "epistemic-graph-server"


class EngineUnavailable(RuntimeError):
    """The real engine could not be obtained (no binary AND no Rust toolchain)."""


def _sibling_epistemic_graph_dir() -> Path | None:
    """The sibling ``epistemic-graph`` source checkout, if present.

    In the dev monorepo it sits two levels up from this package
    (``…/agent-packages/epistemic-graph``); in a worktree the package lives under
    ``…/worktrees/agent-utilities/<branch>`` so we also probe the canonical
    ``agent-packages`` checkout. Returns the first directory that holds a
    ``Cargo.toml``.
    """
    here = Path(__file__).resolve()
    candidates = [
        # …/agent-utilities/tests/_test_engine.py → …/agent-packages/epistemic-graph
        here.parent.parent.parent / "epistemic-graph",
        # canonical checkout (when we run from a worktree under /home/apps/worktrees)
        Path("/home/apps/workspace/agent-packages/epistemic-graph"),
    ]
    for cand in candidates:
        if (cand / "Cargo.toml").is_file():
            return cand
    return None


def resolve_engine_binary() -> Path:
    """Locate (or build once) the real ``epistemic-graph-server`` binary.

    Resolution order:

    1. **Prebuilt wheel binary** — ``Path(sys.executable).parent / "…-server"``
       (how the shipped wheel installs it). This is what production runs, so a
       test that uses it validates the ACTUAL deployed database.
    2. **Sibling checkout** ``target/release`` then ``target/debug``.
    3. **Build once** the lean ``pi``-tier binary in the sibling checkout and
       cache it at ``target/release/…`` (instant on subsequent runs).

    Raises :class:`EngineUnavailable` when there is no binary, no sibling source,
    or no ``cargo`` to build with.
    """
    # 1) Prebuilt wheel binary (production path).
    wheel_bin = Path(sys.executable).resolve().parent / _BINARY_NAME
    if wheel_bin.is_file() and os.access(wheel_bin, os.X_OK):
        return wheel_bin

    rust_dir = _sibling_epistemic_graph_dir()
    if rust_dir is None:
        raise EngineUnavailable(
            "no prebuilt epistemic-graph-server wheel binary and no sibling "
            "epistemic-graph source checkout — cannot obtain a real engine."
        )

    # 2) Already-built sibling binary (release preferred — smaller/faster).
    for profile in ("release", "debug"):
        cand = rust_dir / "target" / profile / _BINARY_NAME
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand

    # 3) Build the lean pi-tier binary ONCE, then cache it.
    cargo = shutil.which("cargo")
    if cargo is None:
        raise EngineUnavailable(
            f"no epistemic-graph-server binary built under {rust_dir}/target and "
            "no `cargo` on PATH to build one — cannot obtain a real engine."
        )
    built = _build_pi_binary(rust_dir, cargo)
    if built is None:
        raise EngineUnavailable(
            f"`cargo build --release --no-default-features --features {BUILD_TIER}`"
            f" did not produce {_BINARY_NAME} under {rust_dir}/target/release."
        )
    return built


def _build_pi_binary(rust_dir: Path, cargo: str) -> Path | None:
    """Build the lean ``pi``-tier server once; return the cached binary or None.

    Pure-Rust + small (no DataFusion, no openraft) so this is fast. The result
    lands at ``target/release/epistemic-graph-server`` and is reused forever.
    """
    print(
        f"[tiny_engine] no prebuilt binary — building lean `{BUILD_TIER}`-tier "
        f"engine once in {rust_dir} (cached after first run)…",
        file=sys.stderr,
    )
    proc = subprocess.run(  # noqa: S603 — fixed argv, no shell
        [
            cargo,
            "build",
            "--release",
            "--no-default-features",
            "--features",
            BUILD_TIER,
            "--bin",
            _BINARY_NAME,
        ],
        cwd=str(rust_dir),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(
            f"[tiny_engine] engine build failed:\n{proc.stderr[-2000:]}",
            file=sys.stderr,
        )
        return None
    built = rust_dir / "target" / "release" / _BINARY_NAME
    return built if built.is_file() else None


def _free_socket_path(root: Path) -> str:
    """A unique, short ephemeral UDS path under ``root``.

    Unix socket paths are length-limited (~108 bytes), so we keep the name short
    and rely on the unique ``root`` (a ``mkdtemp``) for isolation.
    """
    return str(root / f"eg-{uuid.uuid4().hex[:8]}.sock")


class EphemeralEngine:
    """A real ``epistemic-graph-server`` started on an isolated socket + persist dir.

    Use as a context manager or call :meth:`start` / :meth:`stop` explicitly. On
    :meth:`stop` the engine is shut down with a graceful **SIGTERM** (it
    checkpoints and exits cleanly), then the socket and temp persist dir are
    removed — leaving zero residue.
    """

    def __init__(self, binary: Path) -> None:
        self.binary = Path(binary)
        self._root: str | None = None
        self._persist_dir: str | None = None
        self.socket_path: str | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._log: tempfile._TemporaryFileWrapper[bytes] | None = None

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> EphemeralEngine:
        self._root = tempfile.mkdtemp(prefix="au_tiny_engine_")
        root = Path(self._root)
        self._persist_dir = str(root / "persist")
        os.makedirs(self._persist_dir, exist_ok=True)
        self.socket_path = _free_socket_path(root)

        # Keep the engine log so a startup failure is diagnosable, but in a temp
        # file under the engine's own throwaway dir so it's cleaned up with it.
        self._log = open(  # noqa: SIM115 — closed in stop()
            str(root / "engine.log"), "wb"
        )
        env = {
            **os.environ,
            "GRAPH_SERVICE_AUTH_SECRET": TEST_AUTH_SECRET,
            # Be a durable source of truth in tests too (redb authoritative is the
            # default when a persist dir is set) — exactly the shipped behaviour.
            "GRAPH_SERVICE_PERSIST_DIR": self._persist_dir,
        }
        self._proc = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
            [
                str(self.binary),
                "--socket-path",
                self.socket_path,
                "--persist-dir",
                self._persist_dir,
                "--auth-secret",
                TEST_AUTH_SECRET,
                "--idle-shutdown-secs",
                str(IDLE_SHUTDOWN_SECS),
            ],
            stdout=self._log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            self._wait_for_socket()
        except Exception:
            # Startup failed — tear the half-started engine down cleanly so we
            # never leak a process or temp dir, then re-raise for the caller.
            self.stop()
            raise
        return self

    def _wait_for_socket(self) -> None:
        deadline = time.monotonic() + _SOCKET_WAIT_SECS
        assert self.socket_path is not None
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise EngineUnavailable(
                    f"epistemic-graph-server exited early (code "
                    f"{self._proc.returncode}) during startup.\n{self._tail_log()}"
                )
            if os.path.exists(self.socket_path) and self._can_connect():
                return
            time.sleep(0.1)
        raise EngineUnavailable(
            "epistemic-graph-server did not become ready within "
            f"{_SOCKET_WAIT_SECS:.0f}s.\n{self._tail_log()}"
        )

    def _can_connect(self) -> bool:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                s.connect(self.socket_path)  # type: ignore[arg-type]
            return True
        except OSError:
            return False

    def _tail_log(self) -> str:
        if not self._log:
            return ""
        try:
            self._log.flush()
            with open(self._log.name, "rb") as fh:
                return fh.read()[-2000:].decode("utf-8", "replace")
        except OSError:
            return ""

    def stop(self) -> None:
        """Graceful SIGTERM (engine checkpoints + exits), then remove all residue."""
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=_SHUTDOWN_WAIT_SECS)
            except subprocess.TimeoutExpired:
                # Engine ignored SIGTERM (should not happen) — escalate so we
                # never leave an orphan process behind.
                proc.kill()
                proc.wait(timeout=_SHUTDOWN_WAIT_SECS)
            except Exception:
                proc.kill()
        self._proc = None
        if self._log is not None:
            try:
                self._log.close()
            except OSError:
                pass
            self._log = None
        # The engine leaves the socket inode on exit — remove it, then the whole
        # throwaway dir (persist dir + log + socket).
        if self.socket_path and os.path.exists(self.socket_path):
            try:
                os.remove(self.socket_path)
            except OSError:
                pass
        if self._root and os.path.isdir(self._root):
            shutil.rmtree(self._root, ignore_errors=True)
        self._root = self._persist_dir = self.socket_path = None

    def __enter__(self) -> EphemeralEngine:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
