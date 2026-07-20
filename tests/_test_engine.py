#!/usr/bin/python
"""Ephemeral REAL epistemic-graph engine for the test suite (CONCEPT:AU-KG.memory.provides-real-ephemeral-one).

USER DIRECTIVE: tests validate against the ACTUAL database we ship — NOT SQLite,
NOT mocks. This module owns the lifecycle of ONE real ``epistemic-graph-server``
process per test session, deployed ephemerally and destroyed/cleaned up after.

The shape:

* :func:`resolve_engine_binary` finds an explicitly configured binary or the
  feature-complete binary installed by the ``epistemic-graph[full]`` wheel.
  Tests never invoke Cargo implicitly: native builds are serialized pipeline
  stages with a WSL-local target directory, not a pytest side effect.
* :class:`EphemeralEngine` starts that binary on an ISOLATED ephemeral UDS socket
  under a unique temp dir, with an isolated temp ``--persist-dir``, a test auth
  secret, and ``--idle-shutdown-secs`` (self-cleans if the suite dies). Teardown
  is a graceful SIGTERM (the engine checkpoints + exits cleanly), then the temp
  persist dir + socket are removed. Fully ephemeral, no residue.
* :class:`EngineUnavailable` is raised when the full wheel binary cannot be
  obtained so the session fixture can xfail/skip with a clear message.

The ``conftest`` ``tiny_engine`` (session) + ``engine_graph`` (function) fixtures
wrap this; nothing here knows about pytest so it stays unit-testable on its own.
"""

from __future__ import annotations

import json
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
#: without one (CONCEPT:AU-OS.identity.authenticated-identity-enforcement); every client authenticates with this exact
#: value, exported as ``GRAPH_SERVICE_AUTH_SECRET`` so the resolver/EngineResolver
#: pick it up. Test-only; never a real credential.
TEST_AUTH_SECRET = "agent-" + "utilities-test-engine-secret"  # nosec B105 — test-only
TEST_AGENT_ID = "service:agent-utilities-test-suite"
TEST_SIGNER_KEY = "agent-utilities-test-operation-signer"  # nosec B105 - test only
TEST_AUDIENCE = "epistemic-graph-test"
TEST_TENANT = "tenant:test"
TEST_POLICY_VERSION = "policy:test"


def request_context(
    *,
    agent_id: str = TEST_AGENT_ID,
    roles: list[str] | None = None,
    scopes: list[str] | None = None,
) -> dict[str, object]:
    """Return complete, non-personal current-protocol test authority."""

    return {
        "principal": agent_id,
        "tenant": TEST_TENANT,
        "audience": TEST_AUDIENCE,
        "agent_id": agent_id,
        "roles": list(roles if roles is not None else ["test"]),
        "scopes": list(scopes if scopes is not None else ["*"]),
        "policy_version": TEST_POLICY_VERSION,
        "delegation": [],
    }


def bootstrap_context() -> dict[str, object]:
    """Return the narrow one-time authority used to enroll the test signer."""

    return request_context(roles=[], scopes=["security:bootstrap"])


def strict_server_env(state_dir: str, *, auth_secret: str) -> dict[str, str]:
    """Build the complete current-protocol server environment for tests."""

    return {
        "GRAPH_SERVICE_AUTH_SECRET": auth_secret,
        "EPISTEMIC_GRAPH_AUDIENCE": TEST_AUDIENCE,
        "EPISTEMIC_GRAPH_TENANT": TEST_TENANT,
        "EPISTEMIC_GRAPH_POLICY_VERSION": TEST_POLICY_VERSION,
        "EPISTEMIC_GRAPH_SECURITY_STATE_DIR": state_dir,
        "EPISTEMIC_GRAPH_SIGNER_KEYS_JSON": json.dumps(
            {TEST_AGENT_ID: TEST_SIGNER_KEY}
        ),
    }


#: Reference-counted idle-shutdown grace (seconds). If the pytest session dies
#: without running teardown (a crash / SIGKILL), the engine self-terminates this
#: many seconds after its last client disconnects — so a dead suite leaves no
#: orphan process. Short enough to reap promptly, long enough to span a slow
#: session of back-to-back tests sharing the one engine.
IDLE_SHUTDOWN_SECS = 120

#: How long to wait for the engine's socket to appear after spawn, and for the
#: process to exit on graceful SIGTERM.
_SOCKET_WAIT_SECS = 30.0
_SHUTDOWN_WAIT_SECS = 15.0

_BINARY_NAME = "epistemic-graph-server"


class EngineUnavailable(RuntimeError):
    """The mandatory full wheel's real engine binary could not be obtained."""


def resolve_engine_binary() -> Path:
    """Locate the prebuilt, full ``epistemic-graph-server`` binary.

    Resolution order:

    1. Explicit ``EPISTEMIC_GRAPH_TEST_BINARY`` runtime selection.
    2. **Prebuilt wheel binary** — ``Path(sys.executable).parent / "…-server"``
       (how the shipped wheel installs it). This is what production runs, so a
       test that uses it validates the ACTUAL deployed database.

    Raises :class:`EngineUnavailable` when neither binary is available. It never
    compiles implicitly, which prevents parallel native builds, C-drive target
    growth, and surprising resource exhaustion during unrelated unit tests.
    """
    configured = str(os.environ.get("EPISTEMIC_GRAPH_TEST_BINARY", "") or "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
        raise EngineUnavailable("configured test engine binary is unavailable")

    # Prebuilt wheel binary (production path) — TRUST it directly when present.
    #    The hard-base ``epistemic-graph[full]>=2.23.1,<3.0.0`` wheel ships the
    #    feature-complete production server (blob + tsdb substrates), so using it validates
    #    the ACTUAL deployed database. We deliberately do NOT gate the wheel binary
    #    behind the live ``_binary_serves_features`` probe: that probe starts a
    #    throwaway engine, and under a loaded box + the per-test 60s pytest-timeout it
    #    can spuriously fail under engine cold-start contention. The published full
    #    wheel is the authority; pytest never falls through to a native build.
    #    The wheel installs the binary into the SAME bin dir as the interpreter
    #    script (``.venv/bin``). In a venv ``sys.executable`` is a SYMLINK to the
    #    base toolchain interpreter, so ``.resolve()`` would walk OUT of ``.venv/bin``
    #    (where the binary lives) into the base ``python`` dir (where it does not).
    #    Probe the UN-resolved bin dir first (the venv where the wheel was installed),
    #    then the resolved one (system / non-venv installs) — first hit wins.
    seen: set[Path] = set()
    for base in (Path(sys.executable).parent, Path(sys.executable).resolve().parent):
        if base in seen:
            continue
        seen.add(base)
        wheel_bin = base / _BINARY_NAME
        if wheel_bin.is_file() and os.access(wheel_bin, os.X_OK):
            return wheel_bin

    raise EngineUnavailable(
        "no feature-complete epistemic-graph[full] wheel binary is installed; "
        "set EPISTEMIC_GRAPH_TEST_BINARY to a serialized pipeline build"
    )


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
        self._security_dir: str | None = None
        self.socket_path: str | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._log: tempfile._TemporaryFileWrapper[bytes] | None = None

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> EphemeralEngine:
        self._root = tempfile.mkdtemp(prefix="au_tiny_engine_")
        root = Path(self._root)
        self._persist_dir = str(root / "persist")
        self._security_dir = str(root / "security")
        os.makedirs(self._persist_dir, exist_ok=True)
        self.socket_path = _free_socket_path(root)

        # Keep the engine log so a startup failure is diagnosable, but in a temp
        # file under the engine's own throwaway dir so it's cleaned up with it.
        self._log = open(  # noqa: SIM115 — closed in stop()
            str(root / "engine.log"), "wb"
        )
        env = {
            **os.environ,
            **strict_server_env(
                self._security_dir,
                auth_secret=TEST_AUTH_SECRET,
            ),
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
            self._bootstrap_identity()
        except Exception:
            # Startup failed — tear the half-started engine down cleanly so we
            # never leak a process or temp dir, then re-raise for the caller.
            self.stop()
            raise
        return self

    def _bootstrap_identity(self) -> None:
        """Enroll the isolated suite signer before ordinary requests run."""

        from epistemic_graph.client import SyncEpistemicGraphClient

        client = SyncEpistemicGraphClient.connect(
            socket_path=self.socket_path,
            auth_secret=TEST_AUTH_SECRET,
            verified_context=bootstrap_context(),
        )
        try:
            client.consensus.bootstrap_system_identity(
                agent_id=TEST_AGENT_ID,
                signer_id=TEST_AGENT_ID,
                signer_key=TEST_SIGNER_KEY,
            )
        finally:
            client.close()

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
        self._root = self._persist_dir = self._security_dir = self.socket_path = None

    def __enter__(self) -> EphemeralEngine:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
