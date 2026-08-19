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

import base64
import hashlib
import importlib.metadata as importlib_metadata
import json
import os
import re
import shutil
import signal
import socket
import stat
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

#: Shared HMAC secret the test engine runs under. The engine REFUSES to start
#: without one (CONCEPT:AU-OS.identity.authenticated-identity-enforcement); every client authenticates with this exact
#: value, exported as ``GRAPH_SERVICE_AUTH_SECRET`` so the resolver/EngineResolver
#: pick it up. Test-only; never a real credential.
TEST_AUTH_SECRET = "agent-" + "utilities-test-engine-secret"  # nosec B105 — test-only
TEST_AGENT_ID = "service:agent-utilities-test-suite"
TEST_SIGNER_KEY = "agent-utilities-test-operation-signer"  # nosec B105 - test only
# Current epistemic-graph requires the first durable identity registration to be
# backed by an explicitly scoped signer entry.  This signer may bootstrap only
# its own System identity; it has no allowance to register named RBAC roles.
TEST_SIGNER_REGISTRY = {
    TEST_AGENT_ID: {
        "key": TEST_SIGNER_KEY,
        "allowed_roles": [],
        "may_grant_system": True,
    }
}
TEST_AUDIENCE = "epistemic-graph-test"
TEST_TENANT = "tenant:test"
TEST_POLICY_VERSION = "policy:test"
#: Fixed data-at-rest key for the ephemeral test engine's durable transaction/
#: blob substrate (the native engine's txn/blob commit paths refuse to run
#: without one — "transaction durability requires EPISTEMIC_GRAPH_ENCRYPTION_KEY
#: to be configured"). Kept OUT of ``strict_server_env`` deliberately —
#: ``test_ephemeral_engine_environment.py`` asserts that function's return
#: value by exact dict equality, so this is merged directly into
#: :meth:`EphemeralEngine.start`'s env instead of widening that contract.
#: >= 32 bytes as required by ``_ENGINE_ENCRYPTION_KEY_MIN_BYTES``; test-only,
#: never a real credential.
TEST_ENGINE_ENCRYPTION_KEY = (  # nosec B105 - test only
    "agent-utilities-test-engine-encryption-key-0123456789abcdef"
)


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
    """Build the complete authenticated local-server environment for tests.

    The production engine requires OIDC identity binding by default.  This
    isolated test server deliberately opts out of that external verifier while
    retaining transport HMAC authentication and signed request identities.
    """

    return {
        "GRAPH_SERVICE_AUTH_SECRET": auth_secret,
        "EPISTEMIC_GRAPH_AUDIENCE": TEST_AUDIENCE,
        "EPISTEMIC_GRAPH_TENANT": TEST_TENANT,
        "EPISTEMIC_GRAPH_POLICY_VERSION": TEST_POLICY_VERSION,
        "EPISTEMIC_GRAPH_REQUIRE_OIDC": "false",
        "EPISTEMIC_GRAPH_SECURITY_STATE_DIR": state_dir,
        "EPISTEMIC_GRAPH_SIGNER_KEYS_JSON": json.dumps(
            TEST_SIGNER_REGISTRY, sort_keys=True
        ),
    }


#: Reference-counted idle-shutdown grace (seconds). If the pytest session dies
#: without running teardown (a crash / SIGKILL), the engine self-terminates this
#: many seconds after its last client disconnects — so a dead suite leaves no
#: orphan process. Short enough to reap promptly, long enough to span a slow
#: session of back-to-back tests sharing the one engine.
#:
#: D-CDX-32: this grace is measured against the engine's *active-connection*
#: count reaching zero (``run_idle_watcher`` in
#: ``epistemic-graph/src/server/transport.rs``), not wall-clock since the
#: engine started. Individual test clients connect-and-disconnect per
#: operation, so nothing kept this count above zero between tests on its own —
#: a full-suite run is ~16k tests over 45+ minutes where only a small,
#: unevenly-spaced minority touch the engine at all, so a >120s gap with zero
#: connections was essentially guaranteed at least once per run (worse under
#: load, when individual tests stall near the 60s pytest-timeout boundary).
#: Once the watcher fires, the engine exits for good with no re-spawn
#: anywhere in ``conftest.py``, so every remaining engine-dependent test for
#: the rest of the session fails identically with
#: ``ConnectionRefusedError`` — the single largest failure cluster observed
#: in the 2026-08-02 gate run (786 of 899 failing/erroring entries). See
#: :meth:`EphemeralEngine._open_keepalive` for the actual fix (a session-long
#: held connection that keeps the count above zero); this constant only
#: governs the crash-cleanup grace once that connection is gone.
IDLE_SHUTDOWN_SECS = 120

#: D-CDX-32-cascade (2026-08-11): the held keepalive connection alone did NOT
#: keep the engine's active-connection count above zero -- a full-suite run
#: spends several minutes at the START of the session (``tests/consolidation``,
#: ``tests/docs``, ``tests/gates``, ``tests/harness`` -- none of them touch the
#: engine) before the first engine-backed test runs, comfortably longer than
#: ``IDLE_SHUTDOWN_SECS``, and the engine's own log showed
#: ``Idle shutdown: no connections for 120s`` firing during exactly that gap
#: despite :meth:`EphemeralEngine._open_keepalive` having already connected.
#: A merely-open-but-silent connection is therefore not sufficient (whether
#: because the server only counts a connection while a request is in flight,
#: or because an idle-read timeout on the server side closes a silent
#: connection well under 120s -- either way, holding the socket open is not
#: something this test helper can prove from the client side). A periodic
#: heartbeat RPC over the SAME held connection (see
#: :meth:`EphemeralEngine._keepalive_loop`) generates real traffic often
#: enough to guarantee the watcher never observes a gap, independent of which
#: of those mechanisms is the actual explanation.
_KEEPALIVE_HEARTBEAT_INTERVAL_SECS = 20.0

#: How long to wait for the engine's socket to appear after spawn, and for the
#: process to exit on graceful SIGTERM.
_SOCKET_WAIT_SECS = 30.0
_SHUTDOWN_WAIT_SECS = 15.0

_BINARY_NAME = "epistemic-graph-server"
_DISTRIBUTION_NAME = "epistemic-graph"
_EXPLICIT_BINARY_ENV_VARS = (
    "EPISTEMIC_GRAPH_SERVER_BIN",
    # Exact-engine campaign tooling predates the shared acceptance locator.  It
    # remains an explicit path, never a discovery fallback.
    "EPISTEMIC_GRAPH_TEST_BINARY",
)
_EXPECTED_DIGEST_ENV_BY_BINARY_ENV = {
    "EPISTEMIC_GRAPH_SERVER_BIN": "EPISTEMIC_GRAPH_SERVER_BIN_SHA256",
    "EPISTEMIC_GRAPH_TEST_BINARY": "EPISTEMIC_GRAPH_TEST_BINARY_SHA256",
}
_SOURCE_REVISION_ENV_BY_BINARY_ENV = {
    "EPISTEMIC_GRAPH_SERVER_BIN": "EPISTEMIC_GRAPH_SERVER_BIN_SOURCE_REVISION",
    "EPISTEMIC_GRAPH_TEST_BINARY": "EPISTEMIC_GRAPH_TEST_BINARY_SOURCE_REVISION",
}
_EXPLICIT_METADATA_ENV_VARS = (
    "EPISTEMIC_GRAPH_SERVER_BIN_SHA256",
    "EPISTEMIC_GRAPH_TEST_BINARY_SHA256",
    "EPISTEMIC_GRAPH_SERVER_BIN_SOURCE_REVISION",
    "EPISTEMIC_GRAPH_TEST_BINARY_SOURCE_REVISION",
)
_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")
_COMMIT_REVISION_RE = re.compile(r"[0-9a-fA-F]{40}\Z")


@dataclass(frozen=True)
class EngineBinaryIdentity:
    """The exact executable identity bound by a real-engine acceptance run.

    Explicit source-build paths do not have an installed distribution record to
    identify them.  Their content digest is therefore part of the selection
    result and is the separately recorded build identity for the run.  The
    exact campaign's commit revision is retained alongside that digest.  A
    wheel selection additionally carries the active distribution's recorded
    script and version; its RECORD hash and size are verified when present.
    """

    path: Path
    selection: str
    artifact_sha256: str
    artifact_size: int
    distribution_version: str | None = None
    distribution_record: str | None = None
    distribution_record_sha256: str | None = None
    source_env: str | None = None
    source_revision: str | None = None
    expected_sha256: str | None = None

    def verify_for_launch(self) -> None:
        """Re-open this exact artifact and fail closed if it changed."""

        _verify_binary_identity(self)


class EngineUnavailable(RuntimeError):
    """The mandatory full wheel's real engine binary could not be obtained."""


def _read_explicit_identity_contract(
    env_names: tuple[str, ...],
) -> tuple[str, str]:
    """Read one unambiguous digest/revision pair for explicit binaries."""

    configured_metadata = {
        name: str(os.environ.get(name, "") or "").strip()
        for name in _EXPLICIT_METADATA_ENV_VARS
        if str(os.environ.get(name, "") or "").strip()
    }
    expected_names = tuple(
        _EXPECTED_DIGEST_ENV_BY_BINARY_ENV[name] for name in env_names
    )
    revision_names = tuple(
        _SOURCE_REVISION_ENV_BY_BINARY_ENV[name] for name in env_names
    )
    unexpected = set(configured_metadata) - set(expected_names) - set(revision_names)
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise EngineUnavailable(
            f"explicit engine metadata belongs to a different binary authority: {names}"
        )
    missing = [
        name
        for name in (*expected_names, *revision_names)
        if name not in configured_metadata
    ]
    if missing:
        names = ", ".join(missing)
        raise EngineUnavailable(
            "explicit engine path requires expected SHA-256 and source revision: "
            f"{names}"
        )
    digests = {configured_metadata[name].lower() for name in expected_names}
    if any(_SHA256_RE.fullmatch(value) is None for value in digests):
        raise EngineUnavailable("explicit engine SHA-256 authority is malformed")
    if len(digests) != 1:
        raise EngineUnavailable("conflicting explicit engine SHA-256 authorities")
    revisions = {configured_metadata[name].lower() for name in revision_names}
    if any(_COMMIT_REVISION_RE.fullmatch(value) is None for value in revisions):
        raise EngineUnavailable(
            "explicit engine source revision must be a 40-character commit SHA"
        )
    if len(revisions) != 1:
        raise EngineUnavailable("conflicting explicit engine source revisions")
    return next(iter(digests)), next(iter(revisions))


def _resolve_engine_binary_selection() -> tuple[
    Path, str, Any | None, Any | None, tuple[str, ...], str | None, str | None
]:
    """Resolve one exact executable and retain its provenance for identity."""

    configured: list[tuple[str, str]] = []
    for env_name in _EXPLICIT_BINARY_ENV_VARS:
        value = str(os.environ.get(env_name, "") or "").strip()
        if value:
            configured.append((env_name, value))
    if configured:
        values = {value for _name, value in configured}
        if len(values) != 1:
            names = ", ".join(name for name, _value in configured)
            raise EngineUnavailable(f"conflicting explicit engine paths: {names}")
        env_names = tuple(name for name, _value in configured)
        env_name, value = configured[0]
        expected_sha256, source_revision = _read_explicit_identity_contract(env_names)
        return (
            _validate_engine_binary(
                Path(value).expanduser(), source=f"{env_name} explicit path"
            ),
            "explicit",
            None,
            None,
            env_names,
            expected_sha256,
            source_revision,
        )

    stale_metadata = [
        name
        for name in _EXPLICIT_METADATA_ENV_VARS
        if str(os.environ.get(name, "") or "").strip()
    ]
    if stale_metadata:
        raise EngineUnavailable(
            "explicit engine metadata requires an explicit engine path: "
            + ", ".join(sorted(stale_metadata))
        )

    try:
        distribution = importlib_metadata.distribution(_DISTRIBUTION_NAME)
    except importlib_metadata.PackageNotFoundError as exc:
        raise EngineUnavailable(
            "active epistemic-graph distribution is not installed"
        ) from exc

    records = [
        record
        for record in (distribution.files or ())
        if Path(record).name == _BINARY_NAME
    ]
    if len(records) != 1:
        raise EngineUnavailable(
            "active epistemic-graph distribution does not record exactly one "
            f"{_BINARY_NAME} script"
        )
    record = records[0]
    return (
        _validate_engine_binary(
            Path(distribution.locate_file(record)),
            source="active epistemic-graph distribution",
        ),
        "distribution",
        distribution,
        record,
        (),
        None,
        None,
    )


def _sha256_binary(path: Path) -> tuple[str, int]:
    """Return a fresh content identity for one validated executable."""

    hasher = hashlib.sha256()
    open_flags = os.O_RDONLY | int(getattr(os, "O_NOFOLLOW", 0))
    try:
        fd = os.open(path, open_flags)
        with os.fdopen(fd, "rb", closefd=True) as binary:
            file_stat = os.fstat(binary.fileno())
            if not stat.S_ISREG(file_stat.st_mode):
                raise EngineUnavailable("resolved engine executable is not regular")
            for chunk in iter(lambda: binary.read(1024 * 1024), b""):
                hasher.update(chunk)
    except EngineUnavailable:
        raise
    except OSError as exc:
        raise EngineUnavailable("resolved engine executable is unreadable") from exc
    digest = hasher.hexdigest()
    return digest, file_stat.st_size


def resolve_engine_binary() -> Path:
    """Locate the exact ``epistemic-graph-server`` used by the active client.

    Resolution order:

    1. An explicit, absolute regular executable path from
       ``EPISTEMIC_GRAPH_SERVER_BIN`` (or the exact-engine campaign's
       ``EPISTEMIC_GRAPH_TEST_BINARY``).
    2. The uniquely recorded ``epistemic-graph-server`` script co-installed by
       the active ``epistemic-graph`` distribution.

    It deliberately does not search ``PATH``, walk ancestor directories, or
    select a binary merely because it shares a name with the Python client.
    Raises :class:`EngineUnavailable` for missing, ambiguous, symlinked,
    non-regular, or non-executable artifacts. It never compiles implicitly.
    """
    return resolve_engine_binary_identity().path


def resolve_engine_binary_identity() -> EngineBinaryIdentity:
    """Resolve and verify the exact artifact identity for acceptance evidence.

    An explicit executable is accepted only after its complete SHA-256 digest is
    calculated, so a source-built path is never represented by basename/mode
    alone.  A distribution-provided executable remains bound to its one
    recorded script, and any available RECORD hash/size must match the bytes
    that will be launched.
    """

    (
        path,
        selection,
        distribution,
        record,
        source_envs,
        expected_sha256,
        source_revision,
    ) = _resolve_engine_binary_selection()
    digest, size = _sha256_binary(path)
    record_digest: str | None = None
    distribution_version: str | None = None
    distribution_record: str | None = None
    if selection == "distribution":
        assert distribution is not None
        assert record is not None
        distribution_version = str(getattr(distribution, "version", "")) or None
        distribution_record = str(record)
        file_hash = getattr(record, "hash", None)
        record_mode = getattr(file_hash, "mode", None)
        record_value = getattr(file_hash, "value", None)
        if file_hash is not None and record_mode != "sha256":
            raise EngineUnavailable(
                "active epistemic-graph distribution uses an unsupported "
                "engine script RECORD hash"
            )
        if record_value:
            expected = (
                base64.urlsafe_b64encode(bytes.fromhex(digest))
                .rstrip(b"=")
                .decode("ascii")
            )
            if expected != record_value:
                raise EngineUnavailable(
                    "active epistemic-graph distribution RECORD hash does not "
                    "match its engine script"
                )
            record_digest = record_value
        record_size = getattr(record, "size", None)
        if record_size is not None and record_size != size:
            raise EngineUnavailable(
                "active epistemic-graph distribution RECORD size does not "
                "match its engine script"
            )
    elif digest != expected_sha256:
        raise EngineUnavailable(
            "explicit engine SHA-256 does not match the supplied authority"
        )
    return EngineBinaryIdentity(
        path=path,
        selection=selection,
        artifact_sha256=digest,
        artifact_size=size,
        distribution_version=distribution_version,
        distribution_record=distribution_record,
        distribution_record_sha256=record_digest,
        source_env=source_envs[0] if source_envs else None,
        source_revision=source_revision,
        expected_sha256=expected_sha256,
    )


def _verify_binary_identity(identity: EngineBinaryIdentity) -> None:
    """Re-open and verify the exact artifact immediately before spawning."""

    _validate_engine_binary(
        identity.path,
        source=identity.source_env or "active epistemic-graph distribution",
    )
    digest, size = _sha256_binary(identity.path)
    if digest != identity.artifact_sha256 or size != identity.artifact_size:
        raise EngineUnavailable(
            "engine executable changed after identity resolution; refusing to spawn"
        )
    if identity.selection == "explicit" and digest != identity.expected_sha256:
        raise EngineUnavailable(
            "engine executable no longer matches its supplied SHA-256 authority"
        )


def _validate_engine_binary(candidate: Path, *, source: str) -> Path:
    """Validate one exact binary path without resolving an alternate artifact."""

    if not candidate.is_absolute():
        raise EngineUnavailable(f"{source} must be an absolute path")
    if candidate.name != _BINARY_NAME:
        raise EngineUnavailable(f"{source} must name the {_BINARY_NAME} executable")
    try:
        mode = candidate.stat().st_mode
    except OSError as exc:
        raise EngineUnavailable(f"{source} is unavailable") from exc
    if candidate.is_symlink() or not stat.S_ISREG(mode):
        raise EngineUnavailable(f"{source} is not a regular executable file")
    if not os.access(candidate, os.X_OK):
        raise EngineUnavailable(f"{source} is not executable")
    return candidate


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

    def __init__(self, binary_identity: EngineBinaryIdentity) -> None:
        self.binary_identity = binary_identity
        self.binary = binary_identity.path
        self._root: str | None = None
        self._persist_dir: str | None = None
        self._security_dir: str | None = None
        self.socket_path: str | None = None
        self._proc: subprocess.Popen[bytes] | None = None
        self._log: BinaryIO | None = None
        #: A single held connection for the engine's whole ephemeral lifetime
        #: (D-CDX-32). Per-test clients connect and disconnect immediately, so
        #: without this the engine's active-connection count genuinely reaches
        #: zero between tests; holding one keeps ``run_idle_watcher`` (the
        #: engine's own ``--idle-shutdown-secs`` timer) permanently re-armed
        #: for the session, which is the actual crash-safety net
        #: ``--idle-shutdown-secs`` exists for — a suite that dies without
        #: teardown drops this connection along with the process, and the
        #: engine still self-cleans on schedule.
        self._keepalive_client: Any | None = None
        #: Background thread pulsing a cheap RPC over ``_keepalive_client`` so
        #: the held connection is provably active, not merely open — see
        #: :data:`_KEEPALIVE_HEARTBEAT_INTERVAL_SECS`.
        self._keepalive_thread: threading.Thread | None = None
        self._keepalive_stop = threading.Event()

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
            # Required for the engine's durable txn/blob commit paths (media
            # store, ACID cross-modal writes, ...) — see TEST_ENGINE_ENCRYPTION_KEY.
            "EPISTEMIC_GRAPH_ENCRYPTION_KEY": TEST_ENGINE_ENCRYPTION_KEY,
        }
        try:
            # Re-open and hash the retained identity immediately before Popen so
            # a replacement cannot silently turn an exact acceptance run into a
            # different server process.
            self.binary_identity.verify_for_launch()
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
            self._wait_for_socket()
            self._bootstrap_identity()
            self._open_keepalive()
        except Exception:
            # Startup failed — tear the half-started engine down cleanly so we
            # never leak a process or temp dir, then re-raise for the caller.
            self.stop()
            raise
        return self

    def _open_keepalive(self) -> None:
        """Hold one connection open, and actively pulsed, for the engine's
        whole ephemeral lifetime.

        D-CDX-32: ``--idle-shutdown-secs`` (see :data:`IDLE_SHUTDOWN_SECS`) is
        reference-counted against the engine's *active connection* count, not
        wall-clock since start — every per-test client connects and
        disconnects immediately around its own operation, so without a held
        connection the count genuinely reaches zero between tests. Over a
        full-suite run (tens of thousands of tests, 45+ minutes, only a small
        and unevenly-spaced fraction of which touch the engine at all) a
        greater-than-``IDLE_SHUTDOWN_SECS`` gap with zero connections was
        essentially guaranteed at least once — and once the watcher fires the
        engine exits for good with nothing in ``conftest.py`` to notice or
        respawn it, so every remaining engine-dependent test for the rest of
        the session failed identically with ``ConnectionRefusedError``. This
        was the single largest failure cluster in the 2026-08-02 full-suite
        gate run (786 of 899 failing/erroring entries).

        D-CDX-32-cascade (2026-08-11): merely holding the connection OPEN was
        not enough on its own — reproduced live, the idle-shutdown watcher
        still fired during the several-minute, all-non-engine-tests gap at
        the START of a full run (``tests/consolidation``/``docs``/``gates``/
        ``harness`` before the first ``tests/integration/backends`` test),
        with this exact connection already established. See
        :data:`_KEEPALIVE_HEARTBEAT_INTERVAL_SECS` and :meth:`_keepalive_loop`
        for the actual traffic that closes that gap.

        Held for the process lifetime and closed in :meth:`stop`; a crashed
        suite (no teardown) still drops this connection (and stops the
        heartbeat thread, a daemon thread) along with the whole process, so
        ``--idle-shutdown-secs`` keeps doing its actual job — not killing a
        live, in-progress session, just self-cleaning a dead one.
        """
        from epistemic_graph.client import SyncEpistemicGraphClient

        self._keepalive_client = SyncEpistemicGraphClient.connect(
            socket_path=self.socket_path,
            auth_secret=TEST_AUTH_SECRET,
            verified_context=bootstrap_context(),
        )
        self._keepalive_stop.clear()
        self._keepalive_thread = threading.Thread(
            target=self._keepalive_loop,
            name="au-tiny-engine-keepalive",
            daemon=True,
        )
        self._keepalive_thread.start()

    def _keepalive_loop(self) -> None:
        """Pulse a cheap RPC on ``_keepalive_client`` until :meth:`stop`.

        Runs in its own daemon thread for the engine's whole ephemeral
        lifetime, independent of whatever the pytest main thread is doing —
        a long gap of non-engine tests (or a stalled test) must never starve
        this. ``Any`` failure here (a transient reconnect, a slow response)
        is swallowed: this thread's only job is generating traffic, never
        failing the suite, and the real client used by tests reconnects on
        its own (``EpistemicGraphClient._reconnect``) regardless of this
        loop's outcome.
        """
        while not self._keepalive_stop.wait(_KEEPALIVE_HEARTBEAT_INTERVAL_SECS):
            client = self._keepalive_client
            if client is None:
                return
            try:
                client.health()
            except Exception:
                # Best-effort — a single missed beat must not crash the
                # session; the next tick (or a real test's own client) tries
                # again / reconnects.
                pass

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
        self._keepalive_stop.set()
        if self._keepalive_thread is not None:
            self._keepalive_thread.join(timeout=5)
            self._keepalive_thread = None
        if self._keepalive_client is not None:
            try:
                self._keepalive_client.close()
            except Exception:
                # Best-effort — the engine is about to be SIGTERM'd regardless,
                # and a close failure here must never block teardown.
                pass
            self._keepalive_client = None
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
