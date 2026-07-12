#!/usr/bin/python

import os
import sys

# Make this directory importable so ``_test_engine`` (the ephemeral real-engine
# lifecycle helper, CONCEPT:AU-KG.memory.provides-real-ephemeral-one) resolves regardless of pytest import-mode /
# the absence of a ``tests/__init__.py``.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# CONCEPT:AU-KG.memory.provides-real-ephemeral-one — the ephemeral fixture (``tests/_test_engine.py``) deploys the
# PUBLISHED ``epistemic-graph`` 1.0.0 wheel: its bundled ``epistemic-graph-server``
# binary (resolved next to ``sys.executable``) and its client are the SAME release,
# so the client (``epistemic_graph.client.NodeClient``) already carries
# ``compare_and_set`` plus the ``.rdf`` / ``.timeseries`` / ``.streaming`` / ``.txn``
# sub-clients. The historical sys.path shim that prepended the sibling engine SOURCE
# checkout (needed only while the floor was the feature-poor 0.31.0 wheel) is removed:
# the venv's installed 1.0.0 client and binary are in lockstep by construction.

os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LOGFIRE_SEND_TO_LOGFIRE", "false")
os.environ.setdefault("ENABLE_GRAPH_INTEGRATION", "false")
os.environ.setdefault("AGENT_UTILITIES_TESTING", "true")
os.environ.setdefault("KNOWLEDGE_GRAPH_SYNC_BACKGROUND", "False")
# The out-of-box default backend is "epistemic_graph" (the durable engine).
# Pin the unit suite to the pure-ephemeral in-memory backend so tests never
# touch disk / take a file lock. Integration tests that exercise
# fanout/ladybug/postgres override this via their own env or by passing an
# explicit ``backend_type=`` to ``create_backend()``.
os.environ.setdefault("GRAPH_BACKEND", "memory")
import shutil
import tempfile

_pytest_tmp_dir = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), ".pytest_tmp"
)
os.makedirs(_pytest_tmp_dir, exist_ok=True)
_test_db_dir = tempfile.mkdtemp(prefix="agent_utilities_test_db_", dir=_pytest_tmp_dir)
os.environ["GRAPH_DB_PATH"] = os.path.join(_test_db_dir, "test_knowledge_graph.db")

# Isolate the workspace root so self-evolution / SDD writers (golden loop,
# sdd.watcher) emit ``.specify/`` artifacts into a throwaway dir instead of
# mutating the real repo — keeps the test suite hermetic so the pre-commit
# pytest hook never reports "files were modified by this hook".
_test_ws_dir = tempfile.mkdtemp(prefix="agent_utilities_test_ws_", dir=_pytest_tmp_dir)
os.makedirs(os.path.join(_test_ws_dir, ".specify", "specs"), exist_ok=True)
os.environ.setdefault("WORKSPACE_PATH", _test_ws_dir)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "concept(id): mark test as validating a specific documentation concept",
    )
    config.addinivalue_line(
        "markers",
        "engine: test runs against the REAL ephemeral epistemic-graph engine "
        "(requests the ``tiny_engine``/``engine_graph`` fixtures). CONCEPT:AU-KG.memory.provides-real-ephemeral-one",
    )


import pytest

from agent_utilities.knowledge_graph.backends import set_active_backend
from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine


@pytest.fixture(autouse=True)
def _isolate_os_environ():
    """Restore ``os.environ`` to its pre-test state after every test.

    Several production paths (``save_config_item``/``set_config``,
    ``apply_served_security_profile``, backend wiring) write directly to
    ``os.environ`` — NOT via ``monkeypatch`` — so monkeypatch's teardown can't
    undo them and the value leaks into every later test. Concretely a persisted
    ``GRAPH_DB_URI=postgresql://…`` makes a neo4j-mirror build raise
    ``ConfigurationError``, a leaked ``KG_AUTH_REQUIRED=1`` makes unauthenticated
    tool calls raise ``PermissionError``, and ``GRAPH_PG_AGE=1`` flips backend
    selection. Snapshotting the whole environment and restoring the delta at the
    test boundary makes the unit suite order-independent regardless of which test
    leaked. Safe here: no module/session-scoped fixture sets env across tests.
    Runs as the first (autouse) fixture, so a test's own ``monkeypatch.setenv``
    still applies within the test and is restored normally.
    """
    snapshot = dict(os.environ)
    try:
        yield
    finally:
        for key in [k for k in os.environ if k not in snapshot]:
            del os.environ[key]
        for key, value in snapshot.items():
            if os.environ.get(key) != value:
                os.environ[key] = value


@pytest.fixture(autouse=True)
def _isolate_registered_tools():
    """Restore ``kg_server.REGISTERED_TOOLS``/``ACTION_TOOL_ROUTES`` to their pre-test contents.

    ``_build_server()`` populates these two process-wide dicts as a side effect,
    binding tool callables to the engine resolved at build time (and — for a
    handful of tools, e.g. ``nl_query``/the Seam 8 intent verbs,
    CONCEPT:AU-ECO.mcp.intent-surface-condensed-collapse — registering their REST-route twin
    dynamically). A test that builds the server (e.g. via a
    ``server_tools``/``registered_tools`` fixture, or a test that exercises
    ``MCP_TOOL_MODE=intent``) therefore leaves stale tool bindings AND route
    entries in the global registries that corrupt a later test's tool calls (a
    ``graph_query`` returning ``[]`` from the wrong engine) or the MCP⇄REST
    parity contract (an ``ACTION_TOOL_ROUTES`` entry surviving into a test whose
    ``REGISTERED_TOOLS`` snapshot doesn't have the matching tool — a false
    "phantom route"). Snapshotting and restoring both registries at the test
    boundary keeps tool registration order-independent. Only guards a flag is
    needless — the dicts ARE the state. Lazy import so conftest stays cheap.
    """
    try:
        from agent_utilities.mcp import kg_server
    except ImportError:
        # The MCP server module requires the optional ``[mcp]`` extra
        # (fastmcp/starlette/fastapi). In the lean serving/CI install (the
        # Guardrails env) those are absent, so there is no tool registry to
        # isolate — make the autouse fixture a no-op instead of erroring every
        # test at setup. Mirrors the lean-tolerance of the heavy-dep mocks above.
        yield
        return

    tools_snapshot = dict(kg_server.REGISTERED_TOOLS)
    routes_snapshot = dict(kg_server.ACTION_TOOL_ROUTES)
    try:
        yield
    finally:
        kg_server.REGISTERED_TOOLS.clear()
        kg_server.REGISTERED_TOOLS.update(tools_snapshot)
        kg_server.ACTION_TOOL_ROUTES.clear()
        kg_server.ACTION_TOOL_ROUTES.update(routes_snapshot)
        # The named-connection registry (CONCEPT:AU-KG.backend.multi-connection-registry) is a process-wide
        # singleton seeded from config; a test that registers a backend (e.g. a
        # Stardog mirror via setup_environment) leaves it pointing at that
        # backend and corrupts a later test's engine/query routing. Drop it so it
        # rebuilds fresh from the (restored) config on next use.
        kg_server._CONNECTION_REGISTRY = None


@pytest.fixture(autouse=True)
def clean_graph_globals(monkeypatch, tmp_path):
    set_active_backend(None)
    IntelligenceGraphEngine.set_active(None)

    # Engine circuit breakers are shared per-endpoint process-wide
    # (CONCEPT:AU-OS.observability.no-op-without-metrics); reset between tests so a deliberate connect failure
    # in one test never leaves the circuit open for the next.
    from agent_utilities.knowledge_graph.core import engine_breaker

    engine_breaker.reset_breakers()

    monkeypatch.setenv("GRAPH_DB_PATH", str(tmp_path / "test_knowledge_graph.db"))

    yield
    set_active_backend(None)
    IntelligenceGraphEngine.set_active(None)
    engine_breaker.reset_breakers()


@pytest.fixture(autouse=True, scope="session")
def cleanup_build_artifacts():
    """Clean up stale build artifacts after the test session.

    Removes build/, dist/, *.egg-info directories and rogue test .db files
    that accumulate from repeated builds and test runs.
    """
    from pathlib import Path

    yield

    project_root = Path(__file__).resolve().parent.parent
    for pattern in ["build", "dist", "*.egg-info"]:
        for match in project_root.glob(pattern):
            shutil.rmtree(match, ignore_errors=True)
    # Remove rogue test database files (but not the real KG database)
    protected = {"knowledge_graph.db"}
    for db in project_root.glob("*.db"):
        if db.name not in protected:
            db.unlink(missing_ok=True)
    for lock in project_root.glob("*.db.lock"):
        lock.unlink(missing_ok=True)

    # Clean up the temporary test DB directory
    if os.path.isdir(_test_db_dir):
        shutil.rmtree(_test_db_dir, ignore_errors=True)


import uuid


@pytest.fixture(autouse=True)
def isolate_graph_compute_engine(monkeypatch):
    """Give each test a unique graph namespace to prevent cross-test state leakage.

    Monkeypatches GraphComputeEngine.__init__ so every instantiation within a
    test gets a unique graph_name derived from a UUID.  Tests that explicitly
    pass a graph_name kwarg keep their own name; others get the per-test unique
    name.  After the test completes, all graphs created are cleared and deleted
    to free server-side memory.
    """
    from agent_utilities.knowledge_graph.core import graph_compute

    _original_init = graph_compute.GraphComputeEngine.__init__
    _test_graph_name = f"test_{uuid.uuid4().hex[:12]}"
    _created_engines: list = []
    _created_graph_names: set = set()

    def _isolated_init(self, graph_name: str | None = None, **kwargs):
        # Use the fixture's unique name when the caller targets the default
        # commons graph (None, or the explicit "__commons__").
        effective_name = (
            _test_graph_name if graph_name in (None, "__commons__") else graph_name
        )
        _created_graph_names.add(effective_name)
        _original_init(self, graph_name=effective_name, **kwargs)
        _created_engines.append(self)

    monkeypatch.setattr(graph_compute.GraphComputeEngine, "__init__", _isolated_init)

    yield _test_graph_name

    # Teardown: clear and delete all test graphs
    for engine in _created_engines:
        try:
            if hasattr(engine, "_client") and engine._client:
                engine._client.clear()
        except Exception:
            pass
    # Delete all created graphs from the server
    for engine in _created_engines:
        for gn in _created_graph_names:
            try:
                if hasattr(engine, "_client") and engine._client:
                    engine._client.tenants.delete(gn)
            except Exception:
                pass
        break  # Only need one client for deletion


# Set True once the isolated session engine is deployed (or an external
# ``GRAPH_SERVICE_SOCKET`` is provided). When it stays False, no engine is
# reachable in this environment, so engine-backed tests are *skipped* rather than
# hard-failing with ``ConnectionRefused`` — see ``pytest_runtest_makereport``.
_TEST_ENGINE_AVAILABLE = False

#: The session engine's socket path once deployed (``None`` ⇒ unavailable). The
#: ``tiny_engine`` fixture returns this; ``engine_graph`` re-asserts it per test.
_SESSION_ENGINE_SOCKET: "str | None" = None


@pytest.fixture(scope="session", autouse=True)
def _session_engine():
    """Deploy ONE REAL ephemeral epistemic-graph engine for the whole test session.

    USER DIRECTIVE (CONCEPT:AU-KG.memory.provides-real-ephemeral-one): engine-backed tests validate against the
    ACTUAL database we ship — NOT SQLite, NOT mocks — deployed ephemerally and
    destroyed/cleaned up after. This (autouse) session fixture subsumes the old
    ``start_epistemic_graph_server`` autostart so the engine is up for the *whole*
    session when available (the main case), while ``engine_graph`` layers per-test
    tenant isolation on top:

    * resolves the engine binary (prebuilt wheel → sibling ``target`` → build the
      lean ``pi``-tier once and cache it — see ``tests/_test_engine.py``);
    * starts it on an ISOLATED ephemeral UDS socket + temp ``--persist-dir`` with
      a test auth secret and ``--idle-shutdown-secs`` (self-cleans if the suite
      dies);
    * exports ``GRAPH_SERVICE_SOCKET`` (+ the secret) so the client /
      ``EngineResolver`` connect to THIS engine via the **shared** leg — no
      autostart needed (CONCEPT:AU-OS.deployment.engine-resolver-auto-provision);
    * on teardown shuts the engine down with a graceful **SIGTERM** (it
      checkpoints + exits cleanly) and removes the temp persist dir + socket.

    When the engine genuinely cannot be obtained (no binary AND no Rust toolchain),
    this fixture **degrades gracefully** — it yields with no engine and
    engine-backed tests *skip* via ``pytest_runtest_makereport`` (it never skips
    or fails the whole session). An externally-provided ``GRAPH_SERVICE_SOCKET``
    (e.g. a shared host engine) is reused verbatim and never torn down here.
    """
    global _TEST_ENGINE_AVAILABLE, _SESSION_ENGINE_SOCKET
    from _test_engine import (
        TEST_AUTH_SECRET,
        EngineUnavailable,
        EphemeralEngine,
        resolve_engine_binary,
    )

    # An externally-managed engine (its own socket) is reused verbatim — don't
    # start (or stop) one of our own.
    external = os.environ.get("GRAPH_SERVICE_SOCKET")
    if external:
        _TEST_ENGINE_AVAILABLE = True
        _SESSION_ENGINE_SOCKET = external
        yield external
        return

    try:
        binary = resolve_engine_binary()
        engine = EphemeralEngine(binary).start()
    except EngineUnavailable as exc:
        # No real engine obtainable — degrade to hermetic-skip mode (the
        # makereport hook turns engine-unreachable errors into skips).
        print(f"[session-engine] real engine unavailable: {exc}")
        yield None
        return

    # Wire the client / EngineResolver to THIS engine for the whole session.
    # Set in os.environ (not monkeypatch) so it survives the per-test
    # ``_isolate_os_environ`` snapshot/restore.
    from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

    os.environ["GRAPH_SERVICE_SOCKET"] = engine.socket_path  # type: ignore[assignment]
    os.environ["GRAPH_SERVICE_AUTH_SECRET"] = TEST_AUTH_SECRET
    _TEST_ENGINE_AVAILABLE = True
    _SESSION_ENGINE_SOCKET = engine.socket_path
    try:
        yield engine.socket_path
    finally:
        _TEST_ENGINE_AVAILABLE = False
        _SESSION_ENGINE_SOCKET = None
        IntelligenceGraphEngine.set_active(None)
        engine.stop()


@pytest.fixture(scope="session")
def tiny_engine(_session_engine):
    """The session engine's socket path — or ``skip`` when no real engine exists.

    CONCEPT:AU-KG.memory.provides-real-ephemeral-one. Requesting this (or ``engine_graph``) is how a test OPTS IN
    to the real database: on a box with no binary AND no Rust toolchain it skips
    with a clear message, whereas a test that merely tolerates the engine being
    absent should not request it (the autouse ``_session_engine`` already wired
    things up best-effort, and ``pytest_runtest_makereport`` skips it on a
    connection error).
    """
    if not _session_engine:
        pytest.skip(
            "real epistemic-graph engine unavailable (no prebuilt binary and no "
            "Rust toolchain to build the lean pi-tier) — cannot run this "
            "engine-backed test against the real database."
        )
    return _session_engine


@pytest.fixture()
def engine_graph(tiny_engine):
    """A FRESH, isolated REAL tenant graph on the session ``tiny_engine``.

    CONCEPT:AU-KG.memory.provides-real-ephemeral-one — fast per-test isolation without a process per test: create
    a uniquely-named tenant graph on the one running engine, yield a
    :class:`GraphComputeEngine` scoped to it, then DELETE it via the engine's
    tenant-purge (CONCEPT:EG-KG.backend.tenant-delete-recreate-same) so per-test state never leaks into another
    test. Requesting this fixture is how a test opts into the REAL database.

    Yields the bound :class:`GraphComputeEngine`; its ``.graph_name`` is the
    unique tenant. The autouse ``isolate_graph_compute_engine`` namespacing is
    irrelevant here — we pass an explicit ``graph_name`` so the engine targets
    exactly this tenant.
    """
    from _test_engine import TEST_AUTH_SECRET

    from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine

    # Re-assert the engine wiring per test: ``tiny_engine`` exports
    # ``GRAPH_SERVICE_SOCKET`` during its (session) setup, but the per-test
    # ``_isolate_os_environ`` snapshots env BEFORE that setup ran for the first
    # engine test and restores the snapshot (deleting the socket var) at each
    # boundary. Setting it here — inside the function fixture, after that
    # snapshot — guarantees every engine test connects to the running engine.
    if isinstance(tiny_engine, str):
        os.environ["GRAPH_SERVICE_SOCKET"] = tiny_engine
        os.environ["GRAPH_SERVICE_AUTH_SECRET"] = TEST_AUTH_SECRET

    graph_name = f"engtest_{uuid.uuid4().hex[:16]}"
    # GraphComputeEngine auto-creates its tenant graph on connect, so the graph
    # exists immediately (reads on an empty graph succeed).
    compute = GraphComputeEngine(graph_name=graph_name)
    client = getattr(compute, "_client", None)
    try:
        yield compute
    finally:
        # Tenant-purge (CONCEPT:EG-KG.backend.tenant-delete-recreate-same): delete the whole graph so no state
        # leaks into the next test's fresh tenant. The client is intentionally
        # left OPEN: the autouse ``isolate_graph_compute_engine`` teardown (which
        # tracks this engine because we created it during the test) then runs its
        # own ``clear()``/``delete()`` on the live connection. Closing the client
        # here would stop its event-loop thread and make that later, timeout-less
        # ``clear()`` block forever — the per-test client thread is a daemon that
        # the OS reaps at process exit, exactly as the existing teardown relies on.
        try:
            if client is not None:
                client.tenants.delete(graph_name)
        except Exception:
            pass


import importlib
import sys
from unittest.mock import MagicMock


def _is_engine_unreachable_error(exc: BaseException | None) -> bool:
    """True if ``exc`` (or its cause chain) is the epistemic-graph engine being
    unreachable — the message raised by ``GraphComputeEngine`` / the client when
    no engine daemon answers. Matched by message (the client raises a builtin
    ``ConnectionError``/``ConnectionRefusedError``, not a typed exception)."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, (ConnectionError, ConnectionRefusedError)):
            msg = str(exc)
            if (
                "epistemic-graph" in msg
                or "Tokio service" in msg
                or "Connection refused" in msg
            ):
                return True
        exc = exc.__cause__ or exc.__context__
    return False


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hermeticity: when no isolated test engine is reachable in this environment
    (e.g. a bare ``pre-commit`` run on a box with no built test-engine), a test
    that needs the engine *skips* instead of hard-failing with
    ``ConnectionRefused``. Where an engine IS started (CI / canonical, which
    autostart it from the epistemic-graph source) these errors remain real
    failures, so genuine engine bugs are never masked.
    """
    outcome = yield
    if _TEST_ENGINE_AVAILABLE:
        return
    report = outcome.get_result()
    if report.failed and call.excinfo is not None:
        if _is_engine_unreachable_error(call.excinfo.value):
            report.outcome = "skipped"
            report.longrepr = (
                str(getattr(item, "location", ("", 0, item.name))[0]),
                int(getattr(item, "location", ("", 0, item.name))[1] or 0),
                "Skipped: epistemic-graph engine not reachable in this "
                "environment (no isolated test engine started). Set "
                "AGENT_UTILITIES_TESTING=true with the epistemic-graph source "
                "present, or export GRAPH_SERVICE_SOCKET, to run engine-backed "
                "tests.",
            )


# Optional heavy dependencies. These are mocked ONLY when genuinely absent, so the
# suite still *collects* on a minimal env — but when the real library is installed
# we use it. Mocking these unconditionally (the previous behaviour) shadowed the real
# libraries with MagicMocks for the whole session and silently broke every test that
# needed real pandas/scipy/sklearn/torch (e.g. the finance pipeline). numpy is never
# mocked (a MagicMock numpy triggers a PyBind11 PyCapsule crash).
_OPTIONAL_HEAVY_DEPS = [
    "scipy",
    "scipy.stats",
    "pandas",
    "sklearn",
    "sklearn.linear_model",
    "sklearn.metrics",
    "sklearn.preprocessing",
    "statsmodels",
    "statsmodels.tsa.stattools",
    "hmmlearn",
    "hmmlearn.hmm",
    "yfinance",
    "torch",
    "torch.nn",
]
for _mod_name in _OPTIONAL_HEAVY_DEPS:
    try:
        importlib.import_module(_mod_name)
    except Exception:  # noqa: BLE001 — any import failure → fall back to a mock
        mock = MagicMock()
        # Give the mock a plausible __version__ so version-probing code doesn't choke.
        mock.__version__ = "0.0.0-mock"
        sys.modules[_mod_name] = mock


# ── Repo-mutation session guard (test hermeticity) ──────────────────────────
# Self-evolution paths (golden loop / SDD distillers) can fire async and write
# into the repo while the suite runs, which made the pre-commit ``pytest`` hook
# report "files were modified by this hook" non-deterministically. This guard
# snapshots the dirty-tracked + untracked sets at session start and, at session
# end, detects any tracked file that became dirty *during* the session and any
# untracked file that appeared during it.
#
# SAFETY (fix/destructive-test-isolation): the guard is REPORT-ONLY by default.
# The previous behaviour — ``git checkout -- <strays>`` + ``os.remove`` against
# the live working tree — destroyed any *concurrent* uncommitted work: a
# developer or agent editing the repo while the suite ran had their edits
# reverted and new files deleted at session end (the snapshot can't distinguish
# test-generated writes from human/agent writes made mid-run). Destructive
# cleanup now requires an explicit opt-in, used only by the pre-commit
# ``pytest`` hook where the tree must return to its starting state:
#
#   AGENT_UTILITIES_TEST_REPO_GUARD=revert  → revert/delete strays (old behaviour)
#   AGENT_UTILITIES_TEST_REPO_GUARD=warn    → report strays to stderr (default)
#   AGENT_UTILITIES_TEST_REPO_GUARD=off     → disable the guard entirely
import subprocess as _au_subprocess  # noqa: E402

_AU_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_AU_DIRTY_AT_START: set[str] = set()
_AU_UNTRACKED_AT_START: set[str] = set()


def _au_guard_mode() -> str:
    mode = os.environ.get("AGENT_UTILITIES_TEST_REPO_GUARD", "warn").strip().lower()
    return mode if mode in {"warn", "revert", "off"} else "warn"


def _au_git(*args: str, repo_root: str | None = None) -> str:
    try:
        out = _au_subprocess.run(
            ["git", "-C", repo_root or _AU_REPO_ROOT, *args],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return out.stdout
    except Exception:  # noqa: BLE001 — not a git repo / git missing → no-op guard
        return ""


def _au_dirty_tracked(repo_root: str | None = None) -> set[str]:
    return {
        line
        for line in _au_git("diff", "--name-only", repo_root=repo_root).splitlines()
        if line.strip()
    }


def _au_untracked(repo_root: str | None = None) -> set[str]:
    return {
        line
        for line in _au_git(
            "ls-files", "--others", "--exclude-standard", repo_root=repo_root
        ).splitlines()
        if line.strip()
    }


def _au_enforce_session_guard(
    stray_tracked: list[str],
    stray_untracked: list[str],
    *,
    repo_root: str | None = None,
    mode: str | None = None,
) -> None:
    """Apply the session guard policy to stray working-tree mutations.

    ``warn`` (default) only reports — it never touches the working tree, so
    concurrent uncommitted edits made while the suite runs are safe. ``revert``
    restores the tree to its session-start state (pre-commit hook only).
    """
    mode = mode or _au_guard_mode()
    if mode == "off" or not (stray_tracked or stray_untracked):
        return
    if mode == "revert":
        if stray_tracked:
            _au_git("checkout", "--", *stray_tracked, repo_root=repo_root)
        for rel in stray_untracked:
            try:
                os.remove(os.path.join(repo_root or _AU_REPO_ROOT, rel))
            except OSError:
                pass
        return
    lines = [
        "",
        "[repo-guard] tests mutated the repository working tree during this session:",
        *(f"[repo-guard]   modified (tracked): {rel}" for rel in stray_tracked),
        *(f"[repo-guard]   created (untracked): {rel}" for rel in stray_untracked),
        "[repo-guard] left in place (AGENT_UTILITIES_TEST_REPO_GUARD=warn). Fix the",
        "[repo-guard] offending test to write under tmp_path; set =revert only in",
        "[repo-guard] hermetic contexts (pre-commit hook) — it deletes/reverts these.",
    ]
    print("\n".join(lines), file=sys.stderr)


def pytest_sessionstart(session):  # noqa: ARG001 — pytest hook signature
    global _AU_DIRTY_AT_START, _AU_UNTRACKED_AT_START
    _AU_DIRTY_AT_START = _au_dirty_tracked()
    _AU_UNTRACKED_AT_START = _au_untracked()


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001 — pytest hook signature
    stray_tracked = sorted(_au_dirty_tracked() - _AU_DIRTY_AT_START)
    stray_untracked = sorted(
        rel
        for rel in _au_untracked() - _AU_UNTRACKED_AT_START
        if not rel.startswith(".pytest_tmp")
    )
    _au_enforce_session_guard(stray_tracked, stray_untracked)
