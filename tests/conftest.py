#!/usr/bin/python

import os

os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LOGFIRE_SEND_TO_LOGFIRE", "false")
os.environ.setdefault("ENABLE_GRAPH_INTEGRATION", "false")
os.environ.setdefault("AGENT_UTILITIES_TESTING", "true")
os.environ.setdefault("KNOWLEDGE_GRAPH_SYNC_BACKGROUND", "False")
# The out-of-box default backend is "tiered" (epistemic_graph + LadybugDB).
# Pin the unit suite to the pure-ephemeral in-memory backend so tests never
# touch disk / take a file lock. Integration tests that exercise
# tiered/ladybug/postgres override this via their own env or by passing an
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


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "concept(id): mark test as validating a specific documentation concept",
    )


import pytest

from agent_utilities.knowledge_graph.backends import set_active_backend
from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine


@pytest.fixture(autouse=True)
def clean_graph_globals(monkeypatch, tmp_path):
    set_active_backend(None)
    IntelligenceGraphEngine.set_active(None)

    monkeypatch.setenv("GRAPH_DB_PATH", str(tmp_path / "test_knowledge_graph.db"))

    yield
    set_active_backend(None)
    IntelligenceGraphEngine.set_active(None)


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


import subprocess
import time
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

    def _isolated_init(self, graph_name: str = "__bus__", **kwargs):
        # Use the fixture's unique name when the caller uses the default __bus__
        effective_name = _test_graph_name if graph_name == "__bus__" else graph_name
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


@pytest.fixture(scope="session", autouse=True)
def start_epistemic_graph_server():
    import os

    if os.environ.get("AGENT_UTILITIES_TESTING") == "true":
        # Check if epistemic-graph is built, if not build it
        rust_dir = os.path.join(os.path.dirname(__file__), "../../epistemic-graph")
        rust_dir = os.path.abspath(rust_dir)
        if not os.path.isdir(rust_dir):
            # Polyrepo / CI: the epistemic-graph source is not checked out as a
            # sibling here, so we can't build/run the local engine. Skip starting
            # it — config-only tests don't need a live engine; tests that do will
            # skip when GRAPH_SERVICE_SOCKET is unset.
            print(
                "epistemic-graph source not present "
                f"({rust_dir}); skipping engine startup."
            )
            yield None
            return
        socket_path = os.path.join(
            os.path.dirname(__file__), ".test_epistemic_graph.sock"
        )
        socket_path = os.path.abspath(socket_path)

        if os.path.exists(socket_path):
            os.remove(socket_path)

        print("Starting epistemic-graph-server...")
        # Build first
        subprocess.run(["cargo", "build", "--all-features"], cwd=rust_dir, check=False)

        log_file = open(
            os.path.join(tempfile.gettempdir(), ".test_epistemic_graph.log"), "w"
        )
        # Start server
        process = subprocess.Popen(
            [
                "cargo",
                "run",
                "--all-features",
                "--bin",
                "epistemic-graph-server",
                "--",
                "--socket-path",
                socket_path,
            ],
            cwd=rust_dir,
            stdout=log_file,
            stderr=log_file,
        )

        # Wait for socket to be ready
        for _ in range(60):
            if process.poll() is not None:
                log_file.flush()
                with open(log_file.name) as f:
                    logs = f.read()
                raise RuntimeError(
                    f"epistemic-graph-server crashed on startup.\nLogs:\n{logs}"
                )
            if os.path.exists(socket_path):
                break
            time.sleep(0.5)
        else:
            raise RuntimeError(
                "epistemic-graph-server failed to create socket in time."
            )

        # Set environment variable so the client connects to this socket
        os.environ["GRAPH_SERVICE_SOCKET"] = socket_path

        yield process

        # Cleanup
        process.terminate()
        process.wait()
        if os.path.exists(socket_path):
            os.remove(socket_path)
    else:
        yield None


import importlib
import sys
from unittest.mock import MagicMock

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
