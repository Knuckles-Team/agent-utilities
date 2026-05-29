#!/usr/bin/python

import os

os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LOGFIRE_SEND_TO_LOGFIRE", "false")
os.environ.setdefault("ENABLE_GRAPH_INTEGRATION", "false")
os.environ.setdefault("AGENT_UTILITIES_TESTING", "true")
os.environ.setdefault("KNOWLEDGE_GRAPH_SYNC_BACKGROUND", "False")
import shutil
import tempfile

_test_db_dir = tempfile.mkdtemp(prefix="agent_utilities_test_db_")
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
import socket

@pytest.fixture(scope="session", autouse=True)
def start_epistemic_graph_server():
    import os
    if os.environ.get("AGENT_UTILITIES_TESTING") == "true":
        # Check if epistemic-graph is built, if not build it
        rust_dir = os.path.join(os.path.dirname(__file__), "../../epistemic-graph")
        rust_dir = os.path.abspath(rust_dir)
        socket_path = "/tmp/test_epistemic_graph.sock"

        if os.path.exists(socket_path):
            os.remove(socket_path)

        print("Starting epistemic-graph-server...")
        # Build first
        subprocess.run(["cargo", "build"], cwd=rust_dir, check=False)

        # Start server
        process = subprocess.Popen(
            ["cargo", "run", "--bin", "epistemic-graph-server", "--", "--socket-path", socket_path],
            cwd=rust_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # Wait for socket to be ready
        for _ in range(30):
            if os.path.exists(socket_path):
                break
            time.sleep(0.5)

        # Set environment variable so the client connects to this socket
        os.environ["GRAPH_SERVICE_SOCKET"] = socket_path

        yield process

        # Cleanup
        process.terminate()
        process.wait()
        if os.path.exists(socket_path):
            os.remove(socket_path)
