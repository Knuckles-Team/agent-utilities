#!/usr/bin/python

import os

os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LOGFIRE_SEND_TO_LOGFIRE", "false")
os.environ.setdefault("ENABLE_GRAPH_INTEGRATION", "false")
os.environ.setdefault("AGENT_UTILITIES_TESTING", "true")
import tempfile
import atexit
import shutil

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
