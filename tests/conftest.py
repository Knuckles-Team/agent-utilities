#!/usr/bin/python

import os

os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LOGFIRE_SEND_TO_LOGFIRE", "false")
os.environ.setdefault("ENABLE_GRAPH_INTEGRATION", "false")
os.environ.setdefault("AGENT_UTILITIES_TESTING", "true")

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "concept(id): mark test as validating a specific documentation concept"
    )

import pytest

from agent_utilities.knowledge_graph.backends import set_active_backend
from agent_utilities.knowledge_graph.engine import IntelligenceGraphEngine


@pytest.fixture(autouse=True)
def clean_graph_globals():
    set_active_backend(None)
    IntelligenceGraphEngine.set_active(None)
    yield
    set_active_backend(None)
    IntelligenceGraphEngine.set_active(None)
