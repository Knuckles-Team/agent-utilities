"""Deployment profile A — "tiny": zero external dependencies (Raspberry Pi 3).

The headline guarantee: agent-utilities + epistemic-graph cold-boot and *serve*
the Knowledge Graph + local OWL over the gateway with **no external services** —
no Kafka, no Postgres, no remote SPARQL/OWL server — using the self-contained
epistemic-graph authority and a local owlready2 reasoner.

This module is deliberately **not** ``@pytest.mark.live`` — it must pass in the
default PR suite as the continuously-enforced zero-dep contract. (The KG/engine
tests skip when the local epistemic-graph engine isn't running, e.g. a polyrepo
CI without the Rust source; the footprint guard always runs.)

Profile B (the full enterprise stack) lives in ``test_profile_enterprise_full.py``.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from agent_utilities.knowledge_graph.backends import set_active_backend
from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

pytestmark = pytest.mark.integration

# External-service client libraries that the tiny profile must NEVER pull in.
# Their presence in a cold import means a heavyweight dependency leaked into the
# few-MB Pi-3 footprint.
_FORBIDDEN_DRIVERS = (
    "aiokafka",
    "confluent_kafka",
    "psycopg",
    "neo4j",
    "falkordb",
    "pystardog",
)


@pytest.fixture(autouse=True)
def _tiny_profile_env(monkeypatch, tmp_path):
    """Pin the process to the tiny (zero-dep) deployment profile for this module."""
    # The engine is the whole database for tiny; pin the host role for parity with
    # the singleton-host daemon path.
    monkeypatch.setenv("KG_DAEMON_ROLE", "host")
    monkeypatch.setenv("AGENT_UTILITIES_DATA_DIR", str(tmp_path / "agent-data"))
    monkeypatch.setenv("OWL_BACKEND", "owlready2")
    monkeypatch.setenv("TASK_QUEUE_BACKEND", "sqlite")
    for ext in (
        "GRAPH_DB_CONNECTION_PROFILE_REF",
        "STATE_DB_URI",
        "KAFKA_BOOTSTRAP_SERVERS",
    ):
        monkeypatch.delenv(ext, raising=False)
    # Rebuild engine/backend under the tiny env (root conftest also resets these).
    set_active_backend(None)
    IntelligenceGraphEngine._set_active_for_tests(None)


def test_cold_import_pulls_no_external_service_drivers():
    """The served stack imports with zero external-service client libraries.

    Runs in a clean subprocess (a shared pytest session would have other tests'
    imports polluting ``sys.modules``) so this is a true cold-boot footprint check.
    """
    probe = (
        "import os, sys, json\n"
        "os.environ.pop('GRAPH_DB_CONNECTION_PROFILE_REF', None)\n"
        "os.environ.pop('STATE_DB_URI', None)\n"
        "os.environ.pop('KAFKA_BOOTSTRAP_SERVERS', None)\n"
        # Import the served surface: the gateway route layer + the MCP/engine entry.
        "import agent_utilities.gateway.graph_api\n"
        "import agent_utilities.mcp.kg_server\n"
        "import agent_utilities.knowledge_graph.core.engine\n"
        f"forbidden = {list(_FORBIDDEN_DRIVERS)!r}\n"
        "leaked = [m for m in forbidden if m in sys.modules]\n"
        "print(json.dumps(leaked))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, f"cold import failed:\n{result.stderr}"
    leaked = json.loads(result.stdout.strip().splitlines()[-1])
    assert leaked == [], f"tiny profile leaked external-service drivers: {leaked}"
