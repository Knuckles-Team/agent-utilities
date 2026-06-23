"""Deployment profile A — "tiny": zero external dependencies (Raspberry Pi 3).

The headline guarantee: agent-utilities + epistemic-graph cold-boot and *serve*
the Knowledge Graph + local OWL over the gateway with **no external services** —
no Kafka, no Postgres, no remote SPARQL/OWL server — using the in-process
epistemic-graph L1 + embedded LadybugDB L2 and a local owlready2 reasoner.

This module is deliberately **not** ``@pytest.mark.live`` — it must pass in the
default PR suite as the continuously-enforced zero-dep contract. (The KG/engine
tests skip when the local epistemic-graph engine isn't running, e.g. a polyrepo
CI without the Rust source; the footprint guard always runs.)

Profile B (the full enterprise stack) lives in ``test_profile_enterprise_full.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid

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
    monkeypatch.setenv("GRAPH_BACKEND", "epistemic_graph")
    # The engine is the whole database for tiny; pin the host role for parity with
    # the singleton-host daemon path.
    monkeypatch.setenv("KG_DAEMON_ROLE", "host")
    monkeypatch.setenv("GRAPH_DB_PATH", str(tmp_path / "tiny_kg.db"))
    monkeypatch.setenv("OWL_BACKEND", "owlready2")
    monkeypatch.setenv("TASK_QUEUE_BACKEND", "sqlite")
    monkeypatch.setenv("AGENT_DISPATCH_BACKEND", "inline")
    for ext in (
        "GRAPH_DB_URI",
        "STATE_DB_URI",
        "PGGRAPH_DSN",
        "KAFKA_BOOTSTRAP_SERVERS",
    ):
        monkeypatch.delenv(ext, raising=False)
    # Rebuild engine/backend under the tiny env (root conftest also resets these).
    set_active_backend(None)
    IntelligenceGraphEngine.set_active(None)


def test_cold_import_pulls_no_external_service_drivers():
    """The served stack imports with zero external-service client libraries.

    Runs in a clean subprocess (a shared pytest session would have other tests'
    imports polluting ``sys.modules``) so this is a true cold-boot footprint check.
    """
    probe = (
        "import os, sys, json\n"
        "os.environ['GRAPH_BACKEND'] = 'epistemic_graph'\n"
        "os.environ.pop('GRAPH_DB_URI', None)\n"
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


def test_local_owl_reasoner_runs_in_process():
    """OWL runs locally (owlready2), not against a remote Stardog/Fuseki server."""
    pytest.importorskip("owlready2")
    from agent_utilities.knowledge_graph.backends.owl import create_owl_backend
    from agent_utilities.knowledge_graph.backends.owl.owlready2_backend import (
        Owlready2Backend,
    )

    owl = create_owl_backend()  # OWL_BACKEND=owlready2 from the profile env
    try:
        assert isinstance(owl, Owlready2Backend), (
            "tiny profile must use a local OWL reasoner"
        )
        # An in-process reasoner exposes live stats without any network call.
        stats = owl.get_stats()
        assert isinstance(stats, dict)
    finally:
        owl.close()


def test_tiny_profile_serves_kg_over_gateway_with_zero_containers():
    """Write + read the KG through the local gateway REST surface, no containers."""
    if not os.environ.get("GRAPH_SERVICE_SOCKET"):
        pytest.skip(
            "local epistemic-graph engine not running (GRAPH_SERVICE_SOCKET unset)"
        )

    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from agent_utilities.gateway.graph_api import register_graph_routes
    from agent_utilities.mcp import kg_server

    kg_server.ensure_tools_registered()

    app = FastAPI()
    register_graph_routes(app, prefix="/api")

    node_id = f"tiny:{uuid.uuid4().hex[:8]}"
    with TestClient(app) as client:
        write = client.post(
            "/api/graph/write",
            json={
                "action": "add_node",
                "node_id": node_id,
                "node_type": "TinyProfileNode",
                "properties": json.dumps({"served": True, "profile": "tiny"}),
            },
        )
        assert write.status_code == 200, write.text
        assert write.json().get("status") == "success", write.text

        read = client.post(
            "/api/graph/query",
            json={
                "cypher": "MATCH (n:TinyProfileNode) WHERE n.id = $id RETURN n.id AS id",
                # graph_query expects ``params`` as a JSON string, mirroring the MCP tool.
                "params": json.dumps({"id": node_id}),
            },
        )
        assert read.status_code == 200, read.text
        payload = read.json()
        assert payload.get("status") == "success", payload
        assert node_id in json.dumps(payload["result"]), payload

    # The serving engine is backed by the zero-dep tiered stack — never Postgres.
    engine = kg_server._get_engine()
    assert type(engine.backend).__name__ != "PostgreSQLBackend"
