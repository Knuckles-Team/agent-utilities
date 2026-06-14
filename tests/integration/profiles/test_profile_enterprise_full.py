"""Deployment profile B — "enterprise": the full external stack.

Spins up throwaway pggraph (durable KG + state), Kafka (event backbone), and
Apache Jena Fuseki (SPARQL/ontology publish) via testcontainers and asserts the
three integration seams the enterprise deployment depends on:

  1. durable graph writes land in pggraph and survive a fresh connection;
  2. the task queue resolves to Kafka and a put → consume → ack round-trips;
  3. the ontology publishes to Fuseki and is queryable over SPARQL.

This is the cluster-side counterpart to the Pi-3 ``test_profile_tiny_zero_dep``.
It is ``@pytest.mark.live`` (needs Docker) and runs under ``pytest -m live`` /
the nightly job — never in the default PR suite.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

import pytest

from agent_utilities.knowledge_graph.backends import create_backend, set_active_backend
from agent_utilities.knowledge_graph.core.engine import IntelligenceGraphEngine

pytestmark = [pytest.mark.integration, pytest.mark.live]


@pytest.fixture()
def enterprise_env(
    monkeypatch,
    ephemeral_pg_age: dict[str, Any],
    ephemeral_kafka: dict[str, Any],
    ephemeral_fuseki: dict[str, Any],
) -> dict[str, Any]:
    """Pin the process to the full enterprise profile against the live containers."""
    pg_uri = ephemeral_pg_age["uri"]
    monkeypatch.setenv("GRAPH_BACKEND", "postgresql")
    monkeypatch.setenv("GRAPH_DB_URI", pg_uri)
    monkeypatch.setenv("STATE_DB_URI", pg_uri)
    monkeypatch.setenv("TASK_QUEUE_BACKEND", "kafka")
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", ephemeral_kafka["bootstrap_servers"])
    monkeypatch.setenv("AGENT_DISPATCH_BACKEND", "queue")
    monkeypatch.setenv("KG_FUSEKI_PUBLISH", "true")
    monkeypatch.setenv("GRAPH_FUSEKI_URL", ephemeral_fuseki["url"])
    monkeypatch.setenv("GRAPH_FUSEKI_DATASET", ephemeral_fuseki["dataset"])
    set_active_backend(None)
    IntelligenceGraphEngine.set_active(None)
    return {
        "pg_uri": pg_uri,
        "kafka": ephemeral_kafka["bootstrap_servers"],
        "fuseki_url": ephemeral_fuseki["url"],
        "fuseki_dataset": ephemeral_fuseki["dataset"],
    }


def _fresh_config() -> Any:
    """A config snapshot that reflects the per-test monkeypatched environment."""
    from agent_utilities.core.config import AgentConfig

    return AgentConfig()


def test_graph_writes_are_durable_in_pggraph(enterprise_env: dict[str, Any]) -> None:
    """A node written through the engine persists to pggraph across a reconnect."""
    if not os.environ.get("GRAPH_SERVICE_SOCKET"):
        pytest.skip("epistemic-graph engine required (GRAPH_SERVICE_SOCKET unset)")

    backend = create_backend(backend_type="postgresql", uri=enterprise_env["pg_uri"])
    assert backend is not None, "psycopg/pgvector not installed?"
    node_id = f"ent:{uuid.uuid4().hex[:8]}"
    try:
        set_active_backend(backend)
        engine = IntelligenceGraphEngine(backend=backend)
        IntelligenceGraphEngine.set_active(engine)
        engine.add_node(node_id, "Agent", {"name": "enterprise-durable"})
    finally:
        backend.close()

    reopened = create_backend(backend_type="postgresql", uri=enterprise_env["pg_uri"])
    assert reopened is not None
    try:
        set_active_backend(reopened)
        eng2 = IntelligenceGraphEngine(backend=reopened)
        rows = eng2.query_cypher(
            "MATCH (n:Agent) WHERE n.id = $id RETURN n.id AS id", {"id": node_id}
        )
        assert rows and rows[0]["id"] == node_id
    finally:
        reopened.close()


def test_task_queue_resolves_to_kafka_and_roundtrips(
    enterprise_env: dict[str, Any], tmp_path
) -> None:
    """``TASK_QUEUE_BACKEND=kafka`` builds a Kafka queue and a task round-trips."""
    from agent_utilities.knowledge_graph.core.queue_backend import (
        create_task_queue,
        resolve_task_queue_backend,
    )

    config = _fresh_config()
    choice, explicit = resolve_task_queue_backend(config)
    assert (choice, explicit) == ("kafka", True)

    queue, backend_name = create_task_queue(config, str(tmp_path / "fallback.db"))
    assert backend_name == "kafka"
    try:
        marker = uuid.uuid4().hex
        queue.put({"task": "parity-probe", "marker": marker})

        # Kafka consume is poll-based; allow a few attempts for the broker/consumer
        # group to settle before the message is delivered.
        received = None
        for _ in range(15):
            got = queue.get()
            if got is not None:
                item_id, payload = got
                if payload.get("marker") == marker:
                    received = (item_id, payload)
                    break
            time.sleep(1.0)
        assert received is not None, "task never consumed from Kafka within timeout"
        queue.ack(received[0])
    finally:
        import contextlib

        close = getattr(queue, "close", None)
        if callable(close):
            with contextlib.suppress(Exception):
                close()


def test_ontology_publishes_to_fuseki_and_is_queryable(
    enterprise_env: dict[str, Any],
) -> None:
    """The bundled ontology publishes to Fuseki and is reachable over SPARQL."""
    from agent_utilities.knowledge_graph.core.ontology_publisher import (
        publish_ontology_to_fuseki,
    )

    report = publish_ontology_to_fuseki(
        endpoint=enterprise_env["fuseki_url"],
        dataset=enterprise_env["fuseki_dataset"],
    )
    status = report.get("status")
    if status == "skipped":
        pytest.skip(f"no ontology triples to publish in this build: {report}")
    assert status not in ("error", None), f"fuseki publish failed: {report}"

    fuseki = create_backend(
        backend_type="jena_fuseki",
        jena_fuseki_url=enterprise_env["fuseki_url"],
        dataset=enterprise_env["fuseki_dataset"],
    )
    assert fuseki is not None
    try:
        rows = fuseki.execute_sparql_query("SELECT (COUNT(*) AS ?c) WHERE { ?s ?p ?o }")
        assert rows, "SPARQL count returned no rows"
        raw = rows[0].get("c")
        count = int(raw.get("value") if isinstance(raw, dict) else raw)
        assert count > 0, "Fuseki holds no triples after publish"
    finally:
        fuseki.close()
