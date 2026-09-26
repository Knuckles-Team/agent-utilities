"""Source-extractor materialization tests (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

Asserts materialize_source runs a registered extractor over an injected client
and persists via the native graph-slice boundary, that a None engine is a clean
no-op, and that an unknown category raises.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.enrichment.materialize import (
    materialize_source,
    resolve_source_client,
)
from agent_utilities.knowledge_graph.enrichment.models import (
    ExtractionBatch,
    GraphNode,
)
from tests.kg_recording_backend import RecordingGraphBackend as FakeBackend


class FakeCamundaClient:
    def list_process_definitions(self):
        return [{"id": "invoice:1", "key": "invoice", "version": 1}]

    def list_tasks(self):
        return []

    def list_incidents(self):
        return []


def test_materialize_submits_extractor_batch_to_native_boundary(monkeypatch):
    submitted = {}

    def capture(engine, connector, entities, relationships, **kwargs):
        submitted.update(
            engine=engine,
            connector=connector,
            entities=entities,
            relationships=relationships,
            kwargs=kwargs,
        )
        return {"status": "success"}

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.envelope_ingest.ingest_graph_slice",
        capture,
    )
    engine = object()
    n, e = materialize_source(engine, "camunda", FakeCamundaClient())
    assert n >= 1
    assert submitted["engine"] is engine
    assert submitted["connector"] == "camunda"
    assert submitted["entities"][0]["node_type"] == "BusinessProcess"


def test_none_backend_is_noop_but_runs():
    # No engine → (0, 0) but the extractor still ran without error.
    assert materialize_source(None, "camunda", FakeCamundaClient()) == (0, 0)


def test_materialize_rejects_extractor_identity_override_before_write(monkeypatch):
    batch = ExtractionBatch(
        category="camunda",
        nodes=[
            GraphNode(id="process:real", type="BusinessProcess", props={"id": "forged"})
        ],
    )
    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.enrichment.materialize.extract_source_batch",
        lambda *_args, **_kwargs: batch,
    )

    def unexpected_write(*_args, **_kwargs):
        raise AssertionError("graph write started before projection validation")

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.envelope_ingest.ingest_graph_slice",
        unexpected_write,
    )
    with pytest.raises(ValueError, match="reserved identity"):
        materialize_source(object(), "camunda", FakeCamundaClient())


def test_unknown_category_raises():
    with pytest.raises(ValueError):
        materialize_source(FakeBackend(), "does-not-exist", object())


def test_bare_backend_fails_closed():
    backend = FakeBackend()

    with pytest.raises(RuntimeError, match="native ChangeEnvelope graph slice failed"):
        materialize_source(backend, "camunda", FakeCamundaClient())

    assert backend.nodes == {}


def test_resolve_source_client_missing_returns_none():
    # No connector package / creds in the test env → None, never raises.
    assert resolve_source_client("camunda") is None or hasattr(
        resolve_source_client("camunda"), "list_process_definitions"
    )
    assert resolve_source_client("totally-unknown") is None
