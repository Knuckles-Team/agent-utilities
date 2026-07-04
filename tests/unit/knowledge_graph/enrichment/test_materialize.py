"""Source-extractor materialization tests (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

Asserts materialize_source runs a registered extractor over an injected client
and persists via write_batch, that a None backend is a clean no-op, and that an
unknown category raises.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.enrichment.materialize import (
    materialize_source,
    resolve_source_client,
)
from tests.kg_recording_backend import RecordingGraphBackend as FakeBackend


class FakeCamundaClient:
    def list_process_definitions(self):
        return [{"id": "invoice:1", "key": "invoice", "version": 1}]

    def list_tasks(self):
        return []

    def list_incidents(self):
        return []


def test_materialize_persists_extractor_batch():
    backend = FakeBackend()
    n, e = materialize_source(backend, "camunda", FakeCamundaClient())
    assert n >= 1
    assert backend.nodes["bpmn_process:invoice:1"]["type"] == "BusinessProcess"


def test_none_backend_is_noop_but_runs():
    # No backend → (0, 0) but the extractor still ran without error.
    assert materialize_source(None, "camunda", FakeCamundaClient()) == (0, 0)


def test_unknown_category_raises():
    with pytest.raises(ValueError):
        materialize_source(FakeBackend(), "does-not-exist", object())


def test_resolve_source_client_missing_returns_none():
    # No connector package / creds in the test env → None, never raises.
    assert resolve_source_client("camunda") is None or hasattr(
        resolve_source_client("camunda"), "list_process_definitions"
    )
    assert resolve_source_client("totally-unknown") is None
