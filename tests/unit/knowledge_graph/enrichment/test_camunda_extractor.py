"""Camunda BPMN source extractor tests (CONCEPT:KG-2.9).

Uses an injected fake client (no network / no daemon) plus the write_batch
contract to assert node/edge mapping, canonical typing, and persistence.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.extractors.camunda import extract
from agent_utilities.knowledge_graph.enrichment.registry import (
    get_source,
    write_batch,
)


class FakeBackend:
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, node_id, **props):
        self.nodes[node_id] = props

    def add_edge(self, s, t, **props):
        self.edges.append((s, t, props.get("rel_type")))


class FakeCamundaClient:
    """Duck-typed camunda-mcp client returning canned dict records."""

    def list_process_definitions(self):
        return [
            {
                "id": "invoice:1:abc",
                "key": "invoice",
                "name": "Invoice Receipt",
                "version": 1,
            },
            {"id": "onboarding:2:def", "key": "onboarding", "version": 2},
        ]

    def list_tasks(self):
        return [
            {
                "id": "task-1",
                "name": "Approve Invoice",
                "assignee": "alice",
                "processDefinitionId": "invoice:1:abc",
            },
            {"id": "task-2", "name": "Review", "processDefinitionId": "missing"},
        ]

    def list_incidents(self):
        return [
            {
                "id": "inc-1",
                "incidentType": "failedJob",
                "incidentMessage": "boom",
                "processDefinitionId": "invoice:1:abc",
            }
        ]


def test_extract_maps_canonical_nodes():
    batch = extract({"client": FakeCamundaClient()})
    by_id = {n.id: n for n in batch.nodes}

    assert batch.category == "camunda"
    # 2 processes + 2 tasks + 1 incident
    assert len(batch.nodes) == 5

    proc = by_id["bpmn_process:invoice:1:abc"]
    assert proc.type == "BusinessProcess"
    assert proc.props["name"] == "Invoice Receipt"
    assert proc.props["key"] == "invoice"
    assert proc.props["version"] == 1

    task = by_id["bpmn_task:task-1"]
    assert task.type == "BusinessTask"
    assert task.props["assignee"] == "alice"

    inc = by_id["incident:inc-1"]
    assert inc.type == "Incident"
    assert inc.props["short_description"] == "boom"


def test_extract_maps_edges():
    batch = extract({"client": FakeCamundaClient()})
    edges = {(e.source, e.target, e.rel_type) for e in batch.edges}

    assert ("bpmn_task:task-1", "bpmn_process:invoice:1:abc", "PART_OF") in edges
    assert ("bpmn_task:task-2", "bpmn_process:missing", "PART_OF") in edges
    assert ("incident:inc-1", "bpmn_process:invoice:1:abc", "AFFECTS") in edges
    assert len(batch.edges) == 3


def test_extract_no_client_is_tolerant():
    batch = extract({})
    assert batch.category == "camunda"
    assert batch.nodes == []
    assert batch.edges == []


def test_call_tolerates_missing_methods():
    class Sparse:
        def list_process_definitions(self):
            return [{"id": "p1", "key": "p1"}]

    # No list_tasks / list_incidents methods -> tolerated, just the one process.
    batch = extract({"client": Sparse()})
    assert len(batch.nodes) == 1
    assert batch.nodes[0].type == "BusinessProcess"


def test_source_is_registered():
    src = get_source("camunda")
    assert src is not None
    assert src.extract is extract
    assert "Camunda" in src.description


def test_write_batch_persists():
    batch = extract({"client": FakeCamundaClient()})
    backend = FakeBackend()
    n, e = write_batch(backend, batch)

    assert n == 5
    assert e == 3
    assert backend.nodes["bpmn_process:invoice:1:abc"]["type"] == "BusinessProcess"
    assert (
        "bpmn_task:task-1",
        "bpmn_process:invoice:1:abc",
        "PART_OF",
    ) in backend.edges
