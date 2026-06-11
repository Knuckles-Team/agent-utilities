"""Camunda BPMN source extractor tests (CONCEPT:KG-2.9, step lift KG-2.53).

Uses an injected fake client (no network / no daemon) plus the write_batch
contract to assert node/edge mapping, canonical typing, and persistence.
The step-level lift tests feed a real BPMN 2.0 XML fixture through the
optional ``get_process_definition_xml`` capability.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.extractors.camunda import extract
from agent_utilities.knowledge_graph.enrichment.registry import (
    get_source,
    write_batch,
)

from .bpmn_fixtures import XmlCapableClient


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


# ── Step-level structure lift (CONCEPT:KG-2.53) ─────────────────────────────


def _lifted(batch):
    tasks = {
        n.props["element_id"]: n for n in batch.nodes if n.type == "BusinessTask"
    }
    flows = {
        (e.source, e.target): e for e in batch.edges if e.rel_type == "FLOWS_TO"
    }
    return tasks, flows


def test_bpmn_xml_lifts_tasks_and_gateways_as_business_tasks():
    batch = extract({"client": XmlCapableClient()})
    tasks, _ = _lifted(batch)

    # 3 tasks + 1 gateway lifted; start/end/intermediate events are NOT nodes.
    assert set(tasks) == {"review", "decide", "archive", "rework"}
    assert tasks["review"].props["task_type"] == "userTask"
    assert tasks["review"].props["name"] == "Review Invoice"
    assert tasks["archive"].props["task_type"] == "serviceTask"
    # Gateways are typed via the property, not a separate node type.
    assert tasks["decide"].type == "BusinessTask"
    assert tasks["decide"].props["task_type"] == "exclusiveGateway"
    assert tasks["decide"].props["is_gateway"] is True
    assert tasks["review"].props["is_gateway"] is False
    # Every lifted element is PART_OF its process.
    part_of = {
        (e.source, e.target) for e in batch.edges if e.rel_type == "PART_OF"
    }
    for el in ("review", "decide", "archive", "rework"):
        assert (
            f"bpmn_task:invoice:1:abc:{el}",
            "bpmn_process:invoice:1:abc",
        ) in part_of


def test_sequence_flows_become_flows_to_with_conditions():
    batch = extract({"client": XmlCapableClient()})
    _, flows = _lifted(batch)

    def fid(el):
        return f"bpmn_task:invoice:1:abc:{el}"

    # Gateway branching preserved as multiple conditional FLOWS_TO edges.
    assert flows[(fid("decide"), fid("archive"))].props["condition"] == (
        "${approved == true}"
    )
    assert flows[(fid("decide"), fid("rework"))].props["condition"] == (
        "${approved == false}"
    )
    # Plain flow has no condition prop.
    assert flows[(fid("review"), fid("decide"))].props == {}
    # The rework → notify1(event) → archive hop collapses through the
    # pass-through event to a direct FLOWS_TO.
    assert (fid("rework"), fid("archive")) in flows
    # start1/end1 never appear as endpoints.
    assert all("start1" not in pair and "end1" not in pair for pair in flows)


def test_flows_to_condition_persists_through_write_batch():
    batch = extract({"client": XmlCapableClient()})
    backend = FakeBackend()
    write_batch(backend, batch)
    assert (
        "bpmn_task:invoice:1:abc:decide",
        "bpmn_task:invoice:1:abc:archive",
        "FLOWS_TO",
    ) in backend.edges


def test_client_without_xml_capability_degrades_to_metadata_only():
    batch = extract({"client": FakeCamundaClient()})
    assert all(":invoice:1:abc:" not in n.id for n in batch.nodes)
    assert all(e.rel_type != "FLOWS_TO" for e in batch.edges)


def test_xml_fetch_failure_is_tolerated():
    class BrokenXmlClient(XmlCapableClient):
        def get_process_definition_xml(self, id=None, key=None):
            raise RuntimeError("engine down")

    batch = extract({"client": BrokenXmlClient()})
    assert len(batch.nodes) == 1  # just the process definition
    assert batch.nodes[0].type == "BusinessProcess"


def test_unparseable_xml_is_tolerated():
    class GarbageXmlClient(XmlCapableClient):
        def get_process_definition_xml(self, id=None, key=None):
            return "<not-bpmn"

    batch = extract({"client": GarbageXmlClient()})
    assert len(batch.nodes) == 1


def test_egeria_guid_recorded_as_external_id_and_aligned_with_edge():
    class ReconciledClient:
        def list_process_definitions(self):
            return [
                {
                    "id": "invoice:1:abc",
                    "key": "invoice",
                    "egeriaGuid": "guid-123",
                }
            ]

    batch = extract({"client": ReconciledClient()})
    (proc,) = [n for n in batch.nodes if n.type == "BusinessProcess"]
    assert proc.props["externalToolId"] == "guid-123"
    aligned = [
        (e.source, e.target) for e in batch.edges if e.rel_type == "ALIGNED_WITH"
    ]
    assert aligned == [("bpmn_process:invoice:1:abc", "egeria_process:guid-123")]
