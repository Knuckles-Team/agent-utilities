"""ARIS source extractor tests (CONCEPT:KG-2.9, EPC step lift KG-2.53).

Uses an injected fake ARIS client (no network) to assert model-level typing and
the EPC step lift: functions/rule operators → BusinessTask, events collapsed,
connections → FLOWS_TO with conditions, plus ALIGNED_WITH reconciliation.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.extractors.aris import extract
from agent_utilities.knowledge_graph.enrichment.registry import get_source, write_batch
from tests.kg_recording_backend import RecordingGraphBackend as FakeBackend


class FakeArisClient:
    """Duck-typed ARIS client: one process EPC + one architecture model."""

    def list_models(self):
        return [
            {"id": "M1", "name": "Invoice EPC", "type": "EPC", "camundaKey": "invoice"},
            {"id": "A1", "name": "Billing App", "type": "Application"},
        ]

    def list_model_objects(self, model_id):
        if model_id != "M1":
            return []
        return [
            {"id": "f1", "name": "Receive Invoice", "typeName": "OT_FUNC"},
            {"id": "e1", "name": "Invoice Received", "typeName": "OT_EVT"},
            {
                "id": "r1",
                "name": "XOR",
                "typeName": "OT_RULE",
                "symbol": "ST_OPR_XOR_1",
            },
            {"id": "f2", "name": "Approve", "typeName": "OT_FUNC"},
            {"id": "f3", "name": "Reject", "typeName": "OT_FUNC"},
        ]

    def list_model_connections(self, model_id):
        if model_id != "M1":
            return []
        return [
            {"sourceObjectId": "f1", "targetObjectId": "e1"},
            {"sourceObjectId": "e1", "targetObjectId": "r1"},
            {
                "sourceObjectId": "r1",
                "targetObjectId": "f2",
                "condition": "amount<=1000",
            },
            {
                "sourceObjectId": "r1",
                "targetObjectId": "f3",
                "condition": "amount>1000",
            },
        ]


def test_aris_extractor_registered():
    assert get_source("aris") is not None


def test_model_level_typing_and_alignment():
    batch = extract({"client": FakeArisClient()})
    by_id = {n.id: n for n in batch.nodes}
    assert by_id["aris_model:M1"].type == "BusinessProcess"
    assert by_id["aris_model:A1"].type == "ApplicationComponent"
    # camundaKey → externalToolId + ALIGNED_WITH to the Camunda twin
    assert by_id["aris_model:M1"].props["externalToolId"] == "invoice"
    assert (
        "aris_model:M1",
        "bpmn_process:invoice",
        "ALIGNED_WITH",
    ) in [(e.source, e.target, e.rel_type) for e in batch.edges]


def test_epc_step_lift_functions_and_gateway():
    batch = extract({"client": FakeArisClient()})
    by_id = {n.id: n for n in batch.nodes}
    # functions + rule operator lifted to BusinessTask, PART_OF the model
    assert by_id["aris_object:M1:f1"].type == "BusinessTask"
    assert by_id["aris_object:M1:f1"].props["is_gateway"] is False
    assert by_id["aris_object:M1:r1"].props["is_gateway"] is True
    assert by_id["aris_object:M1:r1"].props["gateway_kind"] == "XOR"
    # the event was NOT lifted
    assert "aris_object:M1:e1" not in by_id


def test_epc_flows_collapse_event_and_keep_conditions():
    batch = extract({"client": FakeArisClient()})
    flows = {
        (e.source, e.target): e.props.get("condition")
        for e in batch.edges
        if e.rel_type == "FLOWS_TO"
    }
    # f1 -> r1 collapses through the event e1
    assert ("aris_object:M1:f1", "aris_object:M1:r1") in flows
    # XOR branches keep their conditions
    assert flows[("aris_object:M1:r1", "aris_object:M1:f2")] == "amount<=1000"
    assert flows[("aris_object:M1:r1", "aris_object:M1:f3")] == "amount>1000"


def test_none_client_empty_batch():
    batch = extract({"client": None})
    assert batch.nodes == [] and batch.edges == []


def test_persists_through_write_batch():
    backend = FakeBackend()
    batch = extract({"client": FakeArisClient()})
    n, e = write_batch(backend, batch)
    assert n == len(batch.nodes) and e == len(batch.edges)
    assert backend.nodes["aris_object:M1:r1"]["type"] == "BusinessTask"
