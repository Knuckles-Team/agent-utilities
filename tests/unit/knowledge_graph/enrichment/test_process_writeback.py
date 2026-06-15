"""Outbound process-intelligence writeback tests (CONCEPT:KG-2.8).

Uses a fake graph reader + fake Camunda/ARIS clients to assert all four payload
sections are gathered, the write is hash-idempotent, and one failing client does
not abort the batch.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.process_writeback import (
    gather_intelligence,
    push_process_intelligence,
)


class FakeReader:
    """Minimal graph reader covering all four intelligence sections."""

    def __init__(self):
        self.props = {
            "bpmn_process:invoice": {"type": "BusinessProcess", "key": "invoice"},
            "wf:1": {"type": "Workflow", "name": "Invoice Workflow"},
            "agent:billing": {"type": "Agent", "name": "Billing Agent"},
            "aris_model:M1": {"type": "BusinessProcess"},
            "policy:gdpr": {"type": "Policy", "name": "GDPR"},
            "incident:9": {"type": "Incident", "short_description": "Gateway timeout"},
            "bpmn_task:t1": {"type": "BusinessTask", "name": "Approve"},
            "do:inv": {"type": "DataObject", "name": "Invoice Record"},
            "concept:vat": {"type": "Concept", "name": "VAT"},
        }
        self._out = {
            "bpmn_process:invoice": [
                ("aris_model:M1", {"rel_type": "ALIGNED_WITH"}),
                ("policy:gdpr", {"rel_type": "governedBy"}),
            ],
            "wf:1": [("agent:billing", {"rel_type": "ORCHESTRATES"})],
            "bpmn_task:t1": [
                ("do:inv", {"rel_type": "FLOWS_TO"}),
                ("concept:vat", {"rel_type": "MENTIONS"}),
            ],
        }
        self._in = {
            "bpmn_process:invoice": [
                ("wf:1", {"rel_type": "REALIZES"}),
                ("incident:9", {"rel_type": "AFFECTS"}),
                ("bpmn_task:t1", {"rel_type": "PART_OF"}),
            ],
        }

    def node_props(self, nid):
        return self.props.get(nid, {})

    def out_edges(self, nid):
        return self._out.get(nid, [])

    def in_edges(self, nid):
        return self._in.get(nid, [])

    def process_nodes(self):
        return [("bpmn_process:invoice", self.props["bpmn_process:invoice"])]


class FakeCamunda:
    def __init__(self):
        self.modifications = []
        self.vars = {}

    def list_process_instances(self, params):
        return [{"id": "PI-1"}]

    def get_process_instance_variables(self, iid):
        return self.vars.get(iid, {})

    def modify_process_instance_variables(self, iid, body):
        self.modifications.append((iid, body))
        value = body["modifications"]["kg_intelligence"]["value"]
        self.vars[iid] = {"kg_intelligence": {"value": value}}


def test_gather_all_four_sections():
    payload = gather_intelligence(FakeReader(), "bpmn_process:invoice")
    assert payload["capabilities"] == ["Invoice Workflow (Billing Agent)"]
    assert payload["aligned_with"] == ["aris_model:M1"]
    assert payload["governance"] == ["GDPR"]
    assert payload["incidents"] == ["Gateway timeout"]
    assert payload["glossary_terms"] == ["VAT"]
    assert payload["data_objects"] == ["Invoice Record"]


def test_push_then_idempotent_skip():
    reader, client = FakeReader(), FakeCamunda()
    first = push_process_intelligence(reader, camunda_client=client)
    assert first.camunda_pushed == 1
    assert len(client.modifications) == 1
    second = push_process_intelligence(reader, camunda_client=client)
    assert second.camunda_pushed == 0
    assert second.skipped_unchanged == 1
    assert len(client.modifications) == 1  # no second write


def test_no_running_instances_reports_no_target():
    class NoInstances(FakeCamunda):
        def list_process_instances(self, params):
            return []

    res = push_process_intelligence(FakeReader(), camunda_client=NoInstances())
    assert res.no_target == 1 and res.camunda_pushed == 0


def test_failing_client_does_not_abort():
    class Boom(FakeCamunda):
        def modify_process_instance_variables(self, iid, body):
            raise RuntimeError("transport down")

    res = push_process_intelligence(FakeReader(), camunda_client=Boom())
    assert res.errors == 1 and res.camunda_pushed == 0


def test_empty_payload_process_skipped():
    class Empty(FakeReader):
        def out_edges(self, nid):
            return []

        def in_edges(self, nid):
            return []

    res = push_process_intelligence(Empty(), camunda_client=FakeCamunda())
    assert res.camunda_pushed == 0 and res.skipped_unchanged == 0


def test_aris_writeback_sets_model_attribute():
    class FakeAris:
        def __init__(self):
            self.attrs = {}

        def set_model_attributes(self, model_id, attributes):
            self.attrs[model_id] = attributes

    class ArisReader(FakeReader):
        def process_nodes(self):
            return [("aris_model:M1", {"type": "BusinessProcess"})]

        def in_edges(self, nid):
            return self._in.get("bpmn_process:invoice", []) if nid == "aris_model:M1" else []

        def out_edges(self, nid):
            return self._out.get("bpmn_process:invoice", []) if nid == "aris_model:M1" else self._out.get(nid, [])

    aris = FakeAris()
    res = push_process_intelligence(ArisReader(), aris_client=aris)
    assert res.aris_pushed == 1
    assert "kg_intelligence" in aris.attrs["M1"]
