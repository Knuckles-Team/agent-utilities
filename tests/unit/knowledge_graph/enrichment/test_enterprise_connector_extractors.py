"""ARIS / RSA Archer / Odoo CRM source extractors (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

Injected fake clients (no network) assert node/edge mapping, canonical typing,
the vendor-neutral capability tag, tolerance, and registry discovery.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.extractors.archer import (
    extract as archer_extract,
)
from agent_utilities.knowledge_graph.enrichment.extractors.aris import (
    extract as aris_extract,
)
from agent_utilities.knowledge_graph.enrichment.extractors.odoo import (
    extract as odoo_extract,
)
from agent_utilities.knowledge_graph.enrichment.registry import get_source, write_batch
from tests.kg_recording_backend import RecordingGraphBackend as FakeBackend


# ── ARIS ─────────────────────────────────────────────────────────────────────
class FakeAris:
    def list_models(self):
        return [
            {"id": "m1", "name": "Order to Cash", "type": "EPC Process"},
            {"id": "m2", "name": "App Landscape", "type": "Application Architecture"},
        ]


def test_aris_maps_process_and_architecture():
    batch = aris_extract({"client": FakeAris()})
    by_id = {n.id: n for n in batch.nodes}
    assert batch.category == "aris"
    assert by_id["aris_model:m1"].type == "BusinessProcess"
    assert by_id["aris_model:m1"].props["capability"] == "bpm"
    assert by_id["aris_model:m2"].type == "ApplicationComponent"
    assert by_id["aris_model:m2"].props["capability"] == "enterprise-architecture"


def test_aris_no_client_tolerant():
    batch = aris_extract({})
    assert batch.category == "aris" and batch.nodes == []


# ── RSA Archer ───────────────────────────────────────────────────────────────
class FakeArcher:
    def list_risks(self):
        return [{"id": "R1", "name": "Data breach"}]

    def list_controls(self):
        return [{"id": "C1", "name": "MFA", "riskId": "R1"}]

    def list_findings(self):
        return [{"id": "F1", "name": "MFA gap", "controlId": "C1"}]


def test_archer_maps_grc_graph():
    batch = archer_extract({"client": FakeArcher()})
    by_id = {n.id: n for n in batch.nodes}
    assert by_id["archer_risk:R1"].type == "Risk"
    assert by_id["archer_control:C1"].type == "ComplianceControl"
    assert by_id["archer_finding:F1"].type == "Finding"
    assert all(n.props["capability"] == "grc" for n in batch.nodes)
    edges = {(e.source, e.target, e.rel_type) for e in batch.edges}
    assert ("archer_control:C1", "archer_risk:R1", "MITIGATES") in edges
    assert ("archer_finding:F1", "archer_control:C1", "AFFECTS") in edges


# ── Odoo CRM ─────────────────────────────────────────────────────────────────
class FakeOdoo:
    def list_partners(self):
        return [{"id": 7, "display_name": "Acme Corp"}]

    def list_leads(self):
        return [{"id": 11, "name": "Acme renewal", "partner_id": [7, "Acme Corp"]}]


def test_odoo_maps_crm_graph():
    batch = odoo_extract({"client": FakeOdoo()})
    by_id = {n.id: n for n in batch.nodes}
    assert by_id["odoo_customer:7"].type == "Customer"
    assert by_id["odoo_lead:11"].type == "Lead"
    assert by_id["odoo_customer:7"].props["capability"] == "crm"
    edges = {(e.source, e.target, e.rel_type) for e in batch.edges}
    # many2one partner_id [id,label] resolves to the customer id.
    assert ("odoo_lead:11", "odoo_customer:7", "BELONGS_TO") in edges


def test_odoo_write_batch_persists():
    batch = odoo_extract({"client": FakeOdoo()})
    backend = FakeBackend()
    n, e = write_batch(backend, batch)
    assert n == 2 and e == 1
    assert backend.nodes["odoo_customer:7"]["type"] == "Customer"


# ── registry discovery (Wire-First) ──────────────────────────────────────────
def test_sources_are_discoverable():
    from agent_utilities.knowledge_graph.enrichment.registry import discover_extractors

    discover_extractors()
    for cat in ("aris", "archer", "odoo"):
        src = get_source(cat)
        assert src is not None, f"{cat} not registered"
