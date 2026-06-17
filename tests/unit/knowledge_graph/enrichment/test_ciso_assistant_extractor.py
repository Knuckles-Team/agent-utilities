"""CISO Assistant GRC source extractor tests (CONCEPT:KG-2.110).

Uses an injected fake client (no network / no daemon) plus the write_batch
contract to assert node/edge mapping, canonical typing, the Egeria/Camunda
ALIGNED_WITH crosswalk, and persistence. The fake client mirrors the generated
``ciso_assistant_api.Api`` list-method surface (``api_*_list``) returning a
``Response``-like object whose ``.data`` is the result list.
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.enrichment.extractors.ciso_assistant import (
    extract,
)
from agent_utilities.knowledge_graph.enrichment.registry import get_source, write_batch
from tests.kg_recording_backend import RecordingGraphBackend as FakeBackend


class _Resp:
    """Stand-in for the generated ``Response`` wrapper (exposes ``.data``)."""

    def __init__(self, data):
        self.data = data


class FakeCisoClient:
    """Duck-typed ciso_assistant_api.Api returning canned GRC records."""

    def api_policies_list(self):
        return _Resp(
            [
                {
                    "id": "p1",
                    "name": "Acceptable Use Policy",
                    "ref_id": "AUP-1",
                    "urn": "urn:intuitem:risk:policy:aup",
                    # explicit Egeria twin → ALIGNED_WITH crosswalk
                    "egeria_guid": "egeria-guid-9",
                }
            ]
        )

    def api_applied_controls_list(self):
        return _Resp([{"id": "c1", "name": "MFA Everywhere", "ref_id": "AC-1"}])

    def api_risk_scenarios_list(self):
        return _Resp(
            [
                {
                    "id": "r1",
                    "name": "Phishing leads to breach",
                    "applied_controls": ["c1", {"id": "c2"}],
                    "risk_assessment": {"id": "ra1"},
                    # explicit Camunda twin → ALIGNED_WITH crosswalk
                    "bpmn_process_id": "incident_response:1",
                }
            ]
        )

    def api_compliance_assessments_list(self):
        return _Resp([{"id": "ca1", "name": "ISO 27001 Audit", "framework": "fw1"}])

    def api_frameworks_list(self):
        return _Resp(
            [
                {
                    "id": "fw1",
                    "name": "ISO 27001",
                    "urn": "urn:intuitem:risk:framework:iso27001",
                }
            ]
        )

    # endpoints the extractor probes but this fake doesn't serve → empty
    def api_reference_controls_list(self):
        return _Resp([])


def test_extract_maps_canonical_grc_nodes():
    batch = extract({"client": FakeCisoClient()})
    by_id = {n.id: n for n in batch.nodes}

    assert batch.category == "ciso_assistant"
    policy = by_id["ciso_assistant_policy:p1"]
    assert policy.type == "Policy"
    assert policy.props["name"] == "Acceptable Use Policy"
    assert policy.props["domain"] == "ciso_assistant"
    assert policy.props["externalToolId"] == "p1"
    assert policy.props["qualifiedName"] == "urn:intuitem:risk:policy:aup"

    assert by_id["ciso_assistant_control:c1"].type == "Control"
    assert by_id["ciso_assistant_risk:r1"].type == "Risk"
    assert (
        by_id["ciso_assistant_compliance_assessment:ca1"].type == "ComplianceAssessment"
    )
    assert by_id["ciso_assistant_framework:fw1"].type == "Framework"


def test_extract_emits_internal_and_crosswalk_edges():
    batch = extract({"client": FakeCisoClient()})
    edges = {(e.source, e.rel_type, e.target) for e in batch.edges}

    # internal GRC structure
    assert (
        "ciso_assistant_risk:r1",
        "MITIGATED_BY",
        "ciso_assistant_control:c1",
    ) in edges
    assert (
        "ciso_assistant_risk:r1",
        "MITIGATED_BY",
        "ciso_assistant_control:c2",
    ) in edges
    assert (
        "ciso_assistant_risk:r1",
        "PART_OF",
        "ciso_assistant_risk_assessment:ra1",
    ) in edges
    assert (
        "ciso_assistant_compliance_assessment:ca1",
        "CONFORMS_TO",
        "ciso_assistant_framework:fw1",
    ) in edges

    # bidirectional crosswalk: CISO ↔ Egeria and CISO ↔ Camunda twins
    assert (
        "ciso_assistant_policy:p1",
        "ALIGNED_WITH",
        "egeria_policy:egeria-guid-9",
    ) in edges
    assert (
        "ciso_assistant_risk:r1",
        "ALIGNED_WITH",
        "bpmn_process:incident_response:1",
    ) in edges


def test_extractor_is_registered_and_import_safe():
    # registered under its category
    assert get_source("ciso_assistant") is not None
    # no client → empty batch (import-safe, no network)
    empty = extract({"client": None})
    assert empty.nodes == [] and empty.edges == []


def test_write_batch_persists_canonical_nodes():
    backend = FakeBackend()
    batch = extract({"client": FakeCisoClient()})
    n, e = write_batch(backend, batch, source="ciso_assistant")
    assert n == len(batch.nodes) > 0
    assert e == len(batch.edges) > 0
