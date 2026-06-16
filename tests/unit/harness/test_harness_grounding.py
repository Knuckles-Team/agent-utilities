"""Connector-grounded harness evolution + ARA-Seal (CONCEPT:KG-2.108)."""

from __future__ import annotations

from agent_utilities.harness.harness_grounding import (
    ground_variant,
    link_dimension_to_service,
    seal_level_for,
    seal_variant,
)
from agent_utilities.harness.superhuman_gate import SuperhumanCertifier


def test_ground_variant_emits_grounded_in_edges():
    edges = ground_variant(
        "harness_variant:v1",
        ["trace:gitlab:run42", "test_result:ci:99", "metric_report:obs:7"],
    )
    assert all(e["type"] == "grounded_in" for e in edges)
    assert {e["target"] for e in edges} == {
        "trace:gitlab:run42",
        "test_result:ci:99",
        "metric_report:obs:7",
    }
    assert all(e["source"] == "harness_variant:v1" for e in edges)


def test_seal_levels_track_certification_strength():
    cert = SuperhumanCertifier()
    strong = cert.certify([0.95] * 25, human_baseline=0.6)  # wide margin → L3
    weak_pass = cert.certify([0.66, 0.67, 0.65, 0.68, 0.66] * 4, human_baseline=0.6)
    uncertified = cert.certify([0.61, 0.59, 0.6, 0.58], human_baseline=0.6)
    assert seal_level_for(strong) == "L3"
    assert seal_level_for(weak_pass) in ("L2", "L3")
    assert seal_level_for(uncertified) == "L1"


def test_seal_variant_nodes_and_edges():
    cert = SuperhumanCertifier().certify([0.95] * 25, human_baseline=0.6)
    nodes, edges, level = seal_variant("harness_variant:v1", cert)
    assert level == "L3"
    seal = next(n for n in nodes if n["type"] == "seal_certificate")
    assert seal["level"] == "L3"
    variant = next(n for n in nodes if n["type"] == "harness_variant")
    assert variant["certificationLevel"] == "L3"
    assert edges == [
        {"source": seal["id"], "target": "harness_variant:v1", "type": "grounded_in"}
    ]


def test_dimension_to_service_link():
    e = link_dimension_to_service("D4_tools", "service:cluster:gitlab-mcp")
    assert e == {
        "source": "D4_tools",
        "target": "service:cluster:gitlab-mcp",
        "type": "grounded_in",
    }


def test_harness_runs_preset_registered():
    from agent_utilities.protocols.source_connectors.connectors.mcp_tool import (
        MCP_TOOL_PRESETS,
    )

    assert "harness-runs" in MCP_TOOL_PRESETS
    assert MCP_TOOL_PRESETS["harness-runs"]["doc_type"] == "harness_run"
