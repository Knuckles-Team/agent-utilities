"""Unit tests for the unified ingestion representation (CONCEPT:AU-KG.ingest.enterprise-source-extractor).

The UNWIND path now groups by REAL node/edge type (per-type MERGE) instead of
flattening to :DomainEntity/:EXTERNAL_LINK. Tests the label-safety + grouping
helpers in isolation (no heavy engine init).
"""

from __future__ import annotations

from agent_utilities.knowledge_graph.core.materialization import (
    group_by_label,
    group_by_rel,
    safe_label,
)


def test_safe_label_keeps_valid_identifiers_and_falls_back():
    assert safe_label("Application") == "Application"
    assert safe_label("BusinessProcess") == "BusinessProcess"
    assert safe_label("has space") == "DomainEntity"
    assert safe_label("weird-dash") == "DomainEntity"
    assert safe_label(None) == "DomainEntity"
    assert safe_label(None, fallback="EXTERNAL_LINK") == "EXTERNAL_LINK"


def test_group_by_label_buckets_by_real_type():
    # group_by_label reads the canonical `node_type` key (GraphComputeEngine
    # retired the bare `type` property; see graph_compute.add_node).
    groups = group_by_label(
        [
            {"id": "a:1", "node_type": "Application"},
            {"id": "a:2", "node_type": "Application"},
            {"id": "c:1", "node_type": "Capability"},
            {"id": "x:1"},  # no node_type → DomainEntity fallback
        ]
    )
    assert set(groups) == {"Application", "Capability", "DomainEntity"}
    assert len(groups["Application"]) == 2
    assert groups["DomainEntity"][0]["id"] == "x:1"


def test_group_by_rel_buckets_by_real_rel_type():
    # group_by_rel reads the canonical `relationship` key (GraphComputeEngine
    # retired the `type`/`rel_type` edge aliases; see graph_compute.add_edge).
    groups = group_by_rel(
        [
            {"source": "a", "target": "b", "relationship": "SUPPORTS"},
            {"source": "b", "target": "c", "relationship": "SUPPORTS"},
            {"source": "a", "target": "c"},  # no relationship → EXTERNAL_LINK fallback
        ]
    )
    assert set(groups) == {"SUPPORTS", "EXTERNAL_LINK"}
    assert len(groups["SUPPORTS"]) == 2
