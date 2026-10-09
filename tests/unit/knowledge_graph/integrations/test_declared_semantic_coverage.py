"""AU-SEMANTIC-R021.3 typed model and refusal tests."""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.integrations.declared_semantic_coverage import (
    DeclaredSemanticCoverage,
    DeclaredSemanticCoverageError,
)

_SHAPES = """
:DocumentShape a sh:NodeShape ;
    sh:targetClass :Document ;
    sh:property [ sh:path :sourceRecordRef ] ;
    sh:property [ sh:path :tenantReference ] ;
    sh:property [ sh:path :accessPolicyReference ] ;
    sh:property [ sh:path :provenanceReference ] .
"""


def test_parses_target_classes_and_paths() -> None:
    coverage = DeclaredSemanticCoverage.from_shapes_text(_SHAPES)
    assert "Document" in coverage.target_classes
    assert "sourceRecordRef" in coverage.paths


def test_refuses_incomplete_coverage() -> None:
    coverage = DeclaredSemanticCoverage.from_shapes_text(_SHAPES)
    with pytest.raises(DeclaredSemanticCoverageError):
        coverage.require_covers(frozenset({"Document", "Event"}))


def test_accepts_complete_coverage() -> None:
    coverage = DeclaredSemanticCoverage.from_shapes_text(_SHAPES)
    coverage.require_covers(frozenset({"Document"}))
