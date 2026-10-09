"""AU-SEMANTIC-R022.1: typed EG-generated DTO composition and its refusal behavior."""

from __future__ import annotations

import pytest

from agent_utilities.api.graph_schema_types import (
    GraphSchemaTypes,
    GraphSchemaTypesUnavailable,
)


class _FullEGClient:
    def build_knowledge_graph(self, *args: object, **kwargs: object) -> str:
        return "knowledge_graph"

    def build_schema_definition(self, *args: object, **kwargs: object) -> str:
        return "schema_definition"

    def build_evidence_bundle(self, *args: object, **kwargs: object) -> str:
        return "evidence_bundle"


class _PartialEGClient:
    def build_knowledge_graph(self, *args: object, **kwargs: object) -> str:
        return "knowledge_graph"


def test_for_client_refuses_none() -> None:
    with pytest.raises(GraphSchemaTypesUnavailable):
        GraphSchemaTypes.for_client(None)


def test_for_client_refuses_client_missing_required_methods() -> None:
    with pytest.raises(GraphSchemaTypesUnavailable) as excinfo:
        GraphSchemaTypes.for_client(_PartialEGClient())
    assert "build_schema_definition" in str(excinfo.value)
    assert "build_evidence_bundle" in str(excinfo.value)


def test_for_client_accepts_full_surface_and_delegates() -> None:
    composed = GraphSchemaTypes.for_client(_FullEGClient())
    assert composed.build_knowledge_graph() == "knowledge_graph"
    assert composed.build_schema_definition() == "schema_definition"
    assert composed.build_evidence_bundle() == "evidence_bundle"
