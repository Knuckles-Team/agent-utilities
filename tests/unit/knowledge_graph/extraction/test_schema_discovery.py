"""Schema discovery against the SDK manifest ontology-pack compiler."""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.extraction.schema_discovery import (
    SchemaDiscoveryError,
    discover_schema,
)


@pytest.mark.spec("AU-BOUNDARY-R030.8")
def test_discover_schema_happy_path() -> None:
    schema = discover_schema(
        {
            "connector": "widget",
            "resources": [{"name": "Widget", "id_prefix": "widget"}],
            "schema_mappings": {"Widget": {"fields": {"name": "xsd:string"}}},
            "provenance": {"integrity": {"hash": "0" * 64}},
        }
    )
    assert [c.local for c in schema.classes] == ["Widget"]
    assert schema.classes[0].id_prefix == "widget"
    assert [d.local for d in schema.datatype_properties] == ["name"]
    assert schema.model_config["frozen"] is True


@pytest.mark.spec("AU-BOUNDARY-R030.8")
def test_discover_schema_fails_closed() -> None:
    with pytest.raises(SchemaDiscoveryError):
        discover_schema({"resources": "not-a-list"})
    with pytest.raises(SchemaDiscoveryError):
        discover_schema(42)
