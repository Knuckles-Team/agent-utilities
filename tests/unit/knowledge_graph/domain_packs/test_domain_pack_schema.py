"""Schema-level contract for :class:`DomainPackManifest` — versioning + strictness."""

from __future__ import annotations

import _fixtures
import pytest
from pydantic import ValidationError

from agent_utilities.knowledge_graph.domain_packs.domain_pack import (
    ColumnMapping,
    DomainPackManifest,
    FrontmatterMapping,
    TableMapping,
)


def test_pack_requires_real_semver():
    with pytest.raises(ValidationError):
        _fixtures.build_manifest(version="not-a-version")


def test_pack_rejects_unknown_top_level_field():
    manifest = _fixtures.build_manifest()
    data = manifest.model_dump(mode="json")
    data["unexpected_field"] = "sneaky"
    with pytest.raises(ValidationError):
        DomainPackManifest.model_validate(data)


def test_column_mapping_requires_property_or_relation_matching_produce():
    with pytest.raises(ValidationError):
        ColumnMapping(produce="property")  # missing 'property'
    with pytest.raises(ValidationError):
        ColumnMapping(produce="edge")  # missing 'relation'
    # valid forms do not raise
    ColumnMapping(produce="property", property="name")
    ColumnMapping(produce="edge", relation="assignedTo")


def test_mapping_rules_are_a_discriminated_union_on_kind():
    manifest = _fixtures.build_manifest(
        mappings=[
            FrontmatterMapping(
                key="x", node_type="Document", produce="property", property="x"
            ),
            TableMapping(
                row_node_type="Runbook", row_id_template="{artifact_id}", columns={}
            ),
        ],
        evaluation_cases=[],
    )
    kinds = {rule.kind for rule in manifest.mappings}
    assert kinds == {"frontmatter", "table"}


def test_mapping_version_and_connector_id_trace_a_fact_to_its_pack_revision():
    manifest = _fixtures.build_manifest(pack_name="widgets", version="2.3.1")
    assert manifest.mapping_version == "widgets@2.3.1"
    assert manifest.connector_id == "domain-pack:widgets"


def test_default_classification_must_be_a_real_data_classification_value():
    with pytest.raises(ValidationError, match="default_classification"):
        _fixtures.build_manifest(default_classification="top-secret-typo")
    # every real value is accepted
    for value in ("public", "internal", "confidential", "restricted"):
        _fixtures.build_manifest(default_classification=value)
