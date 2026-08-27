"""Index naming + document shape (`DEC-CA-09` Contract, CA-24-W02)."""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.knowledge_graph.search import doc_shape


def test_index_name_matches_dec_ca_09_pattern() -> None:
    assert doc_shape.index_name("acme", "Person") == "kg-acme-person"


def test_index_name_sanitizes_unsafe_characters() -> None:
    assert (
        doc_shape.index_name("Acme Corp!", "Business Process")
        == "kg-acme_corp-business_process"
    )


def test_index_name_rejects_empty_segments() -> None:
    with pytest.raises(ValueError):
        doc_shape.index_name("", "Person")
    with pytest.raises(ValueError):
        doc_shape.index_name("acme", "")


def test_alias_name_matches_index_name_today() -> None:
    assert doc_shape.alias_name("acme", "Person") == doc_shape.index_name(
        "acme", "Person"
    )


def test_tenant_wildcard() -> None:
    assert doc_shape.tenant_wildcard("acme") == "kg-acme-*"


def test_build_document_shape_matches_contract() -> None:
    doc = doc_shape.build_document(
        node_id="n1",
        node_type="Person",
        tenant="acme",
        marking=["restricted", "restricted", " "],
        properties={"name": "Ada"},
        updated_lsn=7,
        content="Ada",
    )
    assert set(doc.keys()) == {
        "node_id",
        "node_type",
        "tenant",
        "marking",
        "content",
        "properties",
        "updated_lsn",
    }
    assert doc["marking"] == ["restricted"]  # deduped, blanks stripped
    assert doc["updated_lsn"] == 7
    assert doc["properties"] == {"name": "Ada"}


def test_build_document_defaults_empty_marking_and_content() -> None:
    doc = doc_shape.build_document(
        node_id="n1",
        node_type="Person",
        tenant="acme",
        marking=None,
        properties={},
        updated_lsn=0,
    )
    assert doc["marking"] == []
    assert doc["content"] == ""


@pytest.mark.parametrize("field", ["node_id", "node_type", "tenant"])
def test_build_document_requires_identity_fields(field: str) -> None:
    kwargs: dict[str, Any] = {
        "node_id": "n1",
        "node_type": "Person",
        "tenant": "acme",
        "marking": [],
        "properties": {},
        "updated_lsn": 0,
    }
    kwargs[field] = ""
    with pytest.raises(ValueError):
        doc_shape.build_document(**kwargs)
