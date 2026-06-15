"""External-graph profile / imprint / ontology-map (CONCEPT:KG-2.63 extension).

Verifies the seam an agent uses to discover + natively use a registered third-party
graph: introspect its schema, map its labels onto our ontology, and imprint a
self-describing catalog node into our KG — all offline, no live DB.
"""

from __future__ import annotations

import re

import pytest

from agent_utilities.knowledge_graph.core.connection_profiler import (
    map_labels_to_ontology,
    profile_and_imprint,
    profile_connection,
)
from agent_utilities.knowledge_graph.core.connection_registry import ConnectionRegistry

pytestmark = pytest.mark.concept("KG-2.63")


class FakeExternalEngine:
    """A fake Neo4j-ish engine answering the introspection queries."""

    labels = ["Person", "Movie", "Company"]
    rels = ["ACTED_IN", "WORKS_AT"]
    pkeys = ["name", "title", "born"]
    counts = {"Person": 133, "Movie": 38, "Company": 5}
    sample = {"Person": ["name", "born"], "Movie": ["title"], "Company": ["name"]}

    def query_cypher(self, cypher: str):
        if "db.labels()" in cypher:
            return [{"label": x} for x in self.labels]
        if "db.relationshipTypes()" in cypher:
            return [{"relationshipType": x} for x in self.rels]
        if "db.propertyKeys()" in cypher:
            return [{"propertyKey": x} for x in self.pkeys]
        m = re.search(r"MATCH \(n:`([^`]+)`\) RETURN count", cypher)
        if m:
            return [{"c": self.counts.get(m.group(1))}]
        m = re.search(r"MATCH \(n:`([^`]+)`\) RETURN keys", cypher)
        if m:
            return [{"k": self.sample.get(m.group(1), [])}]
        if "MATCH (n) RETURN count(n) AS c" in cypher:
            return [{"c": sum(self.counts.values())}]
        return []


class FakeAuthority:
    """Records imprint writes + supplies our KG's node types."""

    def __init__(self):
        self.nodes: dict[str, tuple] = {}

    def query_cypher(self, cypher: str):
        if "DISTINCT n.type" in cypher:
            return [{"t": "Person"}, {"t": "Document"}, {"t": "Concept"}]
        return []

    def add_node(self, node_id, node_type, properties=None, **_):
        self.nodes[node_id] = (node_type, properties or {})


def test_profile_connection_reads_schema():
    p = profile_connection(FakeExternalEngine(), name="prod-neo4j")
    assert p["labels"] == ["Company", "Movie", "Person"]  # sorted
    assert p["label_count"] == 3
    assert set(p["relationship_types"]) == {"ACTED_IN", "WORKS_AT"}
    assert "name" in p["property_keys"]
    assert p["per_label"]["Person"]["count"] == 133
    assert p["per_label"]["Movie"]["sample_property_keys"] == ["title"]
    assert p["total_nodes"] == 176
    assert isinstance(p["profiled_at"], float)


def test_map_labels_exact_plural_fuzzy_novel():
    our = ["Person", "Document", "Concept", "Organization"]
    out = {
        m["external_label"]: m
        for m in map_labels_to_ontology(
            ["Person", "Documents", "Concpet", "Movie"], our
        )
    }
    assert out["Person"]["method"] == "exact" and out["Person"]["mapped_to"] == "Person"
    assert (
        out["Documents"]["method"] == "plural"
        and out["Documents"]["mapped_to"] == "Document"
    )
    assert (
        out["Concpet"]["method"] == "fuzzy" and out["Concpet"]["mapped_to"] == "Concept"
    )
    assert out["Movie"]["method"] == "novel" and out["Movie"]["mapped_to"] is None


def test_profile_and_imprint_writes_catalog_node():
    auth = FakeAuthority()
    res = profile_and_imprint(
        FakeExternalEngine(),
        name="prod-neo4j",
        spec_summary={"backend": "neo4j", "endpoint": "bolt://host:7687"},
        authority_engine=auth,
        interface_names=[],  # vocab = our KG node types only (Person/Document/Concept)
    )
    assert res["status"] == "success"
    assert res["imprint_node"] == "extgraph:prod-neo4j"
    assert res["label_count"] == 3
    # Person matches our node types; Movie/Company are novel.
    assert res["mapped"] == 1 and res["novel"] == 2

    # the catalog node persisted, carrying schema + mappings, no credentials.
    node_type, props = auth.nodes["extgraph:prod-neo4j"]
    assert node_type == "ExternalGraphReference"
    assert props["backend"] == "neo4j"
    assert props["schema"]["label_count"] == 3
    assert any(m["external_label"] == "Person" for m in props["ontology_mappings"])
    assert "password" not in str(props)


def test_spec_summary_redacts_credentials():
    reg = ConnectionRegistry()
    reg.register(
        "secure", {"backend": "neo4j", "uri": "bolt://neo4j:s3cret@db.internal:7687"}
    )
    s = reg.spec_summary("secure")
    assert s["backend"] == "neo4j"
    assert s["endpoint"] == "bolt://***@db.internal:7687"
    assert "s3cret" not in str(s)
