"""External-graph profile / imprint / ontology-map (CONCEPT:AU-KG.backend.multi-connection-registry extension).

Verifies the seam an agent uses to discover + natively use a registered third-party
graph: introspect its schema, map its labels onto our ontology, and imprint a
self-describing catalog node into our KG — all offline, no live DB.
"""

from __future__ import annotations

import re

import pytest

from agent_utilities.knowledge_graph.core.connection_profiler import (
    map_labels_to_ontology,
    profile_connection,
)
from agent_utilities.knowledge_graph.core.connection_registry import ConnectionRegistry

pytestmark = pytest.mark.concept("AU-KG.backend.multi-connection-registry")


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

def test_spec_summary_never_returns_endpoint_or_database_material():
    reg = ConnectionRegistry()
    reg.register(
        "secure", {"backend": "neo4j", "uri": "bolt://neo4j:s3cret@db.internal:7687"}
    )
    s = reg.spec_summary("secure")
    assert s["backend"] == "neo4j"
    assert s["endpoint_configured"] is True
    assert "endpoint" not in s
    assert "db_name" not in s
    assert "s3cret" not in str(s)
