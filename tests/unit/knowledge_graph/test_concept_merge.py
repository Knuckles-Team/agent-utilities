"""Plan 02 Step 7: merge_similar_concepts must preserve relationship types,
not corrupt the survivor's id, and record MergedFrom provenance.

Uses a fake backend (for the reads + the final DETACH DELETE, all within the
engine's native Cypher write subset) and a fake engine exposing
``link_nodes``/``add_node`` (the typed dispatch every edge/property-merge
write in ``_merge_concept_pair`` now goes through -- a comma-pattern MATCH
paired with an edge MERGE, and a ``SET new += $props`` map-merge assignment,
both exceed that subset;
epistemic-graph/crates/eg-query/src/cypher/parser.rs:1184) so we can assert
the behaviour without a live graph database.
"""

from __future__ import annotations

from typing import Any

from agent_utilities.knowledge_graph.core.maintainer import GraphMaintainer


class FakeBackend:
    """Minimal Cypher-recording backend driving the merge code path's reads
    and the final (in-subset) DETACH DELETE."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def execute(self, query: str, params: dict | None = None):
        params = params or {}
        self.calls.append((query, params))
        q = " ".join(query.split())

        # Two near-identical concepts with embeddings.
        if "c.embedding IS NOT NULL" in q:
            return [
                {"id": "c1", "name": "Vector DB", "embedding": [1.0, 0.0, 0.0]},
                {"id": "c2", "name": "Vector Database", "embedding": [1.0, 0.0, 0.0]},
            ]
        # Outgoing edges of the old node, with a *typed* relationship.
        if "MATCH (old:Concept {id: $old_id})-[r]->(target)" in q and "type(r)" in q:
            return [{"rtype": "DEPENDS_ON", "tid": "tool_x", "props": {"weight": 2}}]
        # Incoming edges of the old node, typed.
        if "MATCH (source)-[r]->(old:Concept {id: $old_id})" in q and "type(r)" in q:
            return [{"rtype": "USED_BY", "sid": "agent_y", "props": {}}]
        # Node property snapshots for the non-destructive merge.
        if "RETURN properties(old) AS old_props" in q:
            return [
                {
                    "old_props": {
                        "id": "c2",
                        "name": "Vector Database",
                        "aliases": ["vdb"],
                        "importance": 9,
                    },
                    "new_props": {
                        "id": "c1",
                        "name": "Vector DB",
                        "aliases": ["vector-db"],
                        "importance": 5,
                    },
                }
            ]
        return []


class FakeEngine:
    """Fake ``IntelligenceGraphEngine`` recording the typed-API calls
    ``_merge_concept_pair``/``_merge_node_properties`` now dispatch edge and
    property-merge writes through."""

    def __init__(self, backend: FakeBackend):
        self.backend = backend
        self.link_calls: list[tuple[str, str, str, dict]] = []
        self.add_node_calls: list[tuple[str, str, dict]] = []

    def link_nodes(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> None:
        self.link_calls.append((source_id, target_id, rel_type, dict(properties or {})))

    def add_node(
        self,
        node_id: str,
        node_type: str,
        properties: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> None:
        self.add_node_calls.append((node_id, node_type, dict(properties or {})))


def _make_maintainer() -> tuple[GraphMaintainer, FakeBackend, FakeEngine]:
    backend = FakeBackend()
    engine = FakeEngine(backend)
    return GraphMaintainer(engine), backend, engine  # type: ignore[arg-type]


def test_merge_preserves_relationship_types():
    maint, _backend, engine = _make_maintainer()
    merged = maint.merge_similar_concepts(similarity_threshold=0.9)
    assert merged == 1

    # The original typed edges survive as typed ``link_nodes`` calls (the
    # survivor "c1" gains DEPENDS_ON -> tool_x and agent_y -[USED_BY]-> "c1")...
    rel_types = {rtype for (_s, _t, rtype, _p) in engine.link_calls}
    assert "DEPENDS_ON" in rel_types
    assert "USED_BY" in rel_types
    # ...and the lossy generic collapse is gone.
    assert "RELATED_TO" not in rel_types


def test_merge_does_not_corrupt_survivor_id():
    maint, backend, engine = _make_maintainer()
    maint.merge_similar_concepts(similarity_threshold=0.9)

    # The old buggy `SET new += old` (which copied old.id onto the survivor)
    # is gone from every raw Cypher call this path still issues.
    for q, _ in backend.calls:
        assert "SET new += old" not in " ".join(q.split())
    for _s, _t, _r, props in engine.link_calls:
        assert "SET new += old" not in str(props)

    # Property merge targets the survivor via the typed node upsert (field-
    # merge: an existing node keeps any field omitted here) and never writes
    # a protected key.
    survivor_upserts = [
        props for node_id, _label, props in engine.add_node_calls if node_id == "c1"
    ]
    assert survivor_upserts, "expected a non-destructive property merge on the survivor"
    props = survivor_upserts[0]
    assert "id" not in props and "name" not in props
    assert set(props["aliases"]) == {"vdb", "vector-db"}  # unioned
    assert props["importance"] == 9  # max


def test_merge_records_provenance_and_deletes_duplicate():
    maint, backend, engine = _make_maintainer()
    maint.merge_similar_concepts(similarity_threshold=0.9)

    assert any(
        rtype == "MERGED_FROM" and source == "c1" and target == "c2"
        for (source, target, rtype, _p) in engine.link_calls
    ), "provenance edge not recorded"
    # The tombstone MergedConcept node itself is a bare-node typed upsert
    # (within the supported subset either way, but now dispatched the same
    # typed way as everything else in this method).
    assert any(
        node_id == "c2" and label == "MergedConcept"
        for node_id, label, _p in engine.add_node_calls
    ), "MergedConcept tombstone not recorded"

    qs = [" ".join(q.split()) for q, _ in backend.calls]
    assert any("DETACH DELETE old" in q for q in qs), "duplicate not deleted"
