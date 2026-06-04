"""Plan 02 Step 7: merge_similar_concepts must preserve relationship types,
not corrupt the survivor's id, and record MergedFrom provenance.

Uses a fake backend that records executed Cypher so we can assert the
behaviour without a live graph database.
"""

from __future__ import annotations

from types import SimpleNamespace

from agent_utilities.knowledge_graph.core.maintainer import GraphMaintainer


class FakeBackend:
    """Minimal Cypher-recording backend driving the merge code path."""

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


def _make_maintainer() -> tuple[GraphMaintainer, FakeBackend]:
    backend = FakeBackend()
    engine = SimpleNamespace(backend=backend)
    return GraphMaintainer(engine), backend  # type: ignore[arg-type]


def test_merge_preserves_relationship_types():
    maint, backend = _make_maintainer()
    merged = maint.merge_similar_concepts(similarity_threshold=0.9)
    assert merged == 1

    merge_queries = [" ".join(q.split()) for q, _ in backend.calls if "MERGE" in q]
    joined = "\n".join(merge_queries)
    # The original typed edges survive...
    assert "DEPENDS_ON" in joined
    assert "USED_BY" in joined
    # ...and the lossy generic collapse is gone.
    assert "MERGE (new)-[:RELATED_TO]->(target)" not in joined


def test_merge_does_not_corrupt_survivor_id():
    maint, backend = _make_maintainer()
    maint.merge_similar_concepts(similarity_threshold=0.9)

    # The old buggy `SET new += old` (which copied old.id onto the survivor) is gone.
    for q, _ in backend.calls:
        assert "SET new += old" not in " ".join(q.split())

    # Property merge targets the survivor and never writes a protected key.
    prop_sets = [p for q, p in backend.calls if "SET new += $props" in " ".join(q.split())]
    assert prop_sets, "expected a non-destructive property merge"
    props = prop_sets[0]["props"]
    assert "id" not in props and "name" not in props
    assert set(props["aliases"]) == {"vdb", "vector-db"}  # unioned
    assert props["importance"] == 9  # max


def test_merge_records_provenance_and_deletes_duplicate():
    maint, backend = _make_maintainer()
    maint.merge_similar_concepts(similarity_threshold=0.9)
    qs = [" ".join(q.split()) for q, _ in backend.calls]
    assert any("MERGED_FROM" in q for q in qs), "provenance edge not recorded"
    assert any("DETACH DELETE old" in q for q in qs), "duplicate not deleted"
