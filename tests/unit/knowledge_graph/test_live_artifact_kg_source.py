"""CONCEPT:KG-2.24 — KG-backed Live Artifact refresh resolver tests (follow-up #1).

Verifies the resolver shapes KG query results into artifact data, raises (→ prior preserved) when the
query is missing/fails, and that a refresh through the RefreshService re-derives from a stubbed KG.
"""

from __future__ import annotations

import pytest

from agent_utilities.knowledge_graph.live_artifacts import (
    LiveArtifact,
    LiveArtifactStore,
    RefreshService,
)
from agent_utilities.knowledge_graph.live_artifacts.kg_source import kg_source_resolver

pytestmark = pytest.mark.concept(id="KG-2.24")


def test_resolver_requires_source_query():
    art = LiveArtifact(template="x", data={}, source_query="")
    with pytest.raises(ValueError):
        kg_source_resolver(art)


def test_resolver_shapes_kg_rows(monkeypatch):
    # Stub KnowledgeGraph.query to avoid a live KG.
    import agent_utilities.knowledge_graph.facade as facade

    class _StubKG:
        def query(self, cypher, params=None):
            assert "MATCH" in cypher
            return [{"n": 1}, {"n": 2}, {"n": 3}]

    monkeypatch.setattr(facade, "KnowledgeGraph", lambda *a, **k: _StubKG())
    art = LiveArtifact(template="{{data.count}}", data={}, source_query="MATCH (n) RETURN n")
    shaped = kg_source_resolver(art)
    assert shaped["count"] == 3
    assert shaped["first"] == {"n": 1}
    assert len(shaped["rows"]) == 3


def test_refresh_via_kg_source_rederives(monkeypatch):
    import agent_utilities.knowledge_graph.facade as facade

    rows = {"v": [{"x": 1}]}

    class _StubKG:
        def query(self, cypher, params=None):
            return rows["v"]

    monkeypatch.setattr(facade, "KnowledgeGraph", lambda *a, **k: _StubKG())
    store = LiveArtifactStore()
    art = store.create(
        LiveArtifact(template="count={{data.count}}", data={"count": 0}, source_query="MATCH (n) RETURN n")
    )
    svc = RefreshService(store)
    res = svc.refresh(art.artifact_id, kg_source_resolver)
    assert res.ok is True
    assert "count=1" in res.rendered

    # KG failure → prior preserved
    def boom(cypher, params=None):
        raise RuntimeError("kg down")

    monkeypatch.setattr(_StubKG, "query", lambda self, c, params=None: boom(c))
    res2 = svc.refresh(art.artifact_id, kg_source_resolver)
    assert res2.ok is False
    assert "count=1" in store.get(art.artifact_id).last_rendered  # prior preserved


def test_install_registers_resolver():
    from agent_utilities.gateway import artifacts_api
    from agent_utilities.knowledge_graph.live_artifacts.kg_source import (
        install_kg_artifact_source,
        kg_source_resolver,
    )

    assert install_kg_artifact_source() is True
    assert artifacts_api._source_resolver is kg_source_resolver
