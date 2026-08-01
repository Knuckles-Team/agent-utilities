"""Unified feed-source bridge + first-class :FeedSource registry (CONCEPT:AU-KG.ingest.rss-feed-connector/2.122)."""

from __future__ import annotations

import uuid

import pytest

from agent_utilities.automation.feed_sources import (
    list_feed_sources,
    register_feed_nodes,
    scholarx_feed_documents,
    upsert_feed_source,
)
from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
    EpistemicGraphBackend,
)
from agent_utilities.knowledge_graph.core.session import GraphSession, use_session


class _Engine:
    """A bare ``EpistemicGraphBackend()`` resolves ``resolve_routing_graph(None)``
    to the ambient tenant's SHARED default graph -- an explicit name the
    autouse ``isolate_graph_compute_engine`` fixture's redirect can't catch
    (it only matches literal ``None``/``"__commons__"``/``"__secrets__"``, not
    an already-tenant-resolved name). Sequential tests each constructing their
    own bare ``_Engine()`` then collide on that ONE shared graph:
    ``RuntimeError: ... STALE_FENCE``. Retargeting a per-instance session at
    an explicit, uniquely-named graph (same shape as
    tests/integration/knowledge_graph/test_engine_helpers.py's ``engine``
    fixture and tests/unit/knowledge_graph/test_topological_analogy.py's
    ``base_graph``) gives every ``_Engine()`` its own isolated graph.
    """

    def __init__(self):
        graph_name = f"test_feed_sources_{uuid.uuid4().hex[:12]}"
        self._session_cm = use_session(GraphSession.from_ambient().with_graph(graph_name))
        self._session_cm.__enter__()
        self.backend = EpistemicGraphBackend(graph_name=graph_name)

    def add_node(self, node_id, node_type, properties=None):
        self.backend.add_node(node_id, node_type=node_type, **(properties or {}))


@pytest.fixture(autouse=True)
def _capture_native_feed_submission(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep registry mapping assertions isolated from native transaction tests."""

    def capture(engine, envelope):
        row = envelope.to_entity_dict()
        node_id = str(row.pop("id"))
        node_type = str(row.pop("type"))
        engine.add_node(node_id, node_type, properties=row)
        return {"status": "success", "watermark_advanced": False}

    monkeypatch.setattr(
        "agent_utilities.knowledge_graph.ingestion.envelope_ingest.ingest_envelope",
        capture,
    )


def test_scholarx_feed_documents_noop_without_scholarx(monkeypatch):
    # No scholarx installed in the unit env → safe no-op (never raises).
    assert scholarx_feed_documents(["cs.AI"]) == []


def test_register_and_list_feed_sources():
    eng = _Engine()
    ids = register_feed_nodes(
        eng,
        native_urls=["http://feed/a", "http://feed/b"],
        scholarx_categories=["cs.AI"],
        freshrss_configured=True,
    )
    assert len(ids) == 4
    listed = {n["id"]: n for n in list_feed_sources(eng)}
    assert len(listed) == 4
    # Native URL feed → RssFeed kind, rss source_system, carries the feed_url.
    a = next(n for n in listed.values() if n.get("feed_url") == "http://feed/a")
    assert a["kind"] == "RssFeed" and a["source_system"] == "rss"
    # FreshRSS → FeedSource kind.
    fr = next(n for n in listed.values() if n.get("source_system") == "freshrss")
    assert fr["kind"] == "FeedSource"


def test_upsert_is_idempotent():
    eng = _Engine()
    a = upsert_feed_source(
        eng, key="http://x", source_system="rss", feed_url="http://x"
    )
    b = upsert_feed_source(
        eng, key="http://x", source_system="rss", feed_url="http://x"
    )
    assert a == b
    assert len(list_feed_sources(eng)) == 1
