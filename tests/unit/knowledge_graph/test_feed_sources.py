"""Unified feed-source bridge + first-class :FeedSource registry (CONCEPT:KG-2.121/2.122)."""

from __future__ import annotations

from agent_utilities.automation.feed_sources import (
    list_feed_sources,
    register_feed_nodes,
    scholarx_feed_documents,
    upsert_feed_source,
)
from agent_utilities.knowledge_graph.backends.epistemic_graph_backend import (
    EpistemicGraphBackend,
)


class _Engine:
    def __init__(self):
        self.backend = EpistemicGraphBackend()

    def add_node(self, node_id, node_type, properties=None):
        self.backend.add_node(node_id, node_type=node_type, **(properties or {}))


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
    a = upsert_feed_source(eng, key="http://x", source_system="rss", feed_url="http://x")
    b = upsert_feed_source(eng, key="http://x", source_system="rss", feed_url="http://x")
    assert a == b
    assert len(list_feed_sources(eng)) == 1
