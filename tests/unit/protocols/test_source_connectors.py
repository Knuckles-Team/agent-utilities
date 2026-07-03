"""Tests for the document-source connector framework (CONCEPT:ECO-4.25–4.28).

Offline + deterministic: the web/rest/mcp connectors are driven by injected
fetch/transport callables, the filesystem connector by a temp dir, and the
database connector by a fake connection — no network, no live services.
"""

from __future__ import annotations

import pytest

from agent_utilities.protocols.source_connectors import (
    ConnectorCheckpoint,
    ExternalAccess,
    LoadConnector,
    PollConnector,
    SourceDocument,
    build_connector,
    list_sources,
    sync_access,
)


@pytest.mark.concept("ECO-4.25")
def test_registry_discovers_builtin_connectors():
    sources = list_sources()
    assert {"web", "filesystem", "rest", "database", "mcp"} <= set(sources)


@pytest.mark.concept("ECO-4.27")
def test_build_connector_unknown_lists_available():
    with pytest.raises(KeyError) as exc:
        build_connector("nope", {})
    assert "Available" in str(exc.value)


@pytest.mark.concept("ECO-4.26")
def test_checkpoint_json_roundtrip():
    cp = ConnectorCheckpoint(
        has_more=True,
        cursor="c1",
        watermark="2026-01-01",
        seen_ids=["a", "b"],
        state={"k": 1},
    )
    restored = ConnectorCheckpoint.from_json(cp.to_json())
    assert restored == cp
    assert ConnectorCheckpoint.from_json(None) is None
    assert ConnectorCheckpoint.from_json("not-json") is None


@pytest.mark.concept("ECO-4.25")
def test_filesystem_connector_load_and_poll(tmp_path):
    (tmp_path / "a.md").write_text("# A\nalpha content about graphs\n")
    (tmp_path / "b.txt").write_text("beta content about ontologies\n")
    (tmp_path / "skip.bin").write_bytes(b"\x00\x01")  # non-doc extension ignored

    conn = build_connector("filesystem", {"root": str(tmp_path)})
    assert isinstance(conn, LoadConnector)
    docs = list(conn.load())
    assert {d.title for d in docs} == {"a.md", "b.txt"}
    assert all(d.text.strip() for d in docs)

    # poll → all on first call, none when unchanged (watermark incrementality).
    batch = conn.poll()
    assert len(batch.documents) == 2
    assert batch.checkpoint.watermark is not None
    again = conn.poll(batch.checkpoint)
    assert again.documents == []


@pytest.mark.concept("ECO-4.28")
def test_filesystem_perm_sync_groups(tmp_path):
    (tmp_path / "c.md").write_text("secret content\n")
    conn = build_connector("filesystem", {"root": str(tmp_path), "public": False})
    access = dict(conn.fetch_access())
    key = str((tmp_path / "c.md").resolve())
    assert key in access
    assert access[key].is_public is False


@pytest.mark.concept("ECO-4.25")
def test_web_connector_offline_crawl():
    pages = {
        "http://x/": "<title>Home</title><a href='/a'>a</a><a href='/b'>b</a>",
        "http://x/a": "<title>A</title>page a body",
        "http://x/b": "<title>B</title>page b body",
    }
    conn = build_connector(
        "web", {"base_url": "http://x/", "max_depth": 1, "fetch_fn": pages.get}
    )
    docs = list(conn.load())
    titles = {d.title for d in docs}
    assert titles == {"Home", "A", "B"}
    assert all(d.external_access and d.external_access.is_public for d in docs)


@pytest.mark.concept("ECO-4.26")
def test_web_connector_poll_dedup():
    pages = {"http://y/": "<title>Y</title>body"}
    conn = build_connector("web", {"base_url": "http://y/", "fetch_fn": pages.get})
    b1 = conn.poll()
    assert len(b1.documents) == 1
    b2 = conn.poll(b1.checkpoint)  # already seen → no new docs
    assert b2.documents == []


@pytest.mark.concept("ECO-4.25")
def test_rest_connector_pagination():
    pages = {
        None: {"items": [{"id": 1, "title": "T1", "body": "x"}], "next": "cur2"},
        "cur2": {"items": [{"id": 2, "title": "T2", "body": "y"}], "next": None},
    }

    def fetch(url, params):
        return pages[params.get("cursor")]

    conn = build_connector(
        "rest",
        {
            "url": "http://api/",
            "records_field": "items",
            "text_field": "body",
            "cursor_field": "next",
            "cursor_param": "cursor",
            "fetch_fn": fetch,
        },
    )
    docs = list(conn.load())
    assert [d.id for d in docs] == ["1", "2"]


@pytest.mark.concept("ECO-4.25")
def test_database_connector_watermark():
    rows = [
        {"id": 1, "title": "A", "body": "alpha", "ts": "2026-01-01"},
        {"id": 2, "title": "B", "body": "beta", "ts": "2026-02-01"},
    ]

    class FakeConn:
        def read(self, q, p=None):
            return rows

        def health_check(self):
            return True

    conn = build_connector(
        "database",
        {
            "dsn": "x",
            "query": "select *",
            "text_field": "body",
            "updated_field": "ts",
            "conn": FakeConn(),
        },
    )
    assert isinstance(conn, PollConnector)
    b1 = conn.poll()
    assert [d.id for d in b1.documents] == ["1", "2"]
    assert b1.checkpoint.watermark == "2026-02-01"
    b2 = conn.poll(b1.checkpoint)  # nothing newer than the watermark
    assert b2.documents == []


@pytest.mark.concept("ECO-4.28")
def test_permission_sync_maps_acl_and_markings():
    access = ExternalAccess(
        is_public=False, group_ids=["eng"], user_emails=["a@x"], markings=["SECRET"]
    )
    acl = sync_access("doc:1", access, [("doc:1", "doc:1::chunk::0")])
    assert acl is not None
    assert "group:eng" in acl.read_roles and "user:a@x" in acl.read_roles

    from agent_utilities.knowledge_graph.ontology.permissioning import markings_for

    assert "SECRET" in markings_for("doc:1")
    assert "SECRET" in markings_for("doc:1::chunk::0")  # propagated to chunk

    # public access registers no restrictive ACL
    assert sync_access("doc:2", ExternalAccess.public(), []) is None


@pytest.mark.concept("ECO-4.25")
def test_source_document_shape():
    doc = SourceDocument(id="1", text="hello", title="T")
    assert doc.doc_type == "document"
    assert doc.external_access is None


# ── Native RSS/Atom connector (CONCEPT:KG-2.121) ─────────────────────────────

_RSS_XML = """<?xml version="1.0"?>
<rss version="2.0"><channel><title>Tech News</title>
  <item><title>GPU launch</title><link>http://n/1</link>
    <guid>http://n/1</guid><pubDate>Tue, 17 Jun 2025 10:00:00 GMT</pubDate>
    <description>A new accelerator from a chipmaker.</description>
    <category>hardware</category></item>
  <item><title>Funding round</title><link>http://n/2</link>
    <guid>http://n/2</guid><pubDate>Wed, 18 Jun 2025 10:00:00 GMT</pubDate>
    <description>Series B for an AI startup.</description></item>
</channel></rss>"""

_ATOM_XML = """<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>arXiv cs.AI</title>
  <entry><title>A paper on agents</title><id>arxiv:2601.0009</id>
    <link href="http://arxiv.org/abs/2601.0009"/>
    <updated>2025-06-18T09:00:00Z</updated>
    <summary>We study self-improving agent harnesses.</summary>
    <category term="research"/></entry>
</feed>"""


@pytest.mark.concept("KG-2.121")
def test_rss_connector_registered():
    assert "rss" in set(list_sources())


@pytest.mark.concept("KG-2.121")
def test_rss_connector_parses_rss_and_atom():
    feeds = {"http://feed/rss": _RSS_XML, "http://feed/atom": _ATOM_XML}
    conn = build_connector("rss", {"feed_urls": list(feeds), "fetch_fn": feeds.get})
    assert isinstance(conn, (LoadConnector, PollConnector))
    docs = list(conn.load())
    assert len(docs) == 3
    by_id = {d.id: d for d in docs}
    # RSS item: record envelope carries categories + origin so the gate can route it.
    rss_doc = by_id["http://n/1"]
    rec = rss_doc.metadata["record"]
    assert rec["categories"] == ["hardware"]
    assert rec["origin"]["streamId"] == "http://feed/rss"
    assert rss_doc.metadata["source_system"] == "rss"
    assert rss_doc.updated_at == "2025-06-17T10:00:00Z"
    # Atom (arXiv) entry parses with its id + research category.
    atom = by_id["arxiv:2601.0009"]
    assert atom.title == "A paper on agents"
    assert "research" in atom.metadata["record"]["categories"]


@pytest.mark.concept("KG-2.121")
def test_rss_connector_poll_watermark_dedup():
    feeds = {"http://feed/rss": _RSS_XML}
    conn = build_connector(
        "rss", {"feed_urls": "http://feed/rss", "fetch_fn": feeds.get}
    )
    b1 = conn.poll()
    assert len(b1.documents) == 2  # first poll → all
    assert b1.checkpoint.watermark == "2025-06-18T10:00:00Z"
    b2 = conn.poll(b1.checkpoint)  # unchanged feed → nothing new (seen-id belt)
    assert b2.documents == []


@pytest.mark.concept("KG-2.121")
def test_rss_connector_dead_feed_is_skipped():
    def _boom(url):
        raise RuntimeError("dns fail")

    conn = build_connector("rss", {"feed_urls": "http://dead/", "fetch_fn": _boom})
    assert list(conn.load()) == []  # a dead feed never aborts


@pytest.mark.concept("KG-2.121")
def test_rss_connector_fetches_feeds_concurrently():
    # Many feeds, each fetch sleeps: a concurrent sweep costs ~one feed's latency,
    # not N×. Guards the serial stall that timed out the 19-feed sweep (>300s) —
    # the throughput unlock for the 2000-reviews/hr path.
    import time

    n, delay = 8, 0.25
    urls = [f"http://feed/{i}" for i in range(n)]

    def _slow_fetch(url):
        time.sleep(delay)
        return _RSS_XML

    conn = build_connector("rss", {"feed_urls": urls, "fetch_fn": _slow_fetch})
    t0 = time.monotonic()
    docs = list(conn.load())
    elapsed = time.monotonic() - t0
    # serial would be n*delay = 2.0s; concurrent (<=12 workers) must be far less
    assert elapsed < n * delay * 0.5, f"feeds not fetched concurrently: {elapsed:.2f}s"
    assert len(docs) == n * 2  # _RSS_XML has 2 entries per feed
