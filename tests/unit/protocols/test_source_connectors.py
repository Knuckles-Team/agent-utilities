"""Tests for the document-source connector framework (CONCEPT:AU-ECO.connector.document-source-framework–4.28).

Offline + deterministic: the web/rest/mcp connectors are driven by injected
fetch/transport callables, the filesystem connector by a temp dir, and the
database connector by a fake connection — no network, no live services.
"""

from __future__ import annotations

import pytest

from agent_utilities.protocols.source_connectors import (
    ConnectorCheckpoint,
    ConnectorGovernanceError,
    ExternalAccess,
    LoadConnector,
    PollConnector,
    SourceDocument,
    build_connector,
    list_sources,
    sync_access,
)


@pytest.mark.concept("AU-ECO.connector.document-source-framework")
def test_registry_discovers_builtin_connectors():
    sources = list_sources()
    assert {"web", "filesystem", "rest", "database", "mcp"} <= set(sources)


@pytest.mark.concept("AU-ECO.connector.factory-ingestion-adaptor")
def test_build_connector_unknown_lists_available():
    with pytest.raises(KeyError) as exc:
        build_connector("nope", {})
    assert "Available" in str(exc.value)


@pytest.mark.concept("AU-KG.ontology.connector-manifest-gate")
def test_build_connector_blocks_uncertified_activation_in_production(monkeypatch):
    monkeypatch.setenv("APP_PROFILE", "production")
    monkeypatch.setenv("AGENT_UTILITIES_TESTING", "true")

    with pytest.raises(ConnectorGovernanceError, match="activation refused"):
        build_connector("mcp_tool", {"server": "uncertified-provider", "tool": "list"})


@pytest.mark.concept("AU-KG.ontology.connector-manifest-gate")
def test_native_connector_bundle_allows_zero_config_governance(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_PROFILE", "production")
    monkeypatch.setenv("AGENT_UTILITIES_TESTING", "false")

    connector = build_connector("filesystem", {"root": str(tmp_path)})

    assert isinstance(connector, LoadConnector)


@pytest.mark.concept("AU-ECO.connector.incremental-poll-watermark")
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


@pytest.mark.concept("AU-ECO.connector.document-source-framework")
def test_filesystem_connector_load_and_poll(tmp_path):
    (tmp_path / "a.md").write_text("# A\nalpha content about graphs\n")
    (tmp_path / "b.txt").write_text("beta content about ontologies\n")
    (tmp_path / "skip.bin").write_bytes(b"\x00\x01")  # non-doc extension ignored

    conn = build_connector("filesystem", {"root": str(tmp_path)})
    assert isinstance(conn, LoadConnector)
    docs = list(conn.load())
    assert {d.title for d in docs} == {"a.md", "b.txt"}
    assert all(d.text.strip() for d in docs)
    assert all(d.source_uri.startswith("filesystem://") for d in docs)
    assert all(str(tmp_path) not in d.source_uri for d in docs)

    # poll → all on first call, none when unchanged (watermark incrementality).
    batch = conn.poll()
    assert len(batch.documents) == 2
    assert batch.checkpoint.watermark is not None
    again = conn.poll(batch.checkpoint)
    assert again.documents == []


@pytest.mark.concept("AU-ECO.connector.external-permission-sync")
def test_filesystem_perm_sync_groups(tmp_path):
    (tmp_path / "c.md").write_text("secret content\n")
    conn = build_connector("filesystem", {"root": str(tmp_path), "public": False})
    access = dict(conn.fetch_access())
    key = next(iter(access))
    assert key.startswith("filesystem://")
    assert str(tmp_path) not in key
    assert key in access
    assert access[key].is_public is False
    assert access[key].markings


@pytest.mark.concept("AU-ECO.connector.document-source-framework")
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
    assert all(d.external_access and not d.external_access.is_public for d in docs)
    assert all(d.external_access and d.external_access.markings for d in docs)


@pytest.mark.concept("AU-ECO.connector.incremental-poll-watermark")
def test_web_connector_poll_dedup():
    pages = {"http://y/": "<title>Y</title>body"}
    conn = build_connector("web", {"base_url": "http://y/", "fetch_fn": pages.get})
    b1 = conn.poll()
    assert len(b1.documents) == 1
    b2 = conn.poll(b1.checkpoint)  # already seen → no new docs
    assert b2.documents == []


@pytest.mark.concept("AU-ECO.connector.document-source-framework")
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


@pytest.mark.concept("AU-ECO.connector.document-source-framework")
def test_database_connector_watermark():
    rows = [
        {"id": 1, "title": "A", "body": "alpha", "ts": "2026-01-01"},
        {"id": 2, "title": "B", "body": "beta", "ts": "2026-02-01"},
    ]

    class FakeConn:
        def read(self, q, p=None, *, max_rows=10_000):
            return rows[:max_rows]

        def health_check(self):
            return True

    conn = build_connector(
        "database",
        {
            "query": "select *",
            "text_field": "body",
            "updated_field": "ts",
            "conn": FakeConn(),
        },
    )
    assert isinstance(conn, PollConnector)
    b1 = conn.poll()
    assert all(d.id.startswith("pref_source_document_") for d in b1.documents)
    assert len({d.id for d in b1.documents}) == 2
    assert all(
        "id=1" not in d.source_uri and "id=2" not in d.source_uri for d in b1.documents
    )
    assert b1.checkpoint.watermark == "2026-02-01"
    b2 = conn.poll(b1.checkpoint)  # nothing newer than the watermark
    assert b2.documents == []


def test_database_connector_rejects_mutating_or_stacked_query():
    from agent_utilities.protocols.source_connectors.connectors.database import (
        DatabaseConnector,
    )

    class FakeConn:
        pass

    for query in (
        "DELETE FROM records",
        "SELECT * FROM records; DELETE FROM records",
        "SELECT * INTO copy FROM records",
        "SELECT * FROM records FOR UPDATE",
    ):
        with pytest.raises(ValueError, match="read-only SELECT"):
            DatabaseConnector(query=query, conn=FakeConn())


def test_database_connector_rejects_literal_connection_value():
    from agent_utilities.protocols.source_connectors.connectors.database import (
        DatabaseConnector,
    )

    with pytest.raises(ValueError, match="legacy database dsn"):
        DatabaseConnector(dsn="postgresql://user:secret@host/db", query="SELECT 1")


def test_rest_connector_rejects_untrusted_cross_host_pagination():
    from agent_utilities.protocols.source_connectors.connectors.rest import (
        RestJsonConnector,
    )

    def fetch(_url, _params):
        return {
            "items": [{"id": "one", "text": "body"}],
            "next": "http://127.0.0.1/private",
        }

    conn = RestJsonConnector(
        url="https://api.example.invalid/items",
        records_field="items",
        next_url_field="next",
        fetch_fn=fetch,
    )
    with pytest.raises(ValueError, match="egress policy"):
        list(conn.load())


def test_http_source_redirect_is_revalidated_before_following():
    import httpx

    from agent_utilities.protocols.source_connectors.http_safety import (
        SourceEgressError,
        safe_get_text,
    )

    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(
            302, headers={"location": "http://127.0.0.1/private"}, request=request
        )

    transport = httpx.MockTransport(handler)
    with pytest.raises(SourceEgressError, match="egress policy"):
        safe_get_text("https://public.example.invalid/start", transport=transport)
    assert calls == 1


def test_http_source_rejects_https_redirect_downgrade():
    import httpx

    from agent_utilities.protocols.source_connectors.http_safety import (
        SourceEgressError,
        safe_get_text,
    )

    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            302,
            headers={"location": "http://public.example.invalid/cleartext"},
            request=request,
        )
    )
    with pytest.raises(SourceEgressError, match="downgrade"):
        safe_get_text("https://public.example.invalid/start", transport=transport)


def test_http_source_dns_resolution_is_pinned_with_original_host_and_sni(monkeypatch):
    import httpx

    import agent_utilities.protocols.source_connectors.http_safety as http_safety

    requests = []
    cleanup_calls = []

    class Trust:
        proxy_url = None

        @staticmethod
        def httpx_kwargs():
            return {"verify": True, "trust_env": True}

        @staticmethod
        def cleanup():
            cleanup_calls.append(True)

    class PinnedStream:
        @staticmethod
        def get_extra_info(name):
            return ("203.0.113.9", 443) if name == "server_addr" else None

    transport = httpx.MockTransport(
        lambda request: (
            requests.append(request)
            or httpx.Response(
                200,
                content=b"bounded",
                extensions={"network_stream": PinnedStream()},
                request=request,
            )
        )
    )
    monkeypatch.setattr(
        "agent_utilities.core.transport_security.resolve_configured_tls_profile",
        lambda _service: Trust(),
    )
    monkeypatch.setattr(
        "agent_utilities.core.http_client.create_http_client",
        lambda **_kwargs: httpx.Client(transport=transport),
    )
    monkeypatch.setattr(
        http_safety,
        "resolve_safe_source_url",
        lambda url, **_kwargs: http_safety.ResolvedSourceURL(
            url=url,
            host="public.example.invalid",
            scheme="https",
            resolved_ips=("203.0.113.9",),
        ),
    )

    body, _encoding = http_safety.safe_get_bytes(
        "https://public.example.invalid/content"
    )

    assert body == b"bounded"
    assert requests[0].url.host == "203.0.113.9"
    assert requests[0].headers["host"] == "public.example.invalid"
    assert requests[0].extensions["sni_hostname"] == "public.example.invalid"
    assert cleanup_calls == [True]


def test_http_source_cross_host_redirect_drops_sensitive_headers():
    import httpx

    from agent_utilities.protocols.source_connectors.http_safety import safe_get_text

    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                302,
                headers={"location": "https://next.example.invalid/content"},
                request=request,
            )
        return httpx.Response(200, content=b"ok", request=request)

    safe_get_text(
        "https://public.example.invalid/start",
        headers={"Authorization": "Bearer runtime-secret", "Cookie": "private=1"},
        allowed_redirect_hosts=["next.example.invalid"],
        transport=httpx.MockTransport(handler),
    )

    assert requests[0].headers.get("authorization") == "Bearer runtime-secret"
    assert requests[1].headers.get("authorization") is None
    assert requests[1].headers.get("cookie") is None


def test_http_source_same_host_new_port_drops_sensitive_headers():
    import httpx

    from agent_utilities.protocols.source_connectors.http_safety import safe_get_text

    requests = []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                302,
                headers={"location": "https://public.example.invalid:8443/content"},
                request=request,
            )
        return httpx.Response(200, content=b"ok", request=request)

    safe_get_text(
        "https://public.example.invalid/start",
        headers={"Authorization": "Bearer runtime-secret"},
        transport=httpx.MockTransport(handler),
    )

    assert requests[0].headers.get("authorization") == "Bearer runtime-secret"
    assert requests[1].headers.get("authorization") is None


def test_http_source_resolution_rejects_rebinding_to_private_address():
    from agent_utilities.protocols.source_connectors.http_safety import (
        SourceEgressError,
        resolve_safe_source_url,
    )

    def resolver(_host, _port):
        return [(2, 1, 6, "", ("127.0.0.1", 0))]

    with pytest.raises(SourceEgressError, match="egress policy"):
        resolve_safe_source_url(
            "https://public.example.invalid/content",
            resolver=resolver,
        )


def test_http_source_peer_must_match_selected_dns_pin():
    from agent_utilities.protocols.source_connectors.http_safety import (
        ResolvedSourceURL,
        SourceEgressError,
        _require_direct_peer,
    )

    class Stream:
        @staticmethod
        def get_extra_info(_name):
            return ("203.0.113.10", 443)

    class Response:
        extensions = {"network_stream": Stream()}

    resolution = ResolvedSourceURL(
        url="https://public.example.invalid/content",
        host="public.example.invalid",
        scheme="https",
        resolved_ips=("203.0.113.9",),
    )
    with pytest.raises(SourceEgressError, match="did not match"):
        _require_direct_peer(Response(), resolution, None)


def test_http_source_body_limit_applies_to_decoded_stream():
    import httpx

    from agent_utilities.protocols.source_connectors.http_safety import (
        SourceEgressError,
        safe_get_text,
    )

    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=b"x" * 32, request=request)
    )
    with pytest.raises(SourceEgressError, match="configured limit"):
        safe_get_text(
            "https://public.example.invalid/data", transport=transport, max_bytes=8
        )


@pytest.mark.asyncio
async def test_async_http_source_body_limit_applies_to_stream():
    import httpx

    from agent_utilities.protocols.source_connectors.http_safety import (
        SourceEgressError,
        safe_get_bytes_async,
    )

    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=b"x" * 32, request=request)
    )
    with pytest.raises(SourceEgressError, match="configured limit"):
        await safe_get_bytes_async(
            "https://public.example.invalid/data",
            transport=transport,
            max_bytes=8,
        )


@pytest.mark.asyncio
async def test_state_changing_source_request_rejects_redirect():
    import httpx

    from agent_utilities.protocols.source_connectors.http_safety import (
        SourceEgressError,
        safe_post_json_async,
    )

    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            307,
            headers={"location": "https://other.example.invalid/submit"},
            request=request,
        )
    )
    with pytest.raises(SourceEgressError, match="redirect"):
        await safe_post_json_async(
            "https://public.example.invalid/submit",
            {"value": "bounded"},
            transport=transport,
        )


@pytest.mark.asyncio
async def test_state_changing_source_request_is_size_bounded_before_transport():
    import httpx

    from agent_utilities.protocols.source_connectors.http_safety import (
        SourceEgressError,
        safe_post_json_async,
    )

    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"ok": True}, request=request)

    with pytest.raises(SourceEgressError, match="request exceeded"):
        await safe_post_json_async(
            "https://public.example.invalid/submit",
            {"value": "too-large"},
            max_request_bytes=4,
            transport=httpx.MockTransport(handler),
        )
    assert calls == 0


def test_sync_json_post_rejects_metadata_ssrf_before_transport():
    import httpx

    from agent_utilities.protocols.source_connectors.http_safety import (
        SourceEgressError,
        safe_post_json,
    )

    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"ok": True}, request=request)

    with pytest.raises(SourceEgressError, match="egress policy"):
        safe_post_json(
            "http://169.254.169.254/latest/meta-data",
            {"value": "blocked"},
            transport=httpx.MockTransport(handler),
        )
    assert calls == 0


def test_sync_json_post_rejects_redirect_and_oversized_response():
    import httpx

    from agent_utilities.protocols.source_connectors.http_safety import (
        SourceEgressError,
        safe_post_json,
    )

    redirect = httpx.MockTransport(
        lambda request: httpx.Response(
            307,
            headers={"location": "https://other.example.invalid/submit"},
            request=request,
        )
    )
    with pytest.raises(SourceEgressError, match="redirect"):
        safe_post_json(
            "https://public.example.invalid/submit",
            {"value": "bounded"},
            transport=redirect,
        )

    oversized = httpx.MockTransport(
        lambda request: httpx.Response(200, content=b"x" * 32, request=request)
    )
    with pytest.raises(SourceEgressError, match="configured limit"):
        safe_post_json(
            "https://public.example.invalid/submit",
            {"value": "bounded"},
            max_bytes=8,
            transport=oversized,
        )


def test_sync_json_post_timeout_is_metadata_only():
    import httpx

    from agent_utilities.protocols.source_connectors.http_safety import (
        SourceEgressError,
        safe_post_json,
    )

    def handler(request):
        raise httpx.ReadTimeout(
            "secret endpoint https://private.example.invalid/?token=value",
            request=request,
        )

    with pytest.raises(SourceEgressError) as caught:
        safe_post_json(
            "https://public.example.invalid/submit",
            {"value": "bounded"},
            timeout=1,
            transport=httpx.MockTransport(handler),
        )
    assert str(caught.value) == "Source request failed"
    assert caught.value.__cause__ is None


def test_sync_json_post_serializes_bounded_json_and_accepts_empty_response():
    import httpx

    from agent_utilities.protocols.source_connectors.http_safety import safe_post_json

    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(204, request=request)

    result = safe_post_json(
        "https://public.example.invalid/submit",
        {"value": "bounded"},
        transport=httpx.MockTransport(handler),
    )
    assert result == {}
    assert requests[0].method == "POST"
    assert requests[0].headers["content-type"] == "application/json"
    assert requests[0].content == b'{"value":"bounded"}'


def test_sync_json_post_dns_pins_host_and_uses_named_tls_profile(monkeypatch):
    import httpx

    import agent_utilities.protocols.source_connectors.http_safety as http_safety

    requests = []
    client_kwargs = []
    selected_services = []

    class Trust:
        proxy_url = None

        @staticmethod
        def httpx_kwargs():
            return {"verify": True, "trust_env": True}

        @staticmethod
        def cleanup():
            return None

    class PinnedStream:
        @staticmethod
        def get_extra_info(name):
            return ("203.0.113.9", 443) if name == "server_addr" else None

    transport = httpx.MockTransport(
        lambda request: (
            requests.append(request)
            or httpx.Response(
                200,
                json={"ok": True},
                extensions={"network_stream": PinnedStream()},
                request=request,
            )
        )
    )

    def resolve_profile(service):
        selected_services.append(service)
        return Trust()

    def create_client(**kwargs):
        client_kwargs.append(kwargs)
        return httpx.Client(transport=transport)

    monkeypatch.setattr(
        "agent_utilities.core.transport_security.resolve_configured_tls_profile",
        resolve_profile,
    )
    monkeypatch.setattr(
        "agent_utilities.core.http_client.create_http_client", create_client
    )
    monkeypatch.setattr(
        http_safety,
        "resolve_safe_source_url",
        lambda url, **_kwargs: http_safety.ResolvedSourceURL(
            url=url,
            host="public.example.invalid",
            scheme="https",
            resolved_ips=("203.0.113.9",),
        ),
    )

    result = http_safety.safe_post_json(
        "https://public.example.invalid/submit",
        {"value": "bounded"},
        tls_service="notification",
    )

    assert result == {"ok": True}
    assert selected_services == ["notification"]
    assert client_kwargs[0]["follow_redirects"] is False
    assert client_kwargs[0]["trust_env"] is False
    assert requests[0].url.host == "203.0.113.9"
    assert requests[0].headers["host"] == "public.example.invalid"
    assert requests[0].extensions["sni_hostname"] == "public.example.invalid"


def test_filesystem_connector_skips_symlink_escape(tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.md"
    outside.write_text("outside content")
    link = tmp_path / "escape.md"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable on this filesystem")

    conn = build_connector("filesystem", {"root": str(tmp_path)})
    assert list(conn.load()) == []


@pytest.mark.concept("AU-ECO.connector.external-permission-sync")
def test_permission_sync_maps_acl_and_markings():
    access = ExternalAccess(
        is_public=False,
        group_ids=["eng"],
        user_emails=["a@x"],
        read_roles=["kg:read"],
        markings=["SECRET"],
    )
    acl = sync_access("doc:1", access, [("doc:1", "doc:1::chunk::0")])
    assert acl is not None
    assert "group:eng" in acl.read_roles and "user:a@x" in acl.read_roles
    assert "kg:read" in acl.read_roles

    from agent_utilities.knowledge_graph.ontology.permissioning import (
        get_company_brain,
        markings_for,
    )

    assert "SECRET" in markings_for("doc:1")
    assert "SECRET" in markings_for("doc:1::chunk::0")  # propagated to chunk
    chunk_acl = get_company_brain().permissions.get_acl("doc:1::chunk::0")
    assert chunk_acl is not None
    assert chunk_acl.read_roles == acl.read_roles

    # Public access is a positive source assertion represented by an explicit
    # public ACL, never by an absent policy row.
    public = sync_access("doc:2", ExternalAccess.public(), [])
    assert public is not None
    assert public.classification.value == "public"

    # Non-public with no principals is deny-all, not default-allow.
    deny_all = sync_access("doc:3", ExternalAccess(is_public=False), [])
    assert deny_all is not None
    assert deny_all.read_roles == []


@pytest.mark.concept("AU-ECO.connector.external-permission-sync")
def test_permission_sync_registers_explicit_platform_and_public_acls():
    internal = sync_access(
        "trace:1",
        ExternalAccess(is_public=False, read_roles=["kg:read", "kg:admin"]),
    )
    assert internal is not None
    assert internal.read_roles == ["kg:read", "kg:admin"]
    assert internal.classification.value == "internal"

    public = sync_access("trace:2", ExternalAccess.public())
    assert public is not None
    assert public.read_roles == []
    assert public.classification.value == "public"


@pytest.mark.concept("AU-ECO.connector.document-source-framework")
def test_source_document_shape():
    doc = SourceDocument(id="1", text="hello", title="T")
    assert doc.doc_type == "document"
    assert doc.external_access is None


# ── Native RSS/Atom connector (CONCEPT:AU-KG.ingest.rss-feed-connector) ─────────────────────────────

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


@pytest.mark.concept("AU-KG.ingest.rss-feed-connector")
def test_rss_connector_registered():
    assert "rss" in set(list_sources())


@pytest.mark.concept("AU-KG.ingest.rss-feed-connector")
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


@pytest.mark.concept("AU-KG.ingest.rss-feed-connector")
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


@pytest.mark.concept("AU-KG.ingest.rss-feed-connector")
def test_rss_connector_dead_feed_is_skipped():
    def _boom(url):
        raise RuntimeError("dns fail")

    conn = build_connector("rss", {"feed_urls": "http://dead/", "fetch_fn": _boom})
    assert list(conn.load()) == []  # a dead feed never aborts


@pytest.mark.concept("AU-KG.ingest.rss-feed-connector")
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
