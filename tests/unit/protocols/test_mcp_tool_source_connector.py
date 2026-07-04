"""Tests for the universal MCP-tool ingestion source (CONCEPT:AU-KG.ingest.mcp-tool-connector).

Offline + deterministic: the MCP transport is an in-process FastMCP server
handed to the connector via the injected ``client`` target — the same
in-memory client pattern the fleet repos use, with canned sql-mcp /
objectstore-mcp envelopes. No package import of any fleet repo, no processes.
"""

from __future__ import annotations

import json

import pytest
from fastmcp import FastMCP

from agent_utilities.protocols.source_connectors import (
    PollConnector,
    build_connector,
    list_sources,
)
from agent_utilities.protocols.source_connectors.connectors.mcp_tool import (
    MCP_TOOL_PRESETS,
    McpToolSourceError,
    get_tool_preset,
    list_tool_presets,
)

# ── canned fleet servers ─────────────────────────────────────────────────────


def make_sql_server(rows: list[dict] | None = None, page_size: int = 2) -> FastMCP:
    """A fake sql-mcp: keyset-paginated sql_query + sql_schema columns."""
    table = (
        rows
        if rows is not None
        else [
            {
                "id": 1,
                "title": "Alpha",
                "body": "alpha body",
                "updated_at": "2026-01-01",
            },
            {"id": 2, "title": "Beta", "body": "beta body", "updated_at": "2026-02-01"},
            {
                "id": 3,
                "title": "Gamma",
                "body": "gamma body",
                "updated_at": "2026-03-01",
            },
        ]
    )
    server = FastMCP("fake-sql-mcp")

    @server.tool
    def sql_query(action: str, params_json: str = "{}", connection: str = "") -> dict:
        assert action == "execute"
        p = json.loads(params_json)
        bound = p.get("params") or {}
        after = bound.get("after", 0)
        since = bound.get("since")
        matched = [r for r in table if r["id"] > after]
        if since is not None:
            matched = [r for r in matched if r["updated_at"] > since]
        cap = int(p.get("max_rows", page_size))
        page = matched[:cap]
        cols = ["id", "title", "body", "updated_at"]
        return {
            "columns": cols,
            "rows": [[r[c] for c in cols] for r in page],
            "row_count": len(page),
            "truncated": len(matched) > len(page),
        }

    @server.tool
    def sql_schema(action: str, params_json: str = "{}", connection: str = "") -> list:
        assert action == "columns"
        assert json.loads(params_json)["table"] == "articles"
        return [
            {"name": "id", "type": "INTEGER", "primary_key": True},
            {"name": "title", "type": "TEXT"},
            {"name": "body", "type": "TEXT"},
            {"name": "updated_at", "type": "TEXT"},
        ]

    return server


def make_objectstore_server() -> FastMCP:
    """A fake objectstore-mcp: paginated objects list + text-mode get."""
    objects = {
        "kb/a.md": "# A\nalpha object content",
        "kb/b.md": "# B\nbeta object content",
        "kb/c.bin": None,  # not valid text → get fails
    }
    server = FastMCP("fake-objectstore-mcp")

    @server.tool
    def objects_tool(action: str, params_json: str = "{}") -> dict:
        p = json.loads(params_json)
        keys = sorted(objects)
        if action == "list":
            start = keys.index(p["token"]) + 1 if p.get("token") else 0
            max_keys = int(p.get("max_keys", 2))
            page = keys[start : start + max_keys]
            truncated = (start + max_keys) < len(keys)
            return {
                "bucket": p["bucket"],
                "objects": [
                    {"key": k, "size": 10, "last_modified": "2026-01-01T00:00:00Z"}
                    for k in page
                ],
                "prefixes": [],
                "next_token": page[-1] if truncated and page else None,
                "truncated": truncated,
            }
        if action == "get":
            body = objects[p["key"]]
            if body is None:
                raise ValueError("Object is not valid UTF-8 text; use mode='base64'.")
            return {
                "bucket": p["bucket"],
                "key": p["key"],
                "size": len(body),
                "encoding": "text",
                "content": body,
            }
        raise ValueError(f"unknown action {action!r}")

    # The real fleet tool is named ``objects``; FastMCP forbids shadowing the
    # local function name, so register under the fleet-visible name explicitly.
    server.tool(objects_tool, name="objects")
    return server


# ── registration + config validation ─────────────────────────────────────────


@pytest.mark.concept("AU-KG.ingest.mcp-tool-connector")
def test_mcp_tool_source_registered():
    assert "mcp_tool" in list_sources()


@pytest.mark.concept("AU-KG.ingest.mcp-tool-connector")
def test_config_validation():
    with pytest.raises(ValueError, match="tool"):
        build_connector("mcp_tool", {"server": "x"})
    with pytest.raises(ValueError, match="transport"):
        build_connector("mcp_tool", {"tool": "t"})
    with pytest.raises(ValueError, match="pagination"):
        build_connector(
            "mcp_tool", {"tool": "t", "server": "x", "pagination": "scroll"}
        )
    with pytest.raises(ValueError, match="preset"):
        build_connector("mcp_tool", {"preset": "nope"})


@pytest.mark.concept("AU-KG.ingest.mcp-tool-connector")
def test_presets_are_data():
    assert {"sql-table", "sql-query", "objectstore-prefix", "servicenow-table"} <= set(
        list_tool_presets()
    )
    # presets are partial configs: plain JSON-serializable data, no callables
    for name, preset in MCP_TOOL_PRESETS.items():
        json.dumps(preset)
        assert get_tool_preset(name) == preset


@pytest.mark.concept("AU-KG.ingest.mcp-tool-connector")
def test_enterprise_presets_build_with_verified_shape():
    # github-mcp + okta-mcp: source-control and identity, grounded in each
    # agent's actual list tool + response envelope (not guessed).
    assert {"github-repos", "okta-users", "keycloak-users"} <= set(list_tool_presets())

    gh = build_connector("mcp_tool", {"preset": "github-repos"})
    assert gh.server == "github-mcp"
    assert gh.tool == "github_repos"
    assert gh.action == "list"
    assert gh.records_path == "data"
    assert (gh.id_field, gh.title_field, gh.text_field, gh.updated_field) == (
        "id",
        "full_name",
        "description",
        "updated_at",
    )

    okta = build_connector("mcp_tool", {"preset": "okta-users"})
    assert okta.server == "okta-mcp"
    assert okta.tool == "okta_users"
    assert okta.records_path == "data"
    # login/email live under the nested ``profile`` object (dotted field maps).
    assert okta.title_field == "profile.login"
    assert okta.text_field == "profile.email"

    kc = build_connector("mcp_tool", {"preset": "keycloak-users"})
    assert kc.server == "keycloak-mcp"
    assert kc.tool == "keycloak_agent_users"
    assert kc.action == "list_users"
    # Keycloak returns a bare user array → records_path stays "" (whole result).
    assert kc.records_path == ""
    assert (kc.id_field, kc.title_field, kc.text_field) == ("id", "username", "email")


@pytest.mark.concept("AU-ECO.connector.mcp-tool-connector")
def test_pulselink_presets_build_with_uniform_shape():
    # PulseLink open-web/social sources: every source is the SAME flat field map
    # over pulse_search/pulse_list's {documents:[...], next_cursor} envelope.
    pl = {p for p in list_tool_presets() if p.startswith("pulselink-")}
    assert {
        "pulselink-x",
        "pulselink-reddit",
        "pulselink-youtube",
        "pulselink-hackernews",
        "pulselink-rss",
    } <= pl

    x = build_connector("mcp_tool", {"preset": "pulselink-x"})
    assert x.server == "pulselink-mcp"
    assert x.tool == "pulse_search"
    assert x.records_path == "documents"
    assert (x.id_field, x.title_field, x.text_field, x.updated_field) == (
        "id",
        "title",
        "text",
        "created_at",
    )
    assert x.cursor_path == "next_cursor"

    # RSS is a list-style source over pulse_list.
    rss = build_connector("mcp_tool", {"preset": "pulselink-rss"})
    assert rss.tool == "pulse_list"

    # YouTube carries a detail phase that fetches the transcript per video.
    yt = MCP_TOOL_PRESETS["pulselink-youtube"]
    assert yt["detail"]["tool"] == "pulse_fetch"
    assert yt["detail"]["params"]["target"] == "{id}"


# ── sql-mcp shapes ───────────────────────────────────────────────────────────


@pytest.mark.concept("AU-KG.ingest.mcp-tool-connector")
def test_sql_query_keyset_sweep_zips_columns_rows():
    conn = build_connector(
        "mcp_tool",
        {
            "preset": "sql-query",
            "client": make_sql_server(),
            "params": {
                "sql": "SELECT id, title, body, updated_at FROM articles "
                "WHERE id > :after ORDER BY id",
                "params": {"after": 0},
                "max_rows": 2,
            },
            "cursor_record_field": "id",
            "text_field": "body",
            "updated_field": "updated_at",
        },
    )
    docs = list(conn.load())
    assert [d.id for d in docs] == ["1", "2", "3"]  # 2 pages: keyset advanced
    assert docs[0].title == "Alpha"
    assert docs[0].text == "alpha body"
    assert docs[0].updated_at == "2026-01-01"
    assert docs[0].metadata["record"]["title"] == "Alpha"
    assert "body" not in docs[0].metadata["record"]  # text not duplicated


@pytest.mark.concept("AU-KG.ingest.mcp-tool-connector")
def test_sql_table_bootstrap_discovers_columns_and_polls_incrementally():
    conn = build_connector(
        "mcp_tool",
        {
            "preset": "sql-table",
            "client": make_sql_server(),
            "sql_table": {
                "table": "articles",
                "key_column": "id",
                "text_column": "body",
                "title_column": "title",
                "updated_column": "updated_at",
                "page_size": 2,
            },
        },
    )
    assert isinstance(conn, PollConnector)
    docs = list(conn.poll_all())
    assert [d.id for d in docs] == ["1", "2", "3"]
    cp = conn.last_checkpoint
    assert cp is not None and not cp.has_more
    assert cp.watermark == "2026-03-01"
    # re-poll with the watermark → server-side delta returns nothing new
    again = conn.poll(cp)
    assert again.documents == []
    assert again.checkpoint.watermark == "2026-03-01"


@pytest.mark.concept("AU-KG.ingest.mcp-tool-connector")
def test_sql_table_rejects_unsafe_identifiers():
    with pytest.raises(ValueError, match="identifier"):
        conn = build_connector(
            "mcp_tool",
            {
                "preset": "sql-table",
                "client": make_sql_server(),
                "sql_table": {"table": "articles; DROP", "text_column": "body"},
            },
        )
        list(conn.load())


# ── objectstore shapes ───────────────────────────────────────────────────────


@pytest.mark.concept("AU-KG.ingest.mcp-tool-connector")
def test_objectstore_prefix_sweep_list_get_detail():
    conn = build_connector(
        "mcp_tool",
        {
            "preset": "objectstore-prefix",
            "client": make_objectstore_server(),
            "params": {"bucket": "docs", "prefix": "kb/", "max_keys": 2},
        },
    )
    docs = list(conn.load())
    # 3 keys over 2 pages; the binary object fails text-mode get and is skipped
    assert [d.id for d in docs] == ["kb/a.md", "kb/b.md"]
    assert docs[0].text.startswith("# A")
    assert docs[0].doc_type == "file"
    assert docs[0].updated_at == "2026-01-01T00:00:00Z"


@pytest.mark.concept("AU-KG.ingest.mcp-tool-connector")
def test_poll_resumes_cursor_state_across_batches():
    conn = build_connector(
        "mcp_tool",
        {
            "preset": "objectstore-prefix",
            "client": make_objectstore_server(),
            "params": {"bucket": "docs", "max_keys": 1},
            "batch_size": 1,
            "detail": None,
            "text_field": "key",  # use the key itself as content (no detail)
        },
    )
    b1 = conn.poll()
    assert [d.id for d in b1.documents] == ["kb/a.md"]
    assert b1.checkpoint.has_more
    b2 = conn.poll(b1.checkpoint)
    assert [d.id for d in b2.documents] == ["kb/b.md"]
    b3 = conn.poll(b2.checkpoint)
    assert [d.id for d in b3.documents] == ["kb/c.bin"]
    assert not b3.checkpoint.has_more
    # the watermark only lands once the sweep exhausts
    assert b3.checkpoint.watermark == "2026-01-01T00:00:00Z"
    assert b1.checkpoint.watermark is None


# ── generic mapping behaviors ────────────────────────────────────────────────


@pytest.mark.concept("AU-KG.ingest.mcp-tool-connector")
def test_acl_fields_map_to_external_access():
    server = FastMCP("acl-server")

    @server.tool
    def records(action: str, params_json: str = "{}") -> dict:
        return {
            "items": [
                {
                    "id": "r1",
                    "text": "restricted",
                    "owners": "a@x, b@x",
                    "teams": ["eng"],
                    "labels": ["SECRET"],
                },
                {"id": "r2", "text": "open", "owners": "", "teams": [], "labels": []},
            ]
        }

    conn = build_connector(
        "mcp_tool",
        {
            "client": server,
            "tool": "records",
            "action": "list",
            "records_path": "items",
            "acl_users_field": "owners",
            "acl_groups_field": "teams",
            "acl_markings_field": "labels",
        },
    )
    docs = {d.id: d for d in conn.load()}
    assert docs["r1"].external_access is not None
    assert docs["r1"].external_access.is_public is False
    assert docs["r1"].external_access.user_emails == ["a@x", "b@x"]
    assert docs["r1"].external_access.group_ids == ["eng"]
    assert docs["r1"].external_access.markings == ["SECRET"]
    assert docs["r2"].external_access.is_public is True


@pytest.mark.concept("AU-KG.ingest.mcp-tool-connector")
def test_page_pagination_offset_exhaustion_and_max_records():
    server = FastMCP("paged")
    all_rows = [{"id": str(i), "text": f"row {i}"} for i in range(7)]

    @server.tool
    def listing(action: str, params_json: str = "{}") -> dict:
        p = json.loads(params_json)
        off, lim = int(p.get("offset", 0)), int(p.get("limit", 3))
        return {"result": all_rows[off : off + lim]}

    config = {
        "client": server,
        "tool": "listing",
        "action": "page",
        "records_path": "result",
        "pagination": "page",
        "page_kind": "offset",
        "page_param": "offset",
        "page_size_param": "limit",
        "page_size": 3,
    }
    docs = list(build_connector("mcp_tool", config).load())
    assert [d.id for d in docs] == [str(i) for i in range(7)]  # 3 pages, short last
    capped = list(build_connector("mcp_tool", {**config, "max_records": 4}).load())
    assert len(capped) == 4


@pytest.mark.concept("AU-KG.ingest.mcp-tool-connector")
def test_args_params_style_for_non_fleet_servers():
    server = FastMCP("flat")

    @server.tool
    def search(query: str, limit: int = 10) -> list:
        assert query == "graphs" and limit == 5
        return [{"id": "x", "text": "found"}]

    conn = build_connector(
        "mcp_tool",
        {
            "client": server,
            "tool": "search",
            "params_style": "args",
            "params": {"query": "graphs", "limit": 5},
        },
    )
    assert [d.id for d in conn.load()] == ["x"]


@pytest.mark.concept("AU-KG.ingest.mcp-tool-connector")
def test_tool_failure_raises_typed_error():
    server = FastMCP("broken")

    @server.tool
    def boom(action: str = "", params_json: str = "{}") -> dict:
        raise RuntimeError("backend down")

    conn = build_connector("mcp_tool", {"client": server, "tool": "boom"})
    with pytest.raises(McpToolSourceError, match="boom"):
        list(conn.load())


@pytest.mark.concept("AU-KG.ingest.mcp-tool-connector")
def test_missing_server_in_mcp_config_is_typed(monkeypatch, tmp_path):
    cfg = tmp_path / "mcp_config.json"
    cfg.write_text(json.dumps({"mcpServers": {}}))
    monkeypatch.setenv("MCP_CONFIG_PATH", str(cfg))
    conn = build_connector("mcp_tool", {"server": "ghost", "tool": "t"})
    with pytest.raises(McpToolSourceError, match="ghost"):
        list(conn.load())


# ── mealie recipes doc-preset (CONCEPT:AU-KG.ingest.mcp-tool-connector) ───────────────────────────────


def make_mealie_server(page_size: int = 2) -> FastMCP:
    """A fake mealie-mcp: action-routed mealie_recipes with Mealie's page envelope."""
    recipes = [
        {"id": "1", "slug": "soup", "name": "Soup", "description": "warm soup"},
        {"id": "2", "slug": "salad", "name": "Salad", "description": "cold salad"},
        {"id": "3", "slug": "stew", "name": "Stew", "description": "hearty stew"},
    ]
    server = FastMCP("fake-mealie-mcp")

    @server.tool
    def mealie_recipes(action: str, params_json: str = "{}") -> dict:
        assert action == "get_recipes"
        p = json.loads(params_json)
        page = int(p.get("page", 1))
        per = int(p.get("per_page", page_size))
        start = (page - 1) * per
        items = recipes[start : start + per]
        return {"items": items, "page": page, "per_page": per, "total": len(recipes)}

    return server


@pytest.mark.concept("AU-KG.ingest.mcp-tool-connector")
def test_mealie_preset_builds_with_verified_shape():
    assert "mealie-recipes" in list_tool_presets()
    conn = build_connector("mcp_tool", {"preset": "mealie-recipes"})
    assert conn.server == "mealie-mcp"
    assert conn.tool == "mealie_recipes"
    assert conn.action == "get_recipes"
    assert conn.params_style == "json"
    assert conn.records_path == "items"
    assert (conn.id_field, conn.title_field, conn.text_field) == (
        "slug",
        "name",
        "description",
    )


@pytest.mark.concept("AU-KG.ingest.mcp-tool-connector")
def test_mealie_recipes_sweep_paginates_pages():
    conn = build_connector(
        "mcp_tool",
        {
            "preset": "mealie-recipes",
            "client": make_mealie_server(page_size=2),
            "params": {"per_page": 2},
            "page_size": 2,
        },
    )
    docs = list(conn.load())
    # 3 recipes across 2 pages (number-based, start_page=1), descriptions as text
    assert [d.id for d in docs] == ["soup", "salad", "stew"]
    assert docs[0].text == "warm soup"
    assert docs[0].doc_type == "record"


# ── connector dual-role presets + per-repo contribution seam (KG-2.59) ───────


def test_connector_presets_well_formed_and_buildable():
    """Every shipped preset has a transport + tool + field map and builds clean."""
    from agent_utilities.protocols.source_connectors.connectors.mcp_tool import (
        all_tool_presets,
    )

    presets = all_tool_presets()
    # The new connector dual-role presets are present and discoverable.
    for name in (
        "jellyfin-media",
        "rom-manager-roms",
        "listmonk-subscribers",
        "ansible-tower-inventories",
        "camunda-tasks",
        "salesforce-sobject",
        "erpnext-doctype",
    ):
        assert name in presets, f"{name} missing from catalog"
        assert name in list_tool_presets()

    for name, preset in presets.items():
        has_sql = "sql_table" in preset
        assert preset.get("server") or has_sql, f"{name} has no server/sql_table"
        assert preset.get("tool") or has_sql, f"{name} has no tool/sql_table"
        # A document needs both an id anchor and a text body (explicit or default).
        assert preset.get("id_field", "id"), name
        assert preset.get("text_field", "text"), name
        # Building the connector binds the preset without raising (server transport).
        conn = build_connector("mcp_tool", {"preset": name})
        assert conn.health_check(), name


def test_contributed_preset_seam_precedence_and_reset():
    """A per-repo contributed preset is discoverable and overrides the central dict."""
    from agent_utilities.protocols.source_connectors.connectors import mcp_tool as M

    M.reset_contributed_presets_cache()
    try:
        # Simulate a connector that ships its own preset beside its MCP tool.
        M._contributed_presets_cache = {
            "acme-widgets": {
                "server": "acme-mcp",
                "tool": "acme_list",
                "action": "list",
                "text_field": "body",
            },
            # Same name as a central preset → contributed wins (lives w/ connector).
            "jellyfin-media": {"server": "override-mcp", "tool": "x"},
        }
        assert "acme-widgets" in M.list_tool_presets()
        assert M.get_tool_preset("acme-widgets")["server"] == "acme-mcp"
        assert M.get_tool_preset("jellyfin-media")["server"] == "override-mcp"
    finally:
        M.reset_contributed_presets_cache()
    # After reset the central preset is authoritative again.
    assert M.get_tool_preset("jellyfin-media")["server"] == "jellyfin-mcp"
