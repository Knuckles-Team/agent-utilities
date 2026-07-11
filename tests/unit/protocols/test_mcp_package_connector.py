"""Tests for the MCP agent-package fleet connector + Onyx parity (CONCEPT:AU-ECO.connector.mcp-package-adapter).

Offline: the MCP transport is an injected ``call_tool`` callable, so no package
servers are spawned.
"""

from __future__ import annotations

import pytest

from agent_utilities.protocols.source_connectors import build_connector
from agent_utilities.protocols.source_connectors.connectors.package_manifest import (
    get_preset,
    list_presets,
    onyx_parity,
    onyx_parity_summary,
)


@pytest.mark.concept("AU-ECO.connector.mcp-package-adapter")
def test_mcp_connector_maps_tool_result_to_documents():
    def fake_call(tool, args):
        assert tool == "search_papers"
        assert args.get("query") == "graphs"
        return {
            "papers": [
                {
                    "id": "p1",
                    "title": "Graph RAG",
                    "abstract": "we study graphs",
                    "published": "2026-01",
                },
                {
                    "id": "p2",
                    "title": "OWL",
                    "abstract": "ontologies",
                    "published": "2026-02",
                },
            ]
        }

    conn = build_connector(
        "mcp", {"package": "scholarx", "query": "graphs", "call_tool": fake_call}
    )
    docs = list(conn.load())
    assert [d.id for d in docs] == ["p1", "p2"]
    assert docs[0].doc_type == "paper"  # from the preset
    assert conn.name == "mcp:scholarx"


@pytest.mark.concept("AU-ECO.connector.mcp-package-adapter")
def test_mcp_connector_requires_tool_without_preset():
    with pytest.raises(ValueError):
        build_connector("mcp", {"package": "unknown-pkg"})


@pytest.mark.concept("AU-P0-4")
def test_mcp_package_connector_defaults_to_quarantined_not_public(monkeypatch):
    """This connector has no ACL surface at all -> fail-closed default (CONCEPT:AU-P0-4).

    Unknown/unconfigured access must never silently default to world-public.
    """
    monkeypatch.delenv("CONNECTOR_DEFAULT_PUBLIC", raising=False)

    def fake_call(tool, args):
        return {"items": [{"id": "p1", "title": "T", "text": "body"}]}

    conn = build_connector(
        "mcp",
        {
            "server": "x-mcp",
            "tool": "list",
            "records_field": "items",
            "call_tool": fake_call,
        },
    )
    docs = list(conn.load())
    assert len(docs) == 1
    access = docs[0].external_access
    assert access is not None
    assert access.is_public is False
    from agent_utilities.protocols.source_connectors.base import (
        CONNECTOR_UNCONFIGURED_MARKING,
    )

    assert CONNECTOR_UNCONFIGURED_MARKING in access.markings


@pytest.mark.concept("AU-P0-4")
def test_mcp_package_connector_default_public_opt_in(monkeypatch):
    """``CONNECTOR_DEFAULT_PUBLIC=true`` restores the legacy public-by-default behavior."""
    monkeypatch.setenv("CONNECTOR_DEFAULT_PUBLIC", "true")

    def fake_call(tool, args):
        return {"items": [{"id": "p1", "title": "T", "text": "body"}]}

    conn = build_connector(
        "mcp",
        {
            "server": "x-mcp",
            "tool": "list",
            "records_field": "items",
            "call_tool": fake_call,
        },
    )
    docs = list(conn.load())
    assert docs[0].external_access.is_public is True


@pytest.mark.concept("AU-ECO.connector.mcp-package-adapter")
def test_mcp_connector_poll_dedup_cursorless():
    def fake_call(tool, args):
        return {"items": [{"id": "1", "text": "a"}, {"id": "2", "text": "b"}]}

    conn = build_connector(
        "mcp",
        {
            "server": "x-mcp",
            "tool": "list",
            "records_field": "items",
            "call_tool": fake_call,
        },
    )
    b1 = conn.poll()
    assert {d.id for d in b1.documents} == {"1", "2"}
    b2 = conn.poll(b1.checkpoint)  # same ids → deduped to nothing
    assert b2.documents == []


@pytest.mark.concept("AU-ECO.connector.mcp-package-adapter")
def test_presets_present():
    presets = list_presets()
    assert "scholarx" in presets and "github-agent" in presets
    assert get_preset("github")["tool"]  # short alias resolves


@pytest.mark.concept("AU-ECO.connector.mcp-package-adapter")
def test_onyx_parity_catalog_covers_sources():
    assert onyx_parity("github")["via"] == "native"
    assert onyx_parity("github")["package"] == "github-agent"
    assert onyx_parity("notion")["route"] == "rest"
    assert onyx_parity("file")["route"] == "filesystem"
    assert onyx_parity("wikipedia")["route"] == "web"

    summary = onyx_parity_summary()
    assert summary["onyx_sources_mapped"] >= 40
    assert summary["by_route"]["native"] >= 10
    # every mapped source resolves to a real connector family
    families = {"mcp", "rest", "web", "filesystem", "database"}
    from agent_utilities.protocols.source_connectors.connectors.package_manifest import (
        ONYX_CONNECTOR_PARITY,
    )

    assert all(s["route"] in families for s in ONYX_CONNECTOR_PARITY.values())
