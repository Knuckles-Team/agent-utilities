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
def test_mcp_package_connector_defaults_to_quarantined_not_public():
    """This connector has no ACL surface at all -> fail-closed default (CONCEPT:AU-P0-4).

    Unknown/unconfigured access must never silently default to world-public.
    """

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


# ── _run_async hard deadline (CONCEPT:AU-ORCH.scheduling.hard-io-deadline) ───────────


@pytest.mark.concept("AU-ORCH.scheduling.hard-io-deadline")
def test_run_async_timeout_frees_the_caller_without_joining_the_stuck_coroutine():
    """Cooperative cancellation cannot stop an uncooperative synchronous/blocked
    call (the proven root cause of a wedged task-queue worker,
    CONCEPT:AU-ORCH.scheduling.acquisition-lane-fairness). ``_run_async``'s
    ``timeout`` bounds how long the CALLER waits, not how long the coroutine is
    allowed to run: the caller must be released on schedule even though the
    coroutine itself never finishes in time, proving the calling worker thread
    is genuinely freed rather than merely asked (and possibly ignored) to stop.
    """
    import asyncio
    import concurrent.futures
    import time

    from agent_utilities.protocols.source_connectors.connectors.mcp_package import (
        _run_async,
    )

    async def _slow_uncooperative_call() -> str:
        # Deliberately longer than the timeout below, standing in for a hung
        # MCP server / subprocess spawn with no cooperative cancellation point
        # the caller can reach in time.
        await asyncio.sleep(2.0)
        return "should never be observed by the caller"

    async def _caller() -> float:
        # Exercise the "called from inside a running loop" branch — the real
        # call site (an async task body calling into _sync_fleet) always has
        # one; this is what makes _run_async spawn the abandoned worker thread
        # instead of running the coroutine on the current thread directly.
        start = time.monotonic()
        with pytest.raises(concurrent.futures.TimeoutError):
            _run_async(_slow_uncooperative_call(), timeout=0.2)
        return time.monotonic() - start

    elapsed = asyncio.run(_caller())
    # Freed close to the requested deadline, nowhere near the coroutine's own
    # 2s runtime — the caller did not wait for (or join) the abandoned thread.
    assert elapsed < 1.0


@pytest.mark.concept("AU-ORCH.scheduling.hard-io-deadline")
def test_run_async_without_timeout_preserves_prior_blocking_behavior():
    """No ``timeout`` given is a purely additive default: identical to the
    pre-fix behavior of waiting for the coroutine to finish."""
    import asyncio

    from agent_utilities.protocols.source_connectors.connectors.mcp_package import (
        _run_async,
    )

    async def _quick() -> str:
        await asyncio.sleep(0.01)
        return "done"

    async def _caller() -> str:
        return _run_async(_quick())

    assert asyncio.run(_caller()) == "done"
