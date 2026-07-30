"""Skills-over-MCP fleet discovery (CONCEPT:AU-ECO.mcp.skills-over-mcp-provider).

Covers the client-side half of Skills-over-MCP: probing a fleet server's
``skill://{name}/SKILL.md`` Resources alongside its Tools, bounding that
catalog the same way the tool catalog is bounded, and ranking skills and
tools together in one ``find_tools`` result set — the unified capability
space (CONCEPT:AU-KG.retrieval.unified-capability-contract).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from agent_utilities.mcp.multiplexer import (
    _bounded_skill_catalog,
)
from tests.test_multiplexer_dynamic_gateway import (
    CNT,
    CNT_TOOL,
    _fake_tool,
    _mux_with_children,
)


def _fake_skill_resource(uri: str, description: str = ""):
    resource = MagicMock()
    resource.uri = uri
    resource.description = description
    return resource


def _fake_session_with_resources(tools, resources):
    sess = AsyncMock()
    tools_result = MagicMock()
    tools_result.tools = [_fake_tool(n, d) for n, d in tools]
    sess.list_tools = AsyncMock(return_value=tools_result)
    resources_result = MagicMock()
    resources_result.resources = resources
    sess.list_resources = AsyncMock(return_value=resources_result)
    return sess


# --------------------------------------------------------------------------- #
# _bounded_skill_catalog
# --------------------------------------------------------------------------- #


def test_bounded_skill_catalog_extracts_only_skill_resources():
    resources = [
        _fake_skill_resource("skill://release-notes/SKILL.md", "draft notes"),
        _fake_skill_resource("skill://release-notes/_manifest", "manifest, ignored"),
        _fake_skill_resource("docs://readme", "unrelated resource, ignored"),
    ]
    skills = _bounded_skill_catalog(resources)
    assert skills == [
        {
            "name": "release-notes",
            "uri": "skill://release-notes/SKILL.md",
            "description": "draft notes",
        }
    ]


def test_bounded_skill_catalog_rejects_non_list_input():
    import pytest

    with pytest.raises(RuntimeError):
        _bounded_skill_catalog("not-a-list")


def test_bounded_skill_catalog_empty_for_no_skill_resources():
    resources = [_fake_skill_resource("docs://readme", "unrelated")]
    assert _bounded_skill_catalog(resources) == []


# --------------------------------------------------------------------------- #
# probe_server: skill resources alongside tools
# --------------------------------------------------------------------------- #


async def test_probe_server_captures_skill_resources_alongside_tools(tmp_path):
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "manage containers")]})

    async def _open(server, cfg, stack):
        return _fake_session_with_resources(
            [(CNT_TOOL, "manage containers")],
            [_fake_skill_resource("skill://onboarding/SKILL.md", "onboard a user")],
        )

    mux._open_one_session = AsyncMock(side_effect=_open)  # type: ignore[method-assign]
    info = await mux.probe_server(CNT)

    assert info["error"] is None
    assert info["tools"][0]["name"] == CNT_TOOL
    assert info["skills"] == [
        {
            "name": "onboarding",
            "uri": "skill://onboarding/SKILL.md",
            "description": "onboard a user",
        }
    ]


async def test_probe_server_degrades_when_list_resources_unsupported(tmp_path):
    """A server (or an mcp SDK build) with no ``resources/list`` support must
    still yield its tools — Skills-over-MCP is optional, never load-bearing."""
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "manage containers")]})

    async def _open(server, cfg, stack):
        sess = AsyncMock()
        tools_result = MagicMock()
        tools_result.tools = [_fake_tool(CNT_TOOL, "manage containers")]
        sess.list_tools = AsyncMock(return_value=tools_result)
        sess.list_resources = AsyncMock(side_effect=RuntimeError("no such method"))
        return sess

    mux._open_one_session = AsyncMock(side_effect=_open)  # type: ignore[method-assign]
    info = await mux.probe_server(CNT)

    assert info["error"] is None
    assert info["tools"][0]["name"] == CNT_TOOL
    assert info["skills"] == []


async def test_probe_server_degrades_on_malformed_skill_catalog(tmp_path):
    """A malformed resource catalog must not fail the (already-succeeded) tool
    probe — best-effort skills, load-bearing tools."""
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "manage containers")]})

    async def _open(server, cfg, stack):
        sess = AsyncMock()
        tools_result = MagicMock()
        tools_result.tools = [_fake_tool(CNT_TOOL, "manage containers")]
        sess.list_tools = AsyncMock(return_value=tools_result)
        resources_result = MagicMock()
        resources_result.resources = "not-a-list"  # malformed
        sess.list_resources = AsyncMock(return_value=resources_result)
        return sess

    mux._open_one_session = AsyncMock(side_effect=_open)  # type: ignore[method-assign]
    info = await mux.probe_server(CNT)

    assert info["error"] is None
    assert info["tools"][0]["name"] == CNT_TOOL
    assert info["skills"] == []


# --------------------------------------------------------------------------- #
# discover_tools: one ranked capability space
# --------------------------------------------------------------------------- #


async def test_discover_tools_ranks_skills_and_tools_in_one_result_set(tmp_path):
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "manage docker containers")]})
    mux._kg_call = AsyncMock(return_value=None)  # type: ignore[method-assign]
    mux._probe_cache[CNT] = {
        "tools": [
            {
                "name": CNT_TOOL,
                "description": "manage docker containers",
                "inputSchema": {},
            }
        ],
        "skills": [
            {
                "name": "container-runbook",
                "uri": "skill://container-runbook/SKILL.md",
                "description": "manage docker containers step by step",
            }
        ],
        "error": None,
    }

    discovery = await mux.discover_tools("manage docker containers", top_k=5)
    results = discovery["results"]
    kinds = {r["kind"] for r in results}

    assert kinds == {"tool", "skill"}
    tool_hit = next(r for r in results if r["kind"] == "tool")
    skill_hit = next(r for r in results if r["kind"] == "skill")
    assert tool_hit["bind"] == {
        "tool_server": CNT,
        "allowed_tools": [CNT_TOOL],
    }
    assert skill_hit["bind"] == {
        "tool_server": CNT,
        "skill_name": "container-runbook",
    }


async def test_discover_tools_skill_absent_when_server_has_no_skills(tmp_path):
    """A server with no skill:// resources contributes no skill entries — the
    ``skills`` probe key is optional and must default to empty, not error."""
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "manage docker containers")]})
    mux._kg_call = AsyncMock(return_value=None)  # type: ignore[method-assign]
    mux._probe_cache[CNT] = {
        "tools": [
            {
                "name": CNT_TOOL,
                "description": "manage docker containers",
                "inputSchema": {},
            }
        ],
        "error": None,
        # no "skills" key at all — mirrors a probe_server in-process-mounted
        # branch or a pre-Skills-over-MCP probe cache entry.
    }

    discovery = await mux.discover_tools("manage docker containers", top_k=5)
    assert all(r["kind"] == "tool" for r in discovery["results"])
