"""The fleet gateway loads tools; it never harvests skill or prompt content.

EH-220 / RF-ADR-009 §2.1 item 5: skill and prompt bodies reach the platform
only as governed EG ``AgentComponent`` records imported from a connector pack
(the SDK connector-sync runner → ``ConnectorPack.Import``). The multiplexer's
former ``skill://``/``prompt://`` body harvest over its probe session was a
second, ungoverned ingestion path and is deleted. These tests pin its absence
at the one place it lived: the probe session.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_utilities.mcp import multiplexer as multiplexer_module
from tests.unit.mcp.test_multiplexer_dynamic_gateway import (
    CNT,
    CNT_TOOL,
    _fake_tool,
    _mux_with_children,
)

_CONTENT_URIS = ("skill://onboarding/SKILL.md", "prompt://ops/triage")


def _resource(uri: str) -> MagicMock:
    resource = MagicMock()
    resource.uri = uri
    resource.name = uri
    resource.description = "served content"
    resource.model_dump = None
    return resource


def _content_serving_session() -> AsyncMock:
    """A child that serves skill:// and prompt:// resources beside its tools."""
    session = AsyncMock()
    tools_result = MagicMock()
    tools_result.tools = [_fake_tool(CNT_TOOL, "manage containers")]
    session.list_tools = AsyncMock(return_value=tools_result)
    resources_result = MagicMock()
    resources_result.resources = [_resource(uri) for uri in _CONTENT_URIS]
    session.list_resources = AsyncMock(return_value=resources_result)
    session.read_resource = AsyncMock(side_effect=AssertionError("body read"))
    return session


async def test_probe_records_descriptors_and_never_reads_content(tmp_path, monkeypatch):
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "manage containers")]})
    session = _content_serving_session()

    async def _open(server, cfg, stack):
        return session

    monkeypatch.setattr(mux, "_open_one_session", AsyncMock(side_effect=_open))
    info = await mux.probe_server(CNT)

    assert info["error"] is None
    assert [tool["name"] for tool in info["tools"]] == [CNT_TOOL]
    assert [entry["uri"] for entry in info["resources"]] == list(_CONTENT_URIS)
    assert "skills" not in info
    assert "prompts" not in info
    session.read_resource.assert_not_awaited()


async def test_discovery_ranks_tools_only(tmp_path):
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "manage docker containers")]})
    mux._probe_cache[CNT] = {
        "tools": [
            {"name": CNT_TOOL, "description": "manage docker containers"},
        ],
        # A cache entry written before EH-220 must not resurrect skill rows.
        "skills": [
            {"name": "container-runbook", "description": "manage docker containers"}
        ],
        "error": None,
    }

    discovery = await mux.discover_tools("manage docker containers", top_k=5)

    assert {row["kind"] for row in discovery["results"]} == {"tool"}


@pytest.mark.parametrize(
    "retired",
    [
        "_probe_skills",
        "_probe_prompts",
        "_harvest_resource_bodies",
        "_live_skills_for_server",
        "_live_prompts_for_server",
        "_ranked_skill_entry",
    ],
)
def test_harvest_methods_are_gone(retired):
    assert not hasattr(multiplexer_module.MCPMultiplexer, retired)
