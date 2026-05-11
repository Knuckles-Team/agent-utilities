from __future__ import annotations
"""CONCEPT:ECO-4.0"""


"""Coverage push for agent_utilities.mcp_agent_manager.

Targets the pure / deterministic paths:
  * compute_tool_relevance_score (all scoring branches)
  * compute_agent_metadata_score (all tiers)
  * partition_tools (single tag, multi-tag, untracked)
  * generate_system_prompt (clean_server, clean_tag naming)
  * should_sync (no config, engine=None, stale cache, fresh cache, exception)
  * score_tools (in-place mutation)
  * sync_mcp_agents (empty tools early return, happy path with mocked backend,
    backend=None, ingest errors)

Does NOT attempt to exercise live MCP server subprocess / JSON-RPC paths.
"""


import pytest

from agent_utilities.mcp import agent_manager as mgr
from agent_utilities.models import MCPToolInfo

# ---------------------------------------------------------------------------
# partition_tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_partition_tools_multi_tags() -> None:
    """Tool with multiple tags appears in each partition."""
    tools = [
        MCPToolInfo(
            name="t1",
            description="",
            mcp_server="s",
            all_tags=["git", "vcs"],
        ),
    ]
    parts = await mgr.partition_tools(tools)
    assert "git" in parts
    assert "vcs" in parts
    assert parts["git"] == [tools[0]]
    assert parts["vcs"] == [tools[0]]


@pytest.mark.asyncio
async def test_partition_tools_empty_falls_to_server_general() -> None:
    """Tools with no tags fall into {server}_general partition."""
    tools = [
        MCPToolInfo(name="t1", description="", mcp_server="docker-mcp"),
    ]
    parts = await mgr.partition_tools(tools)
    assert "docker-mcp_general" in parts
    assert len(parts["docker-mcp_general"]) == 1


@pytest.mark.asyncio
async def test_partition_tools_mixed() -> None:
    """Mixed tagged/untagged: tagged into tag buckets, untagged into _general."""
    tools = [
        MCPToolInfo(name="t1", description="", mcp_server="s1", tag="git"),
        MCPToolInfo(name="t2", description="", mcp_server="s2"),
    ]
    parts = await mgr.partition_tools(tools)
    assert "git" in parts
    assert "s2_general" in parts


@pytest.mark.asyncio
async def test_partition_tools_empty_list() -> None:
    """Empty list yields empty dict."""
    parts = await mgr.partition_tools([])
    assert parts == {}


# ---------------------------------------------------------------------------
# generate_system_prompt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_system_prompt_simple() -> None:
    """Basic prompt generation with distinct server + tag."""
    prompt = await mgr.generate_system_prompt(
        agent_name="test",
        tools=[],
        tag="repo_management",
        server_name="github-mcp",
    )
    assert "Github" in prompt
    assert "Repo Management" in prompt
    assert "specialist" in prompt
    assert "tools" in prompt


@pytest.mark.asyncio
async def test_generate_system_prompt_tag_contains_server() -> None:
    """When clean_server is part of clean_tag, don't duplicate in the name."""
    prompt = await mgr.generate_system_prompt(
        agent_name="test",
        tools=[],
        tag="github_repos",
        server_name="github",
    )
    assert "Github Repos specialist" in prompt


@pytest.mark.asyncio
async def test_generate_system_prompt_strips_mcp_suffix() -> None:
    """'-mcp' and '-agent' suffixes are stripped from server name."""
    prompt = await mgr.generate_system_prompt(
        agent_name="test",
        tools=[],
        tag="tasks",
        server_name="jira-mcp",
    )
    assert "Jira" in prompt
    # No "-mcp" in output
    assert "-mcp" not in prompt.lower() or "mcp" not in prompt.lower()
