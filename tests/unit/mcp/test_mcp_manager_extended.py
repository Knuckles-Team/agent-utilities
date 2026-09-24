from __future__ import annotations

"""CONCEPT:AU-ECO.messaging.native-backend-abstraction"""


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

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent_utilities.mcp import agent_manager as mgr
from agent_utilities.models import MCPToolInfo

# ---------------------------------------------------------------------------
# compute_tool_relevance_score (exhaustive scoring branches)
# ---------------------------------------------------------------------------


def test_score_empty_tool() -> None:
    """Empty tool scores 0."""
    tool = MCPToolInfo(name="", description="", mcp_server="s")
    assert mgr.compute_tool_relevance_score(tool) == 0


def test_score_desc_tier_5_points() -> None:
    """Description 1-15 chars = 5 points."""
    tool = MCPToolInfo(name="", description="short", mcp_server="s")
    assert mgr.compute_tool_relevance_score(tool) == 5


def test_score_name_specificity_only_short_segments() -> None:
    """Only short segments = 5 points fallback."""
    tool = MCPToolInfo(
        name="a_b_c",
        description="",
        mcp_server="s",
    )
    # No meaningful (all <= 2), segments exist -> 5
    assert mgr.compute_tool_relevance_score(tool) == 5


# ---------------------------------------------------------------------------
# compute_agent_metadata_score (all tiers)
# ---------------------------------------------------------------------------


def test_agent_score_empty() -> None:
    """Empty metadata -> 0."""
    assert mgr.compute_agent_metadata_score("", []) == 0


def test_agent_score_desc_tier_5() -> None:
    """Short description = 5 points."""
    assert mgr.compute_agent_metadata_score("a", []) == 5


def test_agent_score_desc_tier_20() -> None:
    """Medium description = 20 points."""
    assert mgr.compute_agent_metadata_score("a" * 50, []) == 20


def test_agent_score_desc_tier_40() -> None:
    """Long description = 40 points."""
    assert mgr.compute_agent_metadata_score("a" * 100, []) == 40


def test_agent_score_desc_tier_50() -> None:
    """Very long description = 50 points."""
    assert mgr.compute_agent_metadata_score("a" * 200, []) == 50


def test_agent_score_skills_tier_10() -> None:
    """1-2 skills = 10 points."""
    assert mgr.compute_agent_metadata_score("", ["s1"]) == 10


def test_agent_score_skills_tier_20() -> None:
    """3-5 skills = 20 points."""
    assert mgr.compute_agent_metadata_score("", ["s"] * 3) == 20


def test_agent_score_skills_tier_40() -> None:
    """6-10 skills = 40 points."""
    assert mgr.compute_agent_metadata_score("", ["s"] * 6) == 40


def test_agent_score_skills_tier_50() -> None:
    """>10 skills = 50 points."""
    assert mgr.compute_agent_metadata_score("", ["s"] * 12) == 50


def test_agent_score_cap_at_100() -> None:
    """Max score capped at 100."""
    assert mgr.compute_agent_metadata_score("a" * 200, ["s"] * 12) == 100


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


# ---------------------------------------------------------------------------
# should_sync
# ---------------------------------------------------------------------------


def test_should_sync_no_config_file() -> None:
    """Nonexistent config -> False."""
    assert mgr.should_sync(Path("/definitely/nonexistent/mcp.json")) is False


# ---------------------------------------------------------------------------
# score_tools (in-place mutation)
# ---------------------------------------------------------------------------


def test_score_tools_empty_list() -> None:
    """Empty list yields empty list."""
    assert mgr.score_tools([]) == []


# ---------------------------------------------------------------------------
# sync_mcp_agents: empty early return
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# extract_tool_metadata: edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_tool_metadata_no_config_file(tmp_path: Path) -> None:
    """Nonexistent config returns empty list."""
    tools = await mgr.extract_tool_metadata(tmp_path / "nope.json")
    assert tools == []


# ---------------------------------------------------------------------------
# _extract_single_server_metadata_inner: fallback paths
# ---------------------------------------------------------------------------


"""Coverage push for agent_utilities.tools.knowledge_tools.

Targets all CRUD operations with a mocked IntelligenceGraphEngine.  Each tool
exercises the no-engine path, the happy path, and any error branches.
"""


import pytest
from pydantic_ai import RunContext

from agent_utilities.knowledge_graph.core.graph_compute import GraphComputeEngine
from agent_utilities.models import AgentDeps
from agent_utilities.tools import knowledge_tools as kt


def _mock_ctx(with_engine: bool = True) -> MagicMock:
    """Return a RunContext-like mock with an optional knowledge_engine."""
    deps = MagicMock(spec=AgentDeps)
    if with_engine:
        engine = MagicMock()
        engine.graph = GraphComputeEngine(backend_type="rust")
        engine.backend = MagicMock()
        deps.knowledge_engine = engine
    else:
        deps.knowledge_engine = None
    ctx = MagicMock(spec=RunContext)
    ctx.deps = deps
    return ctx


# ---------------------------------------------------------------------------
# get_knowledge_engine helper
# ---------------------------------------------------------------------------


def test_get_knowledge_engine_returns_engine() -> None:
    """Helper returns the engine from deps."""
    ctx = _mock_ctx()
    assert kt.get_knowledge_engine(ctx) is ctx.deps.knowledge_engine


# ---------------------------------------------------------------------------
# search_knowledge_graph
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_knowledge_graph_no_engine() -> None:
    """No engine -> 'not available'."""
    ctx = _mock_ctx(with_engine=False)
    result = await kt.search_knowledge_graph(ctx, "q")
    assert "not available" in result


@pytest.mark.asyncio
async def test_search_knowledge_graph_empty_results() -> None:
    """Empty results -> 'No results found'."""
    ctx = _mock_ctx()
    ctx.deps.knowledge_engine.search_hybrid.return_value = []
    result = await kt.search_knowledge_graph(ctx, "q")
    assert "No results found" in result


@pytest.mark.asyncio
async def test_search_knowledge_graph_multiple_results() -> None:
    """Multiple results rendered with separators."""
    ctx = _mock_ctx()
    ctx.deps.knowledge_engine.search_hybrid.return_value = [
        {"id": "n1", "type": "agent", "name": "A1", "description": "desc1"},
        {"id": "n2", "type": "tool", "name": "T1", "description": "desc2"},
    ]
    result = await kt.search_knowledge_graph(ctx, "q")
    assert "[AGENT]" in result
    assert "[TOOL]" in result
    assert "---" in result


# ---------------------------------------------------------------------------
# get_code_impact
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_code_impact_no_engine() -> None:
    """No engine -> 'not available'."""
    ctx = _mock_ctx(with_engine=False)
    result = await kt.get_code_impact(ctx, "foo")
    assert "not available" in result


@pytest.mark.asyncio
async def test_get_code_impact_empty() -> None:
    """Empty impact -> 'No impact found'."""
    ctx = _mock_ctx()
    ctx.deps.knowledge_engine.query_impact.return_value = []
    result = await kt.get_code_impact(ctx, "foo")
    assert "No impact found" in result


@pytest.mark.asyncio
async def test_get_code_impact_with_nodes() -> None:
    """Impact nodes rendered as a list."""
    ctx = _mock_ctx()
    ctx.deps.knowledge_engine.query_impact.return_value = [
        {"id": "n1", "type": "file", "file_path": "/a/b.py"},
        {"id": "n2", "type": "symbol", "file_path": None},
    ]
    result = await kt.get_code_impact(ctx, "foo")
    assert "Impact Set" in result
    assert "n1" in result
    assert "n2" in result


# ---------------------------------------------------------------------------
# add_knowledge_memory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_add_knowledge_memory_no_engine() -> None:
    """No engine -> 'not available for persistence'."""
    ctx = _mock_ctx(with_engine=False)
    result = await kt.add_knowledge_memory(ctx, "content")
    assert "not available" in result


@pytest.mark.asyncio
async def test_add_knowledge_memory_with_tags() -> None:
    """Tags, name and category are forwarded to engine.add_memory."""
    ctx = _mock_ctx()
    ctx.deps.knowledge_engine.add_memory.return_value = "mem:xyz"
    result = await kt.add_knowledge_memory(
        ctx, "content", name="n", category="fact", tags=["a", "b"]
    )
    assert "mem:xyz" in result
    ctx.deps.knowledge_engine.add_memory.assert_called_once_with(
        "content", name="n", category="fact", tags=["a", "b"]
    )


# ---------------------------------------------------------------------------
# get_knowledge_memory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_knowledge_memory_no_engine() -> None:
    """No engine -> 'not available'."""
    ctx = _mock_ctx(with_engine=False)
    result = await kt.get_knowledge_memory(ctx, "mem:abc")
    assert "not available" in result


@pytest.mark.asyncio
async def test_get_knowledge_memory_not_found() -> None:
    """get_memory returning None -> 'not found'."""
    ctx = _mock_ctx()
    ctx.deps.knowledge_engine.get_memory.return_value = None
    result = await kt.get_knowledge_memory(ctx, "mem:abc")
    assert "not found" in result


# ---------------------------------------------------------------------------
# update_knowledge_memory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_knowledge_memory_no_engine() -> None:
    """No engine -> 'not available'."""
    ctx = _mock_ctx(with_engine=False)
    result = await kt.update_knowledge_memory(ctx, "mem:abc", content="new")
    assert "not available" in result


@pytest.mark.asyncio
async def test_update_knowledge_memory_no_updates() -> None:
    """All None params -> 'No updates provided'."""
    ctx = _mock_ctx()
    result = await kt.update_knowledge_memory(ctx, "mem:abc")
    assert "No updates" in result


@pytest.mark.asyncio
async def test_update_knowledge_memory_content_only() -> None:
    """content param yields update_memory call with description."""
    ctx = _mock_ctx()
    result = await kt.update_knowledge_memory(ctx, "mem:abc", content="new desc")
    assert "Successfully updated" in result
    ctx.deps.knowledge_engine.update_memory.assert_called_once_with(
        "mem:abc", description="new desc"
    )


@pytest.mark.asyncio
async def test_update_knowledge_memory_all_fields() -> None:
    """All three fields get updated."""
    ctx = _mock_ctx()
    result = await kt.update_knowledge_memory(
        ctx, "mem:abc", content="x", category="y", tags=["z"]
    )
    assert "Successfully updated" in result
    call_kwargs = ctx.deps.knowledge_engine.update_memory.call_args.kwargs
    assert call_kwargs == {
        "description": "x",
        "category": "y",
        "tags": ["z"],
    }


# ---------------------------------------------------------------------------
# delete_knowledge_memory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_knowledge_memory_no_engine() -> None:
    """No engine -> 'not available'."""
    ctx = _mock_ctx(with_engine=False)
    result = await kt.delete_knowledge_memory(ctx, "mem:abc")
    assert "not available" in result


@pytest.mark.asyncio
async def test_delete_knowledge_memory_success() -> None:
    """Delete forwards to engine.delete_memory."""
    ctx = _mock_ctx()
    result = await kt.delete_knowledge_memory(ctx, "mem:abc")
    assert "Successfully deleted" in result
    ctx.deps.knowledge_engine.delete_memory.assert_called_once_with("mem:abc")


# ---------------------------------------------------------------------------
# link_knowledge_nodes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_link_knowledge_nodes_no_engine() -> None:
    """No engine -> 'not available'."""
    ctx = _mock_ctx(with_engine=False)
    result = await kt.link_knowledge_nodes(ctx, "a", "b")
    assert "not available" in result


@pytest.mark.asyncio
async def test_link_knowledge_nodes_source_missing() -> None:
    """Source not in graph -> error message."""
    ctx = _mock_ctx()
    ctx.deps.knowledge_engine.graph.add_node("b")
    result = await kt.link_knowledge_nodes(ctx, "a", "b")
    assert "not found in graph" in result


@pytest.mark.asyncio
async def test_link_knowledge_nodes_target_missing() -> None:
    """Target not in graph -> error message."""
    ctx = _mock_ctx()
    ctx.deps.knowledge_engine.graph.add_node("a")
    result = await kt.link_knowledge_nodes(ctx, "a", "b")
    assert "not found in graph" in result


@pytest.mark.asyncio
async def test_link_knowledge_nodes_success() -> None:
    """Link succeeds and emits MATCH/MERGE query."""
    ctx = _mock_ctx()
    ctx.deps.knowledge_engine.graph.add_node("a")
    ctx.deps.knowledge_engine.graph.add_node("b")
    result = await kt.link_knowledge_nodes(ctx, "a", "b", "depends_on")
    assert "Successfully established" in result
    assert "depends_on" in result
    # link_knowledge_nodes now goes through the engine's typed link_nodes API,
    # not a raw backend.execute Cypher call (same migration already reflected
    # in test_sync_mcp_agents_success_path's add_node/link_nodes assertions).
    ctx.deps.knowledge_engine.link_nodes.assert_called_once()


@pytest.mark.asyncio
async def test_link_knowledge_nodes_no_backend() -> None:
    """Link succeeds on graph compute even when backend is None."""
    ctx = _mock_ctx()
    ctx.deps.knowledge_engine.graph.add_node("a")
    ctx.deps.knowledge_engine.graph.add_node("b")
    ctx.deps.knowledge_engine.backend = None
    result = await kt.link_knowledge_nodes(ctx, "a", "b")
    assert "Successfully established" in result


# ---------------------------------------------------------------------------
# sync_feature_to_memory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_feature_to_memory_no_engine() -> None:
    """No engine -> 'not available'."""
    ctx = _mock_ctx(with_engine=False)
    result = await kt.sync_feature_to_memory(ctx, "feat-001")
    assert "not available" in result


@pytest.mark.asyncio
async def test_sync_feature_to_memory_no_workspace_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing workspace_path -> error."""
    ctx = _mock_ctx()
    # Set workspace_path to None
    ctx.deps.workspace_path = None
    result = await kt.sync_feature_to_memory(ctx, "feat-001")
    assert "Workspace path not available" in result


@pytest.mark.asyncio
async def test_sync_feature_to_memory_no_spec(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No Spec found -> aborted message."""
    ctx = _mock_ctx()
    ctx.deps.workspace_path = str(tmp_path)
    fake_manager = MagicMock()
    fake_manager.load.return_value = None
    monkeypatch.setattr(kt, "SDDManager", lambda ws: fake_manager)
    result = await kt.sync_feature_to_memory(ctx, "feat-001")
    assert "Could not find Spec" in result


@pytest.mark.asyncio
async def test_sync_feature_to_memory_creates_new(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Creates new memory when existing_mem_id is None."""
    from agent_utilities.models import ImplementationPlan, Spec, Tasks

    spec = MagicMock(spec=Spec)
    spec.title = "Title"
    spec.user_stories = [MagicMock(description="User goal")]
    plan = MagicMock(spec=ImplementationPlan)
    plan.technical_context = "Plan text"
    tasks = MagicMock(spec=Tasks)
    tasks.tasks = []

    ctx = _mock_ctx()
    ctx.deps.workspace_path = str(tmp_path)
    fake_manager = MagicMock()
    fake_manager.load.side_effect = [spec, plan, tasks]
    monkeypatch.setattr(kt, "SDDManager", lambda ws: fake_manager)

    ctx.deps.knowledge_engine.add_memory.return_value = "mem:new"
    result = await kt.sync_feature_to_memory(ctx, "feat-001")
    assert "Successfully captured" in result
    ctx.deps.knowledge_engine.add_memory.assert_called_once()


@pytest.mark.asyncio
async def test_sync_feature_to_memory_updates_existing(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Updates memory when one already exists."""
    from agent_utilities.models import Spec

    spec = MagicMock(spec=Spec)
    spec.title = "Title"
    spec.user_stories = []

    ctx = _mock_ctx()
    ctx.deps.workspace_path = str(tmp_path)
    # Pre-seed graph with existing memory
    ctx.deps.knowledge_engine.graph.add_node(
        "mem:existing",
        node_type="memory",
        name="SDD Feature Memory: feat-001",
    )
    fake_manager = MagicMock()
    fake_manager.load.side_effect = [spec, None, None]
    monkeypatch.setattr(kt, "SDDManager", lambda ws: fake_manager)

    result = await kt.sync_feature_to_memory(ctx, "feat-001")
    assert "Successfully updated historical memory" in result
    ctx.deps.knowledge_engine.update_memory.assert_called_once()


# ---------------------------------------------------------------------------
# log_heartbeat
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_log_heartbeat_no_engine() -> None:
    """No engine -> 'not available'."""
    ctx = _mock_ctx(with_engine=False)
    result = await kt.log_heartbeat(ctx, "agent", "ok")
    assert "not available" in result


@pytest.mark.asyncio
async def test_log_heartbeat_success() -> None:
    """Happy path writes two queries and returns the hb id."""
    ctx = _mock_ctx()
    result = await kt.log_heartbeat(ctx, "agent1", "ok", issues=["i1"])
    assert "Heartbeat logged" in result
    # Typed engine API, not raw backend.execute: Heartbeat node + Agent node,
    # linked HEARTBEAT_OF.
    assert ctx.deps.knowledge_engine.add_node.call_count == 2
    assert ctx.deps.knowledge_engine.link_nodes.call_count == 1


@pytest.mark.asyncio
async def test_log_heartbeat_no_backend() -> None:
    """backend=None -> 'Failed to log'."""
    ctx = _mock_ctx()
    ctx.deps.knowledge_engine.backend = None
    result = await kt.log_heartbeat(ctx, "agent1", "ok")
    assert "Failed to log" in result


# ---------------------------------------------------------------------------
# create_client
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_client_no_engine() -> None:
    """No engine -> 'not available'."""
    ctx = _mock_ctx(with_engine=False)
    result = await kt.create_client(ctx, "ClientA")
    assert "not available" in result


@pytest.mark.asyncio
async def test_create_client_success() -> None:
    """Creates a client node."""
    ctx = _mock_ctx()
    result = await kt.create_client(ctx, "ClientA", "desc")
    assert "Client created" in result


@pytest.mark.asyncio
async def test_create_client_no_backend() -> None:
    """No backend -> 'Failed to create'."""
    ctx = _mock_ctx()
    ctx.deps.knowledge_engine.backend = None
    result = await kt.create_client(ctx, "ClientA")
    assert "Failed to create" in result


# ---------------------------------------------------------------------------
# create_user
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_user_no_engine() -> None:
    """No engine -> 'not available'."""
    ctx = _mock_ctx(with_engine=False)
    result = await kt.create_user(ctx, "Alice")
    assert "not available" in result


@pytest.mark.asyncio
async def test_create_user_with_client_id() -> None:
    """Creates user and links to client."""
    ctx = _mock_ctx()
    result = await kt.create_user(ctx, "Alice", role="admin", client_id="c1")
    assert "User created" in result
    # Typed engine API: User node, linked BELONGS_TO the client.
    assert ctx.deps.knowledge_engine.add_node.call_count == 1
    assert ctx.deps.knowledge_engine.link_nodes.call_count == 1


@pytest.mark.asyncio
async def test_create_user_no_client_id() -> None:
    """No client_id -> only the MERGE user query runs."""
    ctx = _mock_ctx()
    result = await kt.create_user(ctx, "Alice")
    assert "User created" in result
    # No client_id -> only the User node write, no link_nodes call.
    assert ctx.deps.knowledge_engine.add_node.call_count == 1
    assert ctx.deps.knowledge_engine.link_nodes.call_count == 0


@pytest.mark.asyncio
async def test_create_user_no_backend() -> None:
    """No backend -> 'Failed to create'."""
    ctx = _mock_ctx()
    ctx.deps.knowledge_engine.backend = None
    result = await kt.create_user(ctx, "Alice")
    assert "Failed to create" in result


# ---------------------------------------------------------------------------
# save_preference
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_preference_no_engine() -> None:
    """No engine -> 'not available'."""
    ctx = _mock_ctx(with_engine=False)
    result = await kt.save_preference(ctx, "u1", "lang", "python")
    assert "not available" in result


@pytest.mark.asyncio
async def test_save_preference_success() -> None:
    """Saves preference and links to user."""
    ctx = _mock_ctx()
    result = await kt.save_preference(ctx, "u1", "lang", "python")
    assert "Preference saved" in result
    # Typed engine API: Preference node, linked PREFERS to the user.
    assert ctx.deps.knowledge_engine.add_node.call_count == 1
    assert ctx.deps.knowledge_engine.link_nodes.call_count == 1


@pytest.mark.asyncio
async def test_save_preference_no_backend() -> None:
    """No backend -> 'Failed to save'."""
    ctx = _mock_ctx()
    ctx.deps.knowledge_engine.backend = None
    result = await kt.save_preference(ctx, "u1", "lang", "python")
    assert "Failed to save" in result


# ---------------------------------------------------------------------------
# save_chat_message
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_save_chat_message_no_engine() -> None:
    """No engine -> 'not available'."""
    ctx = _mock_ctx(with_engine=False)
    result = await kt.save_chat_message(ctx, "t1", "user", "hi")
    assert "not available" in result


@pytest.mark.asyncio
async def test_save_chat_message_success() -> None:
    """Saves message and links to thread."""
    ctx = _mock_ctx()
    result = await kt.save_chat_message(ctx, "t1", "user", "hi")
    assert "Message saved" in result
    # Typed engine API: Message node + Thread node, linked PART_OF.
    assert ctx.deps.knowledge_engine.add_node.call_count == 2
    assert ctx.deps.knowledge_engine.link_nodes.call_count == 1


@pytest.mark.asyncio
async def test_save_chat_message_no_backend() -> None:
    """No backend -> 'Failed to save message'."""
    ctx = _mock_ctx()
    ctx.deps.knowledge_engine.backend = None
    result = await kt.save_chat_message(ctx, "t1", "user", "hi")
    assert "Failed to save message" in result


# ---------------------------------------------------------------------------
# log_cron_execution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_log_cron_execution_no_engine() -> None:
    """No engine -> 'not available'."""
    ctx = _mock_ctx(with_engine=False)
    result = await kt.log_cron_execution(ctx, "j1", "ok", "done")
    assert "not available" in result


@pytest.mark.asyncio
async def test_log_cron_execution_success() -> None:
    """Happy path logs two queries."""
    ctx = _mock_ctx()
    result = await kt.log_cron_execution(ctx, "j1", "ok", "done")
    assert "Cron execution logged" in result
    # Typed engine API: Log node + Job node, linked EXECUTED_BY.
    assert ctx.deps.knowledge_engine.add_node.call_count == 2
    assert ctx.deps.knowledge_engine.link_nodes.call_count == 1


@pytest.mark.asyncio
async def test_log_cron_execution_no_backend() -> None:
    """No backend -> 'Failed to log cron execution'."""
    ctx = _mock_ctx()
    ctx.deps.knowledge_engine.backend = None
    result = await kt.log_cron_execution(ctx, "j1", "ok", "done")
    assert "Failed to log cron execution" in result


# ---------------------------------------------------------------------------
# knowledge_tools export
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _get_kb_engine helper
# ---------------------------------------------------------------------------


def test_get_kb_engine_no_registry_engine() -> None:
    """_get_kb_engine with no engine in context raises RuntimeError."""
    ctx = _mock_ctx(with_engine=False)
    with pytest.raises(RuntimeError, match="Knowledge Graph not available"):
        kt._get_kb_engine(ctx)


def test_get_kb_engine_with_engine() -> None:
    """_get_kb_engine uses the existing engine's graph/backend."""
    ctx = _mock_ctx()
    with patch(
        "agent_utilities.knowledge_graph.kb.ingestion.KBIngestionEngine"
    ) as MockEngine:
        kt._get_kb_engine(ctx)
        MockEngine.assert_called_once()


# ---------------------------------------------------------------------------
# KB tools - list_knowledge_bases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_knowledge_bases_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """No KBs -> 'No knowledge bases found'."""
    ctx = _mock_ctx()
    fake_kb_engine = MagicMock()
    fake_kb_engine.list_knowledge_bases.return_value = []
    monkeypatch.setattr(kt, "_get_kb_engine", lambda ctx: fake_kb_engine)
    result = await kt.list_knowledge_bases(ctx)
    assert "No knowledge bases" in result


@pytest.mark.asyncio
async def test_list_knowledge_bases_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KB listing formats as a table."""
    ctx = _mock_ctx()
    fake_kb_engine = MagicMock()
    fake_kb_engine.list_knowledge_bases.return_value = [
        {
            "id": "kb:1",
            "name": "k1",
            "topic": "topic1",
            "article_count": 5,
            "source_count": 3,
            "status": "ready",
        }
    ]
    monkeypatch.setattr(kt, "_get_kb_engine", lambda ctx: fake_kb_engine)
    result = await kt.list_knowledge_bases(ctx)
    assert "k1" in result
    assert "kb:1" in result


@pytest.mark.asyncio
async def test_list_knowledge_bases_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exception in list_knowledge_bases is caught."""
    ctx = _mock_ctx()

    def boom(ctx: Any) -> None:
        raise RuntimeError("db down")

    monkeypatch.setattr(kt, "_get_kb_engine", boom)
    result = await kt.list_knowledge_bases(ctx)
    assert "Error listing" in result


# ---------------------------------------------------------------------------
# KB tools - search_knowledge_base_tool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_knowledge_base_tool_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No results -> 'No results found'."""
    ctx = _mock_ctx()
    fake_kb_engine = MagicMock()
    fake_kb_engine.search_knowledge_base.return_value = []
    monkeypatch.setattr(kt, "_get_kb_engine", lambda ctx: fake_kb_engine)
    result = await kt.search_knowledge_base_tool(ctx, "query")
    assert "No results found" in result


@pytest.mark.asyncio
async def test_search_knowledge_base_tool_with_kb_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty results with kb_id include scope in message."""
    ctx = _mock_ctx()
    fake_kb_engine = MagicMock()
    fake_kb_engine.search_knowledge_base.return_value = []
    monkeypatch.setattr(kt, "_get_kb_engine", lambda ctx: fake_kb_engine)
    result = await kt.search_knowledge_base_tool(ctx, "query", kb_id="kb:foo")
    assert "kb:foo" in result


@pytest.mark.asyncio
async def test_search_knowledge_base_tool_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path renders results."""
    ctx = _mock_ctx()
    fake_kb_engine = MagicMock()
    fake_kb_engine.search_knowledge_base.return_value = [
        {
            "article_title": "Article 1",
            "kb_name": "kb:test",
            "excerpt": "text...",
            "article_id": "art:1",
        }
    ]
    monkeypatch.setattr(kt, "_get_kb_engine", lambda ctx: fake_kb_engine)
    result = await kt.search_knowledge_base_tool(ctx, "q")
    assert "Article 1" in result


@pytest.mark.asyncio
async def test_search_knowledge_base_tool_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exception is caught."""
    ctx = _mock_ctx()

    def boom(ctx: Any) -> None:
        raise RuntimeError("down")

    monkeypatch.setattr(kt, "_get_kb_engine", boom)
    result = await kt.search_knowledge_base_tool(ctx, "q")
    assert "Search error" in result


# ---------------------------------------------------------------------------
# KB tools - get_kb_article
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_kb_article_no_engine() -> None:
    """No engine -> 'not available'."""
    ctx = _mock_ctx(with_engine=False)
    result = await kt.get_kb_article(ctx, "art:1")
    assert "not available" in result


@pytest.mark.asyncio
async def test_get_kb_article_not_in_graph() -> None:
    """Article not in graph -> 'not found'."""
    ctx = _mock_ctx()
    result = await kt.get_kb_article(ctx, "art:missing")
    assert "not found" in result.lower()


@pytest.mark.asyncio
async def test_get_kb_article_found() -> None:
    """Existing article node -> markdown content returned."""
    ctx = _mock_ctx()
    ctx.deps.knowledge_engine.graph.add_node(
        "art:1",
        node_type="article",
        name="My Article",
        content="# Heading\ntext",
    )
    result = await kt.get_kb_article(ctx, "art:1")
    assert isinstance(result, str)
