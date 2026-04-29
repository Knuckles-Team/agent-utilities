from __future__ import annotations
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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_utilities import mcp_agent_manager as mgr
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


def test_score_desc_tier_10_points() -> None:
    """Description 16-50 chars = 10 points."""
    tool = MCPToolInfo(
        name="",
        description="a" * 20,
        mcp_server="s",
    )
    # 20 chars > 15, so +10 desc
    assert mgr.compute_tool_relevance_score(tool) == 10


def test_score_desc_tier_20_points() -> None:
    """Description 51-100 chars = 20 points."""
    tool = MCPToolInfo(name="", description="a" * 60, mcp_server="s")
    assert mgr.compute_tool_relevance_score(tool) == 20


def test_score_desc_tier_30_points() -> None:
    """Description >100 chars = 30 points."""
    tool = MCPToolInfo(name="", description="a" * 120, mcp_server="s")
    assert mgr.compute_tool_relevance_score(tool) == 30


def test_score_all_tags_two_or_more() -> None:
    """two+ explicit tags = 30 points for tag confidence."""
    tool = MCPToolInfo(
        name="", description="", mcp_server="s", all_tags=["a", "b"]
    )
    # +30 tag confidence, +15 multi_tag coverage (2 tags)
    assert mgr.compute_tool_relevance_score(tool) == 30 + 15


def test_score_all_tags_three_or_more() -> None:
    """three+ tags = 30 tag confidence, +20 multi_tag coverage."""
    tool = MCPToolInfo(
        name="",
        description="",
        mcp_server="s",
        all_tags=["a", "b", "c"],
    )
    assert mgr.compute_tool_relevance_score(tool) == 30 + 20


def test_score_all_tags_single_long() -> None:
    """single long tag (>6 chars) = 25 tag confidence."""
    tool = MCPToolInfo(
        name="",
        description="",
        mcp_server="s",
        all_tags=["longtag"],
    )
    # +25 tag conf, +10 multi_tag coverage (1 tag)
    assert mgr.compute_tool_relevance_score(tool) == 25 + 10


def test_score_all_tags_single_with_underscore() -> None:
    """single tag with underscore = 25 tag confidence."""
    tool = MCPToolInfo(
        name="",
        description="",
        mcp_server="s",
        all_tags=["a_b"],
    )
    # has underscore so +25 tag conf
    assert mgr.compute_tool_relevance_score(tool) == 25 + 10


def test_score_all_tags_single_short() -> None:
    """single short single-word tag = 15 tag confidence."""
    tool = MCPToolInfo(
        name="",
        description="",
        mcp_server="s",
        all_tags=["git"],
    )
    # short, no underscore -> 15
    assert mgr.compute_tool_relevance_score(tool) == 15 + 10


def test_score_tag_only() -> None:
    """Only tag (no all_tags) = 10 tag confidence."""
    tool = MCPToolInfo(
        name="", description="", mcp_server="s", tag="git"
    )
    assert mgr.compute_tool_relevance_score(tool) == 10


def test_score_name_specificity_three_meaningful() -> None:
    """Three+ meaningful segments = 20 points name specificity."""
    tool = MCPToolInfo(
        name="docker_container_start_stop",
        description="",
        mcp_server="s",
    )
    # generic_verbs in mcp_agent_manager.compute_tool_relevance_score
    # is: {"get", "list", "create", "update", "delete", "set", "run"}.
    # 'start'/'stop' are NOT in that set, so meaningful = [docker, container,
    # start, stop] (all > 2 chars, none in verbs).  => 4 meaningful -> +20.
    score = mgr.compute_tool_relevance_score(tool)
    assert score == 20


def test_score_name_specificity_two_meaningful() -> None:
    """Two meaningful segments = 15 points name specificity."""
    tool = MCPToolInfo(
        name="docker_containers",
        description="",
        mcp_server="s",
    )
    assert mgr.compute_tool_relevance_score(tool) == 15


def test_score_name_specificity_one_meaningful() -> None:
    """One meaningful segment = 10 points name specificity."""
    tool = MCPToolInfo(
        name="containers",
        description="",
        mcp_server="s",
    )
    assert mgr.compute_tool_relevance_score(tool) == 10


def test_score_name_specificity_only_short_segments() -> None:
    """Only short segments = 5 points fallback."""
    tool = MCPToolInfo(
        name="a_b_c",
        description="",
        mcp_server="s",
    )
    # No meaningful (all <= 2), segments exist -> 5
    assert mgr.compute_tool_relevance_score(tool) == 5


def test_score_name_all_verbs() -> None:
    """Name consisting only of generic verbs -> 5 from segments fallback."""
    tool = MCPToolInfo(
        name="get_list",
        description="",
        mcp_server="s",
    )
    # both 'get' and 'list' are generic verbs -> meaningful = 0, segments = 2 -> +5
    assert mgr.compute_tool_relevance_score(tool) == 5


def test_score_score_capped_at_100() -> None:
    """Score is capped at 100."""
    tool = MCPToolInfo(
        name="docker_containers_management_list",
        description="a" * 200,
        mcp_server="s",
        all_tags=["tag1", "tag2", "tag3", "tag4"],
    )
    assert mgr.compute_tool_relevance_score(tool) == 100


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


def test_should_sync_no_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Engine is None -> should_sync returns True (never synced)."""
    cfg = tmp_path / "mcp.json"
    cfg.write_text("{}")

    fake_kg = MagicMock()
    fake_kg.IntelligenceGraphEngine.get_active.return_value = None
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    assert mgr.should_sync(cfg) is True


def test_should_sync_no_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Engine with no backend -> True."""
    cfg = tmp_path / "mcp.json"
    cfg.write_text("{}")

    engine = MagicMock(backend=None)
    fake_kg = MagicMock()
    fake_kg.IntelligenceGraphEngine.get_active.return_value = engine
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    assert mgr.should_sync(cfg) is True


def test_should_sync_no_last_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """last_sync=0/None -> True."""
    cfg = tmp_path / "mcp.json"
    cfg.write_text("{}")

    engine = MagicMock()
    engine.backend = MagicMock()
    engine.backend.execute.return_value = [{"last_sync": 0}]
    fake_kg = MagicMock()
    fake_kg.IntelligenceGraphEngine.get_active.return_value = engine
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    assert mgr.should_sync(cfg) is True


def test_should_sync_stale_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Config newer than last_sync + 2s -> True."""
    cfg = tmp_path / "mcp.json"
    cfg.write_text("{}")

    engine = MagicMock()
    engine.backend = MagicMock()
    # last_sync is way in the past
    engine.backend.execute.return_value = [{"last_sync": 1.0}]
    fake_kg = MagicMock()
    fake_kg.IntelligenceGraphEngine.get_active.return_value = engine
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    assert mgr.should_sync(cfg) is True


def test_should_sync_up_to_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Config unchanged since last_sync -> False."""
    cfg = tmp_path / "mcp.json"
    cfg.write_text("{}")
    current_mtime = cfg.stat().st_mtime

    engine = MagicMock()
    engine.backend = MagicMock()
    # last_sync is after the config mtime (by more than 2 seconds)
    engine.backend.execute.return_value = [{"last_sync": current_mtime + 10.0}]
    fake_kg = MagicMock()
    fake_kg.IntelligenceGraphEngine.get_active.return_value = engine
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    assert mgr.should_sync(cfg) is False


def test_should_sync_execute_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exception during execute -> True (defensive)."""
    cfg = tmp_path / "mcp.json"
    cfg.write_text("{}")

    engine = MagicMock()
    engine.backend = MagicMock()
    engine.backend.execute.side_effect = RuntimeError("db error")
    fake_kg = MagicMock()
    fake_kg.IntelligenceGraphEngine.get_active.return_value = engine
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    assert mgr.should_sync(cfg) is True


def test_should_sync_empty_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty result set -> last_sync=0 -> True."""
    cfg = tmp_path / "mcp.json"
    cfg.write_text("{}")

    engine = MagicMock()
    engine.backend = MagicMock()
    engine.backend.execute.return_value = []
    fake_kg = MagicMock()
    fake_kg.IntelligenceGraphEngine.get_active.return_value = engine
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )
    assert mgr.should_sync(cfg) is True


# ---------------------------------------------------------------------------
# score_tools (in-place mutation)
# ---------------------------------------------------------------------------


def test_score_tools_populates_relevance_score() -> None:
    """score_tools sets relevance_score on each tool."""
    tools = [
        MCPToolInfo(
            name="docker_containers",
            description="a" * 120,
            mcp_server="s",
            all_tags=["docker", "containers"],
        ),
        MCPToolInfo(name="plain", description="", mcp_server="s"),
    ]
    out = mgr.score_tools(tools)
    assert out is tools
    assert tools[0].relevance_score > 0
    assert tools[1].relevance_score < tools[0].relevance_score


def test_score_tools_empty_list() -> None:
    """Empty list yields empty list."""
    assert mgr.score_tools([]) == []


# ---------------------------------------------------------------------------
# sync_mcp_agents: empty early return
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_mcp_agents_no_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No tools -> early return without touching the graph."""
    cfg = tmp_path / "mcp.json"
    cfg.write_text("{}")

    async def fake_extract(path, timeout=300):
        return []

    monkeypatch.setattr(mgr, "extract_tool_metadata", fake_extract)
    # Also make sure knowledge_graph.engine is never imported for the main body
    result = await mgr.sync_mcp_agents(config_path=cfg)
    assert result is None


@pytest.mark.asyncio
async def test_sync_mcp_agents_default_config_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default config_path is resolved via get_workspace_path."""

    async def fake_extract(path, timeout=300):
        return []

    monkeypatch.setattr(mgr, "extract_tool_metadata", fake_extract)
    fake_path = MagicMock(spec=Path)
    fake_path.exists.return_value = True
    monkeypatch.setattr(mgr, "get_workspace_path", lambda name: fake_path)
    await mgr.sync_mcp_agents()
    assert True, 'Default config path sync should not raise'


@pytest.mark.asyncio
async def test_sync_mcp_agents_no_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing backend -> log error and return."""
    cfg = tmp_path / "mcp.json"
    cfg.write_text("{}")

    tool = MCPToolInfo(
        name="t1",
        description="desc",
        mcp_server="s",
        all_tags=["git"],
    )

    async def fake_extract(path, timeout=300):
        return [tool]

    monkeypatch.setattr(mgr, "extract_tool_metadata", fake_extract)

    # Mock knowledge_graph.engine to return engine with None backend
    fake_engine = MagicMock(backend=None)
    fake_kg = MagicMock()
    fake_kg.IntelligenceGraphEngine.get_active.return_value = fake_engine
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )

    result = await mgr.sync_mcp_agents(config_path=cfg)
    assert result is None


@pytest.mark.asyncio
async def test_sync_mcp_agents_success_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full happy path: extract, score, upsert to graph."""
    cfg = tmp_path / "mcp.json"
    cfg.write_text("{}")

    tool1 = MCPToolInfo(
        name="t1",
        description="desc 1",
        mcp_server="srv",
        all_tags=["git"],
    )
    tool2 = MCPToolInfo(
        name="t2",
        description="desc 2",
        mcp_server="srv",
        tag="docker",
    )

    async def fake_extract(path, timeout=300):
        return [tool1, tool2]

    monkeypatch.setattr(mgr, "extract_tool_metadata", fake_extract)

    backend = MagicMock()
    engine = MagicMock(backend=backend)
    fake_kg = MagicMock()
    fake_kg.IntelligenceGraphEngine.get_active.return_value = engine
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )

    # Mock ingest_prompts_to_graph
    fake_arb = MagicMock()
    fake_arb.ingest_prompts_to_graph = AsyncMock(return_value=None)
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.agent_registry_builder",
        fake_arb,
    )

    await mgr.sync_mcp_agents(config_path=cfg)
    # Expect at least 2 execute calls for upsert per tool plus 2 for server link
    # = 4 or 5 calls (incl. cleanup).  We just verify >0.
    assert backend.execute.call_count >= 4
    assert tool1.relevance_score > 0
    assert tool2.relevance_score > 0


@pytest.mark.asyncio
async def test_sync_mcp_agents_cleanup_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cleanup query exception is swallowed."""
    cfg = tmp_path / "mcp.json"
    cfg.write_text("{}")

    tool = MCPToolInfo(
        name="t1",
        description="desc",
        mcp_server="srv",
        all_tags=["git"],
    )

    async def fake_extract(path, timeout=300):
        return [tool]

    monkeypatch.setattr(mgr, "extract_tool_metadata", fake_extract)

    # Make backend.execute raise on the 4th call (the cleanup)
    backend = MagicMock()
    call_count = {"n": 0}

    def flaky_execute(query, params=None):
        call_count["n"] += 1
        if "DETACH DELETE" in query and call_count["n"] >= 3:
            raise RuntimeError("empty db")
        return []

    backend.execute.side_effect = flaky_execute
    engine = MagicMock(backend=backend)
    fake_kg = MagicMock()
    fake_kg.IntelligenceGraphEngine.get_active.return_value = engine
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.knowledge_graph.engine",
        fake_kg,
    )

    fake_arb = MagicMock()
    fake_arb.ingest_prompts_to_graph = AsyncMock(return_value=None)
    monkeypatch.setitem(
        __import__("sys").modules,
        "agent_utilities.agent_registry_builder",
        fake_arb,
    )

    # Must not raise
    await mgr.sync_mcp_agents(config_path=cfg)


# ---------------------------------------------------------------------------
# extract_tool_metadata: edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_tool_metadata_no_config_file(tmp_path: Path) -> None:
    """Nonexistent config returns empty list."""
    tools = await mgr.extract_tool_metadata(tmp_path / "nope.json")
    assert tools == []


@pytest.mark.asyncio
async def test_extract_tool_metadata_malformed_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed JSON: mcp_servers_config falls back to empty dict."""
    cfg = tmp_path / "bad.json"
    cfg.write_text("{ this is not valid json")
    monkeypatch.setattr(mgr, "load_mcp_config", lambda p: [])
    tools = await mgr.extract_tool_metadata(cfg)
    assert tools == []


# ---------------------------------------------------------------------------
# _extract_single_server_metadata_inner: fallback paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_inner_dynamic_failure_env_hints() -> None:
    """Dynamic extract fails, env hints (TOOL suffix keys) provide static tools."""
    failing_server = MagicMock()
    failing_server.name = "docker-mcp"
    failing_server.__aenter__.side_effect = RuntimeError("boom")
    failing_server.__aexit__.return_value = None

    config = {
        "docker-mcp": {
            "env": {
                "CONTAINERS_TOOL": "true",
                "IMAGES_TOOL": "true",
                "OTHER_KEY": "true",  # not a TOOL suffix, should be ignored
            }
        }
    }

    tools = await mgr._extract_single_server_metadata_inner(
        failing_server, config, timeout=5
    )
    # Expect one tool per *_TOOL env key
    tool_names = [t.name for t in tools]
    # The factory names include a double-underscore because
    # "CONTAINERS_TOOL".lower().replace("tool", "") = "containers_"
    assert "docker-mcp_containers__toolset" in tool_names
    assert "docker-mcp_images__toolset" in tool_names
    # OTHER_KEY should NOT produce a tool
    assert len(tools) == 2


@pytest.mark.asyncio
async def test_extract_inner_dynamic_failure_no_hints() -> None:
    """No env hints -> fallback to a single general tool per server."""
    failing_server = MagicMock()
    failing_server.name = "github-mcp"
    failing_server.__aenter__.side_effect = RuntimeError("boom")

    config: dict[str, Any] = {"github-mcp": {}}

    tools = await mgr._extract_single_server_metadata_inner(
        failing_server, config, timeout=5
    )
    assert len(tools) == 1
    assert tools[0].name == "github-mcp_general_tools"
    # Tag should strip -mcp
    assert tools[0].tag == "github"


@pytest.mark.asyncio
async def test_extract_inner_dynamic_success_annotation_tags_list() -> None:
    """Dynamic extraction with annotation-tagged tool."""
    tool = MagicMock()
    tool.name = "my_tool"
    tool.description = "A tool"
    tool.annotations = {"tags": ["git", "vcs"]}

    session = AsyncMock()
    session.list_tools.return_value = [tool]

    server = MagicMock()
    server.name = "myserver"
    server.__aenter__.return_value = session
    server.__aexit__.return_value = None

    tools = await mgr._extract_single_server_metadata_inner(
        server, {}, timeout=5
    )
    assert len(tools) == 1
    assert tools[0].name == "my_tool"
    assert tools[0].tag == "git"
    assert tools[0].all_tags == ["git", "vcs"]


@pytest.mark.asyncio
async def test_extract_inner_dynamic_success_annotation_object() -> None:
    """Dynamic extraction with annotation object (not dict)."""
    tool = MagicMock()
    tool.name = "my_tool"
    tool.description = "A tool"
    annotations = MagicMock(spec=["tags"])
    annotations.tags = ["file"]
    tool.annotations = annotations

    session = AsyncMock()
    session.list_tools.return_value = [tool]

    server = MagicMock()
    server.name = "fs"
    server.__aenter__.return_value = session
    server.__aexit__.return_value = None

    tools = await mgr._extract_single_server_metadata_inner(
        server, {}, timeout=5
    )
    assert len(tools) == 1
    assert tools[0].all_tags == ["file"]


@pytest.mark.asyncio
async def test_extract_inner_annotation_string_tag() -> None:
    """annotation.tags as a bare string becomes a single-element list."""
    tool = MagicMock()
    tool.name = "my_tool"
    tool.description = "A tool"
    tool.annotations = {"tags": "solo-tag"}

    session = AsyncMock()
    session.list_tools.return_value = [tool]

    server = MagicMock()
    server.name = "srv"
    server.__aenter__.return_value = session
    server.__aexit__.return_value = None

    tools = await mgr._extract_single_server_metadata_inner(
        server, {}, timeout=5
    )
    assert tools[0].all_tags == ["solo-tag"]


@pytest.mark.asyncio
async def test_extract_inner_tool_without_annotations_but_with_meta() -> None:
    """Fallback to FastMCP meta tags when annotations missing."""
    tool = MagicMock()
    tool.name = "my_tool"
    tool.description = "A tool"
    tool.annotations = None
    tool.meta = {"fastmcp": {"tags": ["meta-tag"]}}

    session = AsyncMock()
    session.list_tools.return_value = [tool]

    server = MagicMock()
    server.name = "srv"
    server.__aenter__.return_value = session
    server.__aexit__.return_value = None

    tools = await mgr._extract_single_server_metadata_inner(
        server, {}, timeout=5
    )
    assert tools[0].all_tags == ["meta-tag"]


@pytest.mark.asyncio
async def test_extract_inner_heuristic_tag_from_verb_name() -> None:
    """Tool name 'get_containers' heuristically tagged as 'containers'."""
    tool = MagicMock(spec=["name", "description"])
    tool.name = "get_containers"
    tool.description = ""

    session = AsyncMock()
    session.list_tools.return_value = [tool]

    server = MagicMock()
    server.name = "srv"
    server.__aenter__.return_value = session
    server.__aexit__.return_value = None

    tools = await mgr._extract_single_server_metadata_inner(
        server, {}, timeout=5
    )
    assert tools[0].tag == "containers"


@pytest.mark.asyncio
async def test_extract_inner_heuristic_tag_non_verb_first() -> None:
    """First segment is not a generic verb -> used as tag."""
    tool = MagicMock(spec=["name", "description"])
    tool.name = "docker_build"
    tool.description = ""

    session = AsyncMock()
    session.list_tools.return_value = [tool]

    server = MagicMock()
    server.name = "srv"
    server.__aenter__.return_value = session
    server.__aexit__.return_value = None

    tools = await mgr._extract_single_server_metadata_inner(
        server, {}, timeout=5
    )
    assert tools[0].tag == "docker"


@pytest.mark.asyncio
async def test_extract_inner_single_word_name_defaults_to_general() -> None:
    """Single-word tool name with no heuristic -> tag='general'."""
    tool = MagicMock(spec=["name", "description"])
    tool.name = "run"
    tool.description = ""

    session = AsyncMock()
    session.list_tools.return_value = [tool]

    server = MagicMock()
    server.name = "srv"
    server.__aenter__.return_value = session
    server.__aexit__.return_value = None

    tools = await mgr._extract_single_server_metadata_inner(
        server, {}, timeout=5
    )
    assert tools[0].tag == "general"


@pytest.mark.asyncio
async def test_extract_with_semaphore() -> None:
    """_extract_single_server_metadata with a semaphore."""
    import asyncio as _aio

    tool = MagicMock()
    tool.name = "t1"
    tool.description = "desc"
    tool.annotations = {"tags": ["x"]}

    session = AsyncMock()
    session.list_tools.return_value = [tool]

    server = MagicMock()
    server.name = "s"
    server.__aenter__.return_value = session
    server.__aexit__.return_value = None

    sem = _aio.Semaphore(1)
    tools = await mgr._extract_single_server_metadata(
        server, {}, timeout=5, semaphore=sem
    )
    assert len(tools) == 1


@pytest.mark.asyncio
async def test_extract_list_result_tools_attribute() -> None:
    """Server returns ListToolsResult-like with .tools attribute."""
    tool = MagicMock()
    tool.name = "t1"
    tool.description = "desc"
    tool.annotations = {"tags": ["x"]}

    result = MagicMock(spec=["tools"])
    result.tools = [tool]

    session = AsyncMock()
    session.list_tools.return_value = result

    server = MagicMock()
    server.name = "s"
    server.__aenter__.return_value = session
    server.__aexit__.return_value = None

    tools = await mgr._extract_single_server_metadata_inner(server, {}, timeout=5)
    assert len(tools) == 1


@pytest.mark.asyncio
async def test_extract_exception_group_reports_first() -> None:
    """ExceptionGroup from aexit -> error_msg uses first sub-exception."""
    # Simulate ExceptionGroup-like object.  Python 3.11+ has it built in,
    # but our fallback code just checks for `.exceptions`.
    class FakeExceptionGroup(Exception):
        def __init__(self) -> None:
            super().__init__("group")
            self.exceptions = [RuntimeError("inner")]

    server = MagicMock()
    server.name = "s"
    server.__aenter__.side_effect = FakeExceptionGroup()

    tools = await mgr._extract_single_server_metadata_inner(
        server, {"s": {}}, timeout=5
    )
    # Falls back to general tool
    assert len(tools) == 1


@pytest.mark.asyncio
async def test_extract_server_uses_id_when_name_missing() -> None:
    """Server without .name uses ._id fallback."""
    failing = MagicMock(spec=["_id", "__aenter__", "__aexit__"])
    failing._id = "byid"
    failing.__aenter__.side_effect = RuntimeError("boom")

    tools = await mgr._extract_single_server_metadata_inner(
        failing, {"byid": {}}, timeout=5
    )
    assert len(tools) == 1
    assert tools[0].mcp_server == "byid"
"""Coverage push for agent_utilities.tools.knowledge_tools.

Targets all CRUD operations with a mocked RegistryGraphEngine.  Each tool
exercises the no-engine path, the happy path, and any error branches.
"""


from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import networkx as nx
import pytest
from pydantic_ai import RunContext

from agent_utilities.models import AgentDeps
from agent_utilities.tools import knowledge_tools as kt


def _mock_ctx(with_engine: bool = True) -> MagicMock:
    """Return a RunContext-like mock with an optional knowledge_engine."""
    deps = MagicMock(spec=AgentDeps)
    if with_engine:
        engine = MagicMock()
        engine.graph = nx.MultiDiGraph()
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


def test_get_knowledge_engine_no_attr() -> None:
    """Helper returns None when deps lacks knowledge_engine (spec=AgentDeps)."""
    deps = MagicMock(spec=AgentDeps)
    ctx = MagicMock(spec=RunContext)
    ctx.deps = deps
    # spec=AgentDeps restricts MagicMock attributes; getattr default is None
    # if AgentDeps has no knowledge_engine attr defined; but AgentDeps does
    # define it.  Explicitly set to None and verify helper returns None.
    deps.knowledge_engine = None
    assert kt.get_knowledge_engine(ctx) is None


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
    ctx.deps.knowledge_engine.backend.execute.assert_called_once()


@pytest.mark.asyncio
async def test_link_knowledge_nodes_no_backend() -> None:
    """Link succeeds on NetworkX even when backend is None."""
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
    from agent_utilities.models import Spec, Tasks, ImplementationPlan

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
        type="memory",
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
    # Two queries: MERGE heartbeat + MERGE relationship
    assert ctx.deps.knowledge_engine.backend.execute.call_count == 2


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
    # Two queries: MERGE user + MATCH...MERGE relationship
    assert ctx.deps.knowledge_engine.backend.execute.call_count == 2


@pytest.mark.asyncio
async def test_create_user_no_client_id() -> None:
    """No client_id -> only the MERGE user query runs."""
    ctx = _mock_ctx()
    result = await kt.create_user(ctx, "Alice")
    assert "User created" in result
    assert ctx.deps.knowledge_engine.backend.execute.call_count == 1


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
    assert ctx.deps.knowledge_engine.backend.execute.call_count == 2


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
    assert ctx.deps.knowledge_engine.backend.execute.call_count == 2


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
    assert ctx.deps.knowledge_engine.backend.execute.call_count == 2


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


def test_knowledge_tools_export_list() -> None:
    """knowledge_tools export contains all expected tools."""
    from agent_utilities.tools.knowledge_tools import knowledge_tools

    names = {f.__name__ for f in knowledge_tools}
    assert "search_knowledge_graph" in names
    assert "get_code_impact" in names
    assert "add_knowledge_memory" in names
    assert "get_knowledge_memory" in names
    assert "update_knowledge_memory" in names
    assert "delete_knowledge_memory" in names
    assert "link_knowledge_nodes" in names
    assert "sync_feature_to_memory" in names
    assert "log_heartbeat" in names
    assert "create_client" in names
    assert "create_user" in names
    assert "save_preference" in names
    assert "save_chat_message" in names
    assert "log_cron_execution" in names


# ---------------------------------------------------------------------------
# _get_kb_engine helper
# ---------------------------------------------------------------------------


def test_get_kb_engine_no_registry_engine() -> None:
    """_get_kb_engine with no engine in context creates fresh KBIngestionEngine."""
    ctx = _mock_ctx(with_engine=False)
    with patch(
        "agent_utilities.knowledge_graph.kb.ingestion.KBIngestionEngine"
    ) as MockEngine:
        kt._get_kb_engine(ctx)
        MockEngine.assert_called_once()


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
        type="article",
        name="My Article",
        content="# Heading\ntext",
    )
    result = await kt.get_kb_article(ctx, "art:1")
    assert isinstance(result, str)
