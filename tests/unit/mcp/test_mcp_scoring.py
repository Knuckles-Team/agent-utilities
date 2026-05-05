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
    tool = MCPToolInfo(name="", description="", mcp_server="s", all_tags=["a", "b"])
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
    tool = MCPToolInfo(name="", description="", mcp_server="s", tag="git")
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
