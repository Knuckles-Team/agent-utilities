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


def _fake_resource_body(text: str):
    """A ``resources/read`` result carrying one text content part."""
    content = MagicMock()
    content.text = text
    result = MagicMock()
    result.contents = [content]
    return result


def _fake_session_with_resources(tools, resources, bodies=None):
    sess = AsyncMock()
    tools_result = MagicMock()
    tools_result.tools = [_fake_tool(n, d) for n, d in tools]
    sess.list_tools = AsyncMock(return_value=tools_result)
    resources_result = MagicMock()
    resources_result.resources = resources
    sess.list_resources = AsyncMock(return_value=resources_result)
    # CONCEPT:AU-ECO.mcp.cross-process-skill-harvest — a probe now also READS
    # each ``skill://`` body, so a fake child must serve one.
    bodies = bodies or {}

    async def _read(uri):
        return _fake_resource_body(bodies.get(str(uri), f"# {uri}\n\nbody"))

    sess.read_resource = AsyncMock(side_effect=_read)
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
            # The body is what makes the skill promotable to a runnable
            # resource; a name-only catalog entry never could be.
            "instructions": "# skill://onboarding/SKILL.md\n\nbody",
        }
    ]


async def test_probe_server_lists_skills_for_an_already_mounted_child(tmp_path):
    """D-2.2-2.3-1: an already-mounted child's probe used to skip the ``skills``
    key entirely (only the cold-probe branch called ``_probe_skills``), so a
    mounted server's Skills-over-MCP resources were invisible to
    ``find``/``find_tools`` until it happened to be probed cold at least once.
    A mounted child's live primary session is now reused to list resources."""
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "manage containers")]})
    await mux.mount_child(CNT)
    mux.sessions[CNT] = _fake_session_with_resources(
        [(CNT_TOOL, "manage containers")],
        [_fake_skill_resource("skill://onboarding/SKILL.md", "onboard a user")],
    )

    info = await mux.probe_server(CNT)

    assert info["error"] is None
    assert info["tools"][0]["name"] == CNT_TOOL
    assert info["skills"] == [
        {
            "name": "onboarding",
            "uri": "skill://onboarding/SKILL.md",
            "description": "onboard a user",
            # CONCEPT:AU-ECO.mcp.cross-process-skill-harvest landed after this
            # test was first written and now harvests every catalogued skill's
            # body over the same probe session, mounted-child path included.
            "instructions": "# skill://onboarding/SKILL.md\n\nbody",
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
    # A tool binds as "the default delegate, scoped to this one tool" -- the
    # skill_name is load-bearing, not decoration: both execute_capability and
    # run_agent reject tool_server without it.
    assert tool_hit["bind"] == {
        "tool_server": CNT,
        "allowed_tools": [CNT_TOOL],
        "skill_name": "agent-utilities-expert",
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


async def test_semantic_scoring_covers_skills_not_only_tools(tmp_path):
    """Skills must get the SAME embedding term tools do.

    Waves 1-5 merge gate regression: ``_embed_semantic_scores`` iterated
    ``info["tools"]`` only, so ``semantic.get(skill, 0.0)`` in ``discover_tools``
    was always 0.0 and skills were ranked on token overlap alone while tools
    additionally carried a cosine term. Whenever the embedder is warm — the
    production condition this whole feature exists for — skills were
    structurally under-ranked against tools for exactly the intent-similarity
    queries the feature was built to serve. The existing ranking test set
    ``_kg_call`` rather than ``_embed_fn``, so ``_embed_fn is None`` and
    ``_embed_semantic_scores`` returned before reaching the buggy loop — it only
    ever exercised the token-overlap-only regime where the two kinds happen to
    be symmetric.
    """
    mux = _mux_with_children(tmp_path, {CNT: [(CNT_TOOL, "manage docker containers")]})
    mux._kg_call = AsyncMock(return_value=None)  # type: ignore[method-assign]
    mux._probe_cache[CNT] = {
        "tools": [
            {"name": CNT_TOOL, "description": "alpha", "inputSchema": {}},
        ],
        "skills": [
            {
                "name": "container-runbook",
                "uri": "skill://container-runbook/SKILL.md",
                "description": "beta",
            }
        ],
        "error": None,
    }

    # A deterministic "embedder": the query vector matches the SKILL's text and
    # is orthogonal to the tool's, so a correctly-covered skill outranks the tool.
    def _embed(texts):
        out = []
        for text in texts:
            lowered = text.lower()
            if "beta" in lowered or "runbook" in lowered:
                out.append([1.0, 0.0])
            elif "alpha" in lowered or CNT_TOOL in lowered:
                out.append([0.0, 1.0])
            else:
                out.append([1.0, 0.0])  # the query
        return out

    mux._embed_fn = _embed  # type: ignore[assignment]

    semantic: dict[str, float] = {}
    await mux._embed_semantic_scores("beta runbook", mux._probe_cache, semantic)

    assert semantic.get("container-runbook", 0.0) > 0.0, (
        "the skill received no semantic score — skills are not in the same "
        "ranking space as tools"
    )
    assert semantic.get(CNT_TOOL, 0.0) == 0.0

    discovery = await mux.discover_tools("beta runbook", top_k=5)
    results = discovery["results"]
    skill_hit = next(r for r in results if r["kind"] == "skill")
    tool_hits = [r for r in results if r["kind"] == "tool"]
    assert not tool_hits or skill_hit["score"] > tool_hits[0]["score"]


async def test_semantic_cache_key_separates_a_skill_from_a_same_named_tool(tmp_path):
    """A skill and a tool may share a name on one server; they must not share
    one cached embedding."""
    mux = _mux_with_children(tmp_path, {CNT: [("deploy", "tool description")]})
    mux._kg_call = AsyncMock(return_value=None)  # type: ignore[method-assign]
    mux._probe_cache[CNT] = {
        "tools": [{"name": "deploy", "description": "tool description"}],
        "skills": [{"name": "deploy", "description": "skill description"}],
        "error": None,
    }
    embedded: list[str] = []

    def _embed(texts):
        embedded.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    mux._embed_fn = _embed  # type: ignore[assignment]
    await mux._embed_semantic_scores("deploy", mux._probe_cache, {})

    assert f"{CNT}::tools::deploy" in mux._tool_embeddings
    assert f"{CNT}::skills::deploy" in mux._tool_embeddings
    assert any("tool description" in t for t in embedded)
    assert any("skill description" in t for t in embedded)
