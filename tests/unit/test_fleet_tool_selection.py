"""Task-aware tool selection + no-silent-hallucination fallthrough (F1).

CONCEPT:AU-ORCH.execution.task-aware-tool-selection / AU-ORCH.execution.no-silent-hallucination —
a fleet server can expose hundreds of tools (container-manager-mcp: 314). Binding
every schema to one agent makes the LLM call hang and the run silently degrades to a
toolless graph that fabricates a plausible answer. These tests pin the selector (bind
only the top-K task-relevant tools) and the fail-loud path (a resolved fleet-server
failure surfaces as degraded, not a hallucination).
"""

from __future__ import annotations

import asyncio

from agent_utilities.orchestration.agent_runner import (
    _delegation_degraded,
    _fleet_server_failed_result,
    _lexical_top_k_tools,
    _match_designated_to_names,
    _select_relevant_tool_names,
)

# A realistic slice of container-manager-mcp's 314 tools.
TOOLS = [
    {"name": "cm_docker_ps", "description": "List running docker containers"},
    {
        "name": "cm_docker_list_containers",
        "description": "List containers with status/image",
    },
    {
        "name": "cm_volume_operations__prune_volumes",
        "description": "Prune unused volumes",
    },
    {"name": "cm_network_create", "description": "Create a docker network"},
    {"name": "cm_image_pull", "description": "Pull an image from a registry"},
    {"name": "cm_k8s_scale_deployment", "description": "Scale a kubernetes deployment"},
] + [{"name": f"cm_misc_{i}", "description": f"unrelated tool {i}"} for i in range(40)]


def test_lexical_ranks_relevant_tools_first():
    top = _lexical_top_k_tools(
        "list running docker containers name image status", TOOLS, 5
    )
    assert "cm_docker_ps" in top or "cm_docker_list_containers" in top
    # unrelated tools must not crowd out the relevant ones
    assert top[0].startswith("cm_")
    assert not any(n.startswith("cm_misc_") for n in top[:2])


def test_lexical_empty_when_no_overlap():
    assert (
        _lexical_top_k_tools(
            "xyzzy quux", [{"name": "cm_docker_ps", "description": "list"}], 5
        )
        == []
    )


def test_match_designated_ids_to_names():
    ranked = [
        "resource:cm_docker_ps",
        "container-manager-mcp__cm_image_pull",
        "srv:unrelated",
    ]
    names = {"cm_docker_ps", "cm_image_pull", "cm_network_create"}
    matched = _match_designated_to_names(ranked, names)
    assert matched == ["cm_docker_ps", "cm_image_pull"]


def test_select_returns_none_for_small_server():
    # <= _MAX_BOUND_TOOLS: bind wholesale (no filtering).
    small = TOOLS[:6]
    assert (
        asyncio.run(_select_relevant_tool_names(None, "list containers", small)) is None
    )


def test_select_caps_large_server_to_relevant_subset():
    # engine=None -> KG path unavailable -> lexical -> selects the docker/list tools.
    sel = asyncio.run(
        _select_relevant_tool_names(
            None, "list running docker containers", TOOLS, max_tools=20
        )
    )
    assert sel is not None
    assert len(sel) <= 20
    assert any("docker" in n for n in sel)


def test_select_hard_caps_when_nothing_matches():
    # No lexical overlap -> hard cap to max_tools (never hand the agent 300+ schemas).
    sel = asyncio.run(
        _select_relevant_tool_names(None, "zzz nomatch", TOOLS, max_tools=10)
    )
    assert sel is not None and len(sel) == 10


def test_fleet_failed_result_is_degraded():
    res = _fleet_server_failed_result("container-manager-mcp", "connection timeout")
    assert res["metadata"]["degraded"] is True
    assert res["metadata"]["outcome"] == "fleet_server_failed"
    # and the F2/F5 detector flags it so it is never recorded as a clean success
    assert _delegation_degraded(res) is True
