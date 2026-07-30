"""``find`` ranks fleet skills and fleet tools in one result set (CONCEPT:AU-KG.retrieval.unified-capability-contract).

``_find_capability`` (the ``find`` intent verb's handler) widens its local
verb-candidate list with a fleet-wide probe via ``mcp._fleet_mux.discover_tools``.
Since ``discover_tools`` now merges ``Tool`` and Skills-over-MCP ``skill://``
candidates into one ranked list (see ``tests/test_multiplexer_skills_over_mcp.py``),
``find``'s ``fleet_results`` field inherits that unification for free — this
test proves the whole call path, not just the multiplexer method in isolation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from agent_utilities.mcp.tools import intent_tools


class _FakeFleetMux:
    def __init__(self, discovery: dict) -> None:
        self._discovery = discovery
        self.discover_tools = AsyncMock(return_value=discovery)

    def session_loaded(self, _key: object) -> set:
        return set()


class _FakeMCP:
    def __init__(self, fleet_mux: _FakeFleetMux) -> None:
        self._fleet_mux = fleet_mux


@pytest.mark.asyncio
async def test_find_capability_fleet_results_span_both_kinds() -> None:
    discovery = {
        "results": [
            {
                "kind": "tool",
                "server": "github-mcp",
                "tool": "list_issues",
                "prefixed_name": "gith__list_issues",
                "description": "List open issues",
                "score": 0.91,
                "mountable": True,
                "mounted": False,
                "bind": {"tool_server": "github-mcp", "allowed_tools": ["list_issues"]},
            },
            {
                "kind": "skill",
                "server": "docs-mcp",
                "skill": "release-notes-writer",
                "uri": "skill://release-notes-writer/SKILL.md",
                "description": "Draft release notes from a changelog",
                "score": 0.88,
                "mountable": True,
                "mounted": False,
                "bind": {
                    "tool_server": "docs-mcp",
                    "skill_name": "release-notes-writer",
                },
            },
        ],
        "unavailable": {},
    }
    mcp = _FakeMCP(_FakeFleetMux(discovery))

    payload = await intent_tools._find_capability(
        mcp, "draft release notes and list open issues", top_k=5
    )

    fleet_results = payload["fleet_results"]
    kinds = {item["kind"] for item in fleet_results}
    assert kinds == {"tool", "skill"}

    tool_hit = next(item for item in fleet_results if item["kind"] == "tool")
    skill_hit = next(item for item in fleet_results if item["kind"] == "skill")

    # Both bind through the SAME keyword surface — a caller does not need to
    # branch on kind before spreading `bind` into graph_orchestrate.
    assert set(tool_hit["bind"]) <= {"tool_server", "allowed_tools"}
    assert set(skill_hit["bind"]) <= {"tool_server", "skill_name"}
