"""EH-044 / EH-045: A2A routing and tool exposure are assemblies with fallbacks."""

from __future__ import annotations

from typing import Any

from agent_utilities.decide.consumers.assembly import Assembler
from agent_utilities.decide.consumers.graphos import route_a2a_task, tool_subset

SOLVED = {
    "record": {"outcome": {"outcome": "solved"}},
    "agents": [{"agent_id": "a", "tools": [{"component_id": "tool.search"}]}],
}
ABSTAINED = {
    "record": {
        "outcome": {
            "outcome": "abstained",
            "reasons": [{"reason": "uncovered_capability"}],
        }
    }
}


class _Graphs:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result, self.requests, self.commits = result, [], []

    async def assemble(self, request: Any) -> Any:
        self.requests.append(request)
        return self.result

    async def commit_decision(self, request: Any) -> Any:
        self.commits.append(request)


async def _context(record: Any) -> dict[str, Any]:
    return {"record": record}


async def test_tool_exposure_is_the_assembled_subset_and_never_committed() -> None:
    graphs = _Graphs(SOLVED)
    assembler = Assembler(graphs, "t", commit_context=_context)
    tools, answer = await tool_subset(
        assembler, ["eg:capability/retrieval"], 4000, lambda reasons: ["all"]
    )
    assert tools == ["tool.search"] and answer.reason == "solved"
    request = graphs.requests[0]
    assert request["candidates"]["kinds"] == ["tool"]
    assert request["requirements"]["constraints"] == {"context_budget_tokens": 4000}
    assert graphs.commits == [], "evaluate-only: no per-request commit"


async def test_an_uncovered_task_keeps_the_current_exposure() -> None:
    assembler = Assembler(_Graphs(ABSTAINED), "t")
    tools, answer = await tool_subset(
        assembler, ["eg:capability/x"], 4000, lambda reasons: ["all", *reasons]
    )
    assert tools == ["all", "uncovered_capability"]
    assert answer.reason == "abstained: uncovered_capability"


async def test_a2a_routing_assembles_over_the_published_templates() -> None:
    graphs = _Graphs(ABSTAINED)
    template = {"component_id": "graph.triage", "definition_digest": "sha256:x"}
    answer = await route_a2a_task(
        Assembler(graphs, "t"), ["eg:capability/x"], [template], lambda r: "current"
    )
    assert answer.fallback == "current"
    assert graphs.requests[0]["templates"] == [template]
