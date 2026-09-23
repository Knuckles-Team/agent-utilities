"""EH-044 / EH-045: A2A routing and tool exposure are budgeted assemblies with
fallbacks; a routed graph is published with its committed record as evidence."""

from __future__ import annotations

from typing import Any

import pytest

from agent_utilities.decide.consumers.assembly import (
    Assembled,
    Assembler,
    decision_evidence,
    text_digest,
)
from agent_utilities.decide.consumers.graphos import route_a2a_task, tool_subset

GRAPH = {"graph_id": "graph.triage", "version": "1"}
SOLVED = {
    "record": {"record_id": "decision:abc", "outcome": {"outcome": "solved"}},
    "graph": GRAPH,
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
COMMITTED = {
    "record_id": "decision:abc",
    "component": {
        "component": {"kind": "decision_record", "definition_digest": "sha256:rec"}
    },
}


class _Graphs:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.requests: list[Any] = []
        self.commits: list[Any] = []
        self.published: list[Any] = []

    async def assemble(self, request: Any) -> Any:
        self.requests.append(request)
        return self.result

    async def commit_decision(self, request: Any) -> Any:
        self.commits.append(request)
        return COMMITTED

    async def publish_graph(
        self, draft, context, *, evidence=None, idempotency_key=None
    ):
        self.published.append((draft, context, evidence))
        return {"graph_id": draft["graph_id"]}


async def _context(record: Any) -> dict[str, Any]:
    return {"record": record}


async def test_tool_exposure_is_the_budgeted_subset_and_never_committed() -> None:
    graphs = _Graphs(SOLVED)
    assembler = Assembler(graphs, "t", commit_context=_context)
    tools, answer = await tool_subset(
        assembler,
        lambda reasons: ["all"],
        context_budget_tokens=4000,
        capabilities=["eg:capability/retrieval"],
    )
    assert tools == ["tool.search"] and answer.reason == "solved"
    request = graphs.requests[0]
    assert request["candidates"]["kinds"] == ["tool"]
    assert request["requirements"]["constraints"] == {
        "require_tools": True,
        "context_budget_tokens": 4000,
    }
    assert graphs.commits == [], "evaluate-only: no per-request commit"


async def test_an_uncovered_task_keeps_the_current_exposure() -> None:
    assembler = Assembler(_Graphs(ABSTAINED), "t")
    tools, answer = await tool_subset(
        assembler,
        lambda reasons: ["all", *reasons],
        context_budget_tokens=4000,
        capabilities=["eg:capability/x"],
    )
    assert tools == ["all", "uncovered_capability"]
    assert answer.reason == "abstained: uncovered_capability"


async def test_a2a_routing_takes_task_iris_a_budget_and_templates() -> None:
    graphs = _Graphs(ABSTAINED)
    template = {"component_id": "graph.triage", "definition_digest": "sha256:x"}
    answer = await route_a2a_task(
        Assembler(graphs, "t"),
        lambda reasons: "current",
        task_iris=["eg:task/research"],
        context_budget_tokens=8000,
        templates=[template],
    )
    assert answer.fallback == "current"
    request = graphs.requests[0]
    assert request["templates"] == [template]
    assert request["requirements"]["tasks"] == ["eg:task/research"]
    assert request["requirements"]["constraints"]["context_budget_tokens"] == 8000
    assert request["requirements"]["unmapped_task_digests"] == []


async def test_untyped_text_travels_only_as_its_digest() -> None:
    graphs = _Graphs(ABSTAINED)
    await route_a2a_task(
        Assembler(graphs, "t"), lambda r: None, text="find the billing owner"
    )
    requirements = graphs.requests[0]["requirements"]
    assert requirements["unmapped_task_digests"] == [
        text_digest("find the billing owner")
    ]
    assert "billing" not in str(graphs.requests[0])


async def test_a_routed_graph_is_published_with_its_committed_record() -> None:
    graphs = _Graphs(SOLVED)
    assembler = Assembler(graphs, "t", commit_context=_context)
    answer = await route_a2a_task(
        assembler, lambda r: None, task_iris=["eg:task/research"]
    )
    await assembler.publish_routed(answer, {"ctx": "minted"}, idempotency_key="k")
    draft, context, evidence = graphs.published[0]
    assert draft == GRAPH and context == {"ctx": "minted"}
    assert (
        evidence
        == decision_evidence(COMMITTED)
        == {
            "component_id": "decision:abc",
            "kind": "decision_record",
            "definition_digest": "sha256:rec",
        }
    )


async def test_an_uncommitted_answer_is_never_published() -> None:
    assembler = Assembler(_Graphs(SOLVED), "t")
    with pytest.raises(ValueError):
        await assembler.publish_routed(Assembled(SOLVED), {"ctx": "minted"})
