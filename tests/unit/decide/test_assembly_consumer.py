"""graph.assemble() before LLM composition; the mapping is a claim."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from typing import Any

import pytest

from agent_utilities.decide.consumers.assembly import Assembler, install_assembler
from agent_utilities.knowledge_graph.enrichment.synthesize import synthesize_agent
from tests.unit.decide.fakes import FakeGraphs

GOAL = "find the owner of the billing service"
AGENT = {
    "agent_id": "agent:researcher",
    "tools": [{"component_id": "tool.search"}],
    "skills": [{"component_id": "skill.cite"}],
    "system_prompt": {"component_id": "prompt.research"},
    "model_identity": "model:qwen",
}


def _llm(prompt: str) -> str:
    if "task IRIs" in prompt:
        return json.dumps(["eg:task/research", "eg:task/invented"])
    return json.dumps({"name": "LLM agent", "tools": [], "skills": []})


@pytest.fixture
def installed() -> Iterator[list[FakeGraphs]]:
    box: list[FakeGraphs] = []
    yield box
    install_assembler(None)


def _install(box: list[FakeGraphs], result: dict[str, Any], **kw: Any) -> FakeGraphs:
    graphs = FakeGraphs(result)
    box.append(graphs)
    install_assembler(Assembler(graphs, "tenant-t", **kw), asyncio.run)
    return graphs


def test_eg_assembles_the_agent_and_the_goal_is_only_a_claim(installed) -> None:
    solved = {"record": {"outcome": {"outcome": "solved"}}, "agents": [AGENT]}

    async def context(record: Any) -> dict[str, Any]:
        return {"record": record, "context": "minted-by-graph-os"}

    graphs = _install(installed, solved, commit_context=context)
    spec = synthesize_agent(GOAL, lambda goal, limit: [], _llm)
    assert (spec.name, spec.tools, spec.model) == (
        "agent:researcher",
        ["tool.search"],
        "model:qwen",
    )
    request = graphs.requests[0]
    mapping = request["requirements"]["task_mappings"][0]
    assert mapping["task_iris"] == ["eg:task/research"], "unknown IRIs are dropped"
    assert GOAL not in json.dumps(request), "free text never reaches EG"
    assert graphs.commits and graphs.commits[0]["context"] == "minted-by-graph-os"


def test_an_abstention_leaves_the_llm_composition(installed) -> None:
    abstained = {
        "record": {
            "outcome": {
                "outcome": "abstained",
                "reasons": [{"reason": "uncovered_capability", "iri": "eg:x"}],
            }
        }
    }
    graphs = _install(installed, abstained)
    spec = synthesize_agent(GOAL, lambda goal, limit: [], _llm)
    assert spec.name == "LLM agent"
    assert graphs.requests and graphs.commits == []


def test_without_an_assembler_nothing_changes() -> None:
    calls: list[str] = []

    def llm(prompt: str) -> str:
        calls.append(prompt)
        return json.dumps({"name": "LLM agent"})

    assert synthesize_agent(GOAL, lambda goal, limit: [], llm).name == "LLM agent"
    assert len(calls) == 1, "no mapping call when EG is not wired"
