"""AU-CONTROL-R024/R025: one structured, provenanced plan per "how can I" question."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import pytest

from agent_utilities.decide.consumers.assembly import Assembler, install_assembler
from agent_utilities.decide.consumers.task_planner import (
    BASE_GUARDRAILS,
    TaskPlanner,
    install_task_planner,
    is_planning_question,
    keyword_task_mapper,
    process_task_planner,
)
from agent_utilities.decide.consumers.topology import install_topology
from agent_utilities.decide.topology import SWARM_NS
from tests.unit.decide.fakes import FakeGraphs

GOAL = "How can I research the outage, fix the bug and deploy the fix?"
RECORD_ID = "decision:" + "cd" * 32
AGENT = {
    "agent_id": "agent:fixer",
    "tools": [{"component_id": "tool.search"}],
    "skills": [{"component_id": "skill.debug"}],
    "system_prompt": {"component_id": "prompt.fix"},
    "model_identity": "model:qwen",
}
PLAN = {
    "class_iri": SWARM_NS + "FanOutJoin",
    "slots": [
        {"node_id": "lead", "width": 1},
        {"node_id": "worker", "width": 3},
        {"node_id": "merge", "width": 1},
    ],
    "stop": {"rule": "max_rounds", "n": 1},
}
SOLVED = {
    "record": {
        "record_id": RECORD_ID,
        "outcome": {"outcome": "solved", "topology": PLAN},
    },
    "agents": [AGENT],
}
ABSTAINED = {
    "record": {
        "record_id": RECORD_ID,
        "outcome": {
            "outcome": "abstained",
            "reasons": [{"reason": "uncovered_capability"}],
        },
    }
}
TEMPLATE_REF = {"graph_id": "swarm:fan-out-join", "entry_revision": 1}


@pytest.fixture(autouse=True)
def _clean() -> Iterator[None]:
    yield
    install_task_planner(None)
    install_assembler(None)
    install_topology(None)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("How can I deploy the billing service?", True),
        ("how should we migrate the database", True),
        ("What steps are needed to rotate a secret?", True),
        ("Which agents should review this change?", True),
        ("Plan how to onboard a connector", True),
        ("which agents call run_agent?", False),
        ("How does the supply chain affect our services?", False),
        ("list the open incidents", False),
    ],
)
def test_planning_questions_are_recognized(text: str, expected: bool) -> None:
    assert is_planning_question(text) is expected


def test_keyword_mapper_returns_native_tasks_in_plan_order() -> None:
    assert keyword_task_mapper(GOAL) == [
        "eg:task/research",
        "eg:task/implement",
        "eg:task/operate",
    ]


def _plan(planner: TaskPlanner, goal: str = GOAL) -> dict[str, Any]:
    return asyncio.run(planner.plan(goal)).to_dict()


def test_a_plan_without_eg_names_every_gap_and_guesses_nothing() -> None:
    plan = _plan(TaskPlanner())
    assert plan["tasks"] == ["eg:task/research", "eg:task/implement", "eg:task/operate"]
    assert [s["depends_on"] for s in plan["steps"]] == [[], ["step-1"], ["step-2"]]
    assert plan["topology"]["template"] == "swarm:pipeline"
    assert plan["topology"]["decided"] is False
    assert {a["source"] for a in plan["agents"]} == {"unassigned"}
    gaps = " ".join(plan["gaps"])
    for port in ("assembler", "topology_asker", "capability_search", "guardrail"):
        assert port in gaps
    evidence = {p["element"]: p["evidence"] for p in plan["provenance"]}
    assert evidence == {"tasks": "claim", "topology": "fallback"}
    assert len(plan["guardrails"]) == len(BASE_GUARDRAILS)
    assert GOAL not in json.dumps(plan), "the plan carries a digest, not the text"


def test_an_unmapped_goal_returns_only_the_gap() -> None:
    plan = _plan(TaskPlanner(), "How can I zzz the qqq?")
    assert plan["tasks"] == [] and plan["steps"] == [] and plan["agents"] == []
    assert plan["gaps"][0].startswith("unmapped_task")


async def _search(tasks: Sequence[str]) -> list[Mapping[str, Any]]:
    table = {
        "eg:task/research": [{"id": "a2a:researcher", "kind": "a2a_agent"}],
        "eg:task/operate": [{"id": "library:deployer", "kind": "agent_template"}],
    }
    return table.get(tasks[0], [])


async def _rules(tasks: Sequence[str]) -> list[Mapping[str, Any]]:
    return [{"id": "change-window", "rule": "Deploy inside the change window."}]


async def _workflow(tasks: Sequence[str]) -> Mapping[str, Any] | None:
    return {"id": "workflow:incident-fix", "steps": len(tasks)}


def test_eg_decides_components_and_topology_with_provenance() -> None:
    graphs = FakeGraphs(SOLVED)
    planner = TaskPlanner(
        assembler=Assembler(graphs, "tenant-t"),
        templates=lambda: [TEMPLATE_REF],
        capability_search=_search,
        guardrail_source=_rules,
        workflows=_workflow,
    )
    plan = _plan(planner)
    assert plan["topology"]["template"] == "swarm:fan-out-join"
    assert plan["topology"]["decided"] is True
    assert plan["agent_count"] == {"min": 5, "max": 5}
    by_slot = {a["slot"]: a for a in plan["agents"]}
    assert by_slot["lead"]["source"] == "a2a"
    assert by_slot["lead"]["reuses"] == "a2a:researcher"
    assert by_slot["merge"]["source"] == "agent_library"
    assert by_slot["worker"]["source"] == "assembled"
    assert by_slot["worker"]["model"] == "model:qwen"
    assert by_slot["worker"]["skills"] == ["skill.debug"]
    assert plan["workflow"]["source"] == "compiled_workflow"
    assert plan["guardrails"][-1]["evidence"] == "policy"
    decisions = [p for p in plan["provenance"] if p["evidence"] == "decision"]
    assert {p["element"] for p in decisions} == {"agents.components", "topology"}
    assert all(p["record_id"] == RECORD_ID for p in decisions)
    assert plan["gaps"] == []
    for request in graphs.requests:
        assert GOAL not in json.dumps(request), "free text never reaches EG"


def test_an_eg_abstention_is_a_gap_not_a_composition() -> None:
    planner = TaskPlanner(
        assembler=Assembler(FakeGraphs(ABSTAINED), "tenant-t"),
        templates=lambda: [TEMPLATE_REF],
    )
    plan = _plan(planner)
    assert any(g.startswith("assembly_abstained") for g in plan["gaps"])
    assert any(g.startswith("topology_abstained") for g in plan["gaps"])
    assert all(a["model"] is None and a["tools"] == [] for a in plan["agents"])
    assert plan["topology"]["decided"] is False


def test_the_process_planner_uses_the_installed_assembler_and_driver() -> None:
    graphs = FakeGraphs(SOLVED)
    install_assembler(Assembler(graphs, "tenant-t"), asyncio.run)
    install_topology(Assembler(graphs, "tenant-t"), asyncio.run, lambda: [TEMPLATE_REF])
    planner = process_task_planner()
    assert planner.driver is asyncio.run
    plan = _plan(planner)
    assert plan["topology"]["decided"] is True
    assert graphs.requests, "the installed assembler answered"


def test_an_installed_planner_wins() -> None:
    planner = TaskPlanner()
    install_task_planner(planner)
    assert process_task_planner() is planner
