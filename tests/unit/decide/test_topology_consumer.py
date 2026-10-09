"""The swarm topology is ONE AgentAssemble question; the tree is the fallback.

The plan EG certifies decides the executor family and bounds the runtime
admission; there is no AU selector, success rate or outcome store.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Iterator
from typing import Any

import pytest

from agent_utilities.decide.consumers.assembly import Assembler
from agent_utilities.decide.consumers.topology import (
    CAPACITY_DENIED,
    INDEPENDENT_SUBTASKS,
    TaskShape,
    TopologyAsk,
    ask_topology,
    family_of,
    install_topology,
    plan_of,
    topology_request,
)
from agent_utilities.decide.topology import SWARM_NS

# agent_utilities.graph's package __init__ imports .builder, which pulls in
# the compiled epistemic_graph.numeric kernel at import time; skip the whole
# module cleanly when it isn't built, rather than erroring out collection.
pytest.importorskip("epistemic_graph.numeric")
from agent_utilities.graph.plan_admission import admission_from_plan
from agent_utilities.graph.subagent_patterns import (
    SubagentPattern,
    SubagentPatternDecision,
    SubagentPatternRouter,
)
from agent_utilities.graph.topology_engine import TopologyAdmissionError
from tests.unit.decide.fakes import FakeGraphs

RECORD_ID = "decision:" + "ab" * 32
TEMPLATE_REF = {"graph_id": "swarm:fan-out-join", "entry_revision": 1}
PLAN = {
    "class_iri": SWARM_NS + "FanOutJoin",
    "slots": [
        {"node_id": "lead", "width": 1, "rounds": 1, "tokens": 4000},
        {"node_id": "worker", "width": 4, "rounds": 1, "tokens": 16000},
    ],
    "stop": {"rule": "max_rounds", "n": 1},
    "lease": {
        "per_cell": [{"cell_id": "cell-llm", "class": "llm_generator", "amount": 5}],
        "priority": "orchestration",
    },
    "allowances": [],
}
SOLVED = {
    "record": {
        "record_id": RECORD_ID,
        "outcome": {"outcome": "solved", "topology": PLAN},
    },
    "agents": [{"agent_id": "agent:lead"}],
}
ABSTAINED = {
    "record": {
        "record_id": RECORD_ID,
        "outcome": {
            "outcome": "abstained",
            "reasons": [
                {
                    "reason": "infeasible",
                    "constraints": ["lease:cell-llm:llm_generator"],
                }
            ],
        },
    }
}
CAPACITY_DENIED_RESULT = {
    "record": {
        "record_id": RECORD_ID,
        "outcome": {
            "outcome": "abstained",
            "reasons": [{"reason": CAPACITY_DENIED}],
        },
    }
}


class _SequencedGraphs:
    """An L3 agent-graph client answering each successive ``assemble`` call
    with the next result of a fixed sequence (repeats the last past the end).
    """

    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = results
        self.requests: list[Any] = []

    async def assemble(self, request: Any) -> Any:
        self.requests.append(request)
        index = min(len(self.requests) - 1, len(self.results) - 1)
        return self.results[index]


@pytest.fixture
def asker() -> Iterator[list[FakeGraphs]]:
    box: list[FakeGraphs] = []
    yield box
    install_topology(None)


def _install(box: list[FakeGraphs], result: dict[str, Any]) -> FakeGraphs:
    graphs = FakeGraphs(result)
    box.append(graphs)
    install_topology(Assembler(graphs, "tenant-t"), asyncio.run, lambda: [TEMPLATE_REF])
    return graphs


def _select() -> SubagentPatternDecision:
    return SubagentPatternRouter().select_pattern(
        task_complexity=3, parallelizable=True, specialist_count=4
    )


def test_the_topology_question_is_one_assembly_request() -> None:
    ask = TopologyAsk(
        task_classes=(INDEPENDENT_SUBTASKS,),
        subtasks=4,
        deadline_ms=60_000,
        max_width=8,
        cells=("cell-llm",),
    )
    request = topology_request(
        "tenant-t", ask, [TEMPLATE_REF], task_iris=["eg:task/research"]
    )
    assert request["templates"] == [TEMPLATE_REF]
    assert request["requirements"]["tasks"] == ["eg:task/research"]
    topology = request["requirements"]["topology"]
    assert topology["task_classes"] == [INDEPENDENT_SUBTASKS]
    assert topology["deadline_ms"] == 60_000
    assert topology["caps"] == {"max_width": 8, "max_depth": 8, "max_rounds": 8}
    assert topology["capacity"] == {"cells": ["cell-llm"], "priority": "orchestration"}


@pytest.mark.spec("AU-CONTROL-R007")
def test_eg_s_plan_decides_the_family_and_the_reasoning_names_the_record(asker) -> None:
    graphs = _install(asker, SOLVED)
    decision = _select()
    assert decision.pattern is SubagentPattern.FAN_OUT
    assert f"EG topology plan {RECORD_ID}" in decision.reasoning
    asked = graphs.requests[0]["requirements"]["topology"]
    assert asked["task_classes"][-1] == INDEPENDENT_SUBTASKS
    assert asked["subtasks"] == 4


@pytest.mark.spec("AU-CONTROL-R007")
def test_an_abstention_keeps_the_tree_s_family(asker) -> None:
    graphs = _install(asker, ABSTAINED)
    decision = _select()
    assert decision.pattern is SubagentPattern.FAN_OUT
    assert "EG topology plan" not in decision.reasoning
    assert graphs.requests, "the question was still asked"


def test_without_an_asker_the_tree_decides_alone() -> None:
    install_topology(None)
    assert "EG" not in _select().reasoning


def test_plan_and_family_projection() -> None:
    assert plan_of(SOLVED) == PLAN
    assert plan_of(ABSTAINED) is None
    assert family_of(PLAN) == "fan_out"
    assert family_of({"class_iri": "urn:tenant:MyShape"}) is None
    assert TaskShape(3, False, False, 1, False).ask().task_classes == (
        SWARM_NS + "TaskShape",
    )


def test_the_runtime_admission_is_exactly_the_committed_plan() -> None:
    admission = admission_from_plan(
        PLAN, record_id=RECORD_ID, tenant="tenant-t", delegation_id="delegation:1"
    )
    assert (admission.max_fan_out, admission.max_parallelism, admission.max_nodes) == (
        4,
        5,
        5,
    )
    assert (admission.max_depth, admission.max_tokens) == (2, 20_000)
    assert admission.decision_record_id == RECORD_ID
    other = dataclasses.replace(admission, decision_record_id="decision:" + "cd" * 32)
    assert other.digest != admission.digest, "the digest binds the record"


def test_a_plan_admission_names_its_record_and_positive_widths() -> None:
    with pytest.raises(TopologyAdmissionError):
        admission_from_plan(PLAN, record_id="", tenant="t", delegation_id="d:1")
    broken = {**PLAN, "slots": [{"node_id": "w", "width": 0, "rounds": 1}]}
    with pytest.raises(TopologyAdmissionError):
        admission_from_plan(
            broken, record_id=RECORD_ID, tenant="t", delegation_id="d:1"
        )


def _ask(**over: Any) -> TopologyAsk:
    return TopologyAsk(task_classes=(INDEPENDENT_SUBTASKS,), max_width=8, **over)


def test_capacity_denial_gets_exactly_one_narrower_redecision() -> None:
    graphs = _SequencedGraphs([CAPACITY_DENIED_RESULT, SOLVED])
    assembler = Assembler(graphs, "tenant-t")
    answer = asyncio.run(ask_topology(assembler, _ask(), [TEMPLATE_REF]))
    assert answer.reason == "solved"
    assert plan_of(answer.result) == PLAN
    assert len(graphs.requests) == 2
    first_caps = graphs.requests[0]["requirements"]["topology"]["caps"]
    second_caps = graphs.requests[1]["requirements"]["topology"]["caps"]
    assert first_caps["max_width"] == 8
    assert second_caps["max_width"] == 4, "the re-decision asks narrower, never wider"


def test_a_second_capacity_denial_is_not_re_decided_again() -> None:
    graphs = _SequencedGraphs([CAPACITY_DENIED_RESULT, CAPACITY_DENIED_RESULT])
    assembler = Assembler(graphs, "tenant-t")
    answer = asyncio.run(ask_topology(assembler, _ask(), [TEMPLATE_REF]))
    assert answer.reason == f"abstained: {CAPACITY_DENIED}"
    assert len(graphs.requests) == 2, "at most one re-decision, never a loop"


def test_a_non_capacity_abstention_is_not_re_decided() -> None:
    graphs = _SequencedGraphs([ABSTAINED, SOLVED])
    assembler = Assembler(graphs, "tenant-t")
    answer = asyncio.run(ask_topology(assembler, _ask(), [TEMPLATE_REF]))
    assert answer.reason == "abstained: infeasible"
    assert len(graphs.requests) == 1, "only a capacity denial triggers a re-decision"
