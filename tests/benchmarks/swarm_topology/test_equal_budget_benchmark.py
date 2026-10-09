"""The synthetic equal-budget benchmark and the width-quality promotion gate."""

from __future__ import annotations

import asyncio

import pytest

from agent_utilities.decide.consumers.assembly import Assembler
from agent_utilities.decide.topology import SWARM_NS
from agent_utilities.decide.topology.benchmark import (
    BASELINES,
    EQUAL_BUDGET,
    TASKS,
    Answer,
    Task,
    eg_planner,
    run_policy,
    run_suite,
    width_buys_quality,
    wilson,
)
from tests.unit.decide.fakes import FakeGraphs


@pytest.mark.spec("AU-CONTROL-R020")
def test_every_baseline_runs_under_the_same_budget() -> None:
    rows = {row.policy: row for row in run_suite(BASELINES)}
    assert set(rows) == set(BASELINES)
    assert all(row.answered + row.abstentions == len(TASKS) for row in rows.values())
    assert rows["single"].successes == 2, "atomic + chain only"
    assert rows["goose-5"].successes == 2, "pair + survey-4; survey-8 needs 6 of 6"
    for row in rows.values():
        low, high = row.success_interval
        assert 0.0 <= low <= row.successes / row.answered <= high <= 1.0


@pytest.mark.spec("AU-CONTROL-R020")
def test_an_abstention_is_not_a_failure() -> None:
    row = run_policy("abstains", lambda task, budget: Answer(None))
    assert (row.answered, row.abstentions, row.successes) == (0, len(TASKS), 0)
    assert row.success_interval == (0.0, 1.0)


@pytest.mark.spec("AU-CONTROL-R020")
def test_the_eg_planner_asks_the_topology_question_under_the_budget() -> None:
    plan = {
        "class_iri": SWARM_NS + "FanOutJoin",
        "slots": [{"node_id": "worker", "width": 4, "rounds": 1, "tokens": 16000}],
        "makespan_ms": 30000,
    }
    graphs = FakeGraphs(
        {
            "record": {
                "record_id": "decision:1",
                "outcome": {"outcome": "solved", "topology": plan},
            }
        }
    )
    planner = eg_planner(Assembler(graphs, "tenant-t"), asyncio.run, lambda: [])
    task = Task("survey-4", "IndependentSubtasks", 4, frozenset({"FanOutJoin"}))
    assert planner(task, EQUAL_BUDGET) == Answer("FanOutJoin", 4, 16000, 30000)
    asked = graphs.requests[0]["requirements"]["topology"]
    assert asked["caps"]["max_width"] == EQUAL_BUDGET
    assert asked["task_classes"] == [SWARM_NS + "IndependentSubtasks"]


def test_the_width_gate_needs_non_overlapping_intervals() -> None:
    assert wilson(0, 0) == (0.0, 1.0)
    assert width_buys_quality(BASELINES["au-tree"]) is False, (
        "seven synthetic tasks cannot separate the intervals: exploration stays zero"
    )
