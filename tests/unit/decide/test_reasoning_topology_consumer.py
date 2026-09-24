"""EH-474: the reasoning-graph topology is EG's decision, the ladder its fallback."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from agent_utilities import decide
from agent_utilities.decide.consumers.reasoning_topology import (
    QUESTION,
    RUNNABLE,
    reasoning_options,
    select_reasoning_topology,
)
from agent_utilities.decide.points import POINTS, LogMode
from agent_utilities.decide.schemas import feature_schema_body
from agent_utilities.mcp import kg_server
from agent_utilities.mcp.tools import agent_execution_tools
from tests.unit.decide.fakes import FakeTransport, abstained, acted


def _select(task: str = "Compare three plans and pick one") -> Any:
    return asyncio.run(select_reasoning_topology(task))


def test_the_point_is_registered_logged_and_has_a_schema() -> None:
    point = POINTS[QUESTION]
    assert (point.row, point.kind, point.safety) == ("EH-474", "route", "ordinary")
    assert point.log_mode is LogMode.ALWAYS
    names = {f["name"] for f in feature_schema_body(QUESTION)["features"]}
    assert names == {"rung", "loop_budget", "heuristic", "shape_fit"}


def test_every_runnable_topology_is_one_option_keyed_like_the_schema() -> None:
    options = reasoning_options(RUNNABLE, "cot")
    assert [o.option_id for o in options] == [
        "cot",
        "self_consistent_cot",
        "tot_bfs",
        "tot_dfs",
    ]
    assert [o.numbers["rung"] for o in options] == [0.0, 1.0, 2.0, 3.0]
    assert [o.numbers["heuristic"] for o in options] == [1.0, 0.0, 0.0, 0.0]
    assert all(set(o.texts) == {"shape"} and o.texts["shape"] for o in options)


def test_eg_s_calibrated_choice_is_the_topology(eg: FakeTransport) -> None:
    eg.answer = acted("tot_bfs")
    selection = _select()
    assert (selection.topology, selection.decided) == ("tot_bfs", True)
    assert selection.record_id is not None
    request = eg.requests[0]
    assert request["question"]["question_id"] == QUESTION
    assert request["params"] == [
        {
            "name": "task",
            "value": {"type": "text", "value": "Compare three plans and pick one"},
        }
    ]
    assert eg.op_names() == ["commit"], "the record is logged for later crediting"


def test_an_abstention_runs_the_cheapest_rung_and_is_still_recorded(
    eg: FakeTransport,
) -> None:
    eg.answer = abstained("insufficient_confidence")
    selection = _select()
    assert (selection.topology, selection.decided) == ("cot", False)
    assert selection.reason.startswith("abstained")
    assert selection.report()["by"] == "fallback"
    assert eg.op_names() == ["commit"]


def test_a_foreign_option_is_never_run(eg: FakeTransport) -> None:
    eg.answer = acted("rap")
    assert _select().topology == "cot"


def test_without_an_engine_the_ladder_answers() -> None:
    token = decide.use_runner(None)
    try:
        selection = _select()
    finally:
        decide.reset_runner(token)
    assert (selection.topology, selection.decided) == ("cot", False)


class _Step:
    def __init__(self, output: Any) -> None:
        self.output = output


class _FinalAgent:
    def run_sync(self, _prompt: str) -> _Step:
        return _Step(
            agent_execution_tools.ReasoningStepOutput(
                summary="answered", content="42", is_final=True
            )
        )


class _Engine:
    def __init__(self) -> None:
        self.nodes: dict[str, Any] = {}

    def add_node(self, node_id: str, node_type: str, props: dict) -> None:
        self.nodes[node_id] = (node_type, props)


class _Tools:
    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}

    def tool(self, *, name: str, **_metadata: Any) -> Any:
        def capture(function: Any) -> Any:
            self.tools[name] = function
            return function

        return capture


def test_the_served_reason_action_asks_eg_when_topology_is_auto(
    eg: FakeTransport, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = _Engine()
    monkeypatch.setattr(kg_server, "_get_engine", lambda: engine)
    monkeypatch.setattr(
        agent_execution_tools, "_create_reasoning_agent", lambda *_a: _FinalAgent()
    )
    eg.answer = acted("self_consistent_cot")
    mcp = _Tools()
    agent_execution_tools.register_agent_execution_tools(mcp)
    payload = json.loads(
        asyncio.run(
            mcp.tools["graph_agents"](
                action="reason",
                task="What is 6*7?",
                context="",
                context_ref="",
                max_fan_out=5,
                max_steps=10,
                host="",
                container_id="",
                options_json="{}",
                topology="auto",
                num_samples=2,
                branching_factor=3,
                beam_width=3,
                max_depth=5,
            )
        )
    )
    assert payload["topology"] == "self_consistent_cot"
    assert payload["answer"] == "42"
    assert payload["selection"]["by"] == "decide"
    assert payload["selection"]["record_id"] is not None
    assert eg.requests[0]["question"]["question_id"] == QUESTION
