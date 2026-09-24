"""EH-048: the swarm topology is a decision; the cost-ordered tree is the fallback."""

from __future__ import annotations

from agent_utilities.graph.subagent_patterns import (
    SubagentPattern,
    SubagentPatternDecision,
    SubagentPatternRouter,
)
from tests.unit.decide.fakes import FakeTransport, abstained, acted


def _select() -> SubagentPatternDecision:
    return SubagentPatternRouter().select_pattern(
        task_complexity=3, parallelizable=True, specialist_count=4
    )


def test_eg_decides_the_topology_and_the_reasoning_names_the_record(
    eg: FakeTransport,
) -> None:
    eg.answer = acted("agent_pool")
    decision = _select()
    assert decision.pattern is SubagentPattern.AGENT_POOL
    assert "EG decided agent_pool" in decision.reasoning
    options = {
        o["option_id"]: {n["key"]: n["q32"] for n in o["numbers"]}
        for o in eg.requests[0]["candidates"]["options"]
    }
    assert options["fan_out"]["heuristic"] == 1 << 32, "the tree's pick is declared"
    assert options["teams"]["agents"] == 5 << 32


def test_an_abstention_keeps_the_tree_s_pattern(eg: FakeTransport) -> None:
    eg.answer = abstained()
    decision = _select()
    assert decision.pattern is SubagentPattern.FAN_OUT
    assert "EG decided" not in decision.reasoning
    assert eg.op_names() == ["commit"], "the abstention is still recorded"
