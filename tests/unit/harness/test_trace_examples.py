"""KG-trace-derived DSPy training examples (CONCEPT:AU-AHE.optimization.trace-derived-training-examples).

Live-path coverage for :mod:`agent_utilities.harness.trace_examples`: a fake KG engine
stands in for the real ``Episode -[:USED_TOOL]-> ToolCall`` / ``Episode -[:tags]``
``-[:PRODUCED_OUTCOME]-> OutcomeEvaluation`` subgraph (mirroring the ``FakeEngine``
idiom in ``test_dspy_optimization.py``'s ``test_gather_optimization_data_best_effort``
and ``test_dspy_lm_adapter.py``), so we can assert a FAILING trace becomes a labeled
NEGATIVE example carrying the real (low) reward, and that the whole helper degrades to
the caller's self-supervised trainset — never raising — when no engine is reachable.
"""

from __future__ import annotations

import pytest

from agent_utilities.harness.dspy_optimization import get_target
from agent_utilities.harness import trace_examples as trace_examples_mod
from agent_utilities.harness.trace_examples import (
    TraceExample,
    blend_trainset,
    gather_trace_examples,
    record_trace_derived_finding,
    trace_reward_fn,
)


@pytest.fixture()
def no_active_engine(monkeypatch):
    """Force the ``engine=None`` default-on auto-resolution to a real cold-start
    (no process-wide active KG engine), so the "degrades without an engine" tests
    are deterministic regardless of what other tests/process state left active."""
    monkeypatch.setattr(trace_examples_mod, "_auto_engine", lambda: None)


class FakeToolEngine:
    """Stands in for the live engine on the ``Episode -[:USED_TOOL]-> ToolCall`` path."""

    def __init__(self, rows):
        self.rows = rows
        self.calls: list[tuple[str, dict]] = []

    def query_cypher(self, cypher, params=None):
        self.calls.append((cypher, params or {}))
        assert "USED_TOOL" in cypher
        assert params.get("name") == "my_tool"
        return self.rows


class FakeTagEngine:
    """Stands in for the live engine on the tag-attributed (skill/system_prompt) path."""

    def __init__(self, rows):
        self.rows = rows
        self.calls: list[tuple[str, dict]] = []

    def query_cypher(self, cypher, params=None):
        self.calls.append((cypher, params or {}))
        assert "e.tags" in cypher
        assert params.get("name") == "my_skill"
        return self.rows


class RaisingEngine:
    def query_cypher(self, cypher, params=None):
        raise RuntimeError("backend down")


class AddNodeEngine:
    def __init__(self):
        self.nodes: list[tuple[str, str, dict]] = []

    def add_node(self, node_id, label, properties=None):
        self.nodes.append((node_id, label, properties or {}))


# ── gather_trace_examples — tool_description path ───────────────────────────


def test_gather_trace_examples_seeds_a_failing_episode_as_negative_example():
    """A failing :Episode -[:PRODUCED_OUTCOME]-> :OutcomeEvaluation for the target
    tool becomes a TraceExample carrying the REAL (low) reward, success=False, a
    blank response (never a bootstrap-worthy demonstration), and the failure text."""
    engine = FakeToolEngine(
        rows=[
            {
                "context": "user asked to restart the deploy",
                "task_input": "tool_call(args={'service': 'x'})",
                "result": "raised TimeoutError",
                "reward": 0.1,
                "feedback_text": "tool timed out against the live endpoint",
            }
        ]
    )
    target = get_target("tool_description")
    artifact = {"name": "my_tool", "description": "restarts a deploy"}

    examples = gather_trace_examples(engine, target, artifact)

    assert len(examples) == 1
    ex = examples[0]
    assert isinstance(ex, TraceExample)
    assert ex.success is False
    assert ex.reward == pytest.approx(0.1)
    assert ex.response == ""  # a failing trace's output is never a demonstration
    assert ex.failure_reason == "tool timed out against the live endpoint"
    assert ex.source_id == "my_tool"
    assert engine.calls  # the tool-scoped query actually ran


def test_gather_trace_examples_skill_tag_path_marks_success():
    engine = FakeTagEngine(
        rows=[
            {
                "context": "agent completed the skill successfully",
                "task_input": "",
                "result": "",
                "reward": 0.95,
                "feedback_text": "",
            }
        ]
    )
    target = get_target("skill")
    artifact = {"name": "my_skill", "sop": "do the thing"}

    examples = gather_trace_examples(engine, target, artifact)

    assert len(examples) == 1
    assert examples[0].success is True
    assert examples[0].reward == pytest.approx(0.95)
    assert examples[0].failure_reason == ""


def test_gather_trace_examples_degrades_on_query_failure():
    target = get_target("tool_description")
    assert gather_trace_examples(RaisingEngine(), target, {"name": "my_tool"}) == []


def test_gather_trace_examples_empty_without_engine(no_active_engine):
    target = get_target("tool_description")
    assert gather_trace_examples(None, target, {"name": "my_tool"}) == []


def test_gather_trace_examples_empty_when_name_unresolvable():
    class NoNameTarget:
        component_type = "tool_description"

        @staticmethod
        def task_name(_artifact):
            return ""

    assert (
        gather_trace_examples(FakeToolEngine([]), NoNameTarget(), {"name": "x"}) == []
    )


def test_gather_trace_examples_unrouted_component_type_returns_empty():
    class OtherTarget:
        component_type = "extraction"

    assert (
        gather_trace_examples(FakeToolEngine([]), OtherTarget(), {"name": "x"}) == []
    )


# ── blend_trainset — the caller-facing entrypoint ────────────────────────────


def test_blend_trainset_leads_with_trace_derived_negative_example():
    engine = FakeToolEngine(
        rows=[
            {
                "context": "ctx",
                "task_input": "task",
                "result": "bad output",
                "reward": 0.0,
                "feedback_text": "completely wrong",
            }
        ]
    )
    target = get_target("tool_description")
    artifact = {"name": "my_tool"}
    self_supervised = [{"context": "c2", "task": "t2", "response": "r2"}]

    blended, stats = blend_trainset(engine, target, artifact, self_supervised)

    assert stats["trace_derived"] == 1
    assert stats["trace_failures"] == 1
    assert stats["trace_successes"] == 0
    assert stats["self_supervised"] == 1
    assert stats["total"] == 2
    assert len(blended) == 2
    # trace-derived examples lead the blended trainset
    first = blended[0]
    first_reward = (
        first.get("reward") if isinstance(first, dict) else getattr(first, "reward")
    )
    assert first_reward == pytest.approx(0.0)
    assert trace_reward_fn(first) == pytest.approx(0.0)


def test_blend_trainset_degrades_to_self_supervised_without_engine(no_active_engine):
    """No engine reachable → blend_trainset never raises and returns exactly the
    caller's self-supervised trainset (cold-start degrade)."""
    target = get_target("tool_description")
    self_supervised = [
        {"context": "c1", "task": "t1", "response": "r1"},
        {"context": "c2", "task": "t2", "response": "r2"},
    ]

    blended, stats = blend_trainset(None, target, {"name": "my_tool"}, self_supervised)

    assert blended == self_supervised
    assert stats["trace_derived"] == 0
    assert stats["self_supervised"] == 2
    assert stats["total"] == 2


def test_blend_trainset_empty_engine_and_empty_self_supervised_is_empty(
    no_active_engine,
):
    target = get_target("tool_description")
    blended, stats = blend_trainset(None, target, {"name": "my_tool"}, None)
    assert blended == []
    assert stats["total"] == 0


# ── trace_reward_fn — the metric-facing reward reader ────────────────────────


def test_trace_reward_fn_reads_dict_and_object_and_defaults():
    assert trace_reward_fn({"reward": 0.2}) == pytest.approx(0.2)
    assert trace_reward_fn({}) == pytest.approx(0.5)  # neutral default

    class Obj:
        reward = 0.8

    assert trace_reward_fn(Obj()) == pytest.approx(0.8)

    class Malformed:
        reward = "not-a-number"

    assert trace_reward_fn(Malformed()) == pytest.approx(0.5)  # never raises


# ── record_trace_derived_finding — the observability tail ────────────────────


def test_record_trace_derived_finding_persists_when_engine_supports_add_node():
    engine = AddNodeEngine()
    stats = {
        "component_type": "tool_description",
        "identifier": "my_tool",
        "trace_derived": 1,
        "trace_failures": 1,
        "trace_successes": 0,
        "self_supervised": 1,
        "total": 2,
    }
    finding_id = record_trace_derived_finding(engine, stats)
    assert finding_id is not None
    assert len(engine.nodes) == 1
    node_id, label, props = engine.nodes[0]
    assert node_id == finding_id
    assert label == "DSPyTraceOptimizationFinding"
    assert props["trace_derived_count"] == 1


def test_record_trace_derived_finding_noop_without_traces_or_engine():
    assert record_trace_derived_finding(None, {"trace_derived": 0}) is None
    assert record_trace_derived_finding(AddNodeEngine(), {"trace_derived": 0}) is None
    # traces present but engine doesn't support add_node → still never raises
    full_stats = {
        "component_type": "x",
        "identifier": "y",
        "trace_derived": 1,
        "trace_failures": 1,
        "trace_successes": 0,
        "self_supervised": 0,
        "total": 1,
    }
    assert record_trace_derived_finding(object(), full_stats) is None
