"""7.2 — the optimisation loop must consume REAL, TRUTHFUL execution outcomes.

Regression for the exact signal that two prior truthfulness bugs poisoned:
``_execution_succeeded()`` mis-classifying a typed ``EvidenceBundle`` result as
success, and a degraded-grounding run being learned from as clean. Both are
fixed at their own layer (``tests/unit/test_intent_surface.py``,
``tests/unit/test_delegation_degraded_outcome.py``) — this file proves the
chain end to end, one hop at a time, from the point a degraded/failed
:``OutcomeEvaluation`` is recorded through to the exact payload
``run_program_optimization`` hands the native ``eg-program`` backend
(``JobKind::ProgramOptimize``'s Python-side boundary):

    outcome_properties(status=...)  — trace_ontology
        -> _row_to_example(row)      — trace_examples (KG row -> TraceExample)
        -> blend_trainset(...)       — trace_examples (TraceExample -> payload dict)
        -> run_program_optimization  — program_optimization (payload -> native request)

A degraded or failed run must read ``success=False`` at every one of those
hops — never flip to a positive learning signal anywhere along the chain.
"""

from __future__ import annotations

import pytest

from agent_utilities.harness import optimization_backend as ob
from agent_utilities.harness import program_optimization as po
from agent_utilities.harness import trace_examples as te
from agent_utilities.harness.optimization_backend import NativeOptimizationAttempt
from agent_utilities.observability.trace_ontology import outcome_properties


class _FakeEngine:
    """Returns one canned trace/outcome row, matching ``_query_by_tag``'s shape."""

    def __init__(self, row: dict) -> None:
        self._row = row

    def query_cypher(self, query: str, params: dict | None = None) -> list[dict]:
        return [self._row]


def _row_for_status(status: str, *, feedback: str = "grounding was degraded") -> dict:
    """Build a realistic KG row the way ``_query_by_tag`` would return it, with
    the reward computed by the REAL ``outcome_properties`` (never hand-picked),
    so this test exercises the actual truthfulness contract, not a fixture."""
    props = outcome_properties(
        run_id="run-1",
        status=status,
        timestamp="2026-01-01T00:00:00Z",
        event_sequence=1,
        feedback=feedback,
    )
    return {
        "id": "run-1",
        "context": "some agent context",
        "task_input": "do the task",
        "result": "a suspicious-looking output",
        "reward": props["reward"],
        "feedback_text": feedback,
        "event_sequence": 1,
    }


@pytest.mark.parametrize(
    "status,expected_reward",
    [
        ("degraded", 0.25),
        ("failed", 0.0),
        ("error", 0.0),
    ],
)
def test_outcome_properties_never_rewards_a_non_clean_run_as_success(
    status, expected_reward
):
    """Hop 0 — the source of truth. A degraded/failed status must never earn
    the ``reward == 1.0`` a clean success gets."""
    props = outcome_properties(
        run_id="r", status=status, timestamp="t", event_sequence=1
    )
    assert props["success"] is False
    assert props["reward"] == pytest.approx(expected_reward)
    assert props["reward"] < te.FAILURE_REWARD_THRESHOLD


@pytest.mark.parametrize("status", ["degraded", "failed", "error"])
def test_row_to_example_never_classifies_a_degraded_or_failed_row_as_success(status):
    """Hop 1 — trace_examples._row_to_example must read the real reward and
    classify accordingly; a failing example's response is blanked so it can
    never be mistaken for a demonstration to imitate."""
    row = _row_for_status(status)
    example = te._row_to_example(row)
    assert example is not None
    assert example.success is False
    assert example.reward < te.FAILURE_REWARD_THRESHOLD
    assert example.response == ""
    assert example.failure_reason  # the real reason is preserved, not dropped


@pytest.mark.parametrize("status", ["degraded", "failed", "error"])
def test_blend_trainset_never_promotes_a_degraded_or_failed_trace_to_positive(status):
    """Hop 2 — blend_trainset's payload dicts (what actually reaches the
    optimizer) must carry ``success: False`` for a degraded/failed trace."""
    row = _row_for_status(status)
    engine = _FakeEngine(row)
    target = po.OPTIMIZABLE_TARGETS["skill"]
    blended, stats = te.blend_trainset(engine, target, {"name": "some-skill"}, [])
    assert len(blended) == 1
    assert blended[0]["success"] is False
    assert blended[0]["reward"] < te.FAILURE_REWARD_THRESHOLD
    assert stats["trace_failures"] == 1
    assert stats["trace_successes"] == 0


@pytest.mark.parametrize("status", ["degraded", "failed", "error"])
def test_run_program_optimization_hands_the_native_backend_a_negative_example(
    monkeypatch, status
):
    """Hop 3 — THE crux: the exact ``OptimizationRequest`` payload
    ``run_program_optimization`` builds for the native ``eg-program`` backend
    (the Python-side boundary of ``JobKind::ProgramOptimize``) must carry the
    degraded/failed example as ``success: False`` with its real sub-threshold
    reward — never as a clean positive. Captures the request at the native
    call boundary (``try_native_optimization``) rather than hitting a real
    Rust engine."""
    row = _row_for_status(status)
    engine = _FakeEngine(row)
    target = po.OPTIMIZABLE_TARGETS["skill"]

    captured: dict = {}

    def _fake_try_native(_engine, request):
        captured["request"] = request
        return NativeOptimizationAttempt(disposition="unavailable")

    monkeypatch.setattr(ob, "try_native_optimization", _fake_try_native)

    result = po.run_program_optimization(
        target, {"name": "some-skill"}, [], engine=engine
    )

    assert (
        result is None
    )  # native backend reported "unavailable" — no fabricated result
    assert "request" in captured
    trainset = captured["request"].data["trainset"]
    assert len(trainset) == 1
    assert trainset[0]["success"] is False
    assert trainset[0]["reward"] < te.FAILURE_REWARD_THRESHOLD
    # A failing example's response must never reach the optimizer either —
    # nothing here re-inflates it back in between blend_trainset and the
    # native request.
    assert trainset[0]["response"] == ""


def test_a_clean_success_is_the_only_path_to_a_positive_signal(monkeypatch):
    """Control case: a genuinely clean success (status == "completed") DOES
    reach the native request as a positive example — proving the prior tests
    aren't failing to distinguish success from failure, only correctly
    rejecting the non-clean statuses."""
    row = _row_for_status("completed", feedback="")
    engine = _FakeEngine(row)
    target = po.OPTIMIZABLE_TARGETS["skill"]

    captured: dict = {}

    def _fake_try_native(_engine, request):
        captured["request"] = request
        return NativeOptimizationAttempt(disposition="unavailable")

    monkeypatch.setattr(ob, "try_native_optimization", _fake_try_native)

    po.run_program_optimization(target, {"name": "some-skill"}, [], engine=engine)

    trainset = captured["request"].data["trainset"]
    assert len(trainset) == 1
    assert trainset[0]["success"] is True
    assert trainset[0]["reward"] == pytest.approx(1.0)
