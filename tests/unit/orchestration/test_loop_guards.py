"""Unit tests for the harness-enforced loop-exit primitives (CONCEPT:AU-AHE.
harness.loop-exit-conditions).

These cover the reusable guards/evaluator in isolation; the end-to-end
enforcement on ``LoopController.run_loop`` lives in
``tests/unit/knowledge_graph/test_loop_exits.py``.
"""

from __future__ import annotations

from agent_utilities.orchestration.loop_guards import (
    ConsecutiveFailureGuard,
    GoalEvaluation,
    build_goal_evaluator,
    deadline_passed,
    graph_node_transition_cap,
    progress_signature,
    resolve_deadline,
    window_is_stalled,
)


# ── exit 7: ConsecutiveFailureGuard (engine-breaker threshold + reset) ──────────


def test_failure_guard_trips_at_threshold():
    guard = ConsecutiveFailureGuard(threshold=3)
    assert not guard.record_failure()  # 1
    assert not guard.record_failure()  # 2
    assert guard.record_failure()  # 3 -> tripped
    assert guard.tripped
    assert guard.count == 3
    assert guard.total_failures == 3


def test_failure_guard_resets_on_progress():
    guard = ConsecutiveFailureGuard(threshold=3)
    guard.record_failure()
    guard.record_failure()
    guard.record_success()  # any progress resets the consecutive run
    assert guard.count == 0
    assert not guard.tripped
    # ...but the lifetime counter is preserved.
    assert guard.total_failures == 2
    # It takes a fresh run of `threshold` to trip again.
    assert not guard.record_failure()
    assert not guard.record_failure()
    assert guard.record_failure()


def test_failure_guard_threshold_zero_disables_tripping():
    guard = ConsecutiveFailureGuard(threshold=0)
    for _ in range(100):
        assert not guard.record_failure()
    assert not guard.tripped
    assert guard.total_failures == 100


# ── exit 1: GoalEvaluation + build_goal_evaluator ──────────────────────────────


def test_develop_evaluator_is_deterministic():
    ev = build_goal_evaluator({"kind": "develop"})
    # a passing validation command IS a real measured pass
    passed = ev({}, {"status": "completed"})
    assert passed.passed and passed.measured and passed.score == 1.0
    # pending -> measured "not passed yet" (keep iterating)
    pending = ev({}, {"status": "pending"})
    assert not pending.passed and pending.measured
    # a hard failure defers to the callee's terminal status (measured=False),
    # so a decisive give-up still stops the loop rather than being retried.
    failed = ev({}, {"status": "failed"})
    assert not failed.passed and not failed.measured


def test_research_skill_evaluator_degrades_offline():
    # No model endpoint reachable in the unit environment -> the rubric/LLM judge
    # must degrade to measured=False (trust the callee), never fabricate a verdict
    # or attempt a network call.
    ev = build_goal_evaluator({"kind": "skill", "objective": "deploy the service"})
    verdict = ev({"name": "s"}, {"status": "completed", "output": "ran the deploy"})
    assert isinstance(verdict, GoalEvaluation)
    assert verdict.measured is False


def test_evaluator_disabled_always_defers():
    ev = build_goal_evaluator({"kind": "research"}, enable_llm_judge=False)
    verdict = ev({}, {"status": "completed", "output": "x"})
    assert verdict.measured is False


# ── exit 5: progress_signature + window_is_stalled ─────────────────────────────


def test_progress_signature_stable_and_discriminating():
    a = progress_signature("pending", "same output", "")
    b = progress_signature("pending", "same output", "")
    c = progress_signature("pending", "different output", "")
    assert a == b
    assert a != c


def test_window_is_stalled():
    assert not window_is_stalled(["a", "a"], 3)  # not enough samples
    assert window_is_stalled(["x", "a", "a", "a"], 3)  # last 3 identical
    assert not window_is_stalled(["a", "a", "b"], 3)  # last 3 not identical
    assert not window_is_stalled(["a", "a", "a"], 0)  # window<=1 disables
    assert not window_is_stalled(["a", "a", "a"], 1)


# ── exit 4: resolve_deadline + deadline_passed ─────────────────────────────────


def test_resolve_deadline_prefers_absolute_then_relative():
    assert resolve_deadline(123.0, 999.0, start_monotonic=0.0) == 123.0
    assert resolve_deadline(None, 5.0, start_monotonic=100.0) == 105.0
    assert resolve_deadline(None, None, start_monotonic=100.0) is None
    assert resolve_deadline(None, 0.0, start_monotonic=100.0) is None


def test_deadline_passed():
    assert deadline_passed(50.0, now=100.0)
    assert not deadline_passed(200.0, now=100.0)
    assert not deadline_passed(None, now=100.0)


# ── exit 2: TURN CAP threading onto the multi-agent graph ─────────────────────


def test_graph_node_transition_cap_matches_single_server_convention():
    # Same max(max_steps*2, 10) convention as the single-server request_limit.
    assert graph_node_transition_cap(5) == 10
    assert graph_node_transition_cap(3) == 10  # floor
    assert graph_node_transition_cap(30) == 60  # run_agent default -> >= old 50


def test_graphstate_carries_threaded_max_steps():
    # The field the engine threads the caller's turn cap onto; default is the
    # ExecutionBudget's own 50-node-transition guard (unchanged when unset).
    from agent_utilities.graph.state import GraphState

    state = GraphState(query="x")
    assert state.max_steps is None
    assert state.execution_budget.max_node_transitions == 50
    # engine.execute_graph applies exactly this when max_steps is threaded:
    state.max_steps = 5
    state.execution_budget.max_node_transitions = graph_node_transition_cap(5)
    assert state.execution_budget.max_node_transitions == 10
