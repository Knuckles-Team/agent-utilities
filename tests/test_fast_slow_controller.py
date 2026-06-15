#!/usr/bin/python
from __future__ import annotations

"""Tests for the Fast-Slow Training controller (CONCEPT:ORCH-1.56).

Deterministic stubs only — no I/O, no models. Exercises both loops, the live
consumption of the real ``batch_normalized_advantage`` GRPO spine, recurrence
gating, the deferred no-op trainer, the interleave cadence, and model-swap
safety (no learning lost across a frontier-model swap).
"""

from agent_utilities.graph.training_signals import batch_normalized_advantage
from agent_utilities.harness.fast_slow_controller import (
    FastSlowController,
    SlowUpdate,
    Trace,
)


def _counting_harness():
    """A deterministic fast-loop fn: returns an id encoding the call index + size."""
    calls: list[list[Trace]] = []

    def fn(recent: list[Trace]) -> str:
        calls.append(list(recent))
        return f"harness-{len(calls)}-n{len(recent)}"

    return fn, calls


def _recording_trainer():
    """A deterministic slow-loop trainer that records every absorbed group."""
    absorbed: list[tuple[str, int]] = []

    def fn(task_key: str, traces: list[Trace]) -> None:
        absorbed.append((task_key, len(traces)))

    return fn, absorbed


def test_fast_step_returns_injected_harness_id():
    fn, calls = _counting_harness()
    ctrl = FastSlowController(fn)
    ctrl.observe(Trace("a", 1.0, "p1"))
    ctrl.observe(Trace("a", 0.5, "p2"))

    harness_id = ctrl.fast_step()

    assert harness_id == "harness-1-n2"
    assert ctrl.harness_id == "harness-1-n2"
    # The fast loop only saw the traces observed since the last fast step.
    assert len(calls) == 1
    assert len(calls[0]) == 2


def test_fast_step_only_sees_new_traces():
    fn, calls = _counting_harness()
    ctrl = FastSlowController(fn)
    ctrl.observe(Trace("a", 1.0))
    ctrl.fast_step()
    ctrl.observe(Trace("a", 2.0))
    second = ctrl.fast_step()

    assert second == "harness-2-n1"
    assert len(calls[1]) == 1  # only the trace observed after the first step


def test_fast_step_with_no_new_traces_keeps_harness():
    fn, calls = _counting_harness()
    ctrl = FastSlowController(fn)
    ctrl.observe(Trace("a", 1.0))
    first = ctrl.fast_step()
    second = ctrl.fast_step()  # nothing new observed

    assert first == second == "harness-1-n1"
    assert len(calls) == 1  # the fn was not called the second time


def test_slow_step_fires_only_for_recurring_keys():
    trainer, absorbed = _recording_trainer()
    ctrl = FastSlowController(_counting_harness()[0], trainer_fn=trainer, recurrence_threshold=3)
    # "hot" recurs 3 times (>= threshold); "cold" only twice.
    for r in (1.0, 2.0, 3.0):
        ctrl.observe(Trace("hot", r))
    for r in (0.1, 0.2):
        ctrl.observe(Trace("cold", r))

    updates = ctrl.slow_step()

    assert [u.task_key for u in updates] == ["hot"]
    assert absorbed == [("hot", 3)]  # trainer called only for the recurring group


def test_slow_step_advantage_mean_uses_real_grpo_spine():
    ctrl = FastSlowController(_counting_harness()[0], recurrence_threshold=3)
    rewards = [1.0, 2.0, 3.0]
    for r in rewards:
        ctrl.observe(Trace("hot", r))

    updates = ctrl.slow_step()

    expected = batch_normalized_advantage(rewards)
    expected_mean = round(sum(expected) / len(expected), 6)
    assert len(updates) == 1
    assert updates[0].advantage_mean == expected_mean
    assert updates[0].n_traces == 3


def test_default_trainer_is_deferred_noop_but_still_summarizes():
    # No trainer injected -> deferred no-op default; SlowUpdate must still emit.
    ctrl = FastSlowController(_counting_harness()[0], recurrence_threshold=2)
    for r in (1.0, 5.0):
        ctrl.observe(Trace("hot", r))

    updates = ctrl.slow_step()

    assert len(updates) == 1
    assert isinstance(updates[0], SlowUpdate)
    assert updates[0].task_key == "hot"


def test_slow_step_clears_consumed_groups():
    trainer, absorbed = _recording_trainer()
    ctrl = FastSlowController(_counting_harness()[0], trainer_fn=trainer, recurrence_threshold=2)
    for r in (1.0, 2.0):
        ctrl.observe(Trace("hot", r))
    ctrl.observe(Trace("cold", 9.0))

    first = ctrl.slow_step()
    assert [u.task_key for u in first] == ["hot"]

    # Re-running must not re-absorb the consumed group; "cold" still below floor.
    second = ctrl.slow_step()
    assert second == []
    assert absorbed == [("hot", 2)]  # absorbed exactly once


def test_slow_step_keeps_non_recurring_for_later():
    ctrl = FastSlowController(_counting_harness()[0], recurrence_threshold=2)
    ctrl.observe(Trace("cold", 1.0))
    assert ctrl.slow_step() == []  # not recurring yet

    ctrl.observe(Trace("cold", 2.0))  # now it recurs
    updates = ctrl.slow_step()
    assert [u.task_key for u in updates] == ["cold"]
    assert updates[0].n_traces == 2


def test_run_interleaves_fast_and_slow_per_cadence():
    fn, _ = _counting_harness()
    trainer, absorbed = _recording_trainer()
    ctrl = FastSlowController(fn, trainer_fn=trainer, recurrence_threshold=2)
    # 6 traces of the same recurring kind.
    for i in range(6):
        ctrl.observe(Trace("hot", float(i)))

    result = ctrl.run(fast_every=1, slow_every=3)

    # fast_every=1 over 6 batches -> 6 fast updates.
    assert len(result["fast_updates"]) == 6
    # slow_every=3 over 6 batches -> slow steps at batch 3 and batch 6.
    # First slow step absorbs the recurring group and clears it; the second
    # absorbs whatever accumulated after.
    assert len(result["slow_updates"]) >= 1
    assert all(isinstance(u, SlowUpdate) for u in result["slow_updates"])
    assert all(key == "hot" for key, _ in absorbed)


def test_swap_model_preserves_observed_traces_and_harness():
    fn, _ = _counting_harness()
    ctrl = FastSlowController(fn, recurrence_threshold=3)
    ctrl.observe(Trace("hot", 1.0))
    ctrl.observe(Trace("hot", 2.0))
    harness_before = ctrl.fast_step()

    ctrl.swap_model("frontier-model-B")

    # No learning lost: harness id intact, owned model intact.
    assert ctrl.harness_id == harness_before
    assert ctrl.owned_model_id == "owned-v0"

    # The accumulated traces still recur after the swap, proving they survived.
    ctrl.observe(Trace("hot", 3.0))
    updates = ctrl.slow_step()
    assert [u.task_key for u in updates] == ["hot"]
    assert updates[0].n_traces == 3  # all three traces, including pre-swap ones


def test_determinism():
    def build():
        fn, _ = _counting_harness()
        trainer, absorbed = _recording_trainer()
        c = FastSlowController(fn, trainer_fn=trainer, recurrence_threshold=2)
        for i in range(5):
            c.observe(Trace("hot", float(i)))
            c.observe(Trace("warm", float(i) * 2))
        return c, absorbed

    a, abs_a = build()
    b, abs_b = build()
    res_a = a.run(fast_every=1, slow_every=2)
    res_b = b.run(fast_every=1, slow_every=2)

    assert res_a["fast_updates"] == res_b["fast_updates"]
    assert res_a["slow_updates"] == res_b["slow_updates"]
    assert abs_a == abs_b


def test_invalid_thresholds_rejected():
    import pytest

    with pytest.raises(ValueError):
        FastSlowController(_counting_harness()[0], recurrence_threshold=0)
    ctrl = FastSlowController(_counting_harness()[0])
    with pytest.raises(ValueError):
        ctrl.run(fast_every=0)
