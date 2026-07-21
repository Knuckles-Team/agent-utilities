"""End-to-end tests for the EIGHT harness-enforced ``run_loop`` exit conditions
(CONCEPT:AU-AHE.harness.loop-exit-conditions).

Each exit is verified to transition the loop to its DISTINCT terminal
``LoopStatus`` — never a generic ``failed`` — through the real
``LoopController.run_loop`` + the native WorkItem authority (the ``LoopEngine``
double + ``_authority`` session harness shared with ``test_loops``).
"""

from __future__ import annotations

import asyncio
import time

from agent_utilities.knowledge_graph.research.loop_controller import LoopController
from agent_utilities.knowledge_graph.research.loops import (
    LoopStatus,
    is_terminal,
    submit_loop,
    to_status,
)
from agent_utilities.orchestration import work_item as wi
from agent_utilities.orchestration.loop_guards import GoalEvaluation
from tests.unit.knowledge_graph.test_loops import LoopEngine, _authority


def _submit(engine: LoopEngine, *, loop_id: str, kind: str = "develop", **kw) -> dict:
    with _authority():
        return submit_loop(
            engine,
            "bounded objective",
            loop_id=loop_id,
            kind=kind,
            validation_cmd="validate" if kind == "develop" else "",
            skill_ref="runtime-governance" if kind == "skill" else "",
            **kw,
        )


def _run(controller: LoopController, loop: dict, **kw) -> dict:
    with _authority():
        return asyncio.run(controller.run_loop(loop, **kw))


class _FakeOptimizer:
    """Minimal ResourceOptimizer stand-in (duck-typed on is_budget_exceeded)."""

    def __init__(self, exceeded: bool) -> None:
        self._exceeded = exceeded

    def is_budget_exceeded(self) -> bool:
        return self._exceeded

    def summary(self) -> dict:
        return {"tokens_used": 999, "budget_exceeded": self._exceeded}


# ── enum vocabulary / fail-closed foundation ──────────────────────────────────


def test_terminal_status_enum_is_authoritative_and_fail_closed():
    # All eight enforced exits are DISTINCT, diagnosable terminal statuses.
    for s in (
        LoopStatus.COMPLETED,
        LoopStatus.MAX_ITERATIONS_EXCEEDED,
        LoopStatus.BUDGET_EXCEEDED,
        LoopStatus.WALL_CLOCK_EXCEEDED,
        LoopStatus.STALLED,
        LoopStatus.CANCELLED,
        LoopStatus.ERROR_THRESHOLD_EXCEEDED,
        LoopStatus.EXTERNAL_EVENT_SATISFIED,
    ):
        assert is_terminal(s)
    # A misspelled/unknown status can no longer silently read as non-terminal:
    # it fails closed to FAILED (which IS terminal), never "keep looping".
    assert to_status("compltED_typo") is LoopStatus.FAILED
    assert is_terminal("compltED_typo")
    assert not is_terminal("running")


# ── exit 1: GOAL MET (measured pass, not self-declared) ───────────────────────


def test_exit1_goal_met_on_measured_pass():
    engine = LoopEngine()
    loop = _submit(engine, loop_id="loop:develop:exit1")
    dev_calls = {"n": 0}

    def develop(_cmd, _cwd):
        dev_calls["n"] += 1
        return False, f"working {dev_calls['n']}"  # callee never self-declares done

    eval_calls = {"n": 0}

    def evaluator(_loop, _outcome):
        eval_calls["n"] += 1
        return GoalEvaluation(passed=eval_calls["n"] >= 2, score=1.0, measured=True)

    res = _run(
        LoopController(engine, develop_runner=develop),
        loop,
        max_iterations=10,
        goal_evaluator=evaluator,
    )
    assert res["status"] == "completed"
    assert res["iterations"] == 2
    assert dev_calls["n"] == 2 and eval_calls["n"] == 2
    item = wi.get_work_item(engine, wi.loop_work_item_id(loop["id"]))
    assert item["status"] == "succeeded"


def test_exit1_self_declared_done_rejected_by_measurement_does_not_complete():
    """The model self-declaring 'completed' is NOT trusted when the measurement
    rejects it — the loop keeps working (and here hits the turn cap)."""
    engine = LoopEngine()
    loop = _submit(engine, loop_id="loop:develop:exit1-reject")
    dev_calls = {"n": 0}

    def develop(_cmd, _cwd):
        dev_calls["n"] += 1
        return True, "done"  # self-declares completed every iteration

    res = _run(
        LoopController(engine, develop_runner=develop),
        loop,
        max_iterations=3,
        # measurement always fails the goal -> never a real pass
        goal_evaluator=lambda _l, _o: GoalEvaluation(False, 0.1, measured=True),
        no_progress_window=0,  # isolate: don't let identical output trip stall
    )
    assert res["status"] != "completed"
    assert res["status"] == "max_iterations_exceeded"
    assert dev_calls["n"] == 3  # kept working despite the self-declared 'done'


# ── exit 2: TURN CAP ──────────────────────────────────────────────────────────


def test_exit2_turn_cap():
    engine = LoopEngine()
    loop = _submit(engine, loop_id="loop:develop:exit2")
    dev_calls = {"n": 0}

    def develop(_cmd, _cwd):
        dev_calls["n"] += 1
        return False, f"try {dev_calls['n']}"  # changing output -> not stalled

    res = _run(
        LoopController(engine, develop_runner=develop),
        loop,
        max_iterations=4,
    )
    assert res["status"] == "max_iterations_exceeded"
    assert res["iterations"] == 4
    assert dev_calls["n"] == 4
    assert "exit_reason" in res
    item = wi.get_work_item(engine, wi.loop_work_item_id(loop["id"]))
    assert item["status"] == "failed"


# ── exit 3: BUDGET CAP (hard stop) ────────────────────────────────────────────


def test_exit3_budget_exceeded_is_a_hard_terminal_stop():
    engine = LoopEngine()
    loop = _submit(engine, loop_id="loop:develop:exit3")
    dev_calls = {"n": 0}

    def develop(_cmd, _cwd):
        dev_calls["n"] += 1
        return False, "x"

    res = _run(
        LoopController(engine, develop_runner=develop),
        loop,
        max_iterations=10,
        resource_optimizer=_FakeOptimizer(exceeded=True),
    )
    assert res["status"] == "budget_exceeded"
    assert res["iterations"] == 0  # tripped BEFORE running any step
    assert dev_calls["n"] == 0
    assert "budget" in res.get("exit_reason", "").lower()


def test_exit3_budget_not_exceeded_runs_normally():
    engine = LoopEngine()
    loop = _submit(engine, loop_id="loop:develop:exit3-ok")
    res = _run(
        LoopController(engine, develop_runner=lambda _c, _w: (True, "ok")),
        loop,
        max_iterations=5,
        resource_optimizer=_FakeOptimizer(exceeded=False),
    )
    assert res["status"] == "completed"


# ── exit 4: WALL CLOCK ────────────────────────────────────────────────────────


def test_exit4_wall_clock_deadline_in_past():
    engine = LoopEngine()
    loop = _submit(engine, loop_id="loop:develop:exit4")
    dev_calls = {"n": 0}

    def develop(_cmd, _cwd):
        dev_calls["n"] += 1
        return False, "x"

    res = _run(
        LoopController(engine, develop_runner=develop),
        loop,
        max_iterations=10,
        deadline=time.monotonic() - 1.0,  # already elapsed
    )
    assert res["status"] == "wall_clock_exceeded"
    assert res["iterations"] == 0
    assert dev_calls["n"] == 0


def test_exit4_wall_clock_max_duration_trips_mid_run():
    """The overall deadline is checked EVERY iteration in the while-condition,
    independent of the per-substep timeouts."""
    engine = LoopEngine()
    loop = _submit(engine, loop_id="loop:develop:exit4-dur")

    def develop(_cmd, _cwd):
        time.sleep(0.02)  # each step outlasts the tiny overall budget
        return False, "x"

    res = _run(
        LoopController(engine, develop_runner=develop),
        loop,
        max_iterations=50,
        max_duration_s=0.001,
    )
    assert res["status"] == "wall_clock_exceeded"
    assert res["iterations"] >= 1  # ran at least once, then the deadline tripped


# ── exit 5: NO PROGRESS (stalled) ─────────────────────────────────────────────


def test_exit5_stalled_on_identical_iterations():
    engine = LoopEngine()
    loop = _submit(engine, loop_id="loop:develop:exit5")
    dev_calls = {"n": 0}

    def develop(_cmd, _cwd):
        dev_calls["n"] += 1
        return False, "identical"  # same (status, output) every iteration

    res = _run(
        LoopController(engine, develop_runner=develop),
        loop,
        max_iterations=20,
        no_progress_window=3,
    )
    assert res["status"] == "stalled"
    assert res["iterations"] == 3  # tripped at the 3rd identical iteration
    assert dev_calls["n"] == 3
    # the rolling window was persisted on the Loop's progress node
    win = engine.nodes.get(f"{loop['id']}:progress_window")
    assert win is not None and win.get("window") == 3


# ── exit 6: HUMAN INTERRUPT (corrigible kill switch, outside the loop) ─────────


def test_exit6_human_interrupt_kill_precedes_any_step():
    engine = LoopEngine()
    loop = _submit(engine, loop_id="loop:develop:exit6")
    dev_calls = {"n": 0}

    def develop(_cmd, _cwd):
        dev_calls["n"] += 1
        return False, "x"

    res = _run(
        LoopController(engine, develop_runner=develop),
        loop,
        max_iterations=5,
        desired_state=lambda: "kill",
    )
    assert res["status"] == "cancelled"
    assert res["interrupted"] is True
    assert dev_calls["n"] == 0  # kill evaluated OUTSIDE/BEFORE the risky step
    item = wi.get_work_item(engine, wi.loop_work_item_id(loop["id"]))
    assert item["status"] == "cancelled"


def test_exit6_human_interrupt_pause():
    engine = LoopEngine()
    loop = _submit(engine, loop_id="loop:develop:exit6-pause")
    res = _run(
        LoopController(engine, develop_runner=lambda _c, _w: (False, "x")),
        loop,
        max_iterations=5,
        desired_state=lambda: "pause",
    )
    assert res["status"] == "paused"
    assert res["interrupted"] is True


# ── exit 7: ERROR THRESHOLD (N consecutive non-terminal failures) ─────────────


def _retryable_iterate(controller: LoopController, tag: str = "boom"):
    calls = {"n": 0}

    def _iter(_loop):
        calls["n"] += 1
        return {
            "status": "failed",
            "output": f"{tag} {calls['n']}",
            "error": tag,
            "retryable": True,
        }

    controller._iterate = _iter  # type: ignore[method-assign]
    return calls


def test_exit7_error_threshold_on_consecutive_failures():
    engine = LoopEngine()
    loop = _submit(engine, loop_id="loop:develop:exit7")
    controller = LoopController(engine)
    calls = _retryable_iterate(controller)

    res = _run(controller, loop, max_iterations=20, max_consecutive_failures=3)
    assert res["status"] == "error_threshold_exceeded"
    assert res["iterations"] == 3
    assert calls["n"] == 3
    item = wi.get_work_item(engine, wi.loop_work_item_id(loop["id"]))
    assert item["status"] == "failed"


def test_exit7_progress_resets_the_failure_run():
    """A single success between failures resets the consecutive-failure run, so a
    flaky-but-progressing loop is NOT killed by the error threshold."""
    engine = LoopEngine()
    loop = _submit(engine, loop_id="loop:develop:exit7-reset")
    controller = LoopController(engine)
    # fail, fail, <progress>, fail, fail, fail -> trips only at the final run of 3
    script = [
        {"status": "failed", "error": "e", "retryable": True, "output": "a"},
        {"status": "failed", "error": "e", "retryable": True, "output": "b"},
        {"status": "pending", "output": "progress!"},  # resets the guard
        {"status": "failed", "error": "e", "retryable": True, "output": "c"},
        {"status": "failed", "error": "e", "retryable": True, "output": "d"},
        {"status": "failed", "error": "e", "retryable": True, "output": "e"},
    ]
    idx = {"i": 0}

    def _iter(_loop):
        out = script[idx["i"]]
        idx["i"] += 1
        return out

    controller._iterate = _iter  # type: ignore[method-assign]
    res = _run(
        controller,
        loop,
        max_iterations=20,
        max_consecutive_failures=3,
        no_progress_window=0,  # isolate exit 7 from exit 5
    )
    assert res["status"] == "error_threshold_exceeded"
    assert res["iterations"] == 6  # not 3 — the mid-run success reset the count


# ── exit 8: EXTERNAL EVENT ────────────────────────────────────────────────────


def test_exit8_external_event_via_probe_callback():
    engine = LoopEngine()
    loop = _submit(engine, loop_id="loop:develop:exit8")
    polls = {"n": 0}

    def probe() -> bool:
        polls["n"] += 1
        return polls["n"] >= 2  # fires on the second poll

    res = _run(
        LoopController(engine, develop_runner=lambda _c, _w: (False, "x")),
        loop,
        max_iterations=10,
        event_probe=probe,
    )
    assert res["status"] == "external_event_satisfied"
    item = wi.get_work_item(engine, wi.loop_work_item_id(loop["id"]))
    assert item["status"] == "succeeded"  # the signal = success


def test_exit8_external_event_loop_kind_resolves_registered_probe():
    engine = LoopEngine()

    # A not-yet-fired signal keeps the ``external_event`` loop polling until the
    # turn cap (a terminal WorkItem can't be re-run, so use two distinct loops).
    waiting = _submit(engine, loop_id="loop:external_event:waiting", kind="external_event")
    ctrl_wait = LoopController(
        engine, event_probes={"loop:external_event:waiting": lambda: False}
    )
    res_wait = _run(ctrl_wait, waiting, max_iterations=2)
    assert res_wait["status"] == "max_iterations_exceeded"

    # A fired signal completes the same kind on the real-world event, resolved
    # from the registered probe (no event_probe passed to run_loop).
    ready = _submit(engine, loop_id="loop:external_event:ready", kind="external_event")
    ctrl_ready = LoopController(
        engine, event_probes={"loop:external_event:ready": lambda: True}
    )
    res_ready = _run(ctrl_ready, ready, max_iterations=5)
    assert res_ready["status"] == "external_event_satisfied"
    item = wi.get_work_item(engine, wi.loop_work_item_id(ready["id"]))
    assert item["status"] == "succeeded"
