#!/usr/bin/python
from __future__ import annotations

"""Harness-enforced loop-exit primitives (CONCEPT:AU-AHE.harness.loop-exit-conditions).

The eight agent-loop exit conditions are *enforced* by the harness, not merely
requested of the model in a prompt. This module holds the small, reusable,
side-effect-free primitives the enforced exits are built from so they can be
unit-tested in isolation and shared by both the KG ``LoopController.run_loop``
and ``orchestration.agent_runner``'s execution path:

* :class:`ConsecutiveFailureGuard` — the ``engine_breaker.CircuitBreaker``
  semantics (threshold + reset-on-progress) lifted from the transport layer to
  *loop iterations*. Trips a loop to a terminal ``error_threshold_exceeded``
  after N consecutive non-terminal failures; resets to 0 on any progress. This
  is the same counter/reset idiom as the fan-out backend's ``_STALL_THRESHOLD``
  (``consecutive_failures >= threshold``), generalized.
* :class:`GoalEvaluation` + :func:`build_goal_evaluator` — a measured pass/fail
  (with a 0..1 score) so a loop stops on a *real* verified success, not the
  callee self-declaring "done". Deterministic for ``develop`` loops (the
  validation command IS the measurement); a rubric/LLM-judge (``harness.g_eval``)
  for ``research`` / ``skill`` loops, degrading cleanly to trusting the callee
  when no model endpoint is reachable.
* :func:`progress_signature` — a stable hash of ``(status, output, checkpoint)``
  used to detect a stalled (no-progress) loop.
* :func:`resolve_deadline` — turn a relative ``max_duration_s`` / absolute
  ``deadline`` into one monotonic wall-clock deadline.

Each primitive is written so it maps cleanly onto a future explicit loop state
machine: each enforced exit is a guarded transition to a distinct terminal
state, and these are the guards.
"""

import hashlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

#: Default number of consecutive non-terminal failures before a loop halts to
#: ``error_threshold_exceeded`` (config-overridable). Mirrors the transport
#: breaker's default threshold band (3-5).
DEFAULT_FAILURE_THRESHOLD = 3

#: Default number of trailing iterations whose ``progress_signature`` must be
#: identical before a loop is declared ``stalled`` (config-overridable).
DEFAULT_NO_PROGRESS_WINDOW = 3

#: Default rubric-score floor (0..1) at/above which a measured goal evaluation
#: is treated as a real pass for ``research`` / ``skill`` loops.
DEFAULT_GOAL_EVAL_THRESHOLD = 0.7


class ConsecutiveFailureGuard:
    """Per-loop consecutive-failure guard (threshold + reset-on-progress).

    Lifts :class:`~agent_utilities.knowledge_graph.core.engine_breaker.CircuitBreaker`
    semantics from the transport layer to loop iterations. Where the transport
    breaker trips on connect/timeout errors to fail *fast*, this trips a *loop*
    to a terminal ``error_threshold_exceeded`` after ``threshold`` consecutive
    non-terminal step failures, and resets its counter to 0 on any progress — so
    a loop that keeps erroring stops with a diagnosable status instead of
    silently burning its entire turn cap.

    ``threshold <= 0`` disables tripping (the guard still counts), matching the
    transport breaker's ``threshold=0`` "disabled" convention.
    """

    def __init__(self, threshold: int = DEFAULT_FAILURE_THRESHOLD) -> None:
        self.threshold = int(threshold)
        self._consecutive = 0
        self._total_failures = 0

    @property
    def count(self) -> int:
        """Current run of consecutive failures (reset to 0 on progress)."""
        return self._consecutive

    @property
    def total_failures(self) -> int:
        """Lifetime failure count (never reset) — for observability."""
        return self._total_failures

    @property
    def enabled(self) -> bool:
        return self.threshold > 0

    @property
    def tripped(self) -> bool:
        """True once ``count`` reaches the threshold (and the guard is enabled)."""
        return self.enabled and self._consecutive >= self.threshold

    def record_failure(self) -> bool:
        """Record one non-terminal failure. Returns :attr:`tripped` afterwards."""
        self._consecutive += 1
        self._total_failures += 1
        return self.tripped

    def record_success(self) -> None:
        """Any progress resets the consecutive-failure run to 0."""
        self._consecutive = 0

    def reset(self) -> None:
        """Alias for :meth:`record_success` — reset the consecutive run."""
        self._consecutive = 0


@dataclass(frozen=True)
class GoalEvaluation:
    """A measured verdict on whether a loop has met its goal.

    ``measured`` distinguishes a *real* measurement from a degraded one: when a
    rubric/LLM judge cannot reach a model endpoint it returns ``measured=False``
    so the caller falls back to trusting the callee's own terminal status
    (legacy behavior) rather than fabricating a pass or fail.
    """

    passed: bool
    score: float = 0.0
    measured: bool = True
    detail: str = ""


# A goal evaluator maps ``(loop, step_outcome)`` -> a :class:`GoalEvaluation`.
GoalEvaluator = Callable[[dict[str, Any], dict[str, Any]], GoalEvaluation]


def _deterministic_develop_evaluation(
    _loop: dict[str, Any], outcome: dict[str, Any]
) -> GoalEvaluation:
    """The existing deterministic validator for ``develop`` loops.

    For a ``develop`` loop the validation command's exit status IS the
    measurement, so this is fully deterministic and always ``measured``:

    * ``completed`` / ``succeeded`` -> a real measured pass (the command exited 0);
    * ``pending`` / ``running`` -> measured *not-passed*: keep iterating;
    * any other terminal status (``failed`` / ``rejected`` / ...) -> defer to the
      callee's terminal status (``measured=False``) so a hard failure still stops
      the loop instead of being retried forever.
    """
    status = str(outcome.get("status", "")).strip().lower()
    if status in ("completed", "succeeded"):
        return GoalEvaluation(True, 1.0, True, "validation passed (deterministic)")
    if status in ("pending", "running", "validating", ""):
        return GoalEvaluation(False, 0.0, True, "validation pending (deterministic)")
    return GoalEvaluation(
        False, 0.0, measured=False, detail="deferred to callee terminal status"
    )


def _make_llm_judge_evaluator(loop: dict[str, Any], threshold: float) -> GoalEvaluator:
    """A rubric/LLM-judge evaluator for ``research`` / ``skill`` loops.

    Wraps :class:`agent_utilities.harness.g_eval.GEval` (logprob-weighted
    G-Eval). Today these loops only trust the callee's ``ok``/``fail`` or an
    edge-existence check; this scores the produced output against a rubric
    derived from the loop's objective/end-state and only declares a pass when the
    measured score clears ``threshold``. Degrades to ``measured=False`` (trust the
    callee) whenever no model endpoint is reachable — so it is safe to leave on by
    default and offline.
    """
    objective = str(loop.get("objective") or loop.get("name") or "").strip()
    end_state = str(loop.get("end_state") or "").strip()
    criteria = end_state or (
        f"The result fully and correctly accomplishes the objective: {objective}"
    )

    def _evaluate(lp: dict[str, Any], outcome: dict[str, Any]) -> GoalEvaluation:
        output = str(outcome.get("output", "")).strip()
        if not output:
            return GoalEvaluation(
                False, 0.0, measured=False, detail="no output to judge"
            )
        try:
            from agent_utilities.harness import g_eval as _ge

            # Fast, network-free offline check: no live endpoint -> degrade to
            # trusting the callee rather than attempting (and timing out on) a
            # model call in a zero-infra environment.
            if _ge._live_endpoint() is None:
                return GoalEvaluation(
                    False, 0.0, measured=False, detail="no model endpoint (degraded)"
                )
            scorer = _ge.GEval(
                task_introduction=objective
                or "Evaluate the loop result against its goal",
                evaluation_criteria=criteria,
            )
            score, reasoning = scorer.score(
                objective or str(lp.get("name", "")), output
            )
        except Exception as exc:  # noqa: BLE001 — a judge error must never crash the loop
            logger.debug("goal evaluator (g-eval) failed: %s", exc)
            return GoalEvaluation(
                False, 0.0, measured=False, detail=f"judge error: {exc}"
            )
        return GoalEvaluation(
            passed=score >= threshold,
            score=float(score),
            measured=True,
            detail=f"rubric score={score:.2f} (>= {threshold:.2f} passes): {reasoning}",
        )

    return _evaluate


def build_goal_evaluator(
    loop: dict[str, Any],
    *,
    threshold: float = DEFAULT_GOAL_EVAL_THRESHOLD,
    enable_llm_judge: bool = True,
) -> GoalEvaluator:
    """Build the default goal evaluator for a loop, dispatched by ``kind``.

    * ``develop`` -> the deterministic validation-command evaluator (always
      measured);
    * ``research`` / ``skill`` / ``external_event`` -> the rubric/LLM-judge
      evaluator (``harness.g_eval``), degrading to callee-trust offline.

    An injected ``goal_evaluator`` on ``run_loop`` overrides this entirely.
    """
    kind = str(loop.get("kind") or "research").strip().lower()
    if kind == "develop":
        return _deterministic_develop_evaluation
    if not enable_llm_judge:
        # Explicitly disabled -> a no-op evaluator that always defers to the
        # callee's terminal status (measured=False), i.e. exactly legacy behavior.
        return lambda _l, _o: GoalEvaluation(
            False, 0.0, measured=False, detail="llm judge disabled"
        )
    return _make_llm_judge_evaluator(loop, threshold)


def progress_signature(status: str, output: str, checkpoint: str = "") -> str:
    """Stable content hash of a loop iteration's substantive result.

    Hashes ``(status, output, checkpoint)`` — the tuple whose *repetition* across
    iterations means the loop is making no progress. NOTE: the monotonic
    per-iteration fence (``checkpoint:iteration:N``) is deliberately NOT the
    ``checkpoint`` argument here — that always changes and would defeat stall
    detection. Pass a *content* checkpoint (a runner-emitted state token) or ""; a
    develop/skill/research step's ``output`` is the primary progress signal.
    """
    payload = repr((str(status), str(output), str(checkpoint)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def window_is_stalled(signatures: list[str], window: int) -> bool:
    """True when the last ``window`` progress signatures are all identical.

    Requires at least ``window`` samples (a short loop can't be "stalled").
    ``window <= 1`` disables stall detection (a single sample is never a stall).
    """
    if window <= 1 or len(signatures) < window:
        return False
    tail = signatures[-window:]
    return len(set(tail)) == 1


def graph_node_transition_cap(max_steps: int) -> int:
    """Turn a ``run_agent`` ``max_steps`` (interaction ROUNDS) into the multi-agent
    graph's node-transition cap (exit 2 TURN CAP threading).

    The graph dispatcher force-terminates when ``node_transitions >
    execution_budget.max_node_transitions``; a caller's ``max_steps`` is threaded
    onto that existing enforced cap with headroom for the router/agent/verifier
    hops per round — the SAME ``max(max_steps*2, 10)`` convention the single-server
    path (``agent_runner._execute_single_server``) uses for its ``request_limit``,
    so both paths interpret ``max_steps`` identically.
    """
    return max(int(max_steps) * 2, 10)


def resolve_deadline(
    deadline: float | None,
    max_duration_s: float | None,
    start_monotonic: float,
) -> float | None:
    """Resolve one monotonic wall-clock deadline for a loop.

    ``deadline`` (an absolute ``time.monotonic()`` value) wins when given;
    otherwise ``max_duration_s`` is added to ``start_monotonic``. Returns
    ``None`` when neither is set (no wall-clock exit). This deadline is checked in
    the loop's while-condition every iteration, independent of any per-substep
    timeout.
    """
    if deadline is not None:
        return float(deadline)
    if max_duration_s is not None and float(max_duration_s) > 0:
        return start_monotonic + float(max_duration_s)
    return None


def deadline_passed(deadline: float | None, *, now: float | None = None) -> bool:
    """True when ``deadline`` (monotonic) is set and has elapsed."""
    if deadline is None:
        return False
    return (now if now is not None else time.monotonic()) >= deadline


__all__ = [
    "DEFAULT_FAILURE_THRESHOLD",
    "DEFAULT_GOAL_EVAL_THRESHOLD",
    "DEFAULT_NO_PROGRESS_WINDOW",
    "ConsecutiveFailureGuard",
    "GoalEvaluation",
    "GoalEvaluator",
    "build_goal_evaluator",
    "deadline_passed",
    "graph_node_transition_cap",
    "progress_signature",
    "resolve_deadline",
    "window_is_stalled",
]
