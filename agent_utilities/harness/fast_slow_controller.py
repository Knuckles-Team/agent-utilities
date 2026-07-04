#!/usr/bin/python
from __future__ import annotations

"""Fast-Slow Training controller — interleave harness updates with owned weights.

CONCEPT:AU-ORCH.execution.feed-cycle-outcome-fast — Fast-Slow Training (FST)

Distilled from "Owning Your Token Capital / Enterprise AI Learning Loop"
(Fast-Slow Training, arXiv:2605.12484). FST runs two coupled learning loops
over the same stream of production traces:

* The **FAST loop** updates the *harness* — the prompts, scaffolding and tool
  wiring — for what the task in front of the agent needs *right now*. It is
  cheap, immediate, and (critically) **model-swap-safe**: because it lives in
  the harness rather than in any single model's weights, the learning survives
  swapping the frontier model the controller still calls.
* The **SLOW loop** absorbs what *recurs* across the organization's work into an
  *owned* model's weights. Only task kinds seen often enough to be worth the
  capital of a weight update are promoted; the owned model then compounds
  alongside the frontier models it keeps calling.

The thesis is "own your token capital": every production trace is a learning
signal you paid for, so the harness captures the immediate lesson while the
recurring lessons accrue into owned weights you keep.

This controller ships **with the GRPO data spine it consumes** (the AHE-3.x
convention: controller + its data spine together, trainer micro-mechanics
specified-not-implemented). The slow loop is the live consumer of
:func:`agent_utilities.graph.training_signals.batch_normalized_advantage`: each
recurring group's advantage is computed before the (injected) trainer is asked
to absorb it. The *actual* weight trainer is **deferred** — the controller
accepts an injected ``trainer_fn`` (no-op by default) so the full control flow
and the GRPO advantage computation are exercised and testable today, while no
real training is implemented here.

Everything is deterministic and dependency-injected: the caller supplies the
fast-loop harness updater and the slow-loop trainer. No global state, no I/O.

Concept: fast-slow-training
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# NOTE: ``batch_normalized_advantage`` is imported lazily inside the consuming
# method below. A top-level import triggers ``agent_utilities.graph.__init__``,
# which eagerly re-exports the pydantic-ai/pydantic-graph agent runtime (``builder``)
# — an ``[agent]``-extra dependency. Importing it eagerly here would put pydantic_ai
# on the import path of every harness consumer (e.g. the lean Eval-corpus gate),
# breaking ``import`` in the lean CI/serving install (Dependency discipline).

__all__ = [
    "Trace",
    "SlowUpdate",
    "HarnessUpdateFn",
    "TrainerFn",
    "FastSlowController",
]


@dataclass
class Trace:
    """One production trace — the unit of token capital the loops learn from.

    Args:
        task_key: What *kind* of task this was. Recurrence is detected by
            grouping traces with the same ``task_key``; a kind seen often
            enough graduates from the fast loop to the slow loop.
        reward: Scalar outcome reward feeding the GRPO advantage computation.
        prompt: The prompt/scaffold text the harness used for this trace (the
            fast loop reads it to decide the next harness update).
    """

    task_key: str
    reward: float
    prompt: str = ""


@dataclass
class SlowUpdate:
    """Summary of one recurring group absorbed by the slow (owned-weights) loop.

    Args:
        task_key: The recurring task kind that crossed the recurrence threshold.
        n_traces: How many traces of that kind were absorbed.
        advantage_mean: Mean group-normalized advantage over the recurring
            group, from :func:`batch_normalized_advantage` — the GRPO signal the
            (deferred) trainer would optimize against.
    """

    task_key: str
    n_traces: int
    advantage_mean: float


# Fast loop: given recent traces, return the id of the updated harness/prompt.
HarnessUpdateFn = Callable[[list["Trace"]], str]
# Slow loop: absorb a recurring group into owned weights. DEFERRED — the real
# trainer is specified-not-implemented; the default is a pure no-op so the
# control flow and GRPO spine run without any training side effect.
TrainerFn = Callable[[str, list["Trace"]], None]


def _noop_trainer(task_key: str, traces: list[Trace]) -> None:
    """Deferred slow-loop trainer: absorb nothing (specified-not-implemented).

    The real owned-weights training is out of scope for this controller; this
    default lets the slow loop run end-to-end (recurrence detection + GRPO
    advantage computation + summary emission) with no training side effect, so
    a caller can inject a real ``TrainerFn`` later without any other change.
    """
    return None


@dataclass
class _State:
    """Internal mutable state — kept in one place so a model swap can preserve it."""

    observed: list[Trace] = field(default_factory=list)
    harness_id: str = ""
    fast_steps: int = 0
    slow_steps: int = 0
    swaps: list[str] = field(default_factory=list)


class FastSlowController:
    """Interleave fast (harness) + slow (owned-weights) learning (CONCEPT:AU-ORCH.execution.feed-cycle-outcome-fast).

    The controller collects production :class:`Trace` objects via
    :meth:`observe`, runs the FAST loop (:meth:`fast_step`) to update the harness
    for the current work, and runs the SLOW loop (:meth:`slow_step`) to absorb
    task kinds that *recur* across the work into owned weights. :meth:`run`
    interleaves the two on a fixed cadence, and :meth:`swap_model` records a
    frontier-model swap without discarding any accumulated learning.

    Args:
        harness_update_fn: Fast-loop callback. Given the traces observed since
            the last fast step, it returns the id of the updated harness/prompt.
        trainer_fn: Slow-loop callback. Given a recurring ``task_key`` and its
            traces, it absorbs them into owned weights. DEFERRED — defaults to a
            no-op so the control flow and GRPO spine are fully exercised without
            implementing real training here.
        recurrence_threshold: A ``task_key`` is promoted to the slow loop once at
            least this many traces of that kind have been observed.

    The owned model id and harness id are exposed as read-only properties.
    """

    def __init__(
        self,
        harness_update_fn: HarnessUpdateFn,
        *,
        trainer_fn: TrainerFn | None = None,
        recurrence_threshold: int = 5,
    ) -> None:
        if recurrence_threshold < 1:
            raise ValueError("recurrence_threshold must be >= 1")
        self._harness_update_fn = harness_update_fn
        self._trainer_fn: TrainerFn = trainer_fn or _noop_trainer
        self._recurrence_threshold = recurrence_threshold
        self._state = _State()
        self._owned_model_id = "owned-v0"
        # Traces consumed by a fast step but not yet eligible/consumed by a slow
        # step. The fast loop reads recency; the slow loop reads recurrence.
        self._fast_cursor = 0

    # ── observation ──────────────────────────────────────────────────────────
    def observe(self, trace: Trace) -> None:
        """Collect one production trace for both loops to learn from."""
        self._state.observed.append(trace)

    # ── fast loop ──────────────────────────────────────────────────────────--
    def fast_step(self) -> str:
        """Run the FAST loop over traces observed since the last fast step.

        Delegates to the injected ``harness_update_fn`` (the actual harness/prompt
        rewrite is the caller's), records the returned harness id as current, and
        advances the fast cursor. Returns the updated harness id. With no new
        traces the harness id is left unchanged (no spurious update).
        """
        recent = self._state.observed[self._fast_cursor :]
        if recent:
            self._state.harness_id = self._harness_update_fn(recent)
            self._fast_cursor = len(self._state.observed)
        self._state.fast_steps += 1
        return self._state.harness_id

    # ── slow loop ──────────────────────────────────────────────────────────--
    def _recurring_groups(self) -> dict[str, list[Trace]]:
        """Group observed traces by ``task_key``, keeping only recurring kinds."""
        groups: dict[str, list[Trace]] = defaultdict(list)
        for trace in self._state.observed:
            groups[trace.task_key].append(trace)
        return {
            key: traces
            for key, traces in groups.items()
            if len(traces) >= self._recurrence_threshold
        }

    def slow_step(self) -> list[SlowUpdate]:
        """Run the SLOW loop: absorb what RECURS across the work into owned weights.

        Finds every ``task_key`` whose observed trace count is at or above the
        recurrence threshold, computes that group's GRPO advantage via
        :func:`batch_normalized_advantage`, asks the injected ``trainer_fn`` to
        absorb the group (a no-op by default = deferred real training), and emits
        a :class:`SlowUpdate` summary per group. Consumed recurring traces are
        removed from the observation buffer so they are not re-absorbed; traces of
        non-recurring kinds are kept (they may recur later). Groups are processed
        in sorted ``task_key`` order for determinism.
        """
        recurring = self._recurring_groups()
        self._state.slow_steps += 1
        updates: list[SlowUpdate] = []
        consumed: set[str] = set()
        from agent_utilities.graph.training_signals import (
            batch_normalized_advantage,
        )

        for task_key in sorted(recurring):
            traces = recurring[task_key]
            advantages = batch_normalized_advantage([t.reward for t in traces])
            advantage_mean = (
                round(sum(advantages) / len(advantages), 6) if advantages else 0.0
            )
            self._trainer_fn(task_key, traces)
            updates.append(
                SlowUpdate(
                    task_key=task_key,
                    n_traces=len(traces),
                    advantage_mean=advantage_mean,
                )
            )
            consumed.add(task_key)
        if consumed:
            self._state.observed = [
                t for t in self._state.observed if t.task_key not in consumed
            ]
            # The fast cursor indexes into the observation buffer we just
            # shrank; clamp it so the next fast step does not skip live traces.
            self._fast_cursor = min(self._fast_cursor, len(self._state.observed))
        return updates

    # ── interleave ──────────────────────────────────────────────────────────-
    def run(self, *, fast_every: int = 1, slow_every: int = 3) -> dict[str, Any]:
        """Interleave the fast and slow loops over the currently observed traces.

        The traces already collected via :meth:`observe` are processed in
        observe-batches: a fast step fires every ``fast_every`` batches and a slow
        step every ``slow_every`` batches, cycling for as many batches as there
        are observed traces (so a longer trace stream drives more loop cycles).
        Returns ``{"fast_updates": [...harness ids...], "slow_updates":
        [...SlowUpdate...]}``. Deterministic for a given trace stream.
        """
        if fast_every < 1 or slow_every < 1:
            raise ValueError("fast_every and slow_every must be >= 1")
        n_batches = len(self._state.observed)
        fast_updates: list[str] = []
        slow_updates: list[SlowUpdate] = []
        for batch in range(1, n_batches + 1):
            if batch % fast_every == 0:
                fast_updates.append(self.fast_step())
            if batch % slow_every == 0:
                slow_updates.extend(self.slow_step())
        return {"fast_updates": fast_updates, "slow_updates": slow_updates}

    # ── model-swap safety ─────────────────────────────────────────────────────
    def swap_model(self, new_model_id: str) -> None:
        """Record a frontier-model swap without losing accumulated learning.

        The whole FST thesis is that learning lives in the portable harness and
        the owned weights, *not* in the frontier model the controller calls — so
        swapping that model must not reset the harness or drop observed traces.
        This records the swap and leaves observed traces, the harness id, the fast
        cursor and all step counters intact.
        """
        self._state.swaps.append(new_model_id)

    @property
    def owned_model_id(self) -> str:
        """Id of the owned model the slow loop compounds into (read-only)."""
        return self._owned_model_id

    @property
    def harness_id(self) -> str:
        """Id of the current harness/prompt produced by the fast loop (read-only)."""
        return self._state.harness_id
