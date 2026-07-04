"""Generic outcome-learned choice router (CONCEPT:AU-ORCH.execution.shape-policy-learning).

ONE mechanism for *"pick a choice per task-class, learn from the run outcome"*, factored from
the patterns the KG-2.68 ``ReasonerRouter`` (``knowledge_graph/core/reasoner.py``) and the
AHE-3.38 sampling-profile EMA-tournament (``harness/variant_pool.evolve_profile``) already use —
so shape selection (and, later, profile / paradigm selection) **share one learner instead of each
rolling its own bandit**. This is the anti-sprawl spine: it is a thin, **embedding-free** wrapper
over the shared ``CapabilityIndex`` reward-EMA store (the established learner), keyed exactly like
``_profile_id(task_class, choice)``.

Design goals:

* **Minimal overhead** — selection is a couple of O(1) reward-map reads; there is **no per-turn
  embedding** (the choice is gated by a free task-class, not vector similarity), so it never
  reintroduces the latency the lean fast path removed.
* **Maximally dynamic** — the caller's cheap heuristic is a *prior*; the learned reward-EMA refines
  it continuously and **flips the choice** once the learned signal diverges from neutral. An
  untried alternative carries the neutral default (0.5), so a prior that underperforms is
  naturally explored.
* **Self-correcting** — ``record(task_class, choice, reward)`` feeds ``success × speed`` back into
  the same EMA the reasoner/profile learners use; failures pull the EMA down.

The EMA store is in-process per router (mirroring ``ReasonerRouter``'s own ``CapabilityIndex``);
durable cross-process persistence (save/load the reward map, fleet-wide) is the next layer.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Soft weight of the caller's heuristic prior. The learned reward-EMA (in [0, 1], 0.5 neutral)
# overrides the prior once an alternative's EMA exceeds the prior's by more than this — so early
# on the heuristic dominates and the policy only diverges from it on real evidence.
_PRIOR_BIAS = 0.15


class OutcomeRouter:
    """Per-task-class choice selection that learns from outcomes, over the CapabilityIndex spine."""

    def __init__(self, namespace: str) -> None:
        from agent_utilities.knowledge_graph.retrieval.capability_index import (
            CapabilityIndex,
        )

        # dim=1: we use only the reward-EMA map (no vectors) — choices are gated by task-class,
        # not embedding similarity, so this never embeds per call.
        self._index: Any = CapabilityIndex(dim=1)
        self._ns = namespace

    def _id(self, task_class: str, choice: str) -> str:
        return f"{self._ns}:{task_class}:{choice}"

    def select(self, task_class: str, prior: str, candidates: tuple[str, ...]) -> str:
        """Return the chosen candidate: the heuristic ``prior`` nudged by the learned reward-EMA.

        ``score(c) = (prior_bias if c is the prior else 0) + reward_of(c)``. Early (all EMA at the
        0.5 neutral default) the prior wins; as outcomes accumulate the EMA can flip it, and an
        untried alternative (neutral 0.5) is explored when the prior's EMA falls below it.
        """
        best, best_score = prior, float("-inf")
        for c in candidates:
            score = (_PRIOR_BIAS if c == prior else 0.0) + self.reward_of(task_class, c)
            if score > best_score:
                best, best_score = c, score
        return best

    def record(self, task_class: str, choice: str, reward: float) -> None:
        """Feed a run outcome back into the shared reward-EMA (best-effort, never raises)."""
        try:
            self._index.record_outcome(
                self._id(task_class, choice), reward=max(0.0, min(1.0, reward))
            )
        except Exception as e:  # noqa: BLE001 — learning must never break the caller
            logger.debug("[ORCH-1.71] outcome record skipped: %s", e)

    def reward_of(self, task_class: str, choice: str) -> float:
        try:
            return float(self._index.reward_of(self._id(task_class, choice)))
        except Exception:  # noqa: BLE001
            return 0.5  # neutral when the store is unavailable


def outcome_reward(*, success: bool, latency_s: float, budget_s: float = 30.0) -> float:
    """Map a run outcome to a reward in [0, 1] (CONCEPT:AU-ORCH.execution.shape-policy-learning).

    A success is rewarded, discounted by how much of the latency budget it spent (faster is
    better), so the learner prefers the *cheapest shape that still succeeds*; a failure is 0.
    """
    if not success:
        return 0.0
    spent = min(max(latency_s, 0.0) / max(budget_s, 1e-6), 1.0)
    return 1.0 - 0.3 * spent
