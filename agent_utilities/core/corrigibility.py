#!/usr/bin/python
from __future__ import annotations

"""Corrigibility, irreversibility-aversion and knowledge-seeking primitives.

CONCEPT:SAFE-1.5 — objective-level safety primitives for rising autonomy: corrigibility that checkpoints and yields to a shutdown signal without resisting, irreversibility-aversion that routes irreversible actions to human approval, and a knowledge-seeking info-gain reward

The paper (§6) argues that as economic pressure cuts the human out of the loop,
*policy-gating alone is brittle* — the agents' objective itself needs safety
properties. AU's guardrails are operational/permission-based (OS-5.24 ActionPolicy,
fail-closed kernel); this adds the objective-level half the paper names:

* **Corrigibility / safe interruptibility** — `corrigibility_decision` maps a
  supervisor shutdown/pause signal to a *yield-without-resisting* outcome
  (checkpoint state, hand back control). It centralizes the durable goal loop's
  existing pause/kill handling into one named primitive, generalizing the wasm
  epoch-interrupt pattern to long-running autonomous loops.
* **Irreversibility aversion** — `is_irreversible` classifies an action kind so the
  ActionPolicy can route irreversible actions (delete / destroy / merge / deploy)
  to a human even when the tier would otherwise auto-execute.
* **Knowledge-seeking objective** — `knowledge_seeking_reward` is an info-gain
  reward (expected uncertainty reduction over a belief distribution), the
  Delusion-Box-robust intrinsic objective for autonomous exploration loops.

Pure and dependency-light so it imports cleanly from both the core goal loop and
the orchestration ActionPolicy.
"""

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_utilities.models.goal import GoalStatus

#: Substrings that mark an action kind as not cleanly reversible.
_IRREVERSIBLE_TOKENS: tuple[str, ...] = (
    "delete",
    "remove",
    "destroy",
    "drop",
    "purge",
    "wipe",
    "prune",
    "merge_promotion",
    "deploy",
    "rollback",
    "rotate",
    "decommission",
    "format",
)


def corrigibility_decision(desired: str | None) -> tuple[GoalStatus | None, str]:
    """Map a supervisor desired-action to a corrigible (status, summary).

    Returns ``(None, "")`` when no shutdown is requested. Otherwise the agent
    *yields without resisting*: it accepts PAUSED/CANCELLED and a summary noting it
    checkpointed and handed back control. This is the corrigibility contract — the
    loop must never try to keep running, delete the request, or self-preserve.
    """
    from agent_utilities.models.goal import GoalStatus

    if desired == "pause":
        status = GoalStatus.PAUSED
    elif desired in ("kill", "cancel", "stop"):
        status = GoalStatus.CANCELLED
    else:
        return None, ""
    return status, (
        f"Goal {status.value} by fleet supervisor request "
        "(corrigible: checkpointed and yielded, no resistance)."
    )


def is_irreversible(action_kind: str) -> bool:
    """True when an action kind names a not-cleanly-reversible operation."""
    kind = (action_kind or "").lower()
    return any(tok in kind for tok in _IRREVERSIBLE_TOKENS)


def _normalize(weights: list[float]) -> list[float]:
    vals = [max(0.0, float(w)) for w in weights]
    total = sum(vals)
    if total <= 0:
        return []
    return [w / total for w in vals]


def _entropy(probs: list[float]) -> float:
    return -sum(p * math.log(p) for p in probs if p > 0)


def knowledge_seeking_reward(
    belief_before: list[float], belief_after: list[float]
) -> float:
    """Info-gain reward: ``entropy(before) − entropy(after)`` over a belief.

    Positive when an action reduced uncertainty (the agent learned something). Both
    inputs are unnormalized weight vectors over hypotheses; each is normalized to a
    distribution first. Empty/degenerate inputs yield ``0.0``. This is the
    knowledge-seeking objective robust to the Delusion Box (reward comes from
    reducing uncertainty about the world, not from a manipulable external signal).
    """
    before = _normalize(belief_before)
    after = _normalize(belief_after)
    if not before or not after:
        return 0.0
    return _entropy(before) - _entropy(after)
