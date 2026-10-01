"""Routing consumers: choice/model routing and cost-aware routing through EG Decide.

* ``au.route.choice`` -- :class:`~agent_utilities.orchestration.outcome_router.
  OutcomeRouter`, the one chokepoint for "pick a choice per task class"
  (execution shape, intent surface, trace/placement mining). Its prior and
  learned reward are declared as claims; its old prior-plus-EMA rule is the
  fallback.
* ``au.route.model`` -- the role's model pick in ``core.model_router``; the
  tier-gated registry pick is the fallback.
* ``au.route.cost`` -- the same model choice read on cost. Declared cost comes
  from the registry; OBSERVED cost belongs to L5 accounting, which no store
  holds yet, so no option carries ``l5.observed_cost`` and EG abstains naming
  exactly that fact (``UnknownFact``) until an L5 cost source is wired.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from agent_utilities import decide
from agent_utilities.decide.options import Option, text_param

_TIER_RANK = {"light": 0, "medium": 1, "heavy": 2, "reasoning": 3}


def route_choice(
    namespace: str,
    task_class: str,
    prior: str,
    rewards: Mapping[str, float],
    heuristic: str,
) -> str:
    """The routed choice among ``rewards``' keys; ``heuristic`` is the answer when EG does not."""
    options = [
        Option(c, {"prior": 1.0 if c == prior else 0.0, "reward": reward})
        for c, reward in rewards.items()
    ]
    params = [text_param("namespace", namespace), text_param("task_class", task_class)]
    choice = decide.choose("au.route.choice", options, lambda: heuristic, params=params)
    return choice.option_id or heuristic


def _model_options(models: Sequence[Any], picked: Any) -> list[Option]:
    return [
        Option(
            str(m.id),
            {
                "tier_rank": float(_TIER_RANK.get(str(m.tier), 1)),
                "heuristic": 1.0 if m is picked else 0.0,
            },
        )
        for m in models
    ]


def route_model(role: str, models: Sequence[Any], picked: Any) -> Any:
    """The model for ``role`` among eligible ``models``; ``picked`` is the fallback."""
    if picked is None or not models:
        return picked
    by_id = {str(m.id): m for m in models}
    choice = decide.choose(
        "au.route.model",
        _model_options(models, picked),
        lambda: str(picked.id),
        params=[text_param("role", role)],
    )
    return by_id.get(str(choice.option_id), picked)


class ObservedCost(Protocol):
    """L5 accounting: the observed cost of one model, or ``None`` if unknown."""

    def observed_cost(self, model_id: str) -> float | None: ...


def _declared_cost(model: Any) -> float:
    cost = getattr(model, "cost", None)
    return float(getattr(cost, "input", None) or 0.0) + float(
        getattr(cost, "output", None) or 0.0
    )


def _cost_option(model: Any, observed: ObservedCost | None) -> Option:
    numbers = {"declared_cost": _declared_cost(model)}
    seen = None if observed is None else observed.observed_cost(str(model.id))
    if seen is not None:
        numbers["l5.observed_cost"] = seen
    return Option(str(model.id), numbers)


def route_by_cost(
    models: Sequence[Any], picked: Any, observed: ObservedCost | None = None
) -> decide.Choice:
    """Cost-aware model choice; without an L5 cost source EG abstains naming it."""
    options = [_cost_option(m, observed) for m in models]
    return decide.choose("au.route.cost", options, lambda: str(picked.id))


__all__ = ["ObservedCost", "route_by_cost", "route_choice", "route_model"]
