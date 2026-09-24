"""EH-031: enrichment-plane scheduling (run an enricher now, or defer) through EG ``Decide``.

The breaker and the background throttle stay deterministic step-1b
constraints: a tripped breaker or a yielding throttle never reaches the
decision. What remains is the expected-value-versus-cost question
(DECIDE-LAYER-DESIGN §10): ``run`` declares the tick's model-call cost and the
yield the previous tick observed; ``defer`` costs and yields nothing. Running
is the deterministic fallback, exactly the old behaviour. A first tick has no
observed yield, so the fact is absent and EG abstains naming it.
"""

from __future__ import annotations

from agent_utilities import decide
from agent_utilities.decide.options import Option, text_param

RUN = "run"
DEFER = "defer"


def schedule_enricher(
    enricher: str, declared_cost: float, expected_yield: float | None
) -> bool:
    """Whether ``enricher`` runs this tick (``True`` when EG does not decide)."""
    run_numbers = {"declared_cost": declared_cost}
    defer_numbers = {"declared_cost": 0.0}
    if expected_yield is not None:
        run_numbers["expected_yield"] = expected_yield
        defer_numbers["expected_yield"] = 0.0
    choice = decide.choose(
        "au.enrichment.schedule",
        [Option(RUN, run_numbers), Option(DEFER, defer_numbers)],
        lambda: RUN,
        params=[text_param("enricher", enricher)],
    )
    return choice.option_id != DEFER


__all__ = ["DEFER", "RUN", "schedule_enricher"]
