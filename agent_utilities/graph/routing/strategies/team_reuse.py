"""TeamConfig reuse strategy (R2 — CONCEPT:AHE-3.3).

Before invoking the LLM planner, the router checks the KG for a proven
``TeamConfig`` matching the query and reuses it when its success rate clears its
reuse threshold. The reuse *decision* is extracted here as a pure predicate so
it is defined once and independently testable; the monolith still performs the
KG lookup and builds the resulting plan.
"""

from __future__ import annotations

from typing import Any


def select_reusable_team(matching_teams: list[Any] | None) -> Any | None:
    """R2: return the top matching team if it is worth reusing, else ``None``.

    A team is reusable when its ``success_rate`` strictly exceeds its
    ``reuse_threshold`` (the monolith's original condition).
    """
    if not matching_teams:
        return None
    top = matching_teams[0]
    try:
        if top.success_rate > top.reuse_threshold:
            return top
    except AttributeError:
        return None
    return None
