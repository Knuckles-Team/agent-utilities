"""EH-029: retrieval-plan choice through EG ``Decide``.

``HybridRetriever.plan_and_retrieve`` used to take its plan from the caller's
``mode`` argument, and for ``hyde`` asked an LLM planner for a multi-query
plan. The plan TEMPLATE (``standard`` / ``deep`` / ``hyde``) is now a
decision: each template is a declared option with its relevance threshold and
pass count; the caller's requested mode is the deterministic fallback and a
declared fact (``requested``), so a head can learn when to overrule it. The
query itself travels only as a typed ``text`` parameter.
"""

from __future__ import annotations

from agent_utilities import decide
from agent_utilities.decide.options import Option, text_param

#: Template -> (relevance threshold, retrieval passes). ``hyde`` fans out to
#: the planner's queries; one pass here means "one merged plan pass".
PLAN_TEMPLATES: dict[str, tuple[float, float]] = {
    "standard": (0.38, 1.0),
    "deep": (0.28, 1.0),
    "hyde": (0.38, 3.0),
}


def choose_retrieval_plan(query: str, requested: str) -> str:
    """The plan template for ``query``; ``requested`` when EG does not decide."""
    if requested not in PLAN_TEMPLATES:
        return requested
    options = [
        Option(
            name,
            {
                "threshold": threshold,
                "passes": passes,
                "requested": 1.0 if name == requested else 0.0,
            },
        )
        for name, (threshold, passes) in PLAN_TEMPLATES.items()
    ]
    choice = decide.choose(
        "au.retrieval.plan",
        options,
        lambda: requested,
        params=[text_param("query", query)],
    )
    return choice.option_id or requested


__all__ = ["PLAN_TEMPLATES", "choose_retrieval_plan"]
