"""Retrieval-plan choice through EG ``Decide`` (AU-CONTEXT-R001).

``HybridRetriever.plan_and_retrieve`` used to take its plan from the caller's
``mode`` argument, and for ``hyde`` asked an LLM planner for a multi-query
plan. The plan TEMPLATE (``standard`` / ``deep`` / ``hyde``) is now a
decision: each template is a declared option with its relevance threshold and
pass count; the caller's requested mode is the deterministic fallback and a
declared fact (``requested``), so a head can learn when to overrule it. The
query itself travels only as a typed ``text`` parameter.

The task class's PROVEN retrieval paths -- typed plan templates EG
recorded from independently judged successful runs, under the graph's current
composed schema -- join the same choice as further options (``path:<digest>``)
carrying their judged successes and failures. Decide calibrates them like any
option and may abstain, in which case the requested template runs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agent_utilities import decide
from agent_utilities.decide.options import Option, text_param
from agent_utilities.decide.outcome import Choice

#: Template -> (relevance threshold, retrieval passes). ``hyde`` fans out to
#: the planner's queries; one pass here means "one merged plan pass".
PLAN_TEMPLATES: dict[str, tuple[float, float]] = {
    "standard": (0.38, 1.0),
    "deep": (0.28, 1.0),
    "hyde": (0.38, 3.0),
}

#: The option-id prefix of a proven path.
PATH_PREFIX = "path:"


def _template_option(name: str, requested: str) -> Option:
    threshold, passes = PLAN_TEMPLATES[name]
    return Option(
        name,
        {
            "failures": 0.0,
            "passes": passes,
            "requested": 1.0 if name == requested else 0.0,
            "successes": 0.0,
            "threshold": threshold,
        },
    )


def path_option(path: Mapping[str, Any]) -> Option:
    """A proven path as a declared option (its judged record as facts)."""
    return Option(
        PATH_PREFIX + str(path["template_digest"]),
        {
            "failures": float(path.get("failures") or 0),
            "passes": 1.0,
            "requested": 0.0,
            "successes": float(path.get("successes") or 0),
            "threshold": PLAN_TEMPLATES["standard"][0],
        },
    )


def choose_retrieval(
    query: str, requested: str, paths: Sequence[Mapping[str, Any]] = ()
) -> Choice:
    """The plan for ``query`` -- a template or a proven path -- and EG's record.

    An unknown ``requested`` mode is passed through without asking EG.
    """
    if requested not in PLAN_TEMPLATES:
        return Choice(requested, False, "unknown_mode")
    options = [_template_option(name, requested) for name in PLAN_TEMPLATES]
    options.extend(path_option(path) for path in paths)
    return decide.choose(
        "au.retrieval.plan",
        options,
        lambda: requested,
        params=[text_param("query", query)],
    )


def choose_retrieval_plan(query: str, requested: str) -> str:
    """The plan template for ``query``; ``requested`` when EG does not decide."""
    return choose_retrieval(query, requested).option_id or requested


__all__ = [
    "PATH_PREFIX",
    "PLAN_TEMPLATES",
    "choose_retrieval",
    "choose_retrieval_plan",
    "path_option",
]
