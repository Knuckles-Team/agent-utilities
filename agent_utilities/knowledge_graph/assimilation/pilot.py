#!/usr/bin/python
from __future__ import annotations

"""Pilot harness — validate the assimilation engine on a known corpus (CONCEPT:KG-2.7).

The acceptance test for the whole reframe: run the engine over the existing corpus
and confirm it (a) **does not re-propose already-implemented features** — the
machine-checkable invariant that fixes the first attempt's rework — and (b) surfaces
the ranked open gaps for human comparison against the known Waves A/B/C plans.

``run_pilot`` runs the assimilation pass with plan synthesis and asserts the hard
invariant: no proposed feature is one the graph already marks satisfied/built. It
returns metrics + any offenders for inspection; ``passed`` is the invariant result.

Concept: assimilation-pilot
"""

from dataclasses import dataclass, field
from typing import Any

from .gap_analysis import _FEATURE_TYPES, is_closed


@dataclass
class PilotReport:
    total_features: int = 0
    already_built: int = 0  # features the graph marks satisfied/superseded/implemented
    recognized_built: int = 0  # features auto_satisfy matched to a concept THIS pass
    open_gaps: int = 0
    proposed: int = 0
    reproposed_built: list[str] = field(default_factory=list)  # MUST be empty
    ranked_gaps: list[dict[str, Any]] = field(default_factory=list)
    passed: bool = False


def _all_feature_ids(engine: Any, feature_types: tuple[str, ...]) -> list[str]:
    graph = getattr(engine, "graph", None)
    if graph is None:
        return []
    wanted = {t.lower() for t in feature_types}  # case-insensitive (live labels)
    try:
        return [
            nid
            for nid, d in graph.nodes(data=True)
            if isinstance(d, dict) and str(d.get("type", "")).lower() in wanted
        ]
    except TypeError:  # pragma: no cover
        return []


def run_pilot(
    engine: Any,
    *,
    top_n: int = 50,
    feature_types: tuple[str, ...] = _FEATURE_TYPES,
    synth_fn: Any = None,
) -> PilotReport:
    """Run the assimilation pass and validate the no-re-propose-built invariant.

    ``synth_fn`` is forwarded to plan synthesis; pass
    ``assimilation.plan_synthesis._default_synth`` to run the pilot fully offline
    (deterministic, no planner LLM).
    """
    from ..research.loop_controller import run_assimilation_pass

    ids = _all_feature_ids(engine, feature_types)
    # Snapshot what's already built BEFORE synthesis flips features to 'proposed'.
    built = {fid for fid in ids if is_closed(engine, fid)}

    rep = run_assimilation_pass(
        engine, synthesize=True, top_n=top_n, force=True, synth_fn=synth_fn
    )
    proposed = rep.get("proposed_plans", []) or []
    proposed_ids = [p["feature_id"] for p in proposed]

    reproposed_built = sorted(set(proposed_ids) & built)
    report = PilotReport(
        total_features=len(ids),
        already_built=len(built),
        recognized_built=int(rep.get("auto_satisfied", 0)),
        open_gaps=int(rep.get("open_gaps", 0)),
        proposed=len(proposed_ids),
        reproposed_built=reproposed_built,
        ranked_gaps=rep.get("ranked_gaps", []),
        passed=len(reproposed_built) == 0,
    )
    return report


def summarize(report: PilotReport) -> str:
    """Human-readable one-paragraph pilot summary."""
    status = "PASS" if report.passed else "FAIL"
    lines = [
        f"Assimilation pilot: {status}",
        f"  features={report.total_features} already_built={report.already_built} "
        f"recognized_built={report.recognized_built} "
        f"open_gaps={report.open_gaps} proposed={report.proposed}",
        f"  re-proposed already-built: {report.reproposed_built or 'none (invariant holds)'}",
        "  top gaps: "
        + ", ".join(g.get("feature_id", "?") for g in report.ranked_gaps[:10]),
    ]
    return "\n".join(lines)


__all__ = ["PilotReport", "run_pilot", "summarize"]
