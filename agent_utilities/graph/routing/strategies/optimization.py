"""Routing optimization strategies (R6, R7, R8).

Pure functions extracted from the router monolith — each shrinks or formats the
specialist candidate set before LLM planning, defined once and independently
testable:

* R6 (ORCH-1.2) — filtered specialist injection: compact specialist context for
  the planner prompt (prompt-bloat reduction).
* R7 (KG-2.1)  — reward-driven optimization: drop specialists whose average ACO
  pheromone affinity is below a threshold.
* R8 (AHE-3.x) — telemetry-driven optimization: drop specialists with too many
  recent PerformanceAnomalies.

NOTE: the original inline prose carried an ``adaptive_agent_router`` token from a
botched global rename; the extracted text restores the intended ``specialists``
wording (logged as an intended diff in the capability ledger).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

PHEROMONE_DROP_THRESHOLD = 0.1
ANOMALY_DROP_THRESHOLD = 5


def filter_by_pheromone(
    relevant: list[Any],
    pheromone_trails: Mapping[str, Mapping[str, float]] | None,
    *,
    threshold: float = PHEROMONE_DROP_THRESHOLD,
) -> list[Any]:
    """R7: drop specialists whose mean pheromone affinity is below ``threshold``."""
    if not pheromone_trails or not relevant:
        return list(relevant)
    out: list[Any] = []
    for a in relevant:
        trails = pheromone_trails.get(str(getattr(a, "name", "") or ""), {})
        if trails:
            avg_affinity = sum(trails.values()) / len(trails)
            if avg_affinity < threshold:
                continue  # historically poor performer
        out.append(a)
    return out


def prune_by_telemetry(
    relevant: list[Any],
    anomaly_map: Mapping[str, int] | None,
    *,
    max_anomalies: int = ANOMALY_DROP_THRESHOLD,
) -> list[Any]:
    """R8: drop specialists with more than ``max_anomalies`` recent anomalies."""
    if not anomaly_map or not relevant:
        return list(relevant)
    out: list[Any] = []
    for a in relevant:
        if anomaly_map.get(str(getattr(a, "name", "") or ""), 0) > max_anomalies:
            continue  # high recent anomaly count
        out.append(a)
    return out


def format_specialist_step_info(
    relevant: list[Any],
    specialist_tags: Mapping[str, str],
) -> str:
    """R6: compact specialist context for the planner prompt.

    When a filtered ``relevant`` set exists, list it and append the *names* of
    the other available specialists (cheap, request-on-demand) to keep the
    prompt small. Otherwise fall back to the full tag list.
    """
    if relevant:
        step_info = "\n".join(f"- {a.name}: {a.description}" for a in relevant)
        relevant_names = {a.name.lower() for a in relevant}
        other_names = [
            tag for tag in specialist_tags if tag.lower() not in relevant_names
        ]
        if other_names:
            step_info += (
                "\n\nOther available specialists (request if needed): "
                f"{', '.join(other_names)}"
            )
        return step_info
    return "\n".join(f"- {tag}: {desc}" for tag, desc in specialist_tags.items())


def optimize_specialists(
    relevant: list[Any],
    *,
    pheromone_trails: Mapping[str, Mapping[str, float]] | None = None,
    anomaly_map: Mapping[str, int] | None = None,
) -> list[Any]:
    """Apply R7 then R8 in the monolith's original order."""
    relevant = filter_by_pheromone(relevant, pheromone_trails)
    relevant = prune_by_telemetry(relevant, anomaly_map)
    return relevant
