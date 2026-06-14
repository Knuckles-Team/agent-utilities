#!/usr/bin/python
from __future__ import annotations

"""Hierarchical / federated coordination for large collectives.

CONCEPT:ORCH-1.53 — a hierarchical federated coordination protocol that runs consensus within agent neighborhoods and rolls up only cluster representatives, bounding interaction density so a collective coordinates at thousands of agents instead of O(N) global aggregation

The paper (§5.4) sets the bar at coordinating collectives at *vast* scale, where capability
"may scale with population size and interaction density". AU's coordination protocols
(ORCH-1.3) were explicitly small-N — a single global aggregation over all participants,
an O(N) bottleneck and a single convergence-failure point. This adds the scalable form:
partition the collective into neighborhoods, reach consensus *within* each neighborhood,
elect one representative value per neighborhood, and aggregate only the representatives
upward (recursively for very large N). Interaction density is bounded by neighborhood
size, not the whole population. Pure and aggregation-agnostic; production wraps the
ORCH-1.3 protocols inside the ORCH-1.32 MASS neighborhoods.
"""

import statistics
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any


def majority(values: Sequence[Any]) -> Any:
    """Most-common value (ties broken by first-seen) — a discrete consensus."""
    counts: dict[Any, int] = {}
    order: list[Any] = []
    for v in values:
        if v not in counts:
            order.append(v)
        counts[v] = counts.get(v, 0) + 1
    return max(order, key=lambda v: counts[v]) if order else None


def mean(values: Sequence[float]) -> float:
    """Average — a continuous consensus."""
    vals = [float(v) for v in values]
    return statistics.fmean(vals) if vals else 0.0


class HierarchicalCoordinator:
    """Federated consensus: neighborhood-local aggregation then representative roll-up."""

    def __init__(self, aggregate: Callable[[Sequence[Any]], Any] = majority) -> None:
        self.aggregate = aggregate

    def coordinate(
        self,
        votes: Mapping[Hashable, Any],
        neighborhoods: Sequence[Sequence[Hashable]],
    ) -> dict[str, Any]:
        """Aggregate ``votes`` hierarchically over ``neighborhoods``.

        Each neighborhood's members' votes are aggregated to one representative value;
        the representatives are then aggregated to the final consensus. Agents not in any
        neighborhood are ignored; empty neighborhoods are skipped. Returns the final
        consensus plus the per-neighborhood representatives and the interaction bound.
        """
        reps: list[Any] = []
        rep_detail: list[dict[str, Any]] = []
        for i, members in enumerate(neighborhoods):
            member_votes = [votes[m] for m in members if m in votes]
            if not member_votes:
                continue
            rep = self.aggregate(member_votes)
            reps.append(rep)
            rep_detail.append(
                {"neighborhood": i, "size": len(member_votes), "rep": rep}
            )
        consensus = self.aggregate(reps) if reps else None
        max_nb = max((len(n) for n in neighborhoods), default=0)
        return {
            "consensus": consensus,
            "representatives": rep_detail,
            # Interaction is bounded by the largest neighborhood + the rep roll-up,
            # not the whole population — the point of the federation.
            "interaction_bound": max(max_nb, len(reps)),
            "global_bound": len(votes),
        }

    def coordinate_recursive(
        self,
        votes: Mapping[Hashable, Any],
        *,
        fanout: int = 8,
    ) -> dict[str, Any]:
        """Auto-partition into ``fanout``-sized neighborhoods and recurse until one group.

        For very large N, builds the neighborhood tree itself (stable ordering), so the
        caller need only supply the votes. Depth ~ log_fanout(N).
        """
        keys = list(votes.keys())
        level = dict(votes)
        depth = 0
        while len(level) > fanout:
            groups = [list(keys[i : i + fanout]) for i in range(0, len(keys), fanout)]
            result = self.coordinate(level, groups)
            # representatives become the next level's votes
            level = {
                f"rep:{depth}:{j}": r["rep"]
                for j, r in enumerate(result["representatives"])
            }
            keys = list(level.keys())
            depth += 1
        consensus = self.aggregate(list(level.values())) if level else None
        return {"consensus": consensus, "depth": depth + 1, "global_bound": len(votes)}
