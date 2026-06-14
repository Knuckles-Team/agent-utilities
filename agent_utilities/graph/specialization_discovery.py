#!/usr/bin/python
from __future__ import annotations

"""Emergent specialization: discover under-served niches and propose specialists.

CONCEPT:ORCH-1.52 — an emergent-specialization discovery pass that clusters the failing or expensive task stream and proposes a new specialist archetype for any niche no existing archetype covers, so the collective autonomously increases its division of labour

The paper (§5.4/§5.3) makes *cognitive division of labour* — and the pressure to
specialize — the engine of collective superintelligence: specialization should increase
over time, and a homogeneous collective must differentiate to gain synergy. AU's roles
were statically authored; nothing analyzed the task stream to discover where a new
specialist was needed. This adds that discovery: cluster the embeddings of failing /
expensive tasks, and for any cluster that no existing archetype covers (max similarity
below a floor), propose a new specialist archetype for that niche. The proposal is meant
to flow through the regression-gated AHE-3.4 team-evolution path; this module is the pure,
testable discovery half.
"""

import math
from dataclasses import dataclass, field


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _centroid(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]


@dataclass
class SpecialistProposal:
    """A proposed new specialist archetype for an under-served task niche."""

    niche_id: int
    size: int
    centroid: list[float]
    coverage: float  # max similarity of the niche to any existing archetype
    examples: list[str] = field(default_factory=list)


class SpecializationDiscovery:
    """Cluster the under-served task stream and propose missing specialists."""

    def __init__(
        self,
        *,
        cluster_threshold: float = 0.6,
        coverage_floor: float = 0.5,
        min_cluster: int = 3,
    ) -> None:
        self.cluster_threshold = cluster_threshold
        self.coverage_floor = coverage_floor
        self.min_cluster = min_cluster

    def _cluster(
        self, items: list[tuple[str, list[float]]]
    ) -> list[list[tuple[str, list[float]]]]:
        """Greedy single-pass cosine clustering (deterministic, dependency-free)."""
        clusters: list[list[tuple[str, list[float]]]] = []
        centroids: list[list[float]] = []
        for key, vec in items:
            best, best_sim = -1, -1.0
            for ci, c in enumerate(centroids):
                sim = _cosine(vec, c)
                if sim > best_sim:
                    best, best_sim = ci, sim
            if best >= 0 and best_sim >= self.cluster_threshold:
                clusters[best].append((key, vec))
                centroids[best] = _centroid([v for _, v in clusters[best]])
            else:
                clusters.append([(key, vec)])
                centroids.append(list(vec))
        return clusters

    def discover(
        self,
        tasks: list[tuple[str, list[float]]],
        archetype_embeddings: list[list[float]],
    ) -> list[SpecialistProposal]:
        """Propose a specialist for each under-served, well-populated task niche.

        ``tasks`` are ``(task_id, embedding)`` for the failing/expensive stream;
        ``archetype_embeddings`` are the existing specialists' centroids. A cluster is
        a niche worth a new specialist when it has ≥ ``min_cluster`` tasks and its
        coverage (max similarity to any existing archetype) is below ``coverage_floor``.
        """
        proposals: list[SpecialistProposal] = []
        for idx, cluster in enumerate(self._cluster(tasks)):
            if len(cluster) < self.min_cluster:
                continue
            centroid = _centroid([v for _, v in cluster])
            coverage = max(
                (_cosine(centroid, a) for a in archetype_embeddings), default=0.0
            )
            if coverage < self.coverage_floor:
                proposals.append(
                    SpecialistProposal(
                        niche_id=idx,
                        size=len(cluster),
                        centroid=centroid,
                        coverage=round(coverage, 4),
                        examples=[k for k, _ in cluster[:5]],
                    )
                )
        return proposals
