#!/usr/bin/python
from __future__ import annotations

"""Measured-lift benchmark for the latent-native memory mechanisms.

CONCEPT:AHE-3.48 — empirical evidence that the latent-native enhancements move
their metric the right way, mirroring the assimilation-parity suite (AHE-3.47).

Provenance: arXiv:2606.09828 ("Latent Spatial Memory for Video World Models",
Mirage) shows that keeping a *persistent cache in the model's own latent space* —
instead of round-tripping through a reconstructed surface representation — both
removes information loss and keeps a generated trajectory coherent. We adopted that
principle in two places (the world-model rollout latent memory KG-2.73b and the
ontology-prior retrieval ranking KG-2.44b); this module measures, under a fixed
seed, that each one beats the round-tripped / flat baseline it replaces:

* :func:`bench_latent_rollout_memory` — a learned-backend rollout that **carries the
  predicted latent forward** (KG-2.73b) vs the memoryless rollout that discards it
  and re-derives from the bare next-state string each step. Metric: total
  step-to-step latent drift (lower is better) — the trajectory-coherence analogue of
  Mirage's frame consistency.
* :func:`bench_ontology_prior_retrieval` — capability designation that re-projects
  the flat cosine neighbourhood through the ontology type structure (KG-2.44b) vs
  pure cosine. Metric: fraction of the top-k that is ontology-type-coherent with the
  query neighbourhood (higher is better) — the structured-prior analogue of Mirage's
  depth-guided back-projection.

Both tasks are deterministic, CPU-only, dependency-light (numpy + the deterministic
code paths of the two modules), so :func:`run_all` is bit-for-bit reproducible. The
``BenchmarkResult``/``_make_result``/``to_markdown`` shapes are reused from the
sibling assimilation-parity suite so the gateway/MCP reporting block is identical.
"""

from agent_utilities.harness.assimilation_benchmark import (
    BenchmarkResult,
    _make_result,
    to_markdown,
)
from agent_utilities.knowledge_graph.core.world_model import WorldModel
from agent_utilities.knowledge_graph.retrieval.capability_index import CapabilityIndex
from agent_utilities.numeric import xp as np

__all__ = [
    "BenchmarkResult",
    "bench_latent_rollout_memory",
    "bench_ontology_prior_retrieval",
    "run_all",
    "to_markdown",
]


# ----------------------------------------------------------------------------
# KG-2.73b — persistent latent rollout memory
# ----------------------------------------------------------------------------
def bench_latent_rollout_memory(*, seed: int = 0) -> BenchmarkResult:
    """Latent-carry rollout (memory on) vs memoryless rollout (memory off).

    A learned world model is grounded on a clean cyclic corpus, then rolled forward
    under a fixed policy. The baseline carries the latent only to *measure* drift but
    does not blend it (``memory_weight=0`` — bit-identical next-states to the legacy
    memoryless path); OURS EMA-blends the carried latent each step (KG-2.73b), so the
    trajectory's latent moves on-manifold instead of snapping to the nearest discrete
    next-state and re-deriving from its string.

    Metric: total step-to-step latent drift (lower is better). Claim: carrying the
    latent reduces drift.
    """
    del seed  # the corpus + hash embedder are deterministic
    states = ["room_alpha", "room_beta", "room_gamma", "room_delta"]
    wm = WorldModel(backend="latent")
    # Learn the cycle alpha -> beta -> gamma -> delta -> alpha under action "go".
    for _ in range(4):
        for i, s in enumerate(states):
            wm.observe(s, "go", states[(i + 1) % len(states)])

    horizon = 8

    def policy(_s: str) -> str:
        return "go"

    base = wm.rollout("room_alpha", policy, horizon, memory_weight=0.0)
    ours = wm.rollout("room_alpha", policy, horizon, memory_weight=0.25)
    baseline_drift = float(sum(t.drift for t in base))
    ours_drift = float(sum(t.drift for t in ours))

    return _make_result(
        name="Latent rollout memory KG-2.73b",
        metric="trajectory-drift",
        baseline=baseline_drift,
        ours=ours_drift,
        higher_is_better=False,
        detail={
            "horizon": horizon,
            "memory_weight": 0.25,
            "states": len(states),
            "mechanism": "carry+EMA-blend predicted latent (vs re-derive from string)",
        },
    )


# ----------------------------------------------------------------------------
# KG-2.44b — ontology-prior-guided retrieval ranking
# ----------------------------------------------------------------------------
def bench_ontology_prior_retrieval(*, seed: int = 0) -> BenchmarkResult:
    """Ontology-type-prior reranking (KG-2.44b) vs flat cosine ranking.

    A small candidate set sits very close to the query: three of the modal ontology
    type ("Document") and two of a different type ("Widget"), with one Widget tucked
    *between* the Documents on cosine. Flat cosine therefore pulls that Widget into
    the top-k, fragmenting the type-coherent neighbourhood; the ontology prior boosts
    the modal type and recovers a coherent top-k — the structured-prior re-projection
    of the flat neighbourhood.

    Metric: fraction of the top-k whose ontology type is the modal type (higher is
    better). Claim: the ontology prior >= flat cosine on neighbourhood coherence.
    """
    del seed  # vectors are fixed for reproducibility
    dim = 8
    e = np.eye(dim, dtype=np.float32)

    def vec(tilt_dim: int, tilt: float) -> np.ndarray:
        v = e[0] + tilt * e[tilt_dim]
        return v / np.linalg.norm(v)

    # (id, type, off-axis dim, tilt magnitude). Smaller tilt => higher cosine to q.
    candidates = [
        ("doc-1", "Document", 1, 0.10),
        ("wid-1", "Widget", 2, 0.12),  # interleaved between the Documents
        ("doc-2", "Document", 3, 0.14),
        ("doc-3", "Document", 4, 0.16),
        ("wid-2", "Widget", 5, 0.40),
    ]
    index = CapabilityIndex(dim=dim, prefer_backend="numpy")
    for cid, ctype, td, tilt in candidates:
        index.add(cid, vec(td, tilt), capabilities=["answer"], node_type=ctype)

    query = e[0].tolist()
    modal_type = "Document"
    k = 3

    def _coherence(designations: list) -> float:
        if not designations:
            return 0.0
        hits = sum(
            1
            for d in designations
            if index._id_to_type.get(d.id) == modal_type  # noqa: SLF001
        )
        return hits / len(designations)

    # BASELINE: pure cosine (both structured boosts disabled).
    baseline_des = index.designate(query, k=k, reward_weight=0.0, prior_weight=0.0)
    baseline_coherence = _coherence(baseline_des)

    # OURS: default call — the ontology-type prior is on by default (KG-2.44b).
    ours_des = index.designate(query, k=k)
    ours_coherence = _coherence(ours_des)

    return _make_result(
        name="Ontology-prior retrieval KG-2.44b",
        metric=f"top-{k}-type-coherence",
        baseline=baseline_coherence,
        ours=ours_coherence,
        higher_is_better=True,
        detail={
            "k": k,
            "modal_type": modal_type,
            "candidates": len(candidates),
            "baseline_ids": [d.id for d in baseline_des],
            "ours_ids": [d.id for d in ours_des],
            "mechanism": "ontology-type coherence prior (vs flat cosine)",
        },
    )


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------
def run_all(*, seed: int = 0) -> list[BenchmarkResult]:
    """Run every latent-native benchmark under one seed, in order."""
    return [
        bench_latent_rollout_memory(seed=seed),
        bench_ontology_prior_retrieval(seed=seed),
    ]
