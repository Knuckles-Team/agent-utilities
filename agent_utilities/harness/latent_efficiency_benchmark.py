#!/usr/bin/python
from __future__ import annotations

"""Measured-lift benchmark for the latent-native memory mechanisms.

CONCEPT:AU-AHE.harness.empirical-evidence-that-latent — empirical evidence that the latent-native enhancements move
their metric the right way, mirroring the assimilation-parity suite (AHE-3.47).

Provenance: arXiv:2606.09828 ("Latent Spatial Memory for Video World Models",
Mirage) shows that keeping a *persistent cache in the model's own latent space* —
instead of round-tripping through a reconstructed surface representation — both
removes information loss and keeps a generated trajectory coherent. We adopted that
principle in the world-model rollout latent memory (KG-2.73b); this module measures,
under a fixed seed, that it beats the round-tripped baseline it replaces (the
ontology-prior designation benchmark moved to EG with EG-DECISION-ENGINE-R172.10):

* :func:`bench_latent_rollout_memory` — a learned-backend rollout that **carries the
  predicted latent forward** (KG-2.73b) vs the memoryless rollout that discards it
  and re-derives from the bare next-state string each step. Metric: total
  step-to-step latent drift (lower is better) — the trajectory-coherence analogue of
  Mirage's frame consistency.


The task is deterministic, CPU-only, dependency-light (native numeric kernel + the deterministic
code path of the world model), so :func:`run_all` is bit-for-bit reproducible. The
``BenchmarkResult``/``_make_result``/``to_markdown`` shapes are reused from the
sibling assimilation-parity suite so the gateway/MCP reporting block is identical.
"""

from agent_utilities.harness.assimilation_benchmark import (
    BenchmarkResult,
    _make_result,
    to_markdown,
)
from agent_utilities.knowledge_graph.core.world_model import WorldModel

__all__ = [
    "BenchmarkResult",
    "bench_latent_rollout_memory",
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
# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------
def run_all(*, seed: int = 0) -> list[BenchmarkResult]:
    """Run every latent-native benchmark under one seed, in order."""
    return [
        bench_latent_rollout_memory(seed=seed),
    ]
