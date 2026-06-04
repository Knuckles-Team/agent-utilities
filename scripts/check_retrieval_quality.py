#!/usr/bin/env python3
"""Retrieval-quality gate (Plan 10 — retrieval-regression vector).

Builds a tiny, fully synthetic :class:`CapabilityIndex` (no network, no real
embeddings) from a frozen fixture, runs a handful of labelled queries, and FAILS
if Recall@k or MRR drops below configured floors. This catches silent
regressions in the designation/ranking path (e.g. a broken cosine, a backend
swap that reorders, or a capability-filter bug) before they ship.

The fixture plants, for each topical cluster, a set of *relevant* entities whose
embeddings sit near a cluster centroid and a set of *irrelevant* entities near
orthogonal centroids. Each labelled query is the cluster centroid (perturbed);
the gold set is that cluster's relevant ids. A correct ranker must surface the
relevant ids in the top-k.

Vectors are deterministic (seeded numpy) so the gate is reproducible.

Usage::

    python3 scripts/check_retrieval_quality.py [--degrade]

``--degrade`` builds a deliberately broken corpus (relevant and irrelevant
vectors collapsed together) and is used by the meta-test to prove the gate
actually trips. Without it the gate runs the real fixture and must pass on the
current code.

Exit 0 = quality at/above floor. 1 = regression below floor. 2 = build error.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make agent_utilities importable when run from the scripts/ dir or repo root.
_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from agent_utilities.knowledge_graph.retrieval.capability_index import (  # noqa: E402
    CapabilityIndex,
)

DIM = 32
K = 5
# Quality floors. The synthetic fixture is separable enough that a correct
# ranker scores Recall@5 == 1.0 and MRR == 1.0; the floors leave headroom for
# the perturbation noise while still tripping on a genuine regression.
RECALL_FLOOR = 0.9
MRR_FLOOR = 0.9

# Five orthogonal topical clusters. Each query targets one cluster; the relevant
# entities for that query are the ones planted near its centroid.
N_CLUSTERS = 5
RELEVANT_PER_CLUSTER = 3
IRRELEVANT_PER_CLUSTER = 4


def _onehot(dim_a: int, dim_b: int) -> np.ndarray:
    """A near-one-hot vector active on two dims (clusters stay orthogonal)."""
    vec = np.full(DIM, 0.01, dtype=np.float32)
    vec[dim_a % DIM] = 1.0
    vec[dim_b % DIM] = 1.0
    return vec


def _relevant_centroid(cluster: int) -> np.ndarray:
    """Centroid for a query/relevant cluster — lives in the low dims."""
    return _onehot(2 * cluster, 2 * cluster + 1)


def _irrelevant_centroid(cluster: int) -> np.ndarray:
    """Distractor centroid — lives in the *high* dims, orthogonal to every
    query/relevant centroid so a correct ranker never confuses the two."""
    base = 2 * N_CLUSTERS  # start past all relevant dims
    return _onehot(base + 2 * cluster, base + 2 * cluster + 1)


def _jitter(base: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
    return (base + rng.normal(0.0, scale, size=DIM).astype(np.float32)).astype(np.float32)


def build_index(*, degrade: bool = False) -> tuple[CapabilityIndex, list[dict]]:
    """Build the frozen fixture index and the list of labelled queries.

    Relevant and distractor (irrelevant) entities live in disjoint dimension
    blocks so the clusters are mutually near-orthogonal. When ``degrade`` is
    True the distractors are collapsed *onto* the relevant centroid, destroying
    separability so a correct ranker can no longer recover the gold set —
    proving the gate has teeth.
    """
    rng = np.random.default_rng(1234)
    idx = CapabilityIndex(dim=DIM, prefer_backend="numpy")
    queries: list[dict] = []

    for c in range(N_CLUSTERS):
        rel_centroid = _relevant_centroid(c)
        relevant_ids: list[str] = []

        for r in range(RELEVANT_PER_CLUSTER):
            rid = f"c{c}_rel{r}"
            idx.add(rid, _jitter(rel_centroid, 0.02, rng), capabilities=[f"cap{c}"])
            relevant_ids.append(rid)

        for j in range(IRRELEVANT_PER_CLUSTER):
            iid = f"c{c}_irr{j}"
            if degrade:
                far = rel_centroid  # collapse onto the relevant cluster
            else:
                far = _irrelevant_centroid(c)
            idx.add(iid, _jitter(far, 0.02, rng), capabilities=[f"junk{c}"])

        # Labelled query = the cluster centroid, lightly perturbed.
        queries.append(
            {
                "embedding": _jitter(rel_centroid, 0.01, rng),
                "gold": set(relevant_ids),
            }
        )

    return idx, queries


def evaluate(idx: CapabilityIndex, queries: list[dict], k: int = K) -> dict:
    """Compute mean Recall@k and mean MRR over the labelled queries."""
    recalls: list[float] = []
    rrs: list[float] = []
    for q in queries:
        gold: set[str] = q["gold"]
        results = idx.designate(q["embedding"], k=k)
        ranked_ids = [d.id for d in results]

        hits = sum(1 for rid in ranked_ids if rid in gold)
        recalls.append(hits / len(gold) if gold else 0.0)

        rr = 0.0
        for rank, rid in enumerate(ranked_ids, 1):
            if rid in gold:
                rr = 1.0 / rank
                break
        rrs.append(rr)

    return {
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "mrr": float(np.mean(rrs)) if rrs else 0.0,
        "k": k,
        "n_queries": len(queries),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--degrade",
        action="store_true",
        help="Build a deliberately degraded corpus (meta-test use).",
    )
    args = parser.parse_args()

    try:
        idx, queries = build_index(degrade=args.degrade)
        metrics = evaluate(idx, queries)
    except Exception as exc:  # noqa: BLE001 - surface any build/eval error as exit 2
        print(f"Retrieval-quality gate ERROR: {exc}", file=sys.stderr)
        return 2

    recall = metrics["recall_at_k"]
    mrr = metrics["mrr"]
    print(
        f"Retrieval quality over {metrics['n_queries']} queries "
        f"(backend={idx.backend}, k={metrics['k']}): "
        f"Recall@{metrics['k']}={recall:.3f} (floor {RECALL_FLOOR}), "
        f"MRR={mrr:.3f} (floor {MRR_FLOOR})"
    )

    if recall < RECALL_FLOOR or mrr < MRR_FLOOR:
        print(
            "Retrieval-quality gate FAILED: metrics below floor — retrieval "
            "regression detected.",
            file=sys.stderr,
        )
        return 1
    print("OK: retrieval quality at or above floor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
