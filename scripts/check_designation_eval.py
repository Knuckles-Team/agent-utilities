#!/usr/bin/env python3
"""Designation-eval gate (Plan 04 Step 6 / Plan 10 — capability-filter value).

Proves that *capability filtering* genuinely improves designation quality over
pure *embedding-only* ranking on a frozen, fully synthetic corpus (seeded numpy
vectors — no network, no embedding model).

The corpus is built so that capability filtering must help: for each topical
cluster a set of *correct* tools sit near the cluster centroid AND carry the
cluster's required capability, while a set of *distractors* sit in the SAME
embedding neighbourhood (so embedding-only ranking confuses them with the
correct tools) but DO NOT carry the required capability. Capability-filtered
designation removes those distractors before ranking; embedding-only cannot.

Two configs are evaluated against the identical corpus and query set:

* **embedding-only**  — ``designate(emb, required_caps=None)``
* **capability-filtered** — ``designate(emb, required_caps=<correct caps>)``

We report Recall@k and MRR for both and FAIL (exit 1) unless the
capability-filtered config beats embedding-only by at least a configured margin
on BOTH metrics. On the current code the filtered config strictly dominates, so
the gate PASSES.

Usage::

    python3 scripts/check_designation_eval.py [--degrade]

``--degrade`` strips the required capability from the correct tools (and from
the queries) so capability filtering can no longer separate signal from
distractors — the two configs collapse to the same score and the gate trips.
Used by the meta-test to prove the gate has teeth.

Exit 0 = filtered beats embedding-only by the margin. 1 = it does not.
2 = build/eval error.
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

# Topical structure. Each cluster has CORRECT tools (near the centroid, carrying
# the required capability) and DISTRACTORS (near the SAME centroid, lacking it).
N_CLUSTERS = 5
CORRECT_PER_CLUSTER = 3
DISTRACTOR_PER_CLUSTER = 5

# Capability-filtered must beat embedding-only by at least this absolute margin
# on BOTH Recall@k and MRR for the gate to pass.
MARGIN = 0.10

SEED = 20240601


def _centroid(cluster: int) -> np.ndarray:
    """Cluster centroid — a near-one-hot vector active on two cluster dims.

    Distractors share this centroid, so embedding similarity alone cannot tell
    correct tools from distractors; only the capability tag can.
    """
    vec = np.full(DIM, 0.01, dtype=np.float32)
    vec[(2 * cluster) % DIM] = 1.0
    vec[(2 * cluster + 1) % DIM] = 1.0
    return vec


def _jitter(base: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, scale, size=DIM).astype(np.float32)
    return (base + noise).astype(np.float32)


def build_corpus(*, degrade: bool = False) -> tuple[CapabilityIndex, list[dict]]:
    """Build the frozen corpus and the labelled query set.

    Returns the populated :class:`CapabilityIndex` and a list of queries, each
    ``{"embedding", "gold": set[str], "required_caps": list[str]}``.

    When ``degrade`` is True the required capability is stripped from both the
    correct tools and the queries, so capability filtering becomes a no-op and
    can no longer separate correct tools from the co-located distractors.
    """
    rng = np.random.default_rng(SEED)
    idx = CapabilityIndex(dim=DIM, prefer_backend="numpy")
    queries: list[dict] = []

    for c in range(N_CLUSTERS):
        centroid = _centroid(c)
        req_cap = f"cap{c}"
        gold: list[str] = []

        # Correct tools: near the centroid AND providing the required capability
        # (unless degraded, in which case the discriminating cap is removed).
        for r in range(CORRECT_PER_CLUSTER):
            cid = f"c{c}_correct{r}"
            caps = [f"topic{c}"] if degrade else [f"topic{c}", req_cap]
            idx.add(cid, _jitter(centroid, 0.02, rng), capabilities=caps)
            gold.append(cid)

        # Distractors: SAME embedding neighbourhood, but never the required cap.
        # Slightly tighter jitter so embedding-only ranking actually prefers
        # them over the correct tools — making the capability filter load-bearing.
        for d in range(DISTRACTOR_PER_CLUSTER):
            did = f"c{c}_distractor{d}"
            idx.add(did, _jitter(centroid, 0.005, rng), capabilities=[f"topic{c}"])

        queries.append(
            {
                "embedding": _jitter(centroid, 0.01, rng),
                "gold": set(gold),
                # In degrade mode the query asks for nothing discriminating, so
                # the filtered config sees the same candidate pool as embedding-only.
                "required_caps": [] if degrade else [req_cap],
            }
        )

    return idx, queries


def _recall_mrr(ranked_ids: list[str], gold: set[str]) -> tuple[float, float]:
    hits = sum(1 for rid in ranked_ids if rid in gold)
    recall = hits / len(gold) if gold else 0.0
    rr = 0.0
    for rank, rid in enumerate(ranked_ids, 1):
        if rid in gold:
            rr = 1.0 / rank
            break
    return recall, rr


def evaluate(
    idx: CapabilityIndex,
    queries: list[dict],
    *,
    use_capabilities: bool,
    k: int = K,
) -> dict:
    """Mean Recall@k and MRR for one config over the labelled queries."""
    recalls: list[float] = []
    rrs: list[float] = []
    for q in queries:
        req = q["required_caps"] if use_capabilities else None
        results = idx.designate(q["embedding"], required_caps=req, k=k)
        ranked_ids = [d.id for d in results]
        recall, rr = _recall_mrr(ranked_ids, q["gold"])
        recalls.append(recall)
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
        help="Strip the discriminating capability so filtering cannot help "
        "(meta-test use — proves the gate trips).",
    )
    args = parser.parse_args()

    try:
        idx, queries = build_corpus(degrade=args.degrade)
        emb_only = evaluate(idx, queries, use_capabilities=False)
        cap_filtered = evaluate(idx, queries, use_capabilities=True)
    except Exception as exc:  # noqa: BLE001 - surface build/eval errors as exit 2
        print(f"Designation-eval gate ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        f"Designation eval over {emb_only['n_queries']} queries "
        f"(backend={idx.backend}, k={K}, margin {MARGIN}):"
    )
    print(
        f"  embedding-only     : Recall@{K}={emb_only['recall_at_k']:.3f}  "
        f"MRR={emb_only['mrr']:.3f}"
    )
    print(
        f"  capability-filtered: Recall@{K}={cap_filtered['recall_at_k']:.3f}  "
        f"MRR={cap_filtered['mrr']:.3f}"
    )

    recall_gain = cap_filtered["recall_at_k"] - emb_only["recall_at_k"]
    mrr_gain = cap_filtered["mrr"] - emb_only["mrr"]
    print(f"  gain               : Recall +{recall_gain:.3f}  MRR +{mrr_gain:.3f}")

    if recall_gain < MARGIN or mrr_gain < MARGIN:
        print(
            "Designation-eval gate FAILED: capability filtering did not beat "
            f"embedding-only by the required margin ({MARGIN}) on both metrics.",
            file=sys.stderr,
        )
        return 1
    print("OK: capability filtering beats embedding-only by the required margin.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
