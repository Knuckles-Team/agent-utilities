from __future__ import annotations

"""Entity-resolution candidate generation — exact/lexical/type blocking BEFORE ANN (CONCEPT:AU-KG.mining.governed-neural-layer, KG-6.6).

**The hard ordering rule this module exists to enforce:** every candidate pair
first passes through the embedding-free entropy + MinHash/LSH blocking ladder
(:func:`agent_utilities.knowledge_graph.assimilation.entity_resolution.resolve_entities`
— exact normalized-name match, then a Shannon-entropy gate, then MinHash/LSH
Jaccard≥0.9). ANN similarity search (the engine's native HNSW,
``semantic_search``) is invoked ONLY for the ladder's ``residual_ids`` — never
first, never for the whole candidate set. This mirrors the existing
``dedup.py`` escalation pattern, but returns governed, calibrated **proposals**
instead of writing edges directly — dedup.py's ``SIMILAR_TO``/``VARIANT_OF``
writes are an established, separately-owned auto-link path; this module's
output requires :mod:`.governance` review before anything is written as fact.
"""

import hashlib
import logging
from typing import Any

from agent_utilities.knowledge_graph.assimilation.entity_resolution import (
    resolve_entities,
)

from .models import EntityResolutionProposal, GraphNodeRef

logger = logging.getLogger(__name__)

__all__ = ["CALIBRATION_REF", "generate_entity_resolution_proposals"]

#: Versioned identifier for the DEFAULT calibration below (KG-6.6: honestly a
#: placeholder — see the module docstring on why no fitted calibration exists
#: yet). Bump this string whenever the mapping changes so old proposals remain
#: attributable to the calibration that scored them, and a future refit (once
#: ``ReviewOutcome`` history exists) can be distinguished from this default.
CALIBRATION_REF = "default-v1-uncalibrated"

#: Minimum engine ANN cosine similarity to even propose a residual candidate —
#: a floor, not a calibrated probability (candidates below this are not worth a
#: reviewer's time; this is a recall/precision knob, not the returned score).
_DEFAULT_ANN_FLOOR = 0.72


def _calibrate(raw_similarity: float, tier: str) -> float:
    """Map a raw similarity/Jaccard score to a governed ``[0, 1]`` confidence.

    **Explicitly NOT a fitted calibration** — no reviewed accept/reject label
    set exists anywhere in this codebase (confirmed by exhaustive repo survey)
    to fit a Platt/isotonic mapping against, and KG-6.6 forbids fabricating one.
    This is a conservative, DOCUMENTED default:

    * ``exact`` (normalized-name identity) — the highest-confidence tier by
      construction; pinned near-certain but never exactly 1.0 (an exact-name
      collision between distinct real-world entities is rare, not impossible).
    * ``lsh`` — the ladder's own Jaccard≥0.9 threshold already IS a reasonable
      confidence proxy for character-shingle overlap; passed through unscaled.
    * ``ann`` — raw cosine similarity is well known to overstate calibrated
      confidence (embedding geometry ≠ probability); discounted by a fixed
      margin so an uncalibrated 0.9 cosine reads as a cautious ~0.75, not a
      false "as confident as an exact match."

    Once real :class:`~.models.ReviewOutcome` history accumulates, refit this
    against it and bump :data:`CALIBRATION_REF` — see :mod:`.evaluation` for
    the calibration-error measurement this default should be checked against.
    """
    if tier == "exact":
        return 0.98
    if tier == "lsh":
        return max(0.0, min(1.0, raw_similarity))
    # "ann" — discount raw cosine by a fixed margin, floor at the ANN gate.
    return max(0.0, min(1.0, raw_similarity * 0.85))


def _proposal_id(tenant: str, mention_id: str, candidate_id: str, tier: str) -> str:
    key = f"{tenant}|{mention_id}|{candidate_id}|{tier}"
    return f"erprop:{hashlib.sha256(key.encode('utf-8')).hexdigest()[:32]}"


def generate_entity_resolution_proposals(
    engine: Any,
    *,
    tenant: str,
    items: list[tuple[str, str]],
    node_type: str = "",
    embeddings: dict[str, list[float]] | None = None,
    ann_top_k: int = 5,
    ann_floor: float = _DEFAULT_ANN_FLOOR,
) -> list[EntityResolutionProposal]:
    """Blocking-then-ANN candidate generation, returning governed proposals.

    Args:
        items: ``(node_id, display_name)`` pairs to resolve against each other.
        embeddings: Precomputed ``node_id -> vector`` for items in the ladder's
            residual (typically from :func:`.embedding_store.build_tenant_embedding`).
            Residual ids with NO embedding available are simply skipped for the
            ANN tier (never silently escalate without a real vector) — they
            surface in the ladder's own accounting, not as a proposal.
        ann_top_k: Engine ``semantic_search`` candidates fetched per residual id.
        ann_floor: Minimum raw cosine to propose (recall/precision knob, not
            the returned calibrated score — see :func:`_calibrate`).

    Returns:
        Every proposal is ``decision_status="proposed"`` by construction
        (the model type allows nothing else) — none of these are facts yet.
    """
    result = resolve_entities(items)
    names = dict(items)
    proposals: list[EntityResolutionProposal] = []

    for survivor, dup, score, tier in result.merge_pairs:
        proposals.append(
            EntityResolutionProposal(
                proposal_id=_proposal_id(tenant, dup, survivor, tier),
                tenant=tenant,
                mention=GraphNodeRef(node_id=dup, node_type=node_type),
                candidate=GraphNodeRef(node_id=survivor, node_type=node_type),
                score=_calibrate(score, tier),
                raw_similarity=max(0.0, min(1.0, score)),
                blocking_tier=tier,  # "exact" | "lsh"
                calibration_ref=CALIBRATION_REF,
                evidence_refs=(f"blocking:{tier}:{names.get(dup, dup)}",),
            )
        )

    semantic_search = getattr(engine, "semantic_search", None)
    if result.residual_ids and callable(semantic_search) and embeddings:
        for rid in result.residual_ids:
            vector = embeddings.get(rid)
            if not vector:
                continue
            try:
                hits = semantic_search(vector, ann_top_k) or []
            except Exception:  # noqa: BLE001 — an ANN outage never breaks the ladder's results
                logger.debug("ANN escalation failed for %s", rid, exc_info=True)
                continue
            for candidate_id, similarity in hits:
                if candidate_id == rid or similarity < ann_floor:
                    continue
                proposals.append(
                    EntityResolutionProposal(
                        proposal_id=_proposal_id(tenant, rid, candidate_id, "ann"),
                        tenant=tenant,
                        mention=GraphNodeRef(node_id=rid, node_type=node_type),
                        candidate=GraphNodeRef(
                            node_id=candidate_id, node_type=node_type
                        ),
                        score=_calibrate(similarity, "ann"),
                        raw_similarity=max(0.0, min(1.0, similarity)),
                        blocking_tier="ann",
                        calibration_ref=CALIBRATION_REF,
                        evidence_refs=(f"ann:semantic_search:{rid}",),
                    )
                )
    return proposals
