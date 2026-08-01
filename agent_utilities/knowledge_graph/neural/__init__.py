"""Governed neural graph layer (CONCEPT:AU-KG.mining.governed-neural-layer, KG-6.6).

An NGDB-shaped (Ren et al., arXiv 2303.14617) latent-store + calibrated-proposal
surface for incomplete-graph reasoning: versioned/tenant-scoped embeddings over
the engine's EXISTING native HNSW index, exact/lexical/type blocking before any
ANN escalation, calibrated entity-resolution proposals, and an explicit
promotion policy — no neural prediction becomes a fact without governed review.

Deliberately modest (see each module's docstring for what is and is not built):
no relation-aware (ComplEx/RotatE) link scorer (no reviewed label set exists to
fit/validate one against yet — see :mod:`.models`'s ``RelationLinkPrediction``),
no GNN message passing (a later gate per the task brief), no new vector-index
library (the engine's native HNSW is reused as-is).
"""

from .candidate_generation import CALIBRATION_REF, generate_entity_resolution_proposals
from .embedding_store import build_tenant_embedding, content_hash
from .evaluation import (
    CostSample,
    TierEvaluation,
    evaluate_entity_resolution,
    time_block,
)
from .governance import review_entity_resolution_proposal
from .models import (
    EntityResolutionProposal,
    GraphNodeRef,
    RelationLinkPrediction,
    ReviewOutcome,
    TenantScopedEmbedding,
)
from .probabilistic_completion import ExactWithOptionalProbabilistic, compose_result

__all__ = [
    "CALIBRATION_REF",
    "CostSample",
    "EntityResolutionProposal",
    "ExactWithOptionalProbabilistic",
    "GraphNodeRef",
    "RelationLinkPrediction",
    "ReviewOutcome",
    "TenantScopedEmbedding",
    "TierEvaluation",
    "build_tenant_embedding",
    "compose_result",
    "content_hash",
    "evaluate_entity_resolution",
    "generate_entity_resolution_proposals",
    "review_entity_resolution_proposal",
    "time_block",
]
