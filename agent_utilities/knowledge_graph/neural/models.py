from __future__ import annotations

"""Governed neural-graph boundary models (CONCEPT:AU-KG.mining.governed-neural-layer, KG-6.6).

NGDB-shaped (Ren et al., arXiv 2303.14617) latent-store + calibrated-proposal
contracts for ARBITRARY knowledge-graph nodes — the general-purpose sibling of
:mod:`agent_utilities.knowledge_graph.ingestion.semantic_event_model`'s
``NeuralRepresentation``/``NeuralRelationPrediction``/``EntityResolutionProposal``,
which are intentionally scoped to OCEL ``event``/``object``/``object_state``
entities (that module is sibling-lane-owned; see its ``SemanticEntityKind``).
These models follow the SAME governance shape — versioned/immutable, "proposed"
until reviewed, referential evidence required — for the much larger space of
plain KG nodes (``:Person``, ``:Paper``, ``:Concept``, …) that source connectors
and enrichment actually produce.

**Hard rules enforced by this shape, not by convention:**

* A :class:`TenantScopedEmbedding` is a POINTER (``artifact_ref`` + engine-native
  HNSW index entry), never the fact itself — the symbolic graph stays the exact
  source of truth.
* An :class:`EntityResolutionProposal` is immutable and pinned at
  ``decision_status="proposed"`` by its own type; nothing in this package can
  construct one in any other state. The ONLY way a proposal's outcome becomes
  durable is a separate, explicitly authored :class:`ReviewOutcome` — see
  :mod:`.governance`. This is the promotion-policy hard rule (KG-6.6) made
  structural: no code path can accidentally skip review.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "GraphNodeRef",
    "TenantScopedEmbedding",
    "EntityResolutionProposal",
    "ReviewOutcome",
    "RelationLinkPrediction",
]


class _Governed(BaseModel):
    """Strict immutable base — mirrors ``SemanticBoundaryModel``'s discipline."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class GraphNodeRef(_Governed):
    """A plain KG node reference (any OWL class), by id."""

    node_id: str = Field(min_length=1)
    node_type: str = ""


class TenantScopedEmbedding(_Governed):
    """Versioned latent representation for one tenant-scoped KG node.

    Keyed by ``(tenant, target, encoder_id, encoder_version, content_hash)`` so a
    re-embed of unchanged content under the same encoder is a cache hit, not a
    new artifact (mirrors the pipeline-phase embedding cache's ``content_hash``
    skip check). ``artifact_ref`` is the engine's native HNSW pointer (the same
    ``node_id`` used with ``engine.add_embedding``/``semantic_search`` — see
    ``docs/architecture/vector_index_lifecycle.md``); the raw vector itself is
    NEVER duplicated into the KG record.
    """

    representation_id: str = Field(min_length=1)
    tenant: str = ""
    target: GraphNodeRef
    encoder_id: str = Field(min_length=1)
    encoder_version: str = Field(min_length=1)
    dimension: int = Field(gt=0)
    artifact_ref: str = Field(min_length=1)
    graph_epoch: int = Field(ge=0)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_ref: str = ""


class EntityResolutionProposal(_Governed):
    """Calibrated candidate merge, awaiting governed review — never a fact.

    ``blocking_tier`` records which stage of the exact/lexical/type-blocking
    ladder produced this candidate (``"exact"``/``"lsh"`` — the
    entropy+MinHash/LSH ladder, computed with NO embeddings — or ``"ann"``, the
    engine-native HNSW escalation for the ladder's residual). ANN is never the
    first tier; see :mod:`.candidate_generation`.
    """

    proposal_id: str = Field(min_length=1)
    tenant: str = ""
    mention: GraphNodeRef
    candidate: GraphNodeRef
    score: float = Field(ge=0.0, le=1.0)
    raw_similarity: float = Field(ge=0.0, le=1.0)
    blocking_tier: Literal["exact", "lsh", "ann"]
    calibration_ref: str
    evidence_refs: tuple[str, ...] = Field(min_length=1)
    decision_status: Literal["proposed"] = "proposed"


class ReviewOutcome(_Governed):
    """A governed accept/reject decision on an :class:`EntityResolutionProposal`.

    The ONLY record type this package ever writes as a training-eligible
    signal (KG-6.6 hard rule: raw model output never trains its successor —
    only this governed, human/policy-authored outcome does). Referencing the
    proposal by id keeps the immutable proposal untouched; the outcome is the
    durable "what happened next" event.
    """

    outcome_id: str = Field(min_length=1)
    proposal_id: str = Field(min_length=1)
    tenant: str = ""
    decision: Literal["accepted", "rejected"]
    reviewer: str = Field(min_length=1)
    rationale: str = ""
    reviewed_at: datetime


class RelationLinkPrediction(_Governed):
    """Placeholder contract for a FUTURE relation-aware link scorer.

    Deliberately unconstructed today (KG-6.6): no reviewed accept/reject label
    set exists anywhere in this codebase to fit or validate a ComplEx/RotatE-style
    relation-conditioned scorer against (confirmed by exhaustive repo survey —
    see the lane report). This model documents the target shape so a future
    lane can implement scoring once :class:`ReviewOutcome` history accumulates
    — it is intentionally not referenced by any producer in this package.
    """

    prediction_id: str = Field(min_length=1)
    tenant: str = ""
    subject: GraphNodeRef
    predicate: str = Field(min_length=1)
    object: GraphNodeRef
    score: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    model_ref: str = Field(min_length=1)
    trained_on_outcomes_through: str = Field(min_length=1)
    decision_status: Literal["proposed"] = "proposed"
