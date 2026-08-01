from __future__ import annotations

"""The explicit promotion policy — the ONLY path a neural proposal becomes a fact (CONCEPT:AU-KG.mining.governed-neural-layer, KG-6.6).

Hard rules enforced here:

* A :class:`~.models.EntityResolutionProposal` never becomes a graph edge by
  itself — :func:`review_entity_resolution_proposal` is the one function in
  this package that writes a merge edge, and it does so ONLY on
  ``decision="accepted"``.
* Every review — accepted OR rejected — is committed as a durable
  :class:`~.models.ReviewOutcome`. A rejection is not silently discarded: it is
  the durable "do not propose this pair again at this calibration" signal, and
  is exactly the training-eligible data :mod:`.evaluation` measures against.
* Raw model output (the proposal's ``score``/``raw_similarity``) never trains
  anything by itself — only the committed :class:`~.models.ReviewOutcome` does,
  and only once a real training pipeline is built on top of accumulated
  outcomes (explicitly out of scope here — see the lane report).
"""

import hashlib
import logging
from datetime import UTC, datetime
from typing import Any, Literal

from agent_utilities.models.knowledge_graph import RegistryEdgeType

from .models import EntityResolutionProposal, ReviewOutcome

logger = logging.getLogger(__name__)

__all__ = ["review_entity_resolution_proposal"]


def _outcome_id(proposal_id: str, reviewer: str, reviewed_at: datetime) -> str:
    key = f"{proposal_id}|{reviewer}|{reviewed_at.isoformat()}"
    return f"erout:{hashlib.sha256(key.encode('utf-8')).hexdigest()[:32]}"


def review_entity_resolution_proposal(
    engine: Any,
    *,
    proposal: EntityResolutionProposal,
    decision: Literal["accepted", "rejected"],
    reviewer: str,
    rationale: str = "",
) -> ReviewOutcome:
    """Record a governed review decision; promote to a graph edge iff accepted.

    Args:
        proposal: The (immutable, ``decision_status="proposed"``) candidate
            being reviewed — never mutated; the outcome is a new record.
        decision: ``"accepted"`` writes a ``SIMILAR_TO`` edge tagged
            ``governed=True`` (distinct from ``dedup.py``'s auto-computed
            ``SIMILAR_TO`` edges, which carry no such tag) plus the
            ``review_outcome_id`` provenance link. ``"rejected"`` writes NO
            edge — only the outcome record.
        reviewer: Non-empty identity of the human/policy actor deciding —
            required so an outcome can never be anonymous (auditability).

    Returns:
        The committed :class:`ReviewOutcome` — the only durable signal this
        review produces beyond the (conditional) promoted edge.
    """
    if not reviewer.strip():
        raise ValueError(
            "review_entity_resolution_proposal requires a non-empty reviewer"
        )
    reviewed_at = datetime.now(UTC)
    outcome = ReviewOutcome(
        outcome_id=_outcome_id(proposal.proposal_id, reviewer, reviewed_at),
        proposal_id=proposal.proposal_id,
        tenant=proposal.tenant,
        decision=decision,
        reviewer=reviewer,
        rationale=rationale,
        reviewed_at=reviewed_at,
    )
    _commit_outcome(engine, outcome)
    if decision == "accepted":
        _promote_merge(engine, proposal, outcome)
    return outcome


def _commit_outcome(engine: Any, outcome: ReviewOutcome) -> None:
    from ...protocols.source_connectors.base import ExternalAccess
    from ..ingestion.change_envelope import ChangeEnvelope
    from ..ingestion.envelope_ingest import ingest_envelope

    record = {
        "id": outcome.outcome_id,
        "type": "EntityResolutionReviewOutcome",
        **outcome.model_dump(mode="json"),
        "updatedAt": outcome.reviewed_at.isoformat(),
    }
    env = ChangeEnvelope.from_connector_record(
        record,
        connector="neural-layer",
        tenant=outcome.tenant,
        id_field="id",
        version_field="updatedAt",
        source_acl=ExternalAccess.quarantined(),
    )
    applied = ingest_envelope(engine, env)
    if applied.get("status") not in {"success", "skipped"}:
        raise RuntimeError("ReviewOutcome ChangeEnvelope failed")


def _promote_merge(
    engine: Any, proposal: EntityResolutionProposal, outcome: ReviewOutcome
) -> None:
    """Write the ONE edge an accepted proposal is allowed to produce."""
    link_nodes = getattr(engine, "link_nodes", None)
    if not callable(link_nodes):
        raise RuntimeError(
            "engine does not support link_nodes — cannot promote an accepted "
            "entity-resolution proposal"
        )
    link_nodes(
        proposal.mention.node_id,
        proposal.candidate.node_id,
        RegistryEdgeType.SIMILAR_TO,
        properties={
            "_rel": "SIMILAR_TO",
            "score": round(proposal.score, 6),
            "governed": True,
            "review_outcome_id": outcome.outcome_id,
            "blocking_tier": proposal.blocking_tier,
        },
    )
