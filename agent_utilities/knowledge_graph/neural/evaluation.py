from __future__ import annotations

"""Per-tier evaluation: precision, recall, calibration error, drift, cost, latency (CONCEPT:AU-KG.mining.governed-neural-layer, KG-6.6).

Computed ONLY from real, governed :class:`~.models.ReviewOutcome` history — never
fabricated or estimated from the proposals' own scores alone (that would be
circular: grading a calibration against the same numbers it produced). On a
fresh deployment with zero reviews yet, every metric here legitimately reads
as "no data" (``n_reviewed=0``), not a fabricated number — this module is
honest about cold-start, per KG-6.6's "if none exists, say so" instruction.
"""

import contextlib
import time
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from .models import EntityResolutionProposal, ReviewOutcome

__all__ = ["TierEvaluation", "evaluate_entity_resolution", "time_block", "CostSample"]


@dataclass
class TierEvaluation:
    """Evaluation for one ``blocking_tier`` (or any other grouping key)."""

    key: str
    n_reviewed: int = 0
    n_pending: int = 0
    n_accepted: int = 0
    n_rejected: int = 0
    precision: float | None = None
    recall: float | None = None
    calibration_error: float | None = None


def evaluate_entity_resolution(
    proposals: list[EntityResolutionProposal],
    outcomes: list[ReviewOutcome],
    *,
    group_by: str = "blocking_tier",
    known_positive_pairs: set[tuple[str, str]] | None = None,
) -> dict[str, TierEvaluation]:
    """Score every proposal group against its accumulated review outcomes.

    Args:
        group_by: ``"blocking_tier"`` (default) or ``"tenant"`` — the
            per-source/type split the caller wants (KG-6.6 asks for
            "per-source/type" evaluation; the proposal's tier IS its source —
            exact-name-match vs. LSH vs. engine ANN each has a different
            precision profile worth tracking separately).
        known_positive_pairs: Optional ground-truth ``{(mention_id, candidate_id)}``
            set. Without it, recall is left ``None`` (honestly not computable —
            recall needs to know about true positives this pipeline never even
            proposed, which nothing here can infer on its own).

    Returns:
        ``{group_key: TierEvaluation}``. Calibration error is the mean absolute
        gap between each reviewed proposal's ``score`` and the binary outcome
        (1.0 accepted / 0.0 rejected) — a direct, ungamed check of whether
        :data:`.candidate_generation.CALIBRATION_REF`'s default mapping is
        honest, computed ONLY over what has actually been reviewed.
    """
    outcome_by_proposal = {o.proposal_id: o for o in outcomes}
    groups: dict[str, list[tuple[EntityResolutionProposal, ReviewOutcome | None]]] = (
        defaultdict(list)
    )
    for proposal in proposals:
        key = getattr(proposal, group_by, "unknown")
        groups[str(key)].append(
            (proposal, outcome_by_proposal.get(proposal.proposal_id))
        )

    results: dict[str, TierEvaluation] = {}
    for key, pairs in groups.items():
        ev = TierEvaluation(key=key)
        errors: list[float] = []
        true_positive_pairs: set[tuple[str, str]] = set()
        for proposal, outcome in pairs:
            if outcome is None:
                ev.n_pending += 1
                continue
            ev.n_reviewed += 1
            label = 1.0 if outcome.decision == "accepted" else 0.0
            errors.append(abs(proposal.score - label))
            if outcome.decision == "accepted":
                ev.n_accepted += 1
                true_positive_pairs.add(
                    (proposal.mention.node_id, proposal.candidate.node_id)
                )
            else:
                ev.n_rejected += 1
        if ev.n_reviewed:
            ev.precision = ev.n_accepted / ev.n_reviewed
            ev.calibration_error = sum(errors) / len(errors)
        if known_positive_pairs:
            found = true_positive_pairs & known_positive_pairs
            ev.recall = (
                (len(found) / len(known_positive_pairs))
                if known_positive_pairs
                else None
            )
        results[key] = ev
    return results


def accept_rate_drift(
    outcomes_earlier: list[ReviewOutcome], outcomes_later: list[ReviewOutcome]
) -> float | None:
    """Change in accept-rate between two review windows (a simple drift signal).

    ``None`` when either window is empty (honestly not computable, not zero).
    """

    def _rate(batch: list[ReviewOutcome]) -> float | None:
        if not batch:
            return None
        return sum(1 for o in batch if o.decision == "accepted") / len(batch)

    earlier, later = _rate(outcomes_earlier), _rate(outcomes_later)
    if earlier is None or later is None:
        return None
    return later - earlier


@dataclass
class CostSample:
    """One measured cost/latency sample for a neural-layer operation."""

    operation: str
    elapsed_ms: float
    call_count: int = 1
    extra: dict[str, Any] = field(default_factory=dict)


@contextlib.contextmanager
def time_block(
    operation: str, *, call_count: int = 1, **extra: Any
) -> Iterator[CostSample]:
    """Measure wall-clock latency around a neural-layer call (the cost proxy).

    No billing/token-cost meter exists for the embedder in this codebase (a
    real ``$``-cost model is out of scope here), so latency + call-count is the
    honest cost proxy this evaluation surface reports — wrap
    :func:`.embedding_store.build_tenant_embedding` or
    :func:`.candidate_generation.generate_entity_resolution_proposals` calls
    with this to accumulate :class:`CostSample`\\ s for a run.
    """
    sample = CostSample(
        operation=operation, elapsed_ms=0.0, call_count=call_count, extra=extra
    )
    start = time.monotonic()
    try:
        yield sample
    finally:
        sample.elapsed_ms = (time.monotonic() - start) * 1000.0
