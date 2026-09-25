"""EH-398: embedding-admission feedback from retrieval usage -- PROPOSALS only.

EH-269's admission table decides which content classes get a vector. Whether
an admitted class earns its vectors is a retrieval fact: EG aggregates, per
content class, how often the caller's visible runs returned and cited units of
it (the ``decision_class_usage`` relation of EG's decision views, k-anonymised by the policy's
``min_support``). From that aggregate this module derives REVIEWED proposals:

* ``never_retrieved`` -- an admitted class absent from every visible run over
  a window with enough traffic (a class returned fewer than ``min_support``
  times is reported with zero counts, never absent, so absence is a real
  "never");
* ``retrieved_never_cited`` -- a class returned at least ``min_support`` times
  whose units no answer ever cited.

Nothing here edits :data:`NEVER_EMBED_CLASSES` or any other table: a proposal
goes to a reviewer sink, and changing the admission table stays a reviewed
code change. Learned usage can only ever PROPOSE a narrowing.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from epistemic_graph.ingestion.embedding_admission import (
    NEVER_EMBED_CLASSES,
    ContentClass,
)

logger = logging.getLogger(__name__)

#: Fewest visible runs before absence counts as evidence.
DEFAULT_MIN_OUTCOMES = 200


@dataclass(frozen=True, slots=True)
class AdmissionProposal:
    """One proposed admission-table change, for review; never applied here."""

    content_class: str
    finding: str
    evidence: Mapping[str, int] = field(default_factory=dict)
    proposal: str = "demote_to_never_embed"
    status: str = "proposed"


#: Where proposals go for review (a work item, a report, a reviewer queue).
ProposalSink = Callable[[Sequence[AdmissionProposal]], None]


def admitted_classes() -> list[str]:
    """Every content class the admission table currently embeds."""
    return sorted(c.value for c in ContentClass if c not in NEVER_EMBED_CLASSES)


def _rows(usage: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = usage.get("rows") or []
    return {str(r["content_class"]): r for r in rows if isinstance(r, Mapping)}


def _finding(
    row: Mapping[str, Any] | None, outcomes: int, min_support: int, min_outcomes: int
) -> str | None:
    if row is None:
        return "never_retrieved" if outcomes >= min_outcomes else None
    returned, cited = int(row.get("returned") or 0), int(row.get("cited") or 0)
    if returned >= min_support and cited == 0:
        return "retrieved_never_cited"
    return None


def propose_admission_changes(
    usage: Mapping[str, Any],
    *,
    admitted: Iterable[str] | None = None,
    min_outcomes: int = DEFAULT_MIN_OUTCOMES,
) -> list[AdmissionProposal]:
    """Proposals from one EG usage aggregate (deterministic, class order)."""
    rows = _rows(usage)
    outcomes = int(usage.get("outcomes") or 0)
    min_support = int(usage.get("min_support") or 0)
    proposals = []
    for content_class in sorted(
        admitted if admitted is not None else admitted_classes()
    ):
        row = rows.get(content_class)
        finding = _finding(row, outcomes, max(1, min_support), min_outcomes)
        if finding is None:
            continue
        evidence = {
            "outcomes": outcomes,
            "returned": int((row or {}).get("returned") or 0),
            "cited": int((row or {}).get("cited") or 0),
        }
        proposals.append(AdmissionProposal(content_class, finding, evidence))
    return proposals


def log_sink(proposals: Sequence[AdmissionProposal]) -> None:
    """The default reviewer sink: the proposals, logged for a human."""
    for proposal in proposals:
        logger.info(
            "admission proposal (review required): %s %s -> %s %s",
            proposal.content_class,
            proposal.finding,
            proposal.proposal,
            dict(proposal.evidence),
        )


_USAGE_SQL = "SELECT content_class, returned, cited FROM decision_class_usage"
_OUTCOMES_SQL = (
    "SELECT count(DISTINCT record_id) AS runs FROM decision_retrieval_outcomes"
)


async def read_usage(session: Any) -> dict[str, Any]:
    """EG's per-class usage over the caller's visible runs, from the decision
    views. The relation is already k-anonymised (a class below the policy's
    ``min_support`` reports zero counts), so any non-zero count clears it."""
    rows = await session.aquery(_USAGE_SQL)
    runs = await session.aquery(_OUTCOMES_SQL)
    outcomes = int((runs[0] if runs else {}).get("runs") or 0)
    return {"min_support": 1, "outcomes": outcomes, "rows": rows}


async def review_admission_feedback(
    session: Any,
    sink: ProposalSink = log_sink,
    *,
    min_outcomes: int = DEFAULT_MIN_OUTCOMES,
) -> list[AdmissionProposal]:
    """Read EG's per-class usage and hand the resulting proposals to ``sink``."""
    proposals = propose_admission_changes(
        await read_usage(session), min_outcomes=min_outcomes
    )
    sink(proposals)
    return proposals


__all__ = [
    "DEFAULT_MIN_OUTCOMES",
    "AdmissionProposal",
    "ProposalSink",
    "admitted_classes",
    "log_sink",
    "propose_admission_changes",
    "read_usage",
    "review_admission_feedback",
]
