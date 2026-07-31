#!/usr/bin/python
from __future__ import annotations

"""CONCEPT:AU-KG.enrichment.extraction-run-provenance — ExtractionRun provenance.

Implements the ``ExtractionRun`` PROV-O ``Activity`` declared in
``ontology_process_intelligence.ttl`` (``:ExtractionRun``) but, until now, never
given a Python binding: "a deterministic or learned extraction execution bound to
input hash, schema version, thresholds, and metrics."

Every entity/claim :class:`~.entity_claim_extractor.EntityClaimExtractor` persists
is now bound to the :class:`~agent_utilities.models.knowledge_graph.ExtractionRunNode`
that produced it (``WAS_GENERATED_BY``), so "no fact without provenance" is
mechanically true: the run node carries parser/extractor version, the schema pack
in force, the calibration policy + confidence threshold, the graph epoch, and the
exact typed outcome (CONCEPT:AU-KG.enrichment.extraction-outcome-taxonomy) it
landed in — so "no facts found" and "the extractor wasn't there" are two
DIFFERENT, queryable node properties, never the same silent zero.

Two-tier taxonomy
------------------
* **Run-level** (:class:`ExtractionOutcome`) — the ONE headline status persisted
  on the run node: what happened to this extraction execution as a whole.
* **Item-level** (:func:`classify_entity` / :func:`classify_claim`) — every
  candidate entity/claim is bucketed into exactly one of ``accepted`` /
  ``needs_review`` / ``rejected`` / ``quarantined`` BEFORE it is persisted, so an
  uncertain or unsafe candidate is never silently written as an accepted fact
  (constraint 6: "every uncertain result is a proposal ... never a silent fact").
  :func:`decide_outcome` folds the item-level counts into the run-level status.
"""

import hashlib
import logging
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from agent_utilities.models.knowledge_graph import (
    ExtractionRunNode,
    RegistryEdgeType,
)
from agent_utilities.security.persistence_privacy import PersistencePrivacyGuard

logger = logging.getLogger(__name__)

__all__ = [
    "ExtractionOutcome",
    "OutcomeCounts",
    "PARSER_VERSION",
    "EXTRACTOR_VERSION",
    "NEEDS_REVIEW_CONFIDENCE_THRESHOLD",
    "content_hash",
    "classify_entity",
    "classify_claim",
    "decide_outcome",
    "resolve_graph_epoch",
    "build_extraction_run",
    "persist_extraction_run",
    "link_generated_by",
    "record_unavailable_run",
    "record_deferred_run",
    "persist_evidence_span",
]


class ExtractionOutcome(StrEnum):
    """CONCEPT:AU-KG.enrichment.extraction-outcome-taxonomy — the ONE typed vocabulary
    every extraction execution's terminal status is drawn from. Distinguishes, in
    particular, ``no_facts_found`` (the extractor ran cleanly over real content and
    genuinely found nothing) from ``extractor_unavailable`` (the extractor could not
    run at all) — the exact ambiguity that previously caused a misdiagnosis.
    """

    #: At least one entity/claim/relationship was extracted and accepted.
    FACTS_FOUND = "facts_found"
    #: The extractor ran cleanly over real content and found nothing extractable.
    NO_FACTS_FOUND = "no_facts_found"
    #: The extractor itself could not run (import/construction/engine failure) —
    #: NEVER conflated with `no_facts_found`.
    EXTRACTOR_UNAVAILABLE = "extractor_unavailable"
    #: Every candidate failed structural validation (empty/degenerate text).
    REJECTED_BY_VALIDATION = "rejected_by_validation"
    #: Every candidate tripped the persistence-privacy guard (PII/secret-shaped
    #: text) and was withheld rather than persisted.
    QUARANTINED = "quarantined"
    #: The resource-priority cost bound deferred this window/run entirely — it
    #: was never attempted (CONCEPT:AU-ORCH.scheduling.resource-priority-edict).
    BUDGET_DEFERRED = "budget_deferred"
    #: Candidates exist but fell below the calibration confidence threshold —
    #: persisted as proposals awaiting review, never silently accepted.
    NEEDS_REVIEW = "needs_review"


#: The regex/parser ruleset version (`entity_claim_extractor.extract_deterministic`).
#: Bump when the extraction patterns change so old ExtractionRun records remain
#: comparable to the ruleset that actually produced them.
PARSER_VERSION = "regex-deterministic-v1"

#: `EntityClaimExtractor`'s own version — bump on any behavioural change to
#: `extract_and_persist` (item classification, persistence shape, etc).
EXTRACTOR_VERSION = "1.0.0"

#: A claim below this confidence lands in `needs_review` instead of being
#: silently accepted (constraint 6). One correct default (Configuration
#: discipline) — not a new env flag; the value is recorded on every run's
#: `thresholds` so it stays inspectable/auditable per record.
NEEDS_REVIEW_CONFIDENCE_THRESHOLD = 0.65

_MAX_ENTITY_NAME_LEN = 300
_MAX_CLAIM_TEXT_LEN = 2000


def content_hash(text: str) -> str:
    """Stable short digest used for ``input_hash`` / content-addressed run ids."""
    return hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()[
        :32
    ]


@dataclass
class OutcomeCounts:
    """Item-level classification tally for one extraction execution."""

    accepted: int = 0
    needs_review: int = 0
    rejected: int = 0
    quarantined: int = 0
    #: Non-exclusive: relationships are not individually classified (constraint
    #: scope — see module docstring), just tallied for the run's provenance.
    relationships: int = 0

    @property
    def total_candidates(self) -> int:
        return self.accepted + self.needs_review + self.rejected + self.quarantined

    def as_dict(self) -> dict[str, int]:
        return {
            "accepted": self.accepted,
            "needs_review": self.needs_review,
            "rejected": self.rejected,
            "quarantined": self.quarantined,
            "relationships": self.relationships,
        }


def decide_outcome(counts: OutcomeCounts) -> ExtractionOutcome:
    """Fold item-level :class:`OutcomeCounts` into the run's headline status.

    Precedence: any accepted fact makes the run `facts_found` (even if other
    candidates needed review or were dropped); otherwise `needs_review` if any
    candidate is pending; otherwise whichever of rejected/quarantined dominates;
    otherwise `no_facts_found` (nothing was even a candidate).
    """
    if counts.total_candidates == 0:
        return ExtractionOutcome.NO_FACTS_FOUND
    if counts.accepted > 0:
        return ExtractionOutcome.FACTS_FOUND
    if counts.needs_review > 0:
        return ExtractionOutcome.NEEDS_REVIEW
    if counts.rejected >= counts.quarantined and counts.rejected > 0:
        return ExtractionOutcome.REJECTED_BY_VALIDATION
    if counts.quarantined > 0:
        return ExtractionOutcome.QUARANTINED
    return ExtractionOutcome.NO_FACTS_FOUND  # pragma: no cover - unreachable guard


def classify_entity(
    name: str,
    description: str,
    *,
    guard: PersistencePrivacyGuard,
) -> str:
    """Bucket one candidate entity: ``accepted`` / ``rejected`` / ``quarantined``.

    Entities carry no confidence score, so ``needs_review`` never applies to
    them — only claims (below) do.
    """
    stripped = (name or "").strip()
    if (
        not stripped
        or not any(ch.isalpha() for ch in stripped)
        or len(stripped) > _MAX_ENTITY_NAME_LEN
    ):
        return "rejected"
    _, report = guard.sanitize_text(f"{stripped} {description or ''}".strip())
    if report.changed:
        return "quarantined"
    return "accepted"


def classify_claim(
    claim_text: str,
    confidence: float,
    *,
    guard: PersistencePrivacyGuard,
    review_threshold: float = NEEDS_REVIEW_CONFIDENCE_THRESHOLD,
) -> str:
    """Bucket one candidate claim: ``accepted`` / ``needs_review`` / ``rejected`` /
    ``quarantined``. Structural validation and privacy quarantine are checked
    BEFORE confidence, so a malformed or unsafe claim is never merely
    "low-confidence" — it is rejected/quarantined outright.
    """
    stripped = (claim_text or "").strip()
    if (
        not stripped
        or not any(ch.isalpha() for ch in stripped)
        or len(stripped) > _MAX_CLAIM_TEXT_LEN
    ):
        return "rejected"
    _, report = guard.sanitize_text(stripped)
    if report.changed:
        return "quarantined"
    if confidence < review_threshold:
        return "needs_review"
    return "accepted"


def resolve_graph_epoch(engine: Any) -> int:
    """Best-effort ``catalog_epoch`` for provenance (mirrors `envelope_ingest`'s
    ``session.catalog_epoch or authority.compute.catalog_epoch`` fallback). Never
    raises — an unresolved epoch is recorded as ``0``, not a failure.
    """
    try:
        from ..core.session import current_session

        session = current_session()
        if session is not None and session.catalog_epoch is not None:
            return int(session.catalog_epoch)
    except Exception:  # noqa: BLE001 — provenance epoch is best-effort
        logger.debug("graph epoch session lookup failed", exc_info=True)
    try:
        gc = getattr(engine, "graph_compute", None)
        epoch = getattr(gc, "catalog_epoch", None)
        if epoch is not None:
            return int(epoch)
    except Exception:  # noqa: BLE001
        logger.debug("graph epoch compute lookup failed", exc_info=True)
    return 0


def _run_id(
    source_id: str,
    input_hash: str,
    extractor_version: str,
    schema_pack_ref: str,
    outcome: str,
) -> str:
    """Content-addressed run id — idempotent (a repeat run producing the SAME
    outcome for the SAME input upserts, matching `claim_node_id`'s convention),
    while a DIFFERENT outcome for the same input (e.g. a later, unbudgeted retry
    that actually runs) mints a new, historically distinct run record.
    """
    digest = hashlib.sha256(
        f"{source_id}:{input_hash}:{extractor_version}:{schema_pack_ref}:{outcome}".encode()
    ).hexdigest()[:32]
    return f"extraction_run:{digest}"


def build_extraction_run(
    *,
    engine: Any,
    source_id: str,
    input_hash: str,
    outcome: ExtractionOutcome,
    counts: OutcomeCounts | None = None,
    entities_count: int = 0,
    claims_count: int = 0,
    schema_pack_ref: str = "",
    model_ref: str | None = None,
    started_at: float | None = None,
    error: str | None = None,
) -> ExtractionRunNode:
    """Build (but do not persist) the :class:`ExtractionRunNode` for one execution.

    ``counts`` is the COMBINED entity+claim item-level tally (used for
    ``outcome_counts``/precedence); ``entities_count``/``claims_count`` are the
    actual persisted-node counts by type, recorded separately since one
    ``OutcomeCounts`` spans both candidate kinds.
    """
    counts = counts or OutcomeCounts()
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    duration_ms = (time.monotonic() - started_at) * 1000 if started_at else 0.0
    return ExtractionRunNode(
        id=_run_id(
            source_id, input_hash, EXTRACTOR_VERSION, schema_pack_ref, outcome.value
        ),
        name=f"extraction:{source_id}",
        source_id=source_id,
        input_hash=input_hash,
        parser_version=PARSER_VERSION,
        extractor_version=EXTRACTOR_VERSION,
        schema_pack_ref=schema_pack_ref,
        model_ref=model_ref,
        graph_epoch=resolve_graph_epoch(engine),
        calibration_policy="confidence-threshold-v1",
        thresholds={"claim_confidence_review": NEEDS_REVIEW_CONFIDENCE_THRESHOLD},
        outcome=outcome.value,
        outcome_counts=counts.as_dict(),
        entities_count=entities_count,
        claims_count=claims_count,
        relationships_count=counts.relationships,
        started_at=now,
        completed_at=now,
        duration_ms=duration_ms,
        error=error,
    )


def persist_extraction_run(engine: Any, run: ExtractionRunNode) -> str:
    """Persist the run node through the SAME direct engine-write seam
    `EntityClaimExtractor.extract_and_persist` already uses for entities/claims
    (no ChangeEnvelope — this module deliberately mirrors that existing, working
    persistence style rather than introducing a second one). Best-effort: never
    raises into ingestion.
    """
    try:
        graph_data = engine._serialize_node(run)
        engine.graph.add_node(run.id, **graph_data)
        if getattr(engine, "backend", None):
            data = engine._serialize_node(run, label="ExtractionRun")
            engine._upsert_node("ExtractionRun", run.id, data)
    except Exception:  # noqa: BLE001 — provenance write is best-effort
        logger.debug("ExtractionRun persist failed", exc_info=True)
    return run.id


def link_generated_by(
    engine: Any,
    child_id: str,
    run_id: str,
    *,
    review_state: str = "accepted",
    confidence: float | None = None,
) -> None:
    """``child WAS_GENERATED_BY run`` (PROV-O ``wasGeneratedBy``) — binds one
    persisted entity/claim to the run that produced it. Best-effort, mirrors the
    existing ``if source in self.engine.graph and target in self.engine.graph``
    guard `EntityClaimExtractor` already applies before every edge write.
    """
    try:
        if child_id in engine.graph and run_id in engine.graph:
            props: dict[str, Any] = {"review_state": review_state}
            if confidence is not None:
                props["confidence"] = confidence
            engine.link_nodes(
                child_id, run_id, RegistryEdgeType.WAS_GENERATED_BY, props
            )
    except Exception:  # noqa: BLE001 — provenance linking is best-effort
        logger.debug(
            "WAS_GENERATED_BY link failed for %s -> %s", child_id, run_id, exc_info=True
        )


def record_unavailable_run(
    engine: Any, *, source_id: str, error: str
) -> ExtractionRunNode:
    """Persist an `EXTRACTOR_UNAVAILABLE` run — the extractor could not run at
    all, which must NEVER look identical to a clean `NO_FACTS_FOUND` pass.
    """
    run = build_extraction_run(
        engine=engine,
        source_id=source_id,
        input_hash="",
        outcome=ExtractionOutcome.EXTRACTOR_UNAVAILABLE,
        error=error[:500] if error else None,
    )
    persist_extraction_run(engine, run)
    return run


def persist_evidence_span(
    source_id: str,
    content: str,
    text: str,
    *,
    source_type: str,
    confidence: float = 1.0,
) -> str | None:
    """Locate ``text`` verbatim in ``content`` and through-write a ``DocumentSpan``
    evidence locus (CONCEPT:AU-KG.identity.evidence-spine-convergence) — the exact
    byte-span evidence reference constraint 2 requires. Mirrors
    `ingestion.engine._persist_document_span_evidence`'s pattern (same store, same
    best-effort contract) applied to entity/claim text instead of LLM-extracted
    facts. Returns the evidence id, or ``None`` when the span can't be located or
    the evidence store is unavailable — never raises.
    """
    start = content.find(text) if text else -1
    if start < 0:
        return None
    try:
        from ..memory.native_ingest import media_store

        store = media_store()
        locus = store.store_document_span_evidence(
            text.encode("utf-8"),
            document_id=source_id,
            start=start,
            end=start + len(text),
            source=source_type,
            confidence=confidence,
        )
    except Exception:  # noqa: BLE001 — evidence write is best-effort
        logger.debug("entity/claim evidence span write failed", exc_info=True)
        return None
    return locus.evidence_id if locus else None


def record_deferred_run(
    engine: Any, *, source_id: str, input_hash: str
) -> ExtractionRunNode:
    """Persist a `BUDGET_DEFERRED` run — the resource-priority cost bound
    (CONCEPT:AU-ORCH.scheduling.resource-priority-edict) skipped this window
    entirely under background-ingestion contention; it was never attempted, and
    that must be distinguishable from a clean `NO_FACTS_FOUND` pass.
    """
    run = build_extraction_run(
        engine=engine,
        source_id=source_id,
        input_hash=input_hash,
        outcome=ExtractionOutcome.BUDGET_DEFERRED,
    )
    persist_extraction_run(engine, run)
    return run
