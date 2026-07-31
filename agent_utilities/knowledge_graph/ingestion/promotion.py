#!/usr/bin/python
from __future__ import annotations

"""Governed candidate-claim promotion (CONCEPT:AU-KG.ingest.governed-claim-promotion).

The universal-ingestion program's Track A: one assembled path a candidate claim
must survive to become a materialized fact — SHACL validation, policy/
classification checks, PII handling, deduplication, contradiction detection,
per-pack confidence thresholds, and a REAL steward-review hold. This module
does not reimplement any of those checks; it assembles pieces that already
exist elsewhere in the codebase:

* **SHACL** — the connector ingestion boundary's fail-closed gate
  (:func:`~.envelope_ingest.validate_rows_against_shacl`, a public alias of
  ``_shacl_validate_rows``) — NOT the advisory ``SHACLValidator``/
  ``shacl_gate`` phase used elsewhere, which fails OPEN on a missing shapes
  file or an uninstalled ``pyshacl`` and is documented as advisory, not a
  second security authority.
* **Policy/classification** — :func:`~.envelope_ingest.validate_envelope`
  (public alias of ``_validate_envelope``), the SAME fail-closed
  ``DataClassification``/``ExternalAccess`` policy ``ingest_envelope`` itself
  enforces at the write boundary (CONCEPT:AU-P0-4).
* **PII** — :class:`~agent_utilities.security.persistence_privacy.PersistencePrivacyGuard`,
  the same guard :func:`~..kb.extraction_run.classify_claim` already uses.
* **Contradiction detection** — :class:`~..adaptation.contradiction_detector.ContradictionDetector`.
* **Steward review lifecycle** — :class:`~..research.claim_flywheel.ClaimFlywheel`,
  the existing governed ``proposed -> validated -> accepted -> deprecated ->
  retracted`` lifecycle already exposed on the ``graph_claims`` MCP/REST
  surface, whose ``accept`` action already passes through the fail-closed
  ``ActionPolicy`` gate (``kind="claim.accept"``, default tier
  ``approval_required`` -> ``queue_approval`` — a real human approval-queue
  hold, not an advisory flag).
* **Fact materialization** — :func:`~.envelope_ingest.ingest_envelope` (the one
  atomic, idempotent, fail-closed native write path).
* **Supersession/retraction** — :mod:`~.supersession` (this program's Track B).

**The steward-review hold is structural, not a flag.** A :class:`CandidateClaim`
is ALWAYS persisted as a generic ``:Claim`` node (``is_verified=False``,
mirroring the existing mining-flywheel pattern in
``research.candidate_insight``) — never as the real domain-typed entity/edge it
proposes. A "fact" query surface (one that answers domain questions over typed
nodes) structurally cannot see an unpromoted ``:Claim`` node, because it is not
yet the type it proposes to become. Only :func:`materialize_on_claim_accepted`
— invoked after a steward's explicit ``graph_claims(action="accept")`` clears
the ActionPolicy approval gate — writes the real fact via ``ingest_envelope``.
Until then, the claim is unqueryable as a fact by construction, not by a
best-effort filter that a query path could forget to apply.

**Assumed contract (say what we assumed).** No sibling lane has yet published
an ``Artifact``/``Fragment``/``CandidateClaim`` type (the evidence-spine and
candidate-claim-extraction tracks were unstarted at the time this module was
written — verified via a repo-wide, all-branches search). :class:`CandidateClaim`
here is this module's OWN minimal, explicit assumption: one proposed
:class:`~.change_envelope.ChangeEnvelope` (the write this claim asks for, if
accepted) plus a domain/pack name (for per-pack confidence thresholds), a
confidence score, and an optional :class:`~agent_utilities.models.evidence_bundle.EvidenceBundle`.
When the real ``CandidateClaim``/domain-pack contracts land, adapt their shape
into this one (or extend it) rather than duplicating a second promotion path.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ...models.company_brain import DataClassification
from ...models.evidence_bundle import EvidenceBundle
from ...models.knowledge_graph import ClaimNode, RegistryNodeType
from ...security.persistence_privacy import PersistencePrivacyGuard
from ..adaptation.contradiction_detector import Claim as _ContradictionClaim
from ..adaptation.contradiction_detector import ContradictionDetector
from ..research.claim_flywheel import ClaimFlywheel, IllegalTransition
from ..research.promotion_governance import GovernanceCheck
from .change_envelope import ChangeEnvelope

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "CandidateClaim",
    "PromotionDecision",
    "PromotionVerdict",
    "GovernedPromotionValidator",
    "resolve_confidence_threshold",
    "envelope_from_dict",
    "propose_candidate_claim",
    "evaluate_and_advance",
    "materialize_on_claim_accepted",
    "supersede_materialized_claim",
    "retract_and_supersede",
]

#: One conservative default (Configuration discipline: a single correct
#: fallback, not a per-domain env flag family) — used for any domain/pack with
#: no explicit entry in ``AgentConfig.ingestion_confidence_thresholds``.
#: Matches the existing global ``extraction_run.NEEDS_REVIEW_CONFIDENCE_THRESHOLD``
#: so a domain with no stated policy behaves exactly as today.
DEFAULT_CONFIDENCE_THRESHOLD = 0.65

#: Bounded scan for the near-duplicate / contradiction comparison set — a
#: promotion gate must stay fast; this is a review-time signal, not an
#: exhaustive audit.
_EXISTING_CLAIM_SCAN_LIMIT = 200


class PromotionDecision(StrEnum):
    """The governance verdict's outcome (mirrors ``kb.extraction_run``'s item
    taxonomy so the platform has ONE accepted/needs_review/rejected/quarantined
    vocabulary, not two)."""

    #: Every gate passed — eligible to advance to VALIDATED, still pending an
    #: explicit steward ``accept``.
    CLEARED = "cleared"
    #: A soft gate (dedup / contradiction / confidence) held it — stays
    #: PROPOSED, awaiting more evidence or a manual steward decision.
    NEEDS_REVIEW = "needs_review"
    #: A hard structural/policy/SHACL gate failed — retracted, never
    #: materialized.
    REJECTED = "rejected"
    #: The PII guard found sensitive content — retracted, never materialized.
    QUARANTINED = "quarantined"


@dataclass(frozen=True)
class CandidateClaim:
    """One candidate claim awaiting governed promotion (see module docstring
    for the "assumed contract" note).

    ``claim_id`` defaults to a stable, content-addressed id derived from the
    proposed envelope's own ``idempotency_key`` — the SAME identity
    ``ingest_envelope`` uses to detect a redelivery, so re-proposing the
    identical change naturally collapses onto the same claim rather than
    minting a duplicate (Track B idempotency reused for Track A dedup).
    """

    domain: str
    statement: str
    confidence: float
    envelope: ChangeEnvelope
    evidence: EvidenceBundle | None = None
    claim_id: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"CandidateClaim.confidence must be in [0.0, 1.0], got {self.confidence!r}"
            )
        if not self.claim_id:
            object.__setattr__(
                self, "claim_id", f"claim:ingest:{self.envelope.idempotency_key}"
            )


@dataclass
class PromotionVerdict:
    """The full per-gate governance verdict for one candidate claim."""

    claim_id: str
    checks: list[GovernanceCheck] = field(default_factory=list)

    @property
    def decision(self) -> PromotionDecision:
        by_name = {c.name: c for c in self.checks}
        pii = by_name.get("pii")
        if pii is not None and not pii.passed:
            return PromotionDecision.QUARANTINED
        hard = ("classification_policy", "shacl")
        if any(not by_name[n].passed for n in hard if n in by_name):
            return PromotionDecision.REJECTED
        soft = ("dedup", "contradiction", "confidence")
        if any(not by_name[n].passed for n in soft if n in by_name):
            return PromotionDecision.NEEDS_REVIEW
        return PromotionDecision.CLEARED

    @property
    def summary(self) -> str:
        failing = [f"{c.name}: {c.reason}" for c in self.checks if not c.passed]
        return "; ".join(failing) if failing else "all governance checks passed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "decision": self.decision.value,
            "summary": self.summary,
            "checks": [
                {"name": c.name, "passed": c.passed, "reason": c.reason}
                for c in self.checks
            ],
        }


def resolve_confidence_threshold(domain: str) -> float:
    """Per-pack/per-domain promotion confidence threshold.

    Reads ``AgentConfig.ingestion_confidence_thresholds`` (a bounded
    domain -> threshold mapping, ``INGESTION_CONFIDENCE_THRESHOLDS``) — falls
    back to :data:`DEFAULT_CONFIDENCE_THRESHOLD` for any domain without an
    explicit entry. This is the ONE place a domain pack's threshold is
    resolved; a future domain-pack framework overrides it by populating this
    same mapping (or, once the domain-pack contract lands, by extending this
    resolver to consult the pack's own manifest first).
    """
    from ...core.config import config

    thresholds = getattr(config, "ingestion_confidence_thresholds", None) or {}
    try:
        return float(thresholds.get(domain, DEFAULT_CONFIDENCE_THRESHOLD))
    except (TypeError, ValueError):  # pragma: no cover - config already validates
        return DEFAULT_CONFIDENCE_THRESHOLD


def _existing_claims(engine: Any, domain: str) -> list[dict[str, Any]]:
    """Bounded read of other claims in the same domain (best-effort, never raises)."""
    try:
        rows = (
            engine.query_cypher(
                "MATCH (c:Claim) WHERE c.domain = $domain RETURN "
                "c.id AS id, c.claim_text AS claim_text "
                f"LIMIT {_EXISTING_CLAIM_SCAN_LIMIT}",
                {"domain": domain},
            )
            or []
        )
    except Exception as exc:  # noqa: BLE001 — best-effort review signal
        logger.debug("promotion: existing-claim scan failed: %s", exc, exc_info=True)
        rows = []
    return [r for r in rows if isinstance(r, dict) and r.get("id")]


class GovernedPromotionValidator:
    """Assembles the governed promotion gates for one :class:`CandidateClaim`.

    Every check is fail-closed: an exception, an unavailable dependency, or an
    unreadable policy is recorded as a FAILED check (never silently skipped as
    a pass) — the opposite of ``PromotionGovernanceValidator._check_shacl``'s
    fail-open-on-missing-shapes behavior, which this module deliberately does
    NOT reproduce for candidate-claim promotion (that gate faces a different,
    lower-stakes surface: golden-loop code-spec promotion, not external-source
    facts entering the KG).
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def validate(self, claim: CandidateClaim) -> PromotionVerdict:
        verdict = PromotionVerdict(claim_id=claim.claim_id)
        verdict.checks.append(self._check_classification_policy(claim))
        verdict.checks.append(self._check_pii(claim))
        verdict.checks.append(self._check_shacl(claim))
        verdict.checks.append(self._check_dedup(claim))
        verdict.checks.append(self._check_contradiction(claim))
        verdict.checks.append(self._check_confidence(claim))
        return verdict

    def _check_classification_policy(self, claim: CandidateClaim) -> GovernanceCheck:
        """Fail-closed classification/ACL policy — reuses ``ingest_envelope``'s
        OWN write-boundary policy (CONCEPT:AU-P0-4), never a second copy."""
        from .envelope_ingest import validate_envelope

        try:
            violations = validate_envelope(claim.envelope)
        except Exception as exc:  # noqa: BLE001 — unreadable policy denies, never allows
            return GovernanceCheck(
                "classification_policy", False, f"policy check errored: {exc}"
            )
        if violations:
            return GovernanceCheck(
                "classification_policy", False, "; ".join(violations)
            )
        if claim.envelope.classification not in DataClassification:
            return GovernanceCheck(
                "classification_policy",
                False,
                f"unknown classification {claim.envelope.classification!r}",
            )
        return GovernanceCheck("classification_policy", True, "policy conforms")

    def _check_pii(self, claim: CandidateClaim) -> GovernanceCheck:
        try:
            _clean, report = PersistencePrivacyGuard().sanitize_text(claim.statement)
        except Exception as exc:  # noqa: BLE001 — an unscannable claim is never assumed clean
            return GovernanceCheck("pii", False, f"privacy scan errored: {exc}")
        if report.changed:
            return GovernanceCheck(
                "pii",
                False,
                f"sensitive content detected: {', '.join(report.detected_types)}",
            )
        return GovernanceCheck("pii", True, "no sensitive content detected")

    def _check_shacl(self, claim: CandidateClaim) -> GovernanceCheck:
        """Unconditional fail-closed SHACL gate over the claim's proposed
        entity/edge payload — only applicable to an ``upsert`` envelope
        carrying a ``typed_payload`` (a ``delete``/``snapshot_complete``
        marker, or a claim with no proposed materialization, has no shape to
        validate against)."""
        envelope = claim.envelope
        if envelope.operation != "upsert" or envelope.typed_payload is None:
            return GovernanceCheck(
                "shacl", True, "no proposed entity payload to validate"
            )
        client = getattr(self.engine, "client", None) or self.engine
        row = dict(envelope.typed_payload)
        row.setdefault("node_type", envelope.payload_type or "Artifact")
        node_id = envelope.source_object_id or claim.claim_id
        try:
            from .envelope_ingest import validate_rows_against_shacl

            validate_rows_against_shacl(client, [(node_id, row)])
        except Exception as exc:  # noqa: BLE001 — cannot prove conformance ⇒ FAIL, never pass
            return GovernanceCheck("shacl", False, str(exc))
        return GovernanceCheck("shacl", True, "conforms")

    def _check_dedup(self, claim: CandidateClaim) -> GovernanceCheck:
        """Flags a DIFFERENT claim id asserting identical statement text in
        the same domain (a redelivery of the SAME claim id is handled
        separately and benignly by ``ClaimFlywheel.propose``'s own idempotent
        no-op — this check is about content-level duplication across ids)."""
        for row in _existing_claims(self.engine, claim.domain):
            other_id = str(row.get("id") or "")
            if other_id and other_id != claim.claim_id:
                if str(row.get("claim_text") or "").strip() == claim.statement.strip():
                    return GovernanceCheck(
                        "dedup",
                        False,
                        f"duplicate statement text of claim {other_id!r}",
                    )
        return GovernanceCheck("dedup", True, "no duplicate statement found")

    def _check_contradiction(self, claim: CandidateClaim) -> GovernanceCheck:
        existing = [
            _ContradictionClaim(
                id=str(row.get("id") or ""), text=str(row.get("claim_text") or "")
            )
            for row in _existing_claims(self.engine, claim.domain)
            if row.get("id") != claim.claim_id
        ]
        try:
            findings = ContradictionDetector().check(
                _ContradictionClaim(id=claim.claim_id, text=claim.statement),
                existing,
            )
        except Exception as exc:  # noqa: BLE001 — best-effort signal, not a crash
            logger.debug(
                "promotion: contradiction check errored: %s", exc, exc_info=True
            )
            return GovernanceCheck(
                "contradiction", True, "contradiction check unavailable"
            )
        high = [f for f in findings if f.severity == "high"]
        if high:
            names = ", ".join(f.conflict_id for f in high)
            return GovernanceCheck(
                "contradiction", False, f"contradicts existing claim(s): {names}"
            )
        return GovernanceCheck("contradiction", True, "no high-severity contradiction")

    def _check_confidence(self, claim: CandidateClaim) -> GovernanceCheck:
        threshold = resolve_confidence_threshold(claim.domain)
        if claim.confidence < threshold:
            return GovernanceCheck(
                "confidence",
                False,
                f"confidence {claim.confidence:.2f} < domain {claim.domain!r} "
                f"threshold {threshold:.2f}",
            )
        return GovernanceCheck(
            "confidence", True, f"confidence {claim.confidence:.2f} >= {threshold:.2f}"
        )


def _claim_node(claim: CandidateClaim) -> ClaimNode:
    return ClaimNode(
        id=claim.claim_id,
        type=RegistryNodeType.CLAIM,
        name=claim.statement[:120] or claim.claim_id,
        claim_text=claim.statement,
        confidence=claim.confidence,
        claim_type="ingestion_candidate",
        source_ids=list(claim.envelope.provenance.get("source_ids", ()))
        or [claim.envelope.source_object_id or claim.envelope.envelope_id],
        extracted_from=claim.envelope.source_object_id or None,
        domain=claim.domain,
        is_verified=False,
        metadata={
            "proposed_envelope": claim.envelope.as_dict(),
            "connector": claim.envelope.connector,
            "materialized": False,
        },
    )


def propose_candidate_claim(
    engine: Any, claim: CandidateClaim, *, reason: str = ""
) -> dict[str, Any]:
    """Persist ``claim`` as an unverified ``:Claim`` node and record its INITIAL
    lifecycle proposal (idempotent — a redelivery of the same ``claim_id`` is a
    no-op, mirroring ``ClaimFlywheel.propose``'s own contract). Never
    materializes anything — a proposed claim is NEVER queryable as the fact it
    proposes; only :func:`materialize_on_claim_accepted` writes the real fact,
    and only after an explicit steward ``accept``.
    """
    node = _claim_node(claim)
    try:
        # Label is the literal ``"Claim"`` (matches
        # ``loop_controller._run_insight_validation``'s existing convention —
        # ``ClaimNode.type``'s enum VALUE, "claim", is excluded from
        # ``properties`` precisely so it never collides with this label).
        engine.add_node(
            node.id,
            "Claim",
            properties=node.model_dump(mode="json", exclude={"type"}),
        )
    except Exception as exc:  # noqa: BLE001 — the claim node write is best-effort persistence
        logger.debug(
            "promotion: could not persist candidate claim %s: %s",
            claim.claim_id,
            exc,
            exc_info=True,
        )
    flywheel = ClaimFlywheel(engine)
    transition = flywheel.propose(claim.claim_id, reason=reason or "candidate claim")
    return {
        "claim_id": claim.claim_id,
        "current_state": flywheel.current_state(claim.claim_id).value,
        "transition": transition.to_dict() if transition else None,
    }


def evaluate_and_advance(engine: Any, claim: CandidateClaim) -> dict[str, Any]:
    """Run the governed promotion gates and advance (or hold) the claim's
    lifecycle accordingly.

    * ``REJECTED``/``QUARANTINED`` -> retracted (terminal, sticky — never
      re-proposed even if the same content is re-mined/re-ingested later).
    * ``NEEDS_REVIEW`` -> held at its current state (a real, audit-visible
      hold — see :meth:`ClaimFlywheel.record_hold`), awaiting either fresh
      evidence on a later ingestion pass or an explicit steward action.
    * ``CLEARED`` -> advances PROPOSED -> VALIDATED. This is STILL not a fact:
      only an explicit steward ``graph_claims(action="accept")`` call (gated
      by the fail-closed ``ActionPolicy`` approval queue) triggers
      :func:`materialize_on_claim_accepted`.
    """
    propose_candidate_claim(engine, claim)
    verdict = GovernedPromotionValidator(engine).validate(claim)
    flywheel = ClaimFlywheel(engine)
    decision = verdict.decision

    transition = None
    try:
        if decision in (PromotionDecision.REJECTED, PromotionDecision.QUARANTINED):
            transition = flywheel.reject(claim.claim_id, reason=verdict.summary)
        elif decision == PromotionDecision.NEEDS_REVIEW:
            transition = flywheel.record_hold(claim.claim_id, reason=verdict.summary)
        else:
            transition = flywheel.validate(claim.claim_id, True, reason=verdict.summary)
    except IllegalTransition as exc:
        logger.info(
            "promotion: %s already past a state reachable from %s: %s",
            claim.claim_id,
            decision.value,
            exc,
        )

    return {
        "verdict": verdict.to_dict(),
        "current_state": flywheel.current_state(claim.claim_id).value,
        "transition": transition.to_dict() if transition else None,
    }


def envelope_from_dict(data: dict[str, Any]) -> ChangeEnvelope:
    """Reconstruct a :class:`ChangeEnvelope` from :meth:`ChangeEnvelope.as_dict`'s
    JSON-friendly rendering — the one-way round trip this module needs to
    persist a claim's proposed write across process boundaries (the steward's
    ``accept`` may land in a different process than the original proposal),
    and that a caller submitting a candidate claim over MCP/REST (JSON only)
    needs too — see ``graph_claims(action="evaluate")``."""
    from ...protocols.source_connectors.base import ExternalAccess

    payload = dict(data)
    acl = payload.get("source_acl")
    classification = payload.get("classification")
    return ChangeEnvelope(
        envelope_id=payload.get("envelope_id") or "",
        idempotency_key=payload.get("idempotency_key") or "",
        tenant=payload.get("tenant") or "",
        connector=payload.get("connector") or "",
        source_instance=payload.get("source_instance") or "",
        source_object_id=payload.get("source_object_id") or "",
        source_version=payload.get("source_version") or "",
        operation=payload.get("operation") or "upsert",
        event_time=payload.get("event_time"),
        valid_time=payload.get("valid_time"),
        observed_time=payload.get("observed_time") or "",
        schema_version=payload.get("schema_version") or "1",
        ontology_mapping_version=payload.get("ontology_mapping_version") or "",
        payload_type=payload.get("payload_type") or "json",
        typed_payload=payload.get("typed_payload"),
        blob_ref=payload.get("blob_ref"),
        source_acl=ExternalAccess.model_validate(acl) if acl else None,
        classification=DataClassification(classification)
        if classification
        else DataClassification.INTERNAL,
        retention=payload.get("retention"),
        legal_hold=bool(payload.get("legal_hold", False)),
        provenance=dict(payload.get("provenance") or {}),
        confidence=float(payload.get("confidence", 1.0)),
        checkpoint=payload.get("checkpoint"),
        trace_context=payload.get("trace_context"),
        live_ids=tuple(payload.get("live_ids") or ()),
    )


def materialize_on_claim_accepted(engine: Any, claim_id: str) -> dict[str, Any] | None:
    """Materialize an ACCEPTED claim's proposed write — the ONLY function that
    turns a governed candidate claim into a real, queryable fact.

    Best-effort, idempotent-safe hook: intended to be called right after a
    steward's ``graph_claims(action="accept")`` succeeds. Returns ``None``
    (a genuine no-op, not a failure) when the claim carries no proposed
    envelope (a claim family unrelated to ingestion) or was already
    materialized. Returns the ``ingest_envelope`` result dict otherwise —
    including a ``status: "rejected"``/``"failed"`` outcome, which the caller
    MUST surface rather than swallow (an accepted claim whose materialization
    failed must never silently look like it succeeded).
    """
    from .envelope_ingest import ingest_envelope

    try:
        rows = engine.query_cypher(
            "MATCH (c:Claim {id: $id}) RETURN c.metadata AS metadata",
            {"id": claim_id},
        )
    except Exception as exc:  # noqa: BLE001 — cannot read the claim ⇒ nothing to materialize
        logger.warning(
            "promotion: could not read claim %s for materialization: %s",
            claim_id,
            exc,
        )
        return None
    if not rows or not isinstance(rows[0], dict):
        return None
    metadata = rows[0].get("metadata") or {}
    if not isinstance(metadata, dict):
        return None
    if metadata.get("materialized"):
        return {"status": "skipped", "reason": "already materialized"}
    envelope_data = metadata.get("proposed_envelope")
    if not envelope_data:
        return None

    envelope = envelope_from_dict(envelope_data)
    result = ingest_envelope(engine, envelope)
    new_metadata = {
        **metadata,
        "materialized": result.get("status") not in ("rejected", "failed"),
        "materialized_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "materialization_status": result.get("status"),
    }
    try:
        # Whole-value CAS on the single flat ``metadata`` field (never a
        # dotted sub-path — CAS backends are not guaranteed to support
        # nested-key conditions/updates into a JSON property).
        applied = engine.compare_and_set_node_fields(
            claim_id, {"metadata": metadata}, {"metadata": new_metadata}
        )
        if not applied:
            logger.debug(
                "promotion: materialization marker CAS for %s did not apply "
                "(concurrent update) — the write itself already committed",
                claim_id,
            )
    except Exception as exc:  # noqa: BLE001 — the marker write is best-effort bookkeeping
        logger.debug(
            "promotion: could not stamp materialization marker for %s: %s",
            claim_id,
            exc,
            exc_info=True,
        )
    return result


def supersede_materialized_claim(
    engine: Any, claim_id: str, *, reason: str
) -> dict[str, Any] | None:
    """If ``claim_id`` was already materialized, retire the real fact it
    wrote (Track B tie-in — see :mod:`.supersession`); ``None`` when the
    claim was never materialized (nothing to retire).

    Deliberately does NOT touch the claim's own lifecycle state — callers
    that also need to retract the claim (e.g. :func:`retract_and_supersede`,
    or the ``graph_claims`` MCP/REST tool's own ``retract`` action, which
    already calls ``ClaimFlywheel.retract`` itself) call that separately;
    this function is safe to call standalone, any number of times, on a
    claim that is not otherwise being retracted.
    """
    from .supersession import retire_fact

    try:
        rows = engine.query_cypher(
            "MATCH (c:Claim {id: $id}) RETURN c.metadata AS metadata",
            {"id": claim_id},
        )
    except Exception as exc:  # noqa: BLE001 — best-effort lookup
        logger.debug(
            "promotion: supersede lookup failed for %s: %s",
            claim_id,
            exc,
            exc_info=True,
        )
        rows = []
    metadata = rows[0].get("metadata") if rows and isinstance(rows[0], dict) else {}
    if not isinstance(metadata, dict) or not metadata.get("materialized"):
        return None
    envelope_data = metadata.get("proposed_envelope") or {}
    entity_id = envelope_data.get("source_object_id")
    if not entity_id:
        return None
    connector = envelope_data.get("connector") or "governed_promotion"
    return retire_fact(
        engine,
        entity_id=str(entity_id),
        connector=str(connector),
        reason=reason,
        retracted_by_claim=claim_id,
    )


def retract_and_supersede(engine: Any, claim_id: str, *, reason: str) -> dict[str, Any]:
    """Retract a claim AND, if it was already materialized, supersede the
    real fact it wrote.

    Retracting a claim that was never materialized only records the
    (terminal, sticky) ``ClaimLifecycleEvent`` — nothing else exists to
    retire. Retracting a MATERIALIZED claim additionally tombstones the
    written fact through the same fail-closed ``ingest_envelope`` boundary
    (never a direct delete) so the retired fact stays inspectable.
    """
    flywheel = ClaimFlywheel(engine)
    transition = flywheel.retract(claim_id, reason=reason)
    superseded = supersede_materialized_claim(engine, claim_id, reason=reason)
    return {
        "claim_id": claim_id,
        "transition": transition.to_dict() if transition else None,
        "superseded_fact": superseded,
    }
