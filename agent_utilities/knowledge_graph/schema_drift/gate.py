"""The drift gate a connector sync runs between drain and apply (AU-SEC-R004/R005).

``run_gate`` measures the drained delta against the source's approved record
contract and decides whether the delta may be applied:

* no approved contract yet -> the first observation becomes the contract
  (``bootstrap``) once the delta is applied;
* no drift -> apply;
* drift the declared :class:`~.policy.ContractEvolutionPolicy` lets this source
  absorb -> apply, record the report, and widen the contract after the apply;
* any other drift -> QUARANTINE: nothing is applied, so the source checkpoint
  does not advance and the next sync re-drains the same delta. The report is
  recorded as observation evidence, a ``Gap`` (with its WorkItem) is opened,
  and a repair is proposed for approval (:mod:`.repair`).

Before measuring, an approved repair for the source is activated
(:mod:`.activation`), so an approval takes effect on the very next sync.
A contract store that cannot be read quarantines too: unmeasured is not clean.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .activation import activate_approved, proposals
from .classify import classify
from .contract_store import (
    ApprovedContract,
    ContractStoreUnavailable,
    load_contract,
    save_contract,
)
from .policy import ContractEvolutionPolicy, declared_policy
from .repair import RepairContext, propose_repair
from .report import SchemaDriftReport, Verdict, delta_digest, record_report
from .shape import RecordShape, infer_shape, merge_shapes, shape_digest

logger = logging.getLogger(__name__)

#: ``Gap.source`` of a quarantined drift.
GAP_SOURCE = "schema-drift"


@dataclass(frozen=True, slots=True)
class GateOutcome:
    """Whether the delta may be applied, and the contract to write after it is."""

    proceed: bool
    detail: dict[str, Any] = field(default_factory=dict)
    advance: ApprovedContract | None = None


def _verdict(
    policy: ContractEvolutionPolicy, source: str, report_changes: Sequence[Any]
) -> Verdict:
    if not report_changes:
        return Verdict.NO_DRIFT
    if policy.allows(source, report_changes):
        return Verdict.CONTINUE
    return Verdict.QUARANTINE


def _open_gap(engine: Any, report: SchemaDriftReport) -> str | None:
    from ..research.gaps import submit_gap

    kinds = sorted({change.kind.value for change in report.changes})
    gap = submit_gap(
        engine,
        source=GAP_SOURCE,
        signature=f"{report.source}:{report.observed_digest[:16]}",
        statement=(
            f"Schema drift ({', '.join(kinds)}) quarantined {report.records_held} "
            f"records from {report.source}; the checkpoint is held until a repair "
            "is approved or the source reverts."
        ),
        domain="ingestion",
        severity=0.7,
        evidence_refs=[report.report_id],
    )
    return None if gap is None else str(gap.get("id"))


def _existing_proposal(engine: Any, source: str, observed: RecordShape) -> str:
    """Status of a proposal already made for this exact candidate, or ``""``.

    A pending proposal needs no second one, and a refused one is not re-asked:
    the delta stays held and its Gap open until the source or the contract
    changes.
    """
    from .candidate import build_candidate

    try:
        known = proposals(engine, source)
    except ContractStoreUnavailable:
        return ""
    digest = build_candidate(source, observed, {}).digest
    return next((str(r.get("status")) for r in known if r.get("digest") == digest), "")


def _quarantine(
    engine: Any,
    report: SchemaDriftReport,
    observed: RecordShape,
    repair: RepairContext | None,
) -> GateOutcome:
    detail = {"quarantined": True, **report.summary()}
    detail["gap_id"] = _open_gap(engine, report)
    existing = _existing_proposal(engine, report.source, observed)
    if existing:
        detail["repair"] = existing
    elif repair is None:
        detail["repair"] = "unavailable"
    else:
        detail["repair"] = propose_repair(repair, report, observed).summary()
    return GateOutcome(False, detail)


def _activate(repair: RepairContext | None, source: str) -> dict[str, Any] | None:
    if repair is None:
        return None
    try:
        activation = activate_approved(
            repair.engine, repair.port, repair.tenant, source
        )
    except Exception as exc:
        logger.warning("schema repair activation for %s failed: %s", source, exc)
        return {"error": str(exc)}
    return None if activation is None else activation.summary()


@dataclass(frozen=True, slots=True)
class GateRequest:
    """One drained delta, as the sync hands it to the gate."""

    engine: Any
    source: str
    records: Sequence[Mapping[str, Any]]
    record_ids: Sequence[str]
    policy: ContractEvolutionPolicy | None = None
    repair: Callable[[], RepairContext | None] = lambda: None


def _measure(request: GateRequest, repair: RepairContext | None) -> GateOutcome:
    observed = infer_shape(request.records)
    try:
        contract = load_contract(request.engine, request.source)
    except ContractStoreUnavailable as exc:
        logger.warning("schema contract of %s unreadable: %s", request.source, exc)
        return GateOutcome(False, {"quarantined": True, "reason": str(exc)})
    if contract is None:
        advance = ApprovedContract(request.source, observed, "bootstrap")
        return GateOutcome(True, {"contract": "bootstrap"}, advance)
    policy = request.policy or declared_policy()
    changes = classify(contract.shape, observed)
    report = SchemaDriftReport(
        source=request.source,
        approved_digest=contract.digest,
        observed_digest=shape_digest(observed),
        changes=changes,
        verdict=_verdict(policy, request.source, changes),
        records_held=len(request.records),
        delta_digest=delta_digest(request.record_ids),
    )
    if report.verdict is Verdict.NO_DRIFT:
        return GateOutcome(True, {"verdict": Verdict.NO_DRIFT.value})
    record_report(request.engine, report)
    if report.verdict is Verdict.QUARANTINE:
        return _quarantine(request.engine, report, observed, repair)
    widened = ApprovedContract(
        request.source, merge_shapes(contract.shape, observed), "policy"
    )
    return GateOutcome(True, report.summary(), widened)


def run_gate(request: GateRequest) -> GateOutcome:
    """Activate an approved repair, then measure the delta against the contract."""
    repair = request.repair()
    activation = _activate(repair, request.source)
    if not request.records:
        outcome = GateOutcome(True, {"verdict": "no_records"})
    else:
        outcome = _measure(request, repair)
    if activation is not None:
        outcome.detail["activation"] = activation
    return outcome


def commit_contract(engine: Any, outcome: GateOutcome, *, applied_cleanly: bool) -> str:
    """Advance the contract with the checkpoint; returns what happened."""
    if outcome.advance is None:
        return "unchanged"
    if not applied_cleanly:
        return "held"
    try:
        save_contract(engine, outcome.advance)
    except ContractStoreUnavailable as exc:
        logger.warning(
            "schema contract of %s not advanced: %s", outcome.advance.source, exc
        )
        return "unpersisted"
    return outcome.advance.approved_by


__all__ = [
    "GAP_SOURCE",
    "GateOutcome",
    "GateRequest",
    "commit_contract",
    "run_gate",
]
