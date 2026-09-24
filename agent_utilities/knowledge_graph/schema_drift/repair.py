"""Schema repair proposal (EH-403): detect, contain, propose, verify -- never activate.

For a quarantined drift, :func:`propose_repair`:

1. **Decides renames** through EG ``Decide`` (the EH-033 ``au.schema.mapping``
   point: a policy-safety question that never explores and may abstain). Each
   new field is a question over the fields the delta lost plus ``unmapped``;
   the classifier's deterministic rename candidate is the fallback.
2. **Builds the candidate** contract (the observed shape) and renders it as
   SHACL under ``approved:<source>`` (:mod:`.candidate`).
3. **Verifies it on a shadow graph**: the candidate is attached there (EG runs
   its full entering-schema validation, the ABox check) and the held delta is
   ingested into that shadow graph -- never the live one, whose checkpoint the
   quarantine left where it was. The diff report records what happened.
4. **Queues the approval**: an ``action.approval`` control lease (the fleet
   approvals queue a human decides) whose grant binds the exact candidate
   digest. Nothing here attaches to the live graph; :mod:`.activation` does,
   and only through EG ``GraphSchema.AttachApproved``, which refuses without
   that approval.

The proposal is recorded as a ``SchemaRepairProposal`` node (its own evidence,
holding the candidate so activation can re-attach exactly what was approved).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .candidate import RepairCandidate, build_candidate
from .classify import DriftChange, DriftClass
from .eg_port import SchemaRepairPort
from .report import SchemaDriftReport
from .shape import RecordShape

logger = logging.getLogger(__name__)

#: Node label of a recorded proposal.
PROPOSAL_LABEL = "SchemaRepairProposal"
#: The approved action EG requires the approval grant to name.
REPAIR_ACTION = "schema_repair"
#: The approvals queue lease kind (``action_policy.ACTION_APPROVAL_KIND``).
APPROVAL_KIND = "action.approval"
#: A pending approval lives at most EG's 24 h control-lease span.
APPROVAL_TTL_MS = 24 * 60 * 60 * 1000

#: ``(field, candidates, fallback) -> chosen candidate or None``.
RenameDecider = Callable[[str, Sequence[str], str | None], str | None]


def decide_rename(
    label: str, candidates: Sequence[str], fallback: str | None
) -> str | None:
    """EH-033's schema-mapping decision over the lost fields (abstain -> fallback)."""
    from agent_utilities.decide.consumers.schema_mapping import map_label

    choice = map_label(label, list(candidates), fallback)
    chosen = choice.option_id
    return str(chosen) if chosen in candidates else None


def resolve_renames(
    changes: Sequence[DriftChange], decide: RenameDecider = decide_rename
) -> dict[str, str]:
    """New field -> the lost field it replaces; one-to-one, first claim wins."""
    lost = sorted(
        {c.field for c in changes if c.kind is DriftClass.REMOVAL}
        | {c.renamed_from for c in changes if c.renamed_from}
    )
    if not lost:
        return {}
    added = (
        DriftClass.RENAME_CANDIDATE,
        DriftClass.ADDITIVE_NULLABLE,
        DriftClass.ADDITIVE_REQUIRED,
    )
    renames: dict[str, str] = {}
    for change in (c for c in changes if c.kind in added):
        chosen = decide(change.field, lost, change.renamed_from or None)
        if chosen and chosen not in renames.values():
            renames[change.field] = chosen
    return renames


@dataclass(frozen=True, slots=True)
class ShadowResult:
    """What the candidate did on the shadow graph."""

    graph: str
    attached: bool
    ingested: int
    failed: int
    error: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "graph": self.graph,
            "attached": self.attached,
            "ingested": self.ingested,
            "failed": self.failed,
            "error": self.error,
        }


#: Ingest the held delta into the named shadow graph; returns (ingested, failed).
ShadowIngest = Callable[[str], tuple[int, int]]


def shadow_graph_name(source: str, candidate: RepairCandidate) -> str:
    return f"schema-shadow-{candidate.digest[:24]}"


def verify_on_shadow(
    port: SchemaRepairPort, candidate: RepairCandidate, ingest: ShadowIngest
) -> ShadowResult:
    """Attach the candidate to a shadow graph, then ingest the held delta there."""
    graph = shadow_graph_name(candidate.source, candidate)
    shadow = port.for_graph(graph)
    shadow_key = f"admin:schema-candidate.{candidate.source}"
    try:
        shadow.attach_shadow(shadow_key, candidate.shapes_ttl)
    except Exception as exc:
        logger.warning(
            "schema repair shadow attach refused for %s: %s", candidate.source, exc
        )
        return ShadowResult(graph, False, 0, 0, f"attach refused: {exc}")
    try:
        ingested, failed = ingest(graph)
    except Exception as exc:
        logger.warning(
            "schema repair shadow ingest failed for %s: %s", candidate.source, exc
        )
        return ShadowResult(graph, True, 0, 0, f"ingest failed: {exc}")
    return ShadowResult(graph, True, ingested, failed)


@dataclass(frozen=True, slots=True)
class RepairProposal:
    """A queued, unapproved repair."""

    report: SchemaDriftReport
    candidate: RepairCandidate
    shadow: ShadowResult
    approval_lease_id: str = ""
    queue_error: str = ""
    diff: Mapping[str, Any] = field(default_factory=dict)

    @property
    def proposal_id(self) -> str:
        return f"schema-repair:{self.candidate.source}:{self.candidate.digest[:16]}"

    def summary(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "candidate_digest": self.candidate.digest,
            "approval_lease_id": self.approval_lease_id,
            "queue_error": self.queue_error,
            "shadow": self.shadow.to_json(),
            "renames": dict(self.candidate.renames),
        }


def _diff(
    report: SchemaDriftReport, candidate: RepairCandidate, shadow: ShadowResult
) -> dict:
    return {
        "report_id": report.report_id,
        "changes": [change.to_json() for change in report.changes],
        "renames": dict(candidate.renames),
        "records_held": report.records_held,
        "shadow": shadow.to_json(),
    }


def _grant(candidate: RepairCandidate, diff: Mapping[str, Any]) -> dict[str, Any]:
    params = json.dumps(diff, sort_keys=True, default=str)[:2000]
    return {
        "kind": REPAIR_ACTION,
        "target": candidate.source_id,
        "candidate_digest": candidate.digest,
        "request_digest": candidate.digest,
        "params_json": params,
        "source": "schema-drift",
        "reason": f"schema drift repair for {candidate.source}",
        "receipt_schema": "policy-receipt.v1",
    }


def queue_approval(
    port: SchemaRepairPort,
    tenant: str,
    candidate: RepairCandidate,
    diff: Mapping[str, Any],
) -> str:
    """Issue the pending approval lease; returns its id (idempotent per candidate)."""
    now_ms = int(time.time() * 1000)
    lease_id = f"action_approval:schema-repair-{candidate.digest[:32]}"
    request = {
        "tenant": tenant,
        "lease_id": lease_id,
        "kind": APPROVAL_KIND,
        "grant": _grant(candidate, diff),
        "issued_at_ms": now_ms,
        "expires_at_ms": now_ms + APPROVAL_TTL_MS,
        "hard_expires_at_ms": now_ms + APPROVAL_TTL_MS,
        "idempotency_key": f"schema-repair:{candidate.digest}",
    }
    answer = port.issue_approval(request)
    if answer.get("outcome") not in {"issued", "collision"}:
        raise RuntimeError(f"approval queue answered {answer.get('outcome')!r}")
    return lease_id


def record_proposal(engine: Any, proposal: RepairProposal) -> None:
    """Persist the proposal (with its candidate) for activation to re-read."""
    candidate = proposal.candidate
    properties = {
        "name": f"schema repair for {candidate.source}",
        "epistemic_class": "claim",
        "source": candidate.source,
        "status": "pending_approval" if proposal.approval_lease_id else "unqueued",
        "report_id": proposal.report.report_id,
        "candidate_digest": candidate.digest,
        "approval_lease_id": proposal.approval_lease_id,
        "shape": json.dumps(candidate.shape.to_json(), sort_keys=True),
        "shapes_ttl": candidate.shapes_ttl,
        "renames": json.dumps(dict(candidate.renames), sort_keys=True),
        "diff": json.dumps(dict(proposal.diff), sort_keys=True, default=str),
    }
    engine.add_node(proposal.proposal_id, PROPOSAL_LABEL, properties=properties)


@dataclass(frozen=True, slots=True)
class RepairContext:
    """What a proposal needs from the sync that quarantined the delta."""

    engine: Any
    port: SchemaRepairPort
    tenant: str
    ingest: ShadowIngest
    decide: RenameDecider = decide_rename


def propose_repair(
    context: RepairContext, report: SchemaDriftReport, observed: RecordShape
) -> RepairProposal:
    """Decide, build, shadow-verify and queue one repair for approval."""
    renames = resolve_renames(report.changes, context.decide)
    candidate = build_candidate(report.source, observed, renames)
    shadow = verify_on_shadow(context.port, candidate, context.ingest)
    diff = _diff(report, candidate, shadow)
    lease_id, error = "", ""
    try:
        lease_id = queue_approval(context.port, context.tenant, candidate, diff)
    except Exception as exc:
        logger.warning(
            "schema repair approval not queued for %s: %s", report.source, exc
        )
        error = str(exc)
    proposal = RepairProposal(report, candidate, shadow, lease_id, error, diff)
    try:
        record_proposal(context.engine, proposal)
    except Exception as exc:
        logger.warning(
            "schema repair proposal %s not recorded: %s", proposal.proposal_id, exc
        )
    return proposal


__all__ = [
    "APPROVAL_KIND",
    "PROPOSAL_LABEL",
    "REPAIR_ACTION",
    "RenameDecider",
    "RepairContext",
    "RepairProposal",
    "ShadowResult",
    "decide_rename",
    "propose_repair",
    "queue_approval",
    "resolve_renames",
    "shadow_graph_name",
    "verify_on_shadow",
]
