#!/usr/bin/python
from __future__ import annotations

"""The unified Evidence resource (CONCEPT:AU-KG.evolution.unified-evidence-resource, lane 7.1).

Five signal channels feed the evolution/optimisation loop today, each with its
own native shape already persisted somewhere in the graph or computed inline:

    (a) execution_trace      — Langfuse-backed ``RunTrace -[:PRODUCED_OUTCOME]->
                                :OutcomeEvaluation`` (``observability/trace_ontology.py``)
    (b) optimization_signal  — the eg-native ``ProgramOptimize`` sweep's per-target
                                result (``harness/program_optimization.py``)
    (c) graph_health         — ``:HealthAnomaly`` (``observability/health_ingest.py``),
                                plus (not yet gathered live — see module docstring
                                below) retrieval-quality / connector-freshness /
                                ontology-validation reports
    (d) research_finding     — a mined ``CandidateInsight``
                                (``knowledge_graph/research/candidate_insight.py``)
    (e) process_signal       — an OCEL/tEKG import's projection summary
                                (``mcp/tools/engine_surface_tools.py``'s ``process``
                                mining action)

They are **five sources, not five systems**: this module does not build a
per-channel pipeline. It defines ONE contract (:class:`Evidence`), one adapter
per channel that maps the channel's real, already-computed native shape onto
that contract (never fabricating a signal that isn't present on the source),
one writer (:func:`record_evidence`, idempotent/content-addressed — the SAME
dedup discipline ``candidate_insight._stable_id`` uses elsewhere in this
package), and one reader (:func:`gather_evidence`) that channels genuinely
missing a KG-queryable source (b/d/e) do not need — those are recorded AT
THEIR OWN CALL SITE instead of re-derived by a second query, so there is
still only ONE writer per fact, never a shadow copy.

Propose-only, like everything else in this package: recording an
:class:`Evidence` node NEVER mutates, merges, or executes anything — it is a
read-mostly audit/dedup layer the loop (and :mod:`.candidate_insight`'s
governed Claim pipeline, via :func:`candidates_from_evidence`) reads from.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from agent_utilities.models.knowledge_graph import EvolutionEvidenceNode

from .candidate_insight import CandidateInsight

logger = logging.getLogger(__name__)

__all__ = [
    "EvidenceChannel",
    "EvidenceOutcome",
    "Evidence",
    "record_evidence",
    "gather_evidence",
    "candidates_from_evidence",
    "evidence_lineage",
    "from_outcome_evaluation",
    "from_optimization_result",
    "from_health_anomaly",
    "from_candidate_insight",
    "from_process_signal",
]

#: Finding type routed through the SAME CandidateInsight -> ClaimNode ->
#: Validation -> Action-gate pipeline every other mined finding uses (see
#: ``candidate_insight._FINDING_TYPES``) — registered there, not re-derived.
_EVIDENCE_FINDING_TYPE = "EvidenceSignal"

#: A directly-observed outcome (success/failure/degraded — not an inference)
#: carries maximal confidence in the OBSERVATION itself; the payload's own
#: ``signal``/reward is the separate, real magnitude — never conflated.
_OBSERVED_CONFIDENCE = 1.0


class EvidenceChannel(StrEnum):
    """The five sources this contract normalises (CONCEPT:AU-KG.evolution.unified-evidence-resource)."""

    EXECUTION_TRACE = "execution_trace"
    OPTIMIZATION_SIGNAL = "optimization_signal"
    GRAPH_HEALTH = "graph_health"
    RESEARCH_FINDING = "research_finding"
    PROCESS_SIGNAL = "process_signal"


class EvidenceOutcome(StrEnum):
    """What the evidence says happened — never fabricated, always read off the source."""

    SUCCESS = "success"
    FAILURE = "failure"
    DEGRADED = "degraded"
    ANOMALOUS = "anomalous"
    PROPOSED = "proposed"


#: Outcomes worth routing into the governed Claim pipeline (a below-floor/positive
#: outcome is recorded for lineage but never proposed as a reviewable finding —
#: mirrors ``CandidateInsight.clears_floor``'s "counted but not promoted" stance).
_CLAIM_WORTHY_OUTCOMES = frozenset(
    {EvidenceOutcome.FAILURE, EvidenceOutcome.DEGRADED, EvidenceOutcome.ANOMALOUS}
)


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class Evidence:
    """One normalized unit of evidence (CONCEPT:AU-KG.evolution.unified-evidence-resource).

    ``signal`` is the channel's own real magnitude in ``[0, 1]`` (a reward, a
    saturated anomaly score, a quality metric) — the SAME units
    ``observability.trace_ontology.outcome_properties``'s ``reward`` already
    uses, so evidence is comparable across channels. ``confidence`` is how
    reliable the OBSERVATION is (a directly-observed pass/fail is maximally
    confident regardless of how bad ``signal`` is; a heuristic/inferred
    channel's confidence is its own real derivation — never a fabricated
    mid-range guess, mirroring ``EvidenceBundle``'s no-fabrication contract).
    """

    channel: EvidenceChannel
    subject_id: str
    outcome: EvidenceOutcome
    signal: float
    confidence: float
    source_node_id: str | None = None
    source_node_type: str | None = None
    occurred_at: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    lineage: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.channel = EvidenceChannel(self.channel)
        self.outcome = EvidenceOutcome(self.outcome)
        if not self.subject_id:
            raise ValueError("Evidence.subject_id must be non-empty")
        self.signal = _clamp01(self.signal)
        self.confidence = _clamp01(self.confidence)
        if not self.occurred_at:
            self.occurred_at = _now_iso()

    @property
    def evidence_id(self) -> str:
        """Content-addressed id — same identifying parts ⇒ same id every gather
        pass, so re-recording the same source event UPSERTS rather than
        duplicates (the "deduplicate evidence in the graph" loop discipline)."""
        parts = [
            self.channel.value,
            self.subject_id,
            self.source_node_id or "",
            self.occurred_at,
        ]
        payload = json.dumps(parts, sort_keys=True, default=str)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return f"evolution_evidence:{digest}"

    def to_node(self) -> EvolutionEvidenceNode:
        return EvolutionEvidenceNode(
            id=self.evidence_id,
            name=f"{self.channel.value}:{self.subject_id}"[:200],
            channel=self.channel.value,
            outcome=self.outcome.value,
            subject_id=self.subject_id,
            signal=self.signal,
            confidence=self.confidence,
            occurred_at=self.occurred_at,
            source_node_id=self.source_node_id,
            source_node_type=self.source_node_type,
            payload=dict(self.payload),
            lineage=dict(self.lineage),
        )

    def to_candidate_insight(self) -> CandidateInsight | None:
        """Materialize as a reviewable finding through the EXISTING C4 pipeline —
        ONLY for claim-worthy outcomes (failure/degraded/anomalous); a clean
        success/proposed signal is recorded for lineage but never proposed
        (mirrors ``CandidateInsight.clears_floor``'s "counted, not promoted").
        """
        if self.outcome not in _CLAIM_WORTHY_OUTCOMES:
            return None
        statement = (
            f"{self.channel.value} evidence for {self.subject_id}: "
            f"outcome={self.outcome.value} signal={self.signal:.3f}"
        )
        # The evidence node itself MUST be a source id. ``register_claim_materialization``
        # writes one ``(Claim)-[:DERIVED_FROM]->(source_id)`` edge per entry here,
        # and that is the exact edge :func:`evidence_lineage` walks to find the
        # claims a piece of evidence produced. Omitting it left the lineage query
        # structurally unable to match anything, so the chain reported
        # "evidence, no claims" for evidence that HAD been promoted.
        source_ids = [self.evidence_id]
        if self.subject_id and self.subject_id not in source_ids:
            source_ids.append(self.subject_id)
        if self.source_node_id and self.source_node_id not in source_ids:
            source_ids.append(self.source_node_id)
        return CandidateInsight(
            finding_type=_EVIDENCE_FINDING_TYPE,
            finding_id=self.evidence_id,
            statement=statement,
            confidence=self.confidence,
            payload={
                "channel": self.channel.value,
                "outcome": self.outcome.value,
                "signal": self.signal,
                **self.payload,
            },
            source_ids=source_ids,
        )


# --------------------------------------------------------------------------- #
# Per-channel adapters — real native shape -> Evidence, never fabricated.
# --------------------------------------------------------------------------- #


def from_outcome_evaluation(props: dict[str, Any]) -> Evidence:
    """(a) execution_trace — an ``:OutcomeEvaluation`` row's properties
    (``observability.trace_ontology.outcome_properties``'s shape:
    ``trace_id``/``status``/``reward``/``success``/``timestamp``) -> Evidence.

    A directly-observed pass/fail/degraded transition carries
    :data:`_OBSERVED_CONFIDENCE` — this IS the ground truth the reward chain
    was hardened to protect (7.2): ``status == "degraded"`` never reads as a
    clean success here, matching ``trace_ontology.outcome_properties``'s own
    ``reward = 0.25`` (never ``1.0``) for a degraded run.
    """
    status = str(props.get("status") or "").lower()
    success = bool(props.get("success"))
    if status == "degraded":
        outcome = EvidenceOutcome.DEGRADED
    elif success:
        outcome = EvidenceOutcome.SUCCESS
    else:
        outcome = EvidenceOutcome.FAILURE
    trace_id = str(props.get("trace_id") or props.get("id") or "")
    return Evidence(
        channel=EvidenceChannel.EXECUTION_TRACE,
        subject_id=trace_id or "unknown-trace",
        outcome=outcome,
        signal=props.get("reward"),
        confidence=_OBSERVED_CONFIDENCE,
        source_node_id=str(props.get("id") or "") or None,
        source_node_type="OutcomeEvaluation",
        occurred_at=str(props.get("timestamp") or ""),
        payload={k: v for k, v in props.items() if k not in {"trace_id", "id"}},
        lineage={"trace_id": trace_id} if trace_id else {},
    )


def from_optimization_result(target_name: str, result: dict[str, Any]) -> Evidence:
    """(b) optimization_signal — one ``run_component_optimization`` target's
    result dict (``{"status", "result"?, "error_code"?, ...}``) -> Evidence.

    ``status == "error"`` is FAILURE (never silently dropped); ``"optimized"``/
    ``"proposed"`` is SUCCESS (a candidate was found, even if not yet promoted —
    promotion itself stays behind ``should_promote``, unaffected by this
    lineage record); anything else (``"no_data"``, an unrecognised status) is
    PROPOSED — neither a success nor a failure signal, just "nothing to learn
    from yet", never fabricated into either.
    """
    status = str(result.get("status") or "").lower()
    if status == "error":
        outcome = EvidenceOutcome.FAILURE
        signal = 0.0
    elif status in {"optimized", "proposed"}:
        outcome = EvidenceOutcome.SUCCESS
        signal = 1.0
    else:
        outcome = EvidenceOutcome.PROPOSED
        signal = 0.0
    return Evidence(
        channel=EvidenceChannel.OPTIMIZATION_SIGNAL,
        subject_id=str(target_name or "unknown-target"),
        outcome=outcome,
        signal=signal,
        confidence=_OBSERVED_CONFIDENCE,
        source_node_type="ProgramOptimizeResult",
        payload={
            k: v
            for k, v in result.items()
            if k in {"status", "error_code", "duration_s", "backend", "metric"}
        },
    )


def from_health_anomaly(props: dict[str, Any]) -> Evidence:
    """(c) graph_health — a ``:HealthAnomaly`` row's properties
    (``observability.health_ingest.ingest_health_anomaly``'s shape:
    ``entity``/``signal``/``kind``/``zscore``/``observed``/``expected``/
    ``observedAt``) -> Evidence.

    Confidence is the SAME ``|zscore| / 5`` saturation
    ``candidate_insight.candidates_from_anomalies`` already uses for the
    mining-flywheel's own capability-coverage anomaly finding — kept
    identical deliberately so an anomaly's claim-worthiness reads the same
    regardless of which pass mined it.
    """
    try:
        zscore = abs(float(props.get("zscore") or 0.0))
    except (TypeError, ValueError):
        zscore = 0.0
    entity = str(props.get("entity") or "")
    return Evidence(
        channel=EvidenceChannel.GRAPH_HEALTH,
        subject_id=entity or "unknown-entity",
        outcome=EvidenceOutcome.ANOMALOUS,
        signal=_clamp01(zscore / 5.0),
        confidence=_clamp01(zscore / 5.0),
        source_node_id=str(props.get("id") or "") or None,
        source_node_type="HealthAnomaly",
        occurred_at=str(props.get("observedAt") or ""),
        payload={
            k: v
            for k, v in props.items()
            if k in {"signal", "kind", "zscore", "observed", "expected"}
        },
    )


def from_candidate_insight(
    cand: CandidateInsight,
    *,
    governance_valid: bool | None = None,
    action_decision: str | None = None,
    promoted: bool | None = None,
) -> Evidence:
    """(d) research_finding — a mined :class:`CandidateInsight` (association
    rule / anomaly / predicted edge / sequential pattern / ops-causal / an
    evidence-derived finding) -> Evidence, one companion node per claim so
    every finding family's origin is uniformly queryable
    (:func:`evidence_lineage`), regardless of which mining pass produced it.

    Outcome reflects the SAME governance/action-gate result
    ``loop_controller._run_insight_validation`` already computed for this
    candidate — a denied/rejected claim is still recorded (never silently
    dropped; negative results stay queryable, CONCEPT:AU-KG.evolution.unified-evidence-resource
    lineage requirement).
    """
    if promoted:
        outcome = EvidenceOutcome.SUCCESS
    elif governance_valid is False or action_decision == "deny":
        outcome = EvidenceOutcome.FAILURE
    else:
        outcome = EvidenceOutcome.PROPOSED
    return Evidence(
        channel=EvidenceChannel.RESEARCH_FINDING,
        subject_id=cand.finding_id,
        outcome=outcome,
        signal=cand.confidence,
        confidence=cand.confidence,
        source_node_id=cand.claim_id,
        source_node_type="Claim",
        payload={
            "finding_type": cand.finding_type,
            "governance_valid": governance_valid,
            "action_decision": action_decision,
        },
        lineage={"claim_id": cand.claim_id},
    )


def from_process_signal(evidence_dict: dict[str, Any]) -> Evidence:
    """(e) process_signal — an OCEL/tEKG import's projection summary
    (``mcp/tools/engine_surface_tools.py``'s ``process`` mining action builds
    exactly this shape: ``{"mode", "tenant", "content_hash", "idempotency_key",
    "mapping_version", "node_count", "relationship_count"}``) -> Evidence.

    Always PROPOSED (never SUCCESS/FAILURE): an OCEL import is a governed,
    propose-only ingest of process/event data, not a pass/fail judgement — the
    ``node_count``/``relationship_count`` it carries is descriptive volume, not
    a quality signal, so ``signal`` stays ``0.0`` rather than fabricating one
    from an unrelated count.
    """
    tenant = str(evidence_dict.get("tenant") or "")
    idem = str(evidence_dict.get("idempotency_key") or "")
    subject = idem or tenant or "unknown-import"
    return Evidence(
        channel=EvidenceChannel.PROCESS_SIGNAL,
        subject_id=subject,
        outcome=EvidenceOutcome.PROPOSED,
        signal=0.0,
        confidence=_OBSERVED_CONFIDENCE,
        source_node_type="ObjectCentricGraphSlice",
        payload={
            k: v
            for k, v in evidence_dict.items()
            if k
            in {
                "mode",
                "content_hash",
                "mapping_version",
                "node_count",
                "relationship_count",
            }
        },
        lineage={"tenant": tenant} if tenant else {},
    )


# --------------------------------------------------------------------------- #
# Writer / reader
# --------------------------------------------------------------------------- #


def record_evidence(engine: Any, evidence: Evidence) -> str | None:
    """Idempotently upsert one :class:`Evidence` as an ``:EvolutionEvidence``
    node (content-addressed id — a re-record of the SAME source event never
    duplicates), linking ``DERIVED_FROM`` to its source node when known.

    Best-effort (never raises into the caller's pipeline — mirrors every other
    writeback seam in this package, e.g. ``candidate_insight.
    register_claim_materialization``); logs and returns ``None`` on failure so
    a caller can tell recording didn't happen without a try/except of its own.
    """
    if engine is None:
        return None
    node = evidence.to_node()
    try:
        engine.add_node(
            node.id,
            "EvolutionEvidence",
            properties=node.model_dump(mode="json", exclude={"type"}),
        )
    except Exception as e:  # noqa: BLE001 — best-effort writeback
        logger.debug("evidence: record failed for %s: %s", node.id, e)
        return None
    if evidence.source_node_id:
        try:
            engine.add_edge(node.id, evidence.source_node_id, "DERIVED_FROM")
        except Exception as e:  # noqa: BLE001 — provenance edge is best-effort
            logger.debug(
                "evidence: DERIVED_FROM edge failed %s -> %s: %s",
                node.id,
                evidence.source_node_id,
                e,
            )
    return node.id


def _gather_execution_trace_evidence(engine: Any, limit: int) -> list[Evidence]:
    try:
        rows = (
            engine.query_cypher(
                "MATCH (r:RunTrace)-[:PRODUCED_OUTCOME]->(o:OutcomeEvaluation) "
                "WHERE o.reward IS NOT NULL "
                "RETURN o.id AS id, o.trace_id AS trace_id, o.status AS status, "
                "o.success AS success, o.reward AS reward, o.timestamp AS timestamp "
                "ORDER BY o.event_sequence DESC "
                f"LIMIT {int(limit)}",
                {},
            )
            or []
        )
    except Exception as e:  # noqa: BLE001 — a query failure degrades, never raises
        logger.debug("evidence: execution_trace gather failed: %s", e)
        return []
    out: list[Evidence] = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("trace_id"):
            continue
        out.append(from_outcome_evaluation(row))
    return out


def _gather_graph_health_evidence(engine: Any, limit: int) -> list[Evidence]:
    try:
        rows = (
            engine.query_cypher(
                "MATCH (a:HealthAnomaly) "
                "RETURN a.id AS id, a.entity AS entity, a.signal AS signal, "
                "a.kind AS kind, a.zscore AS zscore, a.observed AS observed, "
                "a.expected AS expected, a.observedAt AS observedAt "
                "ORDER BY a.observedAt DESC "
                f"LIMIT {int(limit)}",
                {},
            )
            or []
        )
    except Exception as e:  # noqa: BLE001 — a query failure degrades, never raises
        logger.debug("evidence: graph_health gather failed: %s", e)
        return []
    out: list[Evidence] = []
    for row in rows:
        if not isinstance(row, dict) or not row.get("entity"):
            continue
        out.append(from_health_anomaly(row))
    return out


#: Channels :func:`gather_evidence` can query live from the graph today.
#: (b) optimization_signal, (d) research_finding, and (e) process_signal have
#: no standing "list recent X" query source (b/e are recorded AT their own
#: call site — ``harness.program_optimization.run_optimization_sweep`` /
#: the OCEL import action; d is recorded inline by
#: ``loop_controller._run_insight_validation`` for each candidate it already
#: processes) — querying for them here would be a second, redundant system,
#: exactly what this module's docstring says to avoid.
_QUERYABLE_CHANNELS = (EvidenceChannel.EXECUTION_TRACE, EvidenceChannel.GRAPH_HEALTH)


def gather_evidence(
    engine: Any,
    *,
    channels: tuple[EvidenceChannel, ...] | None = None,
    limit: int = 100,
) -> list[Evidence]:
    """Gather + record fresh :class:`Evidence` for the channels that have a
    live KG query source, returning the normalized, deduplicated list.

    Idempotent: recording is a content-addressed upsert
    (:func:`record_evidence`), so calling this every loop cycle never
    duplicates a fact already seen — it just re-affirms it (the "deduplicate
    evidence in the graph" loop discipline, cheaply, without a second cursor
    mechanism).
    """
    wanted = set(channels) if channels else set(_QUERYABLE_CHANNELS)
    out: list[Evidence] = []
    if EvidenceChannel.EXECUTION_TRACE in wanted:
        out.extend(_gather_execution_trace_evidence(engine, limit))
    if EvidenceChannel.GRAPH_HEALTH in wanted:
        out.extend(_gather_graph_health_evidence(engine, limit))
    for ev in out:
        record_evidence(engine, ev)
    return out


def candidates_from_evidence(
    evidence_list: list[Evidence] | None,
) -> list[CandidateInsight]:
    """The evidence-channel fan-in mirror of ``candidate_insight.
    candidates_from_mine_discovery`` — every claim-worthy :class:`Evidence`
    (failure/degraded/anomalous) becomes a :class:`CandidateInsight`, fed
    into the SAME C4 governance pipeline as a mined association rule or
    anomaly (never a second promotion path)."""
    out: list[CandidateInsight] = []
    for ev in evidence_list or []:
        cand = ev.to_candidate_insight()
        if cand is not None:
            out.append(cand)
    return out


def evidence_lineage(engine: Any, evidence_id: str) -> dict[str, Any]:
    """(7.6) Walk the lineage chain forward from one ``:EvolutionEvidence``
    node: evidence -> the ``Claim`` proposal(s) derived from it (``DERIVED_FROM``,
    the same edge ``candidate_insight.register_claim_materialization`` writes)
    -> any downstream ``SpecProposal``/``SDDFeature`` (the gap -> spec ->
    develop -> publish chain, ``DERIVED_FROM_RESEARCH``/``SATISFIED_BY``).

    Negative results stay in this chain — a rejected/retracted ``Claim`` is
    never deleted (``claim_flywheel.reject``/``retract`` only change its
    ``status``), so a vetoed proposal remains queryable here, not silently
    dropped. Best-effort at every hop: a query failure degrades that hop to
    an empty list rather than aborting the whole chain.
    """
    if engine is None or not evidence_id:
        return {"evidence_id": evidence_id, "found": False, "chain": []}

    def _rows(query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            return list(engine.query_cypher(query, params) or [])
        except Exception as e:  # noqa: BLE001 — one bad hop never aborts the chain
            logger.debug("evidence_lineage: query failed: %s", e)
            return []

    ev_rows = _rows(
        "MATCH (e:EvolutionEvidence {id: $id}) "
        "RETURN e.id AS id, e.channel AS channel, e.outcome AS outcome, "
        "e.signal AS signal, e.subject_id AS subject_id, "
        "e.occurred_at AS occurred_at LIMIT 1",
        {"id": evidence_id},
    )
    if not ev_rows:
        return {"evidence_id": evidence_id, "found": False, "chain": []}

    chain: list[dict[str, Any]] = [{"stage": "evidence", **ev_rows[0]}]

    claim_rows = _rows(
        "MATCH (c:Claim)-[:DERIVED_FROM]->(e:EvolutionEvidence {id: $id}) "
        "RETURN c.id AS id, c.claim_text AS claim_text, c.status AS status, "
        "c.confidence AS confidence, c.is_verified AS is_verified "
        f"LIMIT {50}",
        {"id": evidence_id},
    )
    for claim in claim_rows:
        chain.append({"stage": "claim_proposal", **claim})
        claim_id = claim.get("id")
        if not claim_id:
            continue
        proposal_rows = _rows(
            "MATCH (s)-[:DERIVED_FROM_RESEARCH|SATISFIED_BY]->(c:Claim {id: $cid}) "
            "RETURN s.id AS id, s.status AS status "
            f"LIMIT {20}",
            {"cid": claim_id},
        )
        for proposal in proposal_rows:
            chain.append({"stage": "proposal", "claim_id": claim_id, **proposal})

    return {"evidence_id": evidence_id, "found": True, "chain": chain}
