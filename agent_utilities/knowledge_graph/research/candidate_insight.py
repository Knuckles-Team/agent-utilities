#!/usr/bin/python
from __future__ import annotations

"""Mined finding → reviewable Claim (CONCEPT:AU-KG.evolution.insight-engine-closed-loop, workstream C4).

The discovery-flywheel mining pass (``loop_controller._run_mine_discovery``,
CONCEPT:AU-KG.evolution.mining-flywheel) writes back typed, descriptive KG facts —
``:AssociationRule`` / ``:Anomaly`` / ``:PredictedEdge`` nodes — for a human/agent to
review later. Nothing closes the loop from "mined fact" to "a claim the rest of the
epistemic substrate (C1 EvidenceBundle, C2 belief-revision, the AHE-3.20 promotion
governance stack) can reason about". This module is that seam:

    mined finding → :class:`CandidateInsight` → :class:`~agent_utilities.models.
    evidence_bundle.EvidenceBundle` (evidence) → :class:`~agent_utilities.models.
    knowledge_graph.ClaimNode` (a reviewable assertion)

Deliberately conservative, mirroring the ``EvidenceBundle`` no-fabrication
contract (``evidence_bundle.py`` module docstring):

* A finding's ``confidence`` is derived ONLY from a real signal already present
  on the mined payload (an association rule's mined ``confidence``, an anomaly's
  z-score-like ``anomaly_score`` saturated into ``[0, 1]``, a predicted edge's
  ``score``/``confidence``/``probability`` field) — never invented. A finding
  with no such signal gets ``confidence=0.0`` (unverified), not a guessed
  mid-range default.
* :data:`CONFIDENCE_FLOOR` gates whether a finding is even claim-worthy — below
  it, :func:`candidates_from_mine_discovery` still returns the
  :class:`CandidateInsight` (so callers can report on it / count it), but
  ``clears_floor`` is ``False`` and the loop-controller stage
  (``loop_controller._run_insight_validation``) MUST NOT materialize a
  ``ClaimNode`` for it. This module never promotes anything by itself — see
  the safety notes in ``loop_controller.py`` and ``deploy/action-policy.default.yml``
  for the actual gate (``action_policy.decide(kind="promote_mined_claim")``).
* Claim ids are content-addressed (stable hash of the finding's identifying
  fields) so re-mining the same fact across cycles upserts the same
  ``ClaimNode`` instead of duplicating it — the same idempotency discipline
  ``loop_controller`` uses everywhere else (the assimilation watermark, the
  belief-revision proposal id).
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from agent_utilities.models.evidence_bundle import EvidenceBundle
from agent_utilities.models.knowledge_graph import ClaimNode, RegistryNodeType

__all__ = [
    "CONFIDENCE_FLOOR",
    "CandidateInsight",
    "candidates_from_association_rules",
    "candidates_from_anomalies",
    "candidates_from_predicted_edges",
    "candidates_from_sequential_patterns",
    "candidates_from_mine_discovery",
]

#: Conservative default floor a mined finding's confidence must clear before it
#: is even materialized as a ``ClaimNode`` (never mind promoted — promotion is
#: separately gated by ``action_policy.decide()``, always). Mirrors
#: ``auto_merge.DEFAULT_QUALITY_THRESHOLD``'s "a floor, not a target" philosophy,
#: kept lower here because a Claim is a much smaller commitment (a reviewable
#: assertion, not a promoted artifact) than a golden-loop TeamSpec merge.
CONFIDENCE_FLOOR = 0.6

#: "SequentialPattern" (workstream C6, ``trace_pattern_miner``) is the
#: ``graph_mine action="sequence"`` mining surface's finding kind — a repeated
#: FAILURE tool-call sequence mined from ``AgentTask``/outcome provenance,
#: fed through this SAME CandidateInsight→Claim pipeline as the other three.
_FINDING_TYPES = {"AssociationRule", "Anomaly", "PredictedEdge", "SequentialPattern"}


def _stable_id(prefix: str, *parts: Any) -> str:
    """Content-addressed id: same identifying parts ⇒ same id across cycles."""
    payload = json.dumps(list(parts), sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _clamp01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


@dataclass
class CandidateInsight:
    """One mined finding, on its way to becoming a reviewable ``ClaimNode``.

    Wraps a single ``:AssociationRule`` / ``:Anomaly`` / ``:PredictedEdge``
    finding (as surfaced by ``loop_controller._run_mine_discovery``'s summary
    dicts) with the quality signal + raw payload needed to (a) decide whether
    it clears :data:`CONFIDENCE_FLOOR` and (b) materialize a ``ClaimNode`` +
    ``EvidenceBundle`` for governance review.
    """

    finding_type: str  # "AssociationRule" | "Anomaly" | "PredictedEdge"
    finding_id: str
    statement: str
    confidence: float
    payload: dict[str, Any] = field(default_factory=dict)
    source_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.confidence = _clamp01(self.confidence)
        if self.finding_type not in _FINDING_TYPES:
            raise ValueError(f"unknown finding_type: {self.finding_type!r}")

    @property
    def clears_floor(self) -> bool:
        """``True`` when this finding is claim-worthy (>= :data:`CONFIDENCE_FLOOR`)."""
        return self.confidence >= CONFIDENCE_FLOOR

    @property
    def claim_id(self) -> str:
        return f"claim:insight:{self.finding_id}"

    def to_claim_node(self) -> ClaimNode:
        """Materialize as a ``ClaimNode`` — ALWAYS unverified (never self-verifying).

        ``confidence`` on the claim is exactly the mining quality score (never
        re-derived or inflated); ``is_verified`` stays ``False`` here regardless
        of the score — verification is the governance/action-policy gate's job,
        not this constructor's.
        """
        return ClaimNode(
            id=self.claim_id,
            type=RegistryNodeType.CLAIM,
            name=self.statement[:120] or self.finding_id,
            claim_text=self.statement,
            confidence=self.confidence,
            claim_type="finding",
            source_ids=list(self.source_ids) or [self.finding_id],
            extracted_from=self.finding_id,
            domain="mining_flywheel",
            is_verified=False,
            metadata={"finding_type": self.finding_type},
            importance_score=self.confidence,
        )

    def to_evidence_bundle(self) -> EvidenceBundle:
        """Package the raw finding as an :class:`EvidenceBundle` (C1) for the audit trail.

        No fabrication: ``confidence`` is the same real mining signal as the
        claim's; ``evidence_spans`` carries the untouched mined payload so a
        reviewer can see exactly what was mined, not just the templated
        ``statement``.
        """
        return EvidenceBundle(
            answer_candidate=self.statement,
            claims=[{"id": self.finding_id, "text": self.statement}],
            evidence_spans=[
                {"id": self.finding_id, "type": self.finding_type, **self.payload}
            ],
            confidence=self.confidence,
            reasoning_trace=[
                {
                    "step": "mine_discovery",
                    "finding_type": self.finding_type,
                    "payload": self.payload,
                }
            ],
        )


# ── per-finding-type extraction (see loop_controller._run_mine_discovery docstring
#    for the exact shape of each ``examples`` entry) ──────────────────────────────


def candidates_from_association_rules(
    result: dict[str, Any] | None,
) -> list[CandidateInsight]:
    """``{"examples": [{"antecedent", "consequent", "confidence", "lift"}, ...]}`` → insights.

    An association rule's mined ``confidence`` (already a [0,1] probability —
    "X co-occurs with Y with confidence C") maps straight through as the
    finding's confidence — no transform needed.
    """
    out: list[CandidateInsight] = []
    for ex in (result or {}).get("examples") or []:
        if not isinstance(ex, dict):
            continue
        antecedent = ex.get("antecedent")
        consequent = ex.get("consequent")
        statement = (
            f"{antecedent} ⇒ {consequent} "
            f"(confidence={ex.get('confidence')}, lift={ex.get('lift')})"
        )
        source_ids = [str(x) for x in (antecedent, consequent) if x]
        out.append(
            CandidateInsight(
                finding_type="AssociationRule",
                finding_id=_stable_id("assoc", antecedent, consequent),
                statement=statement,
                confidence=_clamp01(ex.get("confidence")),
                payload=dict(ex),
                source_ids=source_ids,
            )
        )
    return out


def candidates_from_anomalies(result: dict[str, Any] | None) -> list[CandidateInsight]:
    """``{"examples": [{"capability", "covered_concepts", "anomaly_score"}, ...]}`` → insights.

    ``anomaly_score`` is a z-score-like coverage-divergence proxy (see
    ``loop_controller._mine_capability_anomalies``), not a calibrated
    probability — saturated into ``[0, 1]`` via ``|score| / 5`` (a |z| >= 5
    divergence reads as maximally confident; a borderline one stays low/below
    the floor). Documented simplification, mirrors the mining pass's own
    "defensible simplification" stance.
    """
    out: list[CandidateInsight] = []
    for ex in (result or {}).get("examples") or []:
        if not isinstance(ex, dict):
            continue
        cap = ex.get("capability")
        if not cap:
            continue
        statement = (
            f"Capability {cap} is a coverage-divergence outlier "
            f"(covered_concepts={ex.get('covered_concepts')}, "
            f"anomaly_score={ex.get('anomaly_score')})"
        )
        try:
            raw_score = abs(float(ex.get("anomaly_score") or 0.0))
        except (TypeError, ValueError):
            raw_score = 0.0
        out.append(
            CandidateInsight(
                finding_type="Anomaly",
                finding_id=_stable_id("anomaly", cap),
                statement=statement,
                confidence=_clamp01(raw_score / 5.0),
                payload=dict(ex),
                source_ids=[str(cap)],
            )
        )
    return out


def candidates_from_predicted_edges(
    result: dict[str, Any] | None,
) -> list[CandidateInsight]:
    """``{"examples": [{...}]}`` (``graph_learn`` predicted-edge rows) → insights.

    The engine's predicted-edge row shape is not part of this repo's own
    contract (it comes from the Rust ``graph_learn`` surface) so this reads
    whichever of ``score``/``confidence``/``probability`` is present,
    defensively. Absent any of those, confidence is ``0.0`` — NEVER a fabricated
    mid-range guess (mirrors the ``EvidenceBundle`` no-fabrication contract).
    """
    out: list[CandidateInsight] = []
    for ex in (result or {}).get("examples") or []:
        if not isinstance(ex, dict):
            continue
        src = ex.get("source") or ex.get("src") or ex.get("from")
        dst = ex.get("target") or ex.get("dst") or ex.get("to")
        confidence = 0.0
        for key in ("score", "confidence", "probability"):
            if ex.get(key) is not None:
                confidence = _clamp01(ex.get(key))
                break
        statement = f"Predicted concept relation: {src} → {dst}"
        out.append(
            CandidateInsight(
                finding_type="PredictedEdge",
                finding_id=_stable_id("predicted_edge", src, dst),
                statement=statement,
                confidence=confidence,
                payload=dict(ex),
                source_ids=[str(x) for x in (src, dst) if x],
            )
        )
    return out


def candidates_from_sequential_patterns(
    result: dict[str, Any] | None,
) -> list[CandidateInsight]:
    """``{"patterns": [{"items", "support", "count"}, ...]}`` → insights (workstream C6).

    Mirrors the ``graph_mine action="sequence"`` mining surface's own result
    shape (frequent ORDERED subsequences — see ``engine_surface_tools.
    graph_mine``'s docstring): ``items`` is the ordered tool-name sequence,
    ``support`` is already a ``[0, 1]`` frequency fraction (the same
    ``min_support`` units the mining call itself gates on) so it maps
    straight through as the finding's confidence — no transform needed,
    exactly like an association rule's mined ``confidence``. Used by
    :mod:`.trace_pattern_miner` to turn repeated FAILURE tool-call sequences
    into reviewable claims through this SAME pipeline.
    """
    out: list[CandidateInsight] = []
    for ex in (result or {}).get("patterns") or []:
        if not isinstance(ex, dict):
            continue
        items = ex.get("items") or []
        if not items:
            continue
        statement = (
            f"Repeated failure tool-call sequence: {' → '.join(str(i) for i in items)} "
            f"(support={ex.get('support')}, count={ex.get('count')})"
        )
        out.append(
            CandidateInsight(
                finding_type="SequentialPattern",
                finding_id=_stable_id("trace_pattern", items),
                statement=statement,
                confidence=_clamp01(ex.get("support")),
                payload=dict(ex),
                source_ids=[str(i) for i in items],
            )
        )
    return out


def candidates_from_mine_discovery(
    mine_result: dict[str, Any] | None,
) -> list[CandidateInsight]:
    """The full fan-out: a ``_run_mine_discovery`` report → every mined ``CandidateInsight``.

    ``mine_result`` is the dict ``_run_mine_discovery`` returns (``{"association_rules",
    "anomalies", "predicted_edges", "errors"}``); missing/malformed sections degrade to
    no candidates from that section rather than raising, matching the mining
    pass's own best-effort tolerance.
    """
    mine_result = mine_result or {}
    out: list[CandidateInsight] = []
    out.extend(candidates_from_association_rules(mine_result.get("association_rules")))
    out.extend(candidates_from_anomalies(mine_result.get("anomalies")))
    out.extend(candidates_from_predicted_edges(mine_result.get("predicted_edges")))
    return out
