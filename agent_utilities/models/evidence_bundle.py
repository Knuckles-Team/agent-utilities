#!/usr/bin/python
from __future__ import annotations

"""EvidenceBundle — a unified epistemic envelope over the KG's structured answers.

Epistemic Substrate Program, workstream C1. Today every retrieval surface returns
its own bespoke, well-typed shape — :class:`~agent_utilities.knowledge_graph.retrieval.
code_context.CodeContextAnswer`, :class:`~agent_utilities.knowledge_graph.retrieval.
executable_rag.RagResult`, the ``nl_to_query``/``nl_query`` payload dict — each grounded
in its own way (file:line citations, an execution trace, a generated+audited query).
:class:`EvidenceBundle` does NOT replace any of those: it is an additive, opt-in
WRAPPER that projects whichever shape a caller has into one common envelope, so a
downstream consumer (a synthesis step, a UI, a future engine-side reasoner) can
reason about "what evidence backs this answer, how fresh is it, does it conflict
with anything else, how confident should I be" the same way regardless of which
retrieval surface produced it.

The wrapping is deliberately conservative:

* Every ``from_*`` classmethod only ever POPULATES fields it has real signal for;
  everything else is left at its safe default (``[]``/``{}``/``None``) — never
  fabricated. This applies hardest to ``confidence``: none of today's retrieval
  surfaces computes a calibrated probability, so it is always ``None`` unless a
  caller explicitly threads one in.
* Nothing is silently dropped. Fields with no dedicated slot in the envelope
  (e.g. code_context's ``sections``/``capability_id``, nl_query's ``schema``) are
  folded into ``reasoning_trace`` as extra trace entries, so the full source
  payload is still recoverable from the bundle.

Concept: evidence-bundle-envelope
"""

import re
from dataclasses import asdict
from typing import Any

from pydantic import BaseModel, Field

from agent_utilities.knowledge_graph.adaptation.contradiction_detector import (
    Claim,
    ContradictionDetector,
)
from agent_utilities.models.company_brain import MergeStrategy

__all__ = ["EvidenceBundle"]

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> list[str]:
    """Split templated prose into non-empty, stripped sentences (deterministic)."""
    if not text or not text.strip():
        return []
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


def _claim_text(claim: dict[str, Any], fallback_id: str) -> tuple[str, str]:
    """Best-effort ``(id, text)`` extraction from a heterogeneous claim dict.

    Claims arriving from different sources have different shapes (a
    ``{"id", "text"}`` pair synthesized from prose, or a raw KG result row with
    no ``text`` field at all) — this normalizes whichever shape shows up into
    something :class:`Claim` (and thus :class:`ContradictionDetector`) can
    compare, without ever inventing content that is not in the dict.
    """
    cid = str(claim.get("id") or fallback_id)
    text = claim.get("text")
    if isinstance(text, str) and text.strip():
        return cid, text
    # No explicit "text" — fall back to the most name-like field, else a stable
    # (not model-generated) repr of the whole row so the detector still has
    # *something* concrete to compare.
    for key in ("name", "definition", "note", "answer"):
        val = claim.get(key)
        if isinstance(val, str) and val.strip():
            return cid, val
    return cid, str(claim)


def _scan_contradictions(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run :class:`ContradictionDetector` over ``claims`` and return plain dicts.

    Zero/one claims trivially scan to no findings (the detector only compares
    pairs) — this is always safe to call, even on an empty claim set.
    """
    if len(claims) < 2:
        return []
    detector_claims = [
        Claim(id=cid, text=text)
        for cid, text in (
            _claim_text(c, fallback_id=f"claim:{i}")
            for i, c in enumerate(claims)
            if isinstance(c, dict)
        )
    ]
    findings = ContradictionDetector().scan(detector_claims)
    return [asdict(f) for f in findings]


def _source_authority_from_citations(
    citations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate a ``source_system`` authority signal, only when one is present.

    Mirrors :class:`~agent_utilities.models.company_brain.MergeStrategy`'s
    ``SOURCE_AUTHORITY_WINS`` resolution: tallies how many citations came from
    each source system. Returns ``{}`` (never a fabricated ranking) when no
    citation carries a real ``source_system``.
    """
    counts: dict[str, int] = {}
    for c in citations:
        if not isinstance(c, dict):
            continue
        source_system = c.get("source_system")
        if source_system:
            counts[source_system] = counts.get(source_system, 0) + 1
    if not counts:
        return {}
    return {
        "strategy": MergeStrategy.SOURCE_AUTHORITY_WINS.value,
        "by_source_system": counts,
    }


class EvidenceBundle(BaseModel):
    """Unified epistemic envelope wrapping any of the KG's structured answers.

    Never a replacement for the wrapped answer — callers keep the original
    ``CodeContextAnswer``/``RagResult``/``nl_query`` payload and additionally get
    this common projection when they opt in (``envelope="bundle"`` on the MCP
    tools). See the module docstring for the no-fabrication contract.
    """

    answer_candidate: str = Field(
        default="", description="The best current natural-language answer string."
    )
    claims: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Atomic assertions the answer rests on (id/text at minimum).",
    )
    evidence_spans: list[dict[str, Any]] = Field(
        default_factory=list,
        description="The grounding evidence (citations / result rows / doc spans).",
    )
    source_authority: dict[str, Any] = Field(
        default_factory=dict,
        description="Source-authority resolution signal (empty when unknown).",
    )
    contradictions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="FrictionFinding-shaped conflicts detected across `claims`.",
    )
    confidence: float | None = Field(
        default=None,
        description="Calibrated confidence in [0, 1]. None when no real epistemic "
        "signal exists — NEVER a fabricated number.",
    )
    freshness: dict[str, Any] = Field(
        default_factory=dict,
        description="Coverage / staleness signal for the underlying evidence.",
    )
    policy_exclusions: list[str] = Field(
        default_factory=list,
        description="Policy-driven redactions/exclusions applied before this bundle.",
    )
    reasoning_trace: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Inspectable steps that produced the answer (primitives used, "
        "retrieval steps, generated queries, plus any source fields with no "
        "dedicated slot above — nothing is silently dropped).",
    )
    next_actions: list[str] = Field(
        default_factory=list,
        description="Concrete, grounded follow-ups (e.g. re-ingest, retry with a "
        "different mode) — only populated when the source signal actually implies one.",
    )

    # ------------------------------------------------------------------
    # CodeContextAnswer
    # ------------------------------------------------------------------
    @classmethod
    def from_code_context_answer(cls, ans: Any, **overrides: Any) -> EvidenceBundle:
        """Wrap ``build_code_context``'s output (a ``CodeContextAnswer`` or its ``as_dict()``).

        Reuses: ``citations`` -> ``evidence_spans``; ``coverage`` -> ``freshness``;
        ``used_primitives`` (plus ``sections``/``capability_id``/``intent``/
        ``cross_repo``/``query``, which have no dedicated slot) -> ``reasoning_trace``;
        the synthesized ``answer`` is sentence-split into ``claims`` and scanned for
        internal contradictions. ``confidence`` is always ``None`` — code_context is a
        deterministic, templated composition, not a scored estimate.
        """
        payload = ans.as_dict() if hasattr(ans, "as_dict") else dict(ans)

        answer = str(payload.get("answer") or "")
        citations = list(payload.get("citations") or [])
        used_primitives = list(payload.get("used_primitives") or [])
        coverage = dict(payload.get("coverage") or {})
        anchors = list(payload.get("anchors") or [])

        claims = [
            {"id": f"claim:{i}", "text": s} for i, s in enumerate(_sentences(answer))
        ]

        reasoning_trace: list[dict[str, Any]] = [
            {"primitive": p} for p in used_primitives
        ]
        reasoning_trace.append(
            {
                "step": "meta",
                "query": payload.get("query"),
                "intent": payload.get("intent"),
                "capability_id": payload.get("capability_id"),
                "cross_repo": payload.get("cross_repo"),
            }
        )
        if payload.get("sections"):
            reasoning_trace.append(
                {"step": "sections", "sections": payload.get("sections")}
            )

        next_actions: list[str] = []
        if not anchors:
            next_actions.append(
                "source_sync source=all mode=delta (re-ingest so this area resolves), "
                "or refine the query with a more specific symbol name."
            )

        fields: dict[str, Any] = {
            "answer_candidate": answer,
            "claims": claims,
            "evidence_spans": citations,
            "source_authority": _source_authority_from_citations(citations),
            "contradictions": _scan_contradictions(claims),
            "confidence": None,
            "freshness": coverage,
            "policy_exclusions": [],
            "reasoning_trace": reasoning_trace,
            "next_actions": next_actions,
        }
        fields.update(overrides)
        return cls(**fields)

    # ------------------------------------------------------------------
    # RagResult
    # ------------------------------------------------------------------
    @classmethod
    def from_rag_result(
        cls,
        res: Any,
        *,
        evidence: list[dict[str, Any]] | None = None,
        **overrides: Any,
    ) -> EvidenceBundle:
        """Wrap an executable-RAG :class:`RagResult`.

        ``RagResult`` only retains de-duped ``evidence_ids`` (not the retrieved
        content), so pass the raw retrieved ``evidence`` dicts (as seen by
        ``answer_fn``) when available for richer ``evidence_spans``/``claims`` —
        this degrades gracefully to id-only spans when omitted. ``trace`` (the
        ``StepTrace`` list) plus the final ``success`` flag map to
        ``reasoning_trace``. ``confidence`` is always ``None`` — ``success`` is a
        boolean compiler signal, not a calibrated probability, and turning it into
        one would be exactly the fabrication this envelope refuses to do.
        """
        answer = str(getattr(res, "answer", "") or "")
        evidence_ids = list(getattr(res, "evidence_ids", []) or [])
        trace = list(getattr(res, "trace", []) or [])
        success = bool(getattr(res, "success", False))

        if evidence is not None:
            evidence_spans = list(evidence)
            claims = [dict(e) for e in evidence if isinstance(e, dict)]
        else:
            evidence_spans = [{"id": eid} for eid in evidence_ids]
            claims = [
                {"id": f"claim:{i}", "text": s}
                for i, s in enumerate(_sentences(answer))
            ]

        reasoning_trace = [
            st.model_dump() if hasattr(st, "model_dump") else dict(st) for st in trace
        ]
        reasoning_trace.append({"step": "final", "success": success})

        next_actions: list[str] = []
        if not success:
            next_actions.append(
                "insufficient evidence — consider re-running with a boosted top_k "
                "or an additional retrieval mode."
            )

        fields: dict[str, Any] = {
            "answer_candidate": answer,
            "claims": claims,
            "evidence_spans": evidence_spans,
            "source_authority": {},
            "contradictions": _scan_contradictions(claims),
            "confidence": None,
            "freshness": {},
            "policy_exclusions": [],
            "reasoning_trace": reasoning_trace,
            "next_actions": next_actions,
        }
        fields.update(overrides)
        return cls(**fields)

    # ------------------------------------------------------------------
    # nl_query / nl_to_query payload
    # ------------------------------------------------------------------
    @classmethod
    def from_nl_query(cls, payload: dict[str, Any], **overrides: Any) -> EvidenceBundle:
        """Wrap the ``nl_to_query``/``nl_planner.nl_query`` result dict.

        ``nl_to_query`` returns ``{question, dialect, generated_query, results,
        row_count, citations, schema}``; ``nl_planner.nl_query`` returns the same
        shape but keys the question as ``request`` and adds ``planner`` — both are
        accepted. Each KG result row IS an atomic fact, so ``results`` maps
        straight through to ``claims``; the bare provenance-id strings in
        ``citations`` become minimal ``evidence_spans``. There is no prose
        "answer" field in this payload, so ``answer_candidate`` is a
        deterministic, templated restatement of the row count — never an
        invented summary. ``confidence`` is always ``None``.
        """
        error = payload.get("error")
        results = list(payload.get("results") or [])
        row_count = payload.get("row_count", len(results))
        citations = list(payload.get("citations") or [])
        question = str(payload.get("question") or payload.get("request") or "")

        if error:
            answer_candidate = ""
        elif results or "results" in payload:
            answer_candidate = f"{row_count} row(s) for: {question}".strip()
        else:
            answer_candidate = ""

        claims = [c for c in results if isinstance(c, dict)]
        evidence_spans = [{"ref": c} for c in citations]

        reasoning_trace: list[dict[str, Any]] = [
            {
                "step": "nl_query",
                "question": question,
                "dialect": payload.get("dialect"),
                "generated_query": payload.get("generated_query"),
                "schema": payload.get("schema"),
                "planner": payload.get("planner"),
            }
        ]
        if error:
            reasoning_trace.append({"step": "error", "error": error})

        next_actions: list[str] = []
        if error:
            next_actions.append(
                "review the generated_query / retry with an explicit dialect."
            )

        fields: dict[str, Any] = {
            "answer_candidate": answer_candidate,
            "claims": claims,
            "evidence_spans": evidence_spans,
            "source_authority": {},
            "contradictions": _scan_contradictions(claims),
            "confidence": None,
            "freshness": {},
            "policy_exclusions": [],
            "reasoning_trace": reasoning_trace,
            "next_actions": next_actions,
        }
        fields.update(overrides)
        return cls(**fields)

    # ------------------------------------------------------------------
    # Engine wire (future epistemic-graph KnowledgeSet/EvidenceBundle)
    # ------------------------------------------------------------------
    @classmethod
    def from_engine_wire(cls, ws: dict[str, Any]) -> EvidenceBundle:
        """Map a future engine-side ``KnowledgeSet``/``EvidenceBundle`` wire dict.

        NOTE: the engine-side wire type is being designed separately (a different
        workstream of the Epistemic Substrate Program); its exact field names are
        not final yet. Until that lands, this is a minimal, best-effort passthrough
        that assumes the wire dict already uses (a subset of) this class's own
        field names — safe because every lookup defaults to this class's own safe
        defaults, so an unrecognized/partial payload degrades cleanly rather than
        raising. Replace with a real 1:1 mapping once the engine wire schema is
        finalized.
        """
        return cls(
            answer_candidate=str(ws.get("answer_candidate") or ws.get("answer") or ""),
            claims=list(ws.get("claims") or []),
            evidence_spans=list(ws.get("evidence_spans") or ws.get("evidence") or []),
            source_authority=dict(ws.get("source_authority") or {}),
            contradictions=list(ws.get("contradictions") or []),
            confidence=ws.get("confidence"),
            freshness=dict(ws.get("freshness") or {}),
            policy_exclusions=list(ws.get("policy_exclusions") or []),
            reasoning_trace=list(ws.get("reasoning_trace") or []),
            next_actions=list(ws.get("next_actions") or []),
        )
