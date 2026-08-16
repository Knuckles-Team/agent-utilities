#!/usr/bin/python
from __future__ import annotations

"""EvidenceBundle — a unified epistemic envelope over the KG's structured answers.

Epistemic Substrate Program, workstream C1. Today every retrieval surface returns
its own bespoke, well-typed shape — :class:`~agent_utilities.knowledge_graph.retrieval.
code_context.CodeContextAnswer`, :class:`~agent_utilities.knowledge_graph.retrieval.
executable_rag.RagResult`, the ``nl_to_query``/``nl_query`` payload dict — each grounded
in its own way (file:line citations, an execution trace, a generated+audited query).
:class:`EvidenceBundle` is the current public response contract for graph query and
analysis surfaces. It projects whichever internal shape produced an answer into one
common envelope, so a
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

import json
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


# --- U-124 / U-131: contradiction analysis is opt-in, never generic-row-wide ---
#
# Different values in independent query rows are NOT contradictions. Sibling
# metadata rows (e.g. three ``SourceArtifact`` rows differing only in
# path/size/hash) were being synthesized into a generic templated repr (via
# :func:`_claim_text`'s ``str(claim)`` fallback) and compared as if that repr
# were a real assertion — schema-similar rows land at ~0.81-0.91 lexical
# similarity and, on any differing number, trip :func:`opposes`'s numeric
# rule. A row is now eligible for contradiction analysis ONLY when it
# explicitly declares how it should be compared:
#
# * **Natural-language** — a ``semantic_role`` drawn from the closed
#   assertion-role set below, PLUS a non-empty ``text`` (see
#   :func:`_is_nl_assertion`). Ordinary query/result rows never carry
#   ``semantic_role`` at all, so they are excluded by construction — never by
#   guessing at their shape.
# * **Structured** — a stable ``comparison_key``/``contradiction_key`` PLUS
#   either ``comparison_mode`` in {"exclusive", "single_value"} or an explicit
#   ``mutually_exclusive: true`` (see :func:`_is_structured_comparable`).
#   Shared column names, differing hashes/sizes/timestamps, or sibling
#   ``kind``s are NEVER, by themselves, inferred as declaring exclusivity.
#
# Every row in ``claims`` is retained regardless of eligibility — this stage
# only ever produces findings; it never filters the bundle's ``claims``.
_NL_ASSERTION_ROLES = frozenset(
    {"assertion", "belief", "claim", "statement", "opinion"}
)
_STRUCTURED_COMPARISON_MODES = frozenset({"exclusive", "single_value"})


def _is_nl_assertion(claim: dict[str, Any]) -> bool:
    """True when ``claim`` explicitly opts into natural-language contradiction scanning.

    Requires BOTH a ``semantic_role`` in the closed assertion-role set AND a
    non-empty ``text`` — never inferred from any other field (name/definition/
    note/answer), unlike :func:`_claim_text`'s display-oriented fallback.
    """
    if claim.get("semantic_role") not in _NL_ASSERTION_ROLES:
        return False
    text = claim.get("text")
    return isinstance(text, str) and bool(text.strip())


def _structured_comparison_key(claim: dict[str, Any]) -> str | None:
    """The claim's declared comparison domain, or ``None`` when absent/blank."""
    key = claim.get("comparison_key") or claim.get("contradiction_key")
    return str(key) if isinstance(key, str) and key.strip() else None


def _is_structured_comparable(claim: dict[str, Any]) -> bool:
    """True when ``claim`` declares a shared comparison domain AND mutual exclusivity.

    A stable comparison key alone is not enough — the row must also assert
    that values under that key are mutually exclusive (``comparison_mode`` in
    {"exclusive", "single_value"}) or ``mutually_exclusive: true``.
    """
    if _structured_comparison_key(claim) is None:
        return False
    if bool(claim.get("mutually_exclusive")):
        return True
    return claim.get("comparison_mode") in _STRUCTURED_COMPARISON_MODES


def _canonicalize_comparison_value(value: Any) -> Any:
    """Canonicalize a structured comparison value while preserving its polarity.

    Booleans stay booleans (checked before the numeric branch so ``True`` is
    never conflated with ``1.0``), numbers normalize to ``float``, strings
    trim/lowercase for whitespace/case-insensitive comparison, and any other
    type compares as-is.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return value.strip().lower()
    return value


def _structured_comparison_value(claim: dict[str, Any]) -> Any:
    """The claim's declared comparison value (``comparison_value``, else ``value``)."""
    if "comparison_value" in claim:
        return claim.get("comparison_value")
    return claim.get("value")


def _scan_nl_contradictions(
    indexed_claims: list[tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Lexical opposition scan restricted to explicit natural-language assertions.

    Builds :class:`Claim` objects only from rows that opt in via
    :func:`_is_nl_assertion`, then delegates to the existing
    :class:`ContradictionDetector`. Every finding is tagged with the
    comparison rule that produced it.
    """
    eligible = [
        Claim(id=cid, text=claim["text"])
        for cid, claim in indexed_claims
        if _is_nl_assertion(claim)
    ]
    if len(eligible) < 2:
        return []
    findings = [asdict(f) for f in ContradictionDetector().scan(eligible)]
    for finding in findings:
        finding["comparison_rule"] = "natural_language_opposition"
    return findings


def _scan_structured_contradictions(
    indexed_claims: list[tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Pairwise structured-value contradiction scan restricted to declared domains.

    Groups claims by their declared ``comparison_key``/``contradiction_key``
    (only rows passing :func:`_is_structured_comparable`), canonicalizes each
    claim's comparison value, and flags a FRICTION finding for every pair
    within a group whose canonicalized values differ. The comparison domain
    and mutual-exclusivity mode are exactly what the rows themselves declared
    — never inferred from column overlap.
    """
    groups: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for cid, claim in indexed_claims:
        if not _is_structured_comparable(claim):
            continue
        key = _structured_comparison_key(claim)
        if key is None:  # pragma: no cover - guaranteed by the check above
            continue
        groups.setdefault(key, []).append((cid, claim))

    findings: list[dict[str, Any]] = []
    for key, members in groups.items():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                id_a, claim_a = members[i]
                id_b, claim_b = members[j]
                value_a = _canonicalize_comparison_value(
                    _structured_comparison_value(claim_a)
                )
                value_b = _canonicalize_comparison_value(
                    _structured_comparison_value(claim_b)
                )
                if value_a == value_b:
                    continue
                mode = (
                    claim_a.get("comparison_mode")
                    or claim_b.get("comparison_mode")
                    or "mutually_exclusive"
                )
                lo_id, hi_id = sorted((id_a, id_b))
                lo_val, hi_val = (
                    (value_a, value_b) if id_a <= id_b else (value_b, value_a)
                )
                reason = (
                    f"[FRICTION] structured comparison_key='{key}' mode='{mode}': "
                    f"claim '{lo_id}' asserts {lo_val!r} while "
                    f"claim '{hi_id}' asserts {hi_val!r}"
                )
                findings.append(
                    {
                        "new_id": lo_id,
                        "conflict_id": hi_id,
                        "similarity": 1.0,
                        "reason": reason,
                        "severity": "high",
                        "comparison_rule": "structured_mutual_exclusivity",
                        "comparison_key": key,
                        "comparison_mode": mode,
                    }
                )
    return findings


def _dedupe_and_sort_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop duplicate claim-id-pair findings (keep the first) and sort deterministically.

    Sorted most-similar first, then by the pair's own ids, so the result is
    stable regardless of which rule (natural-language vs structured)
    produced a given finding.
    """
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for finding in findings:
        id_a, id_b = sorted(
            (str(finding.get("new_id")), str(finding.get("conflict_id")))
        )
        pair: tuple[str, str] = (id_a, id_b)
        if pair in seen:
            continue
        seen.add(pair)
        deduped.append(finding)
    deduped.sort(
        key=lambda f: (
            -float(f.get("similarity") or 0.0),
            str(f.get("new_id")),
            str(f.get("conflict_id")),
        )
    )
    return deduped


def _scan_contradictions(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run opt-in contradiction analysis over ``claims`` and return plain dicts.

    See the module-level comment above :data:`_NL_ASSERTION_ROLES` for the
    full eligibility contract. Zero/one claims trivially scan to no findings
    — this is always safe to call, even on an empty claim set. Every claim in
    ``claims`` is preserved by the caller regardless of what this returns;
    this function only ever produces findings, never filters rows. Each
    finding cites both claim ids (``new_id``/``conflict_id``) and the
    ``comparison_rule`` that produced it.
    """
    if len(claims) < 2:
        return []
    indexed = [
        (str(c.get("id") or f"claim:{i}"), c)
        for i, c in enumerate(claims)
        if isinstance(c, dict)
    ]
    findings = _scan_nl_contradictions(indexed) + _scan_structured_contradictions(
        indexed
    )
    return _dedupe_and_sort_findings(findings)


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

    The complete current response contract for public graph analysis and query
    tools. Source payload fields without dedicated slots are retained in
    ``reasoning_trace``. See the module docstring for the no-fabrication contract.
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
        description="FrictionFinding-shaped conflicts detected across `claims`. "
        "Opt-in only (U-124/U-131): a claim participates ONLY when it declares "
        "itself a natural-language assertion (`semantic_role` + `text`) or a "
        "structured comparison (`comparison_key`/`contradiction_key` + "
        "`comparison_mode` in {exclusive, single_value} or "
        "`mutually_exclusive: true`) — different values across ordinary query "
        "rows are never treated as contradictions.",
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
    error: dict[str, Any] | None = Field(
        default=None,
        description="The structured operation error the source payload carried, when "
        "the underlying operation genuinely FAILED (as distinct from a healthy query "
        "that simply found no/low-confidence evidence). This is the single dedicated "
        "success/failure signal for this bundle — dispatch-level callers (e.g. the "
        "intent surface's ``_execution_succeeded``) must treat it as authoritative "
        "rather than re-deriving status from `claims`/`reasoning_trace` shape. "
        "None means no operation-level failure was observed — never fabricated.",
    )

    @classmethod
    def from_payload(
        cls,
        payload: Any,
        *,
        operation: str = "graph",
    ) -> EvidenceBundle:
        """Project an arbitrary internal result into the sole public bundle type.

        This is deliberately deterministic and lossless: dict/list payloads are
        retained as claims and/or a trace entry, while plain text becomes the
        answer candidate. A payload that already contains an ``evidence_bundle``
        is validated and its sibling fields are appended to the trace rather than
        silently discarded.
        """

        if isinstance(payload, cls):
            return payload
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except (TypeError, ValueError):
                return cls(
                    answer_candidate=payload,
                    reasoning_trace=[{"step": operation, "text": payload}],
                )

        if isinstance(payload, dict):
            embedded = payload.get("evidence_bundle")
            if isinstance(embedded, dict):
                bundle = cls.model_validate(embedded)
                siblings = {k: v for k, v in payload.items() if k != "evidence_bundle"}
                if siblings:
                    bundle.reasoning_trace.append(
                        {"step": operation, "payload": siblings}
                    )
                return bundle
            if set(payload).issubset(set(cls.model_fields)):
                return cls.model_validate(payload)
            rows = payload.get("rows")
            results = payload.get("results")
            candidate_rows = rows if isinstance(rows, list) else results
            claims = (
                [dict(row) for row in candidate_rows if isinstance(row, dict)]
                if isinstance(candidate_rows, list)
                else [dict(payload)]
            )
            error = payload.get("error")
            answer = str(payload.get("answer") or payload.get("output") or "")
            if isinstance(error, dict):
                bundle_error: dict[str, Any] | None = error
            elif error and payload.get("status") == "failed":
                # A truthy non-dict ``error`` alongside an explicit ``status:
                # "failed"`` still names a real failure — echo the status the
                # payload already carries rather than dropping the signal.
                bundle_error = {"code": "operation_failed", "message": str(error)}
            elif payload.get("status") == "failed":
                bundle_error = {"code": "operation_failed"}
            else:
                bundle_error = None
            return cls(
                answer_candidate="" if error else answer,
                claims=claims,
                contradictions=_scan_contradictions(claims),
                reasoning_trace=[{"step": operation, "payload": payload}],
                next_actions=["review the structured error and retry"] if error else [],
                error=bundle_error,
            )

        if isinstance(payload, list):
            claims = [dict(row) for row in payload if isinstance(row, dict)]
            return cls(
                answer_candidate=f"{len(payload)} row(s)",
                claims=claims,
                contradictions=_scan_contradictions(claims),
                reasoning_trace=[{"step": operation, "payload": payload}],
            )
        return cls(
            answer_candidate=str(payload),
            reasoning_trace=[{"step": operation, "payload": payload}],
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

        # semantic_role="assertion" opts these system-synthesized sentences into
        # contradiction analysis (U-124/U-131) — they are genuine natural-language
        # claims, unlike a raw KG query row with no distinguishing assertion text.
        claims = [
            {"id": f"claim:{i}", "text": s, "semantic_role": "assertion"}
            for i, s in enumerate(_sentences(answer))
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
            # semantic_role="assertion" opts these system-synthesized sentences
            # into contradiction analysis (U-124/U-131) — see the matching note
            # in from_code_context_answer.
            claims = [
                {"id": f"claim:{i}", "text": s, "semantic_role": "assertion"}
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
            "error": (
                error
                if isinstance(error, dict)
                else (
                    {"code": "operation_failed", "message": str(error)}
                    if error
                    else None
                )
            ),
        }
        fields.update(overrides)
        return cls(**fields)

    # ------------------------------------------------------------------
    # Engine wire (the epistemic-graph engine's KnowledgeSet/EvidenceBundle, D11)
    # ------------------------------------------------------------------
    @classmethod
    def from_engine_wire(cls, ws: dict[str, Any]) -> EvidenceBundle:
        """Map the engine's ``KnowledgeSet``/``EvidenceBundle`` wire dict.

        Epistemic Substrate Program, control-plane closeout D11. The engine's E3
        ``KnowledgeSet`` returns a ``{"rows": [...]}`` shape — one row per
        candidate answer/fact, each carrying ``id``/``kind``/``score``/
        ``confidence``/``valid_time``/``tx_time`` (bitemporal E2 belief-op
        provenance) /``source_refs``/``evidence_refs``/``policy_labels``. When
        ``rows`` is present this method does the REAL 1:1 mapping documented
        below; when it is absent (a caller-assembled dict already shaped like
        this class's own fields — the pre-D11 stub's contract, still exercised
        by ``test_from_engine_wire_passthrough``) it falls back to the original
        best-effort passthrough so existing callers keep working unchanged.

        Row mapping (no fabrication — mirrors the module docstring's contract):

        * ``claims`` — one entry per row (``id``/``text`` via the same
          heterogeneous-shape extraction :func:`_claim_text` uses elsewhere in
          this module, plus the row's own ``kind``).
        * ``evidence_spans`` — every ``evidence_refs``/``source_refs`` entry,
          tagged with the row it grounds and whether it is an evidence or a
          source reference.
        * ``confidence`` — the TOP-SCORED row's own ``confidence`` (never an
          invented cross-row average) — falls back to a top-level
          ``ws["confidence"]`` when no row carries one.
        * ``freshness`` — the min/max ``valid_time``/``tx_time`` observed
          across rows (a real bitemporal coverage signal, not derived from
          nothing).
        * ``policy_exclusions`` — every row's ``policy_labels``, deduped,
          order-preserving.
        * ``reasoning_trace`` — one ``knowledge_set_row`` entry per row
          (nothing dropped: score/confidence/valid_time/tx_time/refs/labels
          all carried through) plus a trailing ``meta`` entry for any
          top-level wire fields with no dedicated slot.
        * ``answer_candidate`` — the wire's own ``answer_candidate``/``answer``
          when present; otherwise a deterministic, templated row-count
          restatement (mirrors :meth:`from_nl_query` — never an invented
          summary).
        """
        rows = ws.get("rows")
        if isinstance(rows, list) and rows:
            return cls._from_knowledge_set_rows(ws, rows)

        # -- forward-compat passthrough: a wire dict already shaped like this
        # class's own fields (no "rows") — every lookup defaults safely, so an
        # unrecognized/partial payload degrades cleanly rather than raising. --
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
            error=ws.get("error") if isinstance(ws.get("error"), dict) else None,
        )

    @classmethod
    def _from_knowledge_set_rows(
        cls, ws: dict[str, Any], rows: list[Any]
    ) -> EvidenceBundle:
        """The real E3 ``KnowledgeSet`` row → bundle mapping (see :meth:`from_engine_wire`)."""
        rows = [r for r in rows if isinstance(r, dict)]
        claims: list[dict[str, Any]] = []
        evidence_spans: list[dict[str, Any]] = []
        policy_labels: list[str] = []
        valid_times: list[Any] = []
        tx_times: list[Any] = []
        reasoning_trace: list[dict[str, Any]] = []
        best_row: dict[str, Any] | None = None
        best_score = float("-inf")

        for i, row in enumerate(rows):
            rid = row.get("id")
            kind = row.get("kind")
            score = row.get("score")
            row_confidence = row.get("confidence")
            valid_time = row.get("valid_time")
            tx_time = row.get("tx_time")
            source_refs = list(row.get("source_refs") or [])
            evidence_refs = list(row.get("evidence_refs") or [])
            labels = list(row.get("policy_labels") or [])

            cid, text = _claim_text(row, fallback_id=str(rid or f"row:{i}"))
            claims.append({"id": cid, "text": text, "kind": kind})

            for ref in evidence_refs:
                evidence_spans.append(
                    {"ref": ref, "row_id": rid, "type": "evidence_ref"}
                )
            for ref in source_refs:
                evidence_spans.append({"ref": ref, "row_id": rid, "type": "source_ref"})

            for lbl in labels:
                if lbl not in policy_labels:
                    policy_labels.append(lbl)

            if valid_time is not None:
                valid_times.append(valid_time)
            if tx_time is not None:
                tx_times.append(tx_time)

            reasoning_trace.append(
                {
                    "step": "knowledge_set_row",
                    "id": rid,
                    "kind": kind,
                    "score": score,
                    "confidence": row_confidence,
                    "valid_time": valid_time,
                    "tx_time": tx_time,
                    "source_refs": source_refs,
                    "evidence_refs": evidence_refs,
                    "policy_labels": labels,
                }
            )

            try:
                numeric_score = float(score) if score is not None else None
            except (TypeError, ValueError):
                numeric_score = None
            if numeric_score is not None and numeric_score > best_score:
                best_score = numeric_score
                best_row = row

        # confidence: the top-scoring row's own confidence — never an invented
        # average across heterogeneous rows.
        confidence: float | None = None
        if best_row is not None and best_row.get("confidence") is not None:
            try:
                confidence = max(0.0, min(1.0, float(best_row["confidence"])))
            except (TypeError, ValueError):
                confidence = None
        if confidence is None and ws.get("confidence") is not None:
            try:
                confidence = max(0.0, min(1.0, float(ws["confidence"])))
            except (TypeError, ValueError):
                confidence = None

        answer_candidate = str(ws.get("answer_candidate") or ws.get("answer") or "")
        if not answer_candidate:
            query = str(ws.get("query") or ws.get("question") or "")
            answer_candidate = (
                f"{len(rows)} row(s) for: {query}".strip()
                if query
                else f"{len(rows)} row(s) from the engine KnowledgeSet"
            )

        freshness: dict[str, Any] = {}
        try:
            if valid_times:
                freshness["valid_time"] = {
                    "min": min(valid_times),
                    "max": max(valid_times),
                }
            if tx_times:
                freshness["tx_time"] = {"min": min(tx_times), "max": max(tx_times)}
        except TypeError:
            # Heterogeneous/uncomparable timestamp types — degrade to no
            # freshness signal rather than raising.
            freshness = {}

        _mapped_keys = {
            "rows",
            "answer_candidate",
            "answer",
            "confidence",
            "query",
            "question",
            "next_actions",
        }
        meta_extras = {k: v for k, v in ws.items() if k not in _mapped_keys}
        if meta_extras:
            reasoning_trace.append({"step": "meta", **meta_extras})

        return cls(
            answer_candidate=answer_candidate,
            claims=claims,
            evidence_spans=evidence_spans,
            source_authority=_source_authority_from_citations(evidence_spans),
            contradictions=_scan_contradictions(claims),
            confidence=confidence,
            freshness=freshness,
            policy_exclusions=policy_labels,
            reasoning_trace=reasoning_trace,
            next_actions=list(ws.get("next_actions") or []),
        )
