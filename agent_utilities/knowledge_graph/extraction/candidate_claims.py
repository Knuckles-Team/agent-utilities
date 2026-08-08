"""Candidate-claim extraction (CONCEPT:AU-KG.enrichment.candidate-claim-extraction).

Universal-ingestion program, Track 4 (``reports/program/universal-ingestion.md``):
a model-backed, schema-constrained service for ambiguous prose that proposes
entities/relations with **exact evidence spans**, a **non-fabricated confidence**,
and **NO direct write authority**.

Survey first (the standing rule for this program): the whole *extraction*
machinery already exists and is extended here, not rebuilt —

* :mod:`.extraction_schema` (ontology-guided closed-vocabulary prompting),
* :mod:`.fact_extractor` (the canonical prompt, the incremental streaming JSON
  parser, the streaming-model factory, and semantic dedup — all reused
  directly). ``propose`` below drives its own multi-round loop over these
  same primitives rather than calling the higher-level
  :func:`~.fact_extractor.extract_facts` wrapper, for one deliberate reason:
  ``extract_facts`` hands callers an already-``ExtractedFact``-coerced dict,
  and that coercion (``_coerce_fact``) defaults a missing/unparsable
  ``confidence`` to ``0`` — exactly the fabrication the "Confidence honesty"
  section below refuses to do. Reading the raw parsed JSON (before any
  coercion) is the only way to tell "the model said zero" apart from "the
  model said nothing".
* :mod:`.ontology_grounding` (surface-form -> canonical OWL type),
* :mod:`agent_utilities.knowledge_graph.kb.extraction_run` (the 7-state
  ``ExtractionOutcome`` taxonomy + the accepted/needs_review/rejected/quarantined
  item classification).

The one genuine gap the survey found: **no `CandidateClaim` type exists
repo-wide**, and the closest existing analog
(:class:`~agent_utilities.knowledge_graph.kb.entity_claim_extractor.
EntityClaimExtractor`) persists directly — ``engine.graph.add_node``/
``engine.link_nodes`` calls live inside its own extraction method. That is the
opposite of what this track requires: a proposal-only extractor whose
no-write-authority boundary is **structural**, not a convention someone could
forget.

Structural no-write-authority
------------------------------
:class:`CandidateClaimExtractor` and every free function in this module take
**no engine, backend, store, or session parameter anywhere in their public
API** — not optional, not defaulted to ``None``: the parameter simply does not
exist. There is therefore no code path through which a live engine handed to
this module could ever be written to; the only way to persist a
:class:`CandidateClaim` is for a caller outside this module to build a
``ClaimNode`` from it and drive it through the governed
:class:`~agent_utilities.knowledge_graph.research.claim_flywheel.ClaimFlywheel`
(``propose`` -> ``validate`` -> ``accept``) or an equivalent governed-promotion
path (a sibling lane's Track 6) — this module never calls either. The unit test
(`tests/unit/knowledge_graph/test_candidate_claims_wiring.py`) proves this two
ways: a signature-introspection contract test (no parameter across the public
surface is engine/backend/store/session-shaped) and a live-double test that
hands a fully "live" engine mock (with working ``add_node``/``add_edge``/
``link_nodes``) into every extension point this module *does* accept and
asserts zero mutating calls ever fire.

Confidence honesty
-------------------
The underlying model DOES self-report a ``confidence`` (0..100) per extracted
fact (:class:`~.fact_extractor.ExtractedFact`). That raw self-report is real
signal, not fabricated, so it is carried through as
:attr:`CandidateClaim.model_confidence` — but it is never defaulted or
manufactured: when the field is missing or fails to parse as a number, this
module returns ``None`` (an explicit *abstain*, mirroring
``harness/reliability_scorers.py``'s Brier-score abstain convention) rather
than reusing ``fact_extractor._coerce_fact``'s ``0`` fallback, which reads as
"very low confidence" when the true state is "no signal at all". A claim never
carries an invented calibrated probability distinct from what the model said.

Evidence spans address Fragments
---------------------------------
Track 4 evidence must cite a ``Fragment`` (the addressable evidence-spine unit
a sibling lane is building — paragraph/heading/table/row/page/span with a
stable id + content hash), not a raw character offset into an opaque blob.
That dataclass did not exist anywhere in the repo when this module was first
written (confirmed by survey), so :class:`EvidenceSpan` here is a deliberately
minimal, duck-typed interim: it only requires whatever object it is given to
expose ``fragment_id``/``text`` (see :class:`FragmentLike`).

**Re-verified against the now-published, now-MERGED evidence-spine contract**
(``agent_utilities.knowledge_graph.ingestion.evidence_spine.Fragment``, on
``main`` since the evidence-spine lane merged): the real ``Fragment``
dataclass exposes its stable address as ``.fragment_id``, **not** ``.id`` — it
has no ``id`` attribute at all. :class:`FragmentLike` originally duck-typed on
``.id``/``.text``, which the real class would **not** have satisfied. Renamed
the protocol's address attribute to ``.fragment_id`` to match the real
contract exactly, so a real ``Fragment`` instance already satisfies
:class:`FragmentLike` with zero changes — no import line change needed either:
this module deliberately keeps the structural ``Protocol`` (never a concrete
``evidence_spine.Fragment`` import) so a plain dict/``SimpleNamespace`` test
double stays satisfying too; see :class:`_FragmentDict` for the JSON-caller
adapter :func:`~agent_utilities.mcp.tools.candidate_claim_tools.
register_candidate_claim_tools`'s ``graph_candidate_claims(action="propose")``
MCP/REST surface (D-CE-2) uses. Tracked as ``D-CE-1`` in
``reports/deferred/lane-claims-er.md`` (closed: contract re-verified and
aligned).
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from agent_utilities.knowledge_graph.kb.extraction_run import (
    OutcomeCounts,
    build_extraction_run,
    classify_claim,
    content_hash,
    decide_outcome,
)
from agent_utilities.security.persistence_privacy import PersistencePrivacyGuard

from .extraction_schema import ExtractionSchema
from .fact_extractor import (
    FACT_EXTRACTION_PROMPT,
    ExtractedFact,
    FactDeduper,
    StreamFn,
    make_streaming_extract_fn,
    parse_facts_incremental,
)
from .ontology_grounding import ground_fact

__all__ = [
    "FragmentLike",
    "EvidenceSpan",
    "CandidateClaim",
    "CandidateClaimBatch",
    "CandidateClaimExtractor",
    "claim_confidence",
]


@runtime_checkable
class FragmentLike(Protocol):
    """The minimal shape an evidence-spine ``Fragment`` must satisfy.

    Deliberately duck-typed (``@runtime_checkable`` structural ``Protocol``,
    not a concrete import) so this module never imports a class that has not
    merged to ``main`` yet — see the module docstring's "Evidence spans
    address Fragments" section. Any object exposing ``fragment_id`` (the
    stable fragment address) and ``text`` (the fragment's own extracted text,
    the substring evidence spans are located within) satisfies this — the
    real ``agent_utilities.knowledge_graph.ingestion.evidence_spine.Fragment``
    dataclass, a ``SimpleNamespace``, or a plain dict-backed shim in a test
    all qualify. Named ``fragment_id`` (not ``id``) specifically because the
    real ``Fragment`` has no ``id`` attribute — only ``fragment_id``.
    """

    @property
    def fragment_id(self) -> str: ...  # noqa: D102

    @property
    def text(self) -> str: ...  # noqa: D102


class _FragmentDict:
    """Adapts a ``{"id"|"fragment_id": ..., "text": ...}`` mapping to
    :class:`FragmentLike`.

    Callers that only have JSON (an MCP/REST caller, e.g. ``graph_candidate_claims``,
    D-CE-2) may pass plain dicts rather than a real evidence-spine ``Fragment``
    object; this wrapper is the ONE place that tolerance lives so
    :meth:`CandidateClaimExtractor.propose` itself only ever handles the
    protocol shape (``fragment_id``/``text``). Accepts BOTH key spellings:
    ``fragment_id`` (the real ``evidence_spine.Fragment``'s own attribute
    name — what a caller serializing a real fragment to JSON actually has)
    and the pre-existing ``id`` shorthand (kept for the callers already using
    it) — ``fragment_id`` wins if a dict somehow has both.
    """

    __slots__ = ("fragment_id", "text")

    def __init__(self, fragment_id: str, text: str) -> None:
        self.fragment_id = fragment_id
        self.text = text


def _as_fragment(obj: Any) -> FragmentLike | None:
    if isinstance(obj, FragmentLike):
        return obj
    if isinstance(obj, dict) and "text" in obj:
        fragment_id = obj.get("fragment_id", obj.get("id"))
        if fragment_id is not None:
            return _FragmentDict(str(fragment_id), str(obj["text"]))
    return None


class EvidenceSpan(BaseModel):
    """One verbatim evidence citation, addressed to a ``Fragment``, not a raw offset.

    ``start``/``end`` are character offsets **within the cited fragment's own
    text** (``fragment.text[start:end] == quote``), never a raw byte offset
    into an opaque document blob — so the span survives independent of how
    the source document as a whole is chunked or re-fetched.
    """

    fragment_id: str = Field(description="The Fragment this evidence cites.")
    quote: str = Field(description="Verbatim substring of the fragment's text.")
    start: int = Field(ge=0, description="Offset of `quote` within fragment.text.")
    end: int = Field(ge=0, description="End offset (exclusive) within fragment.text.")

    @classmethod
    def locate(cls, quote: str, fragment: FragmentLike) -> EvidenceSpan | None:
        """Locate ``quote`` verbatim inside ``fragment.text``; ``None`` if absent.

        Never fabricates an offset: a quote that is not a real substring of the
        fragment (a model hallucination, or a fragment mismatch) produces no
        :class:`EvidenceSpan` at all rather than a guessed one.
        """
        if not quote:
            return None
        text = fragment.text or ""
        start = text.find(quote)
        if start < 0:
            return None
        return cls(
            fragment_id=fragment.fragment_id,
            quote=quote,
            start=start,
            end=start + len(quote),
        )


def claim_confidence(raw: Any) -> float | None:
    """Coerce a model's raw ``confidence`` (0..100) to ``[0, 1]``, or ``None``.

    ``None`` is an explicit abstain — missing/unparsable is NOT the same as
    "the model said zero" and must never be silently coerced to ``0.0``
    (that would fabricate a confident-sounding low score where there is
    actually no signal at all).
    """
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, value / 100.0))


def _claim_id(source_id: str, subject: str, predicate: str, obj: str) -> str:
    """Content-addressed id, matching the repo's established ``claim:<sha256>``
    convention (:func:`agent_utilities.knowledge_graph.kb.entity_claim_extractor.
    claim_node_id`) so a governed-promotion caller that later persists a real
    ``ClaimNode`` from this candidate mints the SAME id — idempotent re-runs
    upsert rather than duplicate, with no extra plumbing required of that
    caller.
    """
    claim_text = f"{subject} {predicate} {obj}"
    digest = hashlib.sha256(f"{source_id}:{claim_text}".encode()).hexdigest()[:32]
    return f"claim:{digest}"


class CandidateClaim(BaseModel):
    """One proposed entity/relation triple — a PROPOSAL, never a persisted fact.

    Carries everything a governed-promotion step needs to decide whether to
    accept it, and nothing that lets it be mistaken for an already-accepted
    graph edge: no engine reference, no persisted-node id, and a
    ``review_bucket`` that is never silently ``"accepted"`` without having
    actually passed :func:`~agent_utilities.knowledge_graph.kb.extraction_run.
    classify_claim`'s structural + privacy + confidence gates.
    """

    id: str = Field(description="Content-addressed, matches claim_node_id's scheme.")
    source_id: str
    subject: str
    subject_type: str | None = Field(
        default=None, description="Canonical OWL type via ontology_grounding."
    )
    predicate: str
    object: str
    object_type: str | None = Field(default=None)
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    model_confidence: float | None = Field(
        default=None,
        description="The model's own self-reported [0,1] confidence. None means "
        "the model gave no usable signal — NEVER a fabricated number (see "
        "claim_confidence).",
    )
    review_bucket: str = Field(
        description="accepted | needs_review | rejected | quarantined "
        "(agent_utilities.knowledge_graph.kb.extraction_run.classify_claim)."
    )
    tags: list[str] = Field(default_factory=list)
    extraction_run_id: str = Field(
        default="", description="The ExtractionRun this candidate is bound to."
    )

    @property
    def abstained(self) -> bool:
        """True when the model gave no usable confidence signal at all."""
        return self.model_confidence is None


@dataclass
class CandidateClaimBatch:
    """The full, inspectable result of one :meth:`CandidateClaimExtractor.propose` call."""

    candidates: list[CandidateClaim]
    run: Any  # ExtractionRunNode — built, deliberately NEVER persisted here.
    counts: OutcomeCounts
    unresolved_evidence: int = 0  # facts whose evidence_span located in NO fragment


class CandidateClaimExtractor:
    """Model-backed, schema-constrained extraction with NO write authority.

    See the module docstring's "Structural no-write-authority" section: no
    method here, now or in any future change, may add an ``engine``/
    ``backend``/``store``/``session`` parameter without breaking that
    guarantee's own unit test (a signature-introspection contract test) — if a
    future change needs one, it belongs in a SEPARATE governed-promotion
    module, not here.
    """

    def __init__(
        self,
        *,
        stream_fn: StreamFn | None = None,
        schema: ExtractionSchema | None = None,
        review_threshold: float | None = None,
    ) -> None:
        self._stream_fn = stream_fn
        self._schema = schema
        self._review_threshold = review_threshold
        self._guard = PersistencePrivacyGuard()

    async def propose(
        self,
        text: str,
        fragments: Sequence[FragmentLike | dict[str, Any]],
        *,
        source_id: str,
        rounds: int = 1,
        dedup: bool = True,
        deduper: FactDeduper | None = None,
    ) -> CandidateClaimBatch:
        """Extract candidate claims from ``text``, citing evidence in ``fragments``.

        ``text`` is the window actually sent to the model (typically the
        concatenation of ``fragments``' own text, but this module does not
        require that invariant — it only requires that a real evidence quote
        the model returns is a genuine substring of *some* fragment's text;
        see :meth:`EvidenceSpan.locate`). Returns a :class:`CandidateClaimBatch`
        — never writes anything, never raises into a caller's ingestion loop
        for a model/stream failure (mirrors ``extract_facts``' own
        best-effort contract).
        """
        started = time.monotonic()
        input_hash = content_hash(text)
        frags = [f for f in (_as_fragment(x) for x in fragments) if f is not None]

        counts = OutcomeCounts()
        candidates: list[CandidateClaim] = []
        unresolved_evidence = 0

        review_threshold_kwargs: dict[str, float] = {}
        if self._review_threshold is not None:
            review_threshold_kwargs["review_threshold"] = self._review_threshold

        prompt = FACT_EXTRACTION_PROMPT
        if self._schema is not None and not self._schema.is_empty:
            prompt = f"{self._schema.prompt_block()}\n{prompt}"
        stream_fn = self._stream_fn or make_streaming_extract_fn()
        if dedup and deduper is None:
            deduper = FactDeduper()

        for r in range(1, max(1, rounds) + 1):
            seed = 1000 + r * 7919
            full_prompt = f"{prompt}\n\nDocument:\n  text: {text}"
            buf = ""
            seen_hashes: set[int] = set()
            async for delta in stream_fn(full_prompt, seed):
                buf += delta
                for raw in parse_facts_incremental(buf, seen_hashes):
                    outcome_item = self._classify_and_build(
                        raw,
                        source_id=source_id,
                        fragments=frags,
                        dedup=dedup,
                        deduper=deduper,
                        review_threshold_kwargs=review_threshold_kwargs,
                    )
                    if outcome_item is None:
                        continue
                    candidate, bucket, evidence_unresolved = outcome_item
                    if bucket == "rejected":
                        counts.rejected += 1
                        continue
                    if bucket == "quarantined":
                        counts.quarantined += 1
                        continue
                    if bucket == "needs_review":
                        counts.needs_review += 1
                    else:
                        counts.accepted += 1
                    if evidence_unresolved:
                        unresolved_evidence += 1
                    # `_classify_and_build` only returns `candidate=None` for
                    # the rejected/quarantined buckets, both already handled
                    # (and `continue`d past) above.
                    assert candidate is not None
                    candidates.append(candidate)

        outcome = decide_outcome(counts)
        run = build_extraction_run(
            engine=None,  # never resolves/persists against a live engine (see module docstring)
            source_id=source_id,
            input_hash=input_hash,
            outcome=outcome,
            counts=counts,
            entities_count=0,
            claims_count=len(candidates),
            schema_pack_ref=self._schema.name if self._schema else "",
            started_at=started,
        )
        for c in candidates:
            c.extraction_run_id = run.id

        return CandidateClaimBatch(
            candidates=candidates,
            run=run,
            counts=counts,
            unresolved_evidence=unresolved_evidence,
        )

    def _classify_and_build(
        self,
        raw: dict[str, Any],
        *,
        source_id: str,
        fragments: list[FragmentLike],
        dedup: bool,
        deduper: FactDeduper | None,
        review_threshold_kwargs: dict[str, float],
    ) -> tuple[CandidateClaim | None, str, bool] | None:
        """Validate + classify one raw parsed-JSON fact into a candidate.

        Operates on the RAW dict :func:`~.fact_extractor.parse_facts_incremental`
        handed back — BEFORE any coercion defaults a missing ``confidence`` to
        ``0`` — so :func:`claim_confidence` can tell "absent" apart from
        "the model said zero" (see the module docstring's "Confidence
        honesty" section). Returns ``None`` for a structurally malformed fact
        (missing subject/predicate/object), mirroring
        ``fact_extractor._coerce_fact``'s own drop-silently contract.
        """
        subject = str(raw.get("subject", "")).strip()
        obj = str(raw.get("object", "")).strip()
        predicate = str(raw.get("predicate", "")).strip()
        if not subject or not obj or not predicate:
            return None
        evidence_span = str(raw.get("evidence_span", ""))
        tags = raw.get("tags") or []
        if not isinstance(tags, list):
            tags = []

        model_confidence = claim_confidence(raw.get("confidence"))
        # Classification needs SOME numeric signal to bucket on; an abstained
        # (None) model confidence is treated exactly like a below-threshold
        # one for that decision ONLY — it is never written back as a
        # fabricated `model_confidence` on the candidate itself.
        classification_confidence = (
            model_confidence if model_confidence is not None else 0.0
        )

        if dedup and deduper is not None:
            probe = ExtractedFact(
                subject=subject,
                predicate=predicate,
                object=obj,
                evidence_span=evidence_span,
                confidence=int(classification_confidence * 100),
                tags=[str(t) for t in tags],
                source_file=source_id,
            )
            is_dup, _max_sim = deduper.check(probe)
            if is_dup:
                return None

        claim_text = f"{subject} {predicate} {obj}"
        bucket = classify_claim(
            claim_text,
            classification_confidence,
            guard=self._guard,
            **review_threshold_kwargs,
        )
        if bucket in ("rejected", "quarantined"):
            return None, bucket, False

        grounding = ground_fact(subject, predicate, obj)
        evidence: list[EvidenceSpan] = []
        for frag in fragments:
            span = EvidenceSpan.locate(evidence_span, frag)
            if span is not None:
                evidence.append(span)
        unresolved = bool(evidence_span) and not evidence

        candidate = CandidateClaim(
            id=_claim_id(source_id, subject, predicate, obj),
            source_id=source_id,
            subject=subject,
            subject_type=grounding.get("subject_type"),
            predicate=predicate,
            object=obj,
            object_type=grounding.get("object_type"),
            evidence=evidence,
            model_confidence=model_confidence,
            review_bucket=bucket,
            tags=[str(t) for t in tags],
        )
        return candidate, bucket, unresolved
