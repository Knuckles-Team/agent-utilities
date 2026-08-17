"""``ClassificationClaim`` — typed, provenance-bearing multi-category claims (U-47).

CONCEPT:AU-KG.ontology.classification-claim-multi-category
CONCEPT:AU-KG.ontology.classification-claim-promotion-lifecycle
CONCEPT:AU-KG.ontology.cross-source-identity-proposal

**What already existed, and the gap this module closes.** The raw
artifact/evidence layer this lane must extend (never fork) already ships:
:class:`~..ingestion.evidence_spine.Artifact` (immutable, source-scoped, keyed
by ``connector`` + ``source_instance`` + ``source_object_id`` — the "source
authority + object identity" half of identity) and
:class:`~..ingestion.evidence_spine.Fragment` (the stable, content-hashed,
addressable evidence-citation unit); :mod:`~..extraction.candidate_claims`
(a structurally no-write-authority LLM proposer emitting subject/predicate/
object triples with evidence spans and an honest, never-fabricated
confidence); and :mod:`~..research.claim_flywheel` (a FIVE-state governed
lifecycle — ``proposed -> validated -> accepted -> deprecated -> retracted``
— but over *mined research insights*, a different subject and a different
vocabulary entirely).

What did **not** exist anywhere in the repo (confirmed by survey before
writing a line of this module — the standing rule for this program): a claim
type that lets ONE subject carry **multiple simultaneous, independently
true-or-false classifications** — "this file is code AND a policy
implementation AND a skill resource AND security-critical" — with each
classification's own method (was it *declared* by the source, *observed* by
a deterministic parser, *derived* by a deterministic computation over other
claims, or *generated* by an LLM guess?) and its own promotion state, spelled
out by the U-47 audit item as a **closed, four-value method enum** crossed
with a **five-value status enum** (``candidate -> reviewed ->
promoted/rejected/superseded``) that is deliberately DIFFERENT from
``claim_flywheel``'s vocabulary — see :class:`ClassificationPromotionLedger`'s
docstring for exactly why the two are not unified.

Everything in this module is additive to the evidence spine, never a second
identity or evidence scheme: :attr:`ClassificationClaim.subject_id` is
expected to be an ``evidence_spine.Artifact.artifact_id`` (or a domain
entity's own id — ``CodeSymbol``/``Module``/``Skill``/... — the model does
not care which, it only requires a stable id), and
:attr:`ClassificationClaim.evidence_refs` are expected to be
``evidence_spine.Fragment.fragment_id`` values. Neither is imported
concretely (this module stays duck-typed on plain strings, exactly the way
:mod:`~..extraction.candidate_claims` stays duck-typed on ``FragmentLike``)
so it has zero coupling to the extraction pipeline's own object graph.

Structural invariants (Authority #1-#6 of the lane contract), each enforced
by the TYPE, not a convention someone could forget:

* **#3 — deterministic observations are never overwritten by candidates.**
  :class:`ClassificationClaim` structurally CANNOT represent a
  ``declared``/``observed`` claim at any status other than ``promoted`` or
  ``superseded`` — ``__post_init__`` raises otherwise. There is no code path,
  anywhere, that produces a ``declared``/``observed`` claim sitting in
  ``candidate``/``reviewed``/``rejected`` — those three statuses are, by
  construction, reachable ONLY by ``derived``/``generated`` claims. A reader
  filtering ``WHERE status = 'candidate'`` can therefore never receive a
  deterministic observation misread as a guess, and a reader filtering
  ``WHERE method IN ('declared','observed')`` can never receive a claim that
  has not already cleared to ground truth.
* **#4 — a candidate never becomes a policy fact without an explicit
  promotion transition.** :meth:`ClassificationClaim.propose` is the ONLY
  constructor for ``derived``/``generated`` claims and it always mints
  ``status="candidate"`` — the ``status`` field is not even a parameter.
  The only way to reach ``promoted`` from there is
  :meth:`ClassificationClaim.with_status`, whose transition table matches
  the state machine ``candidate -> reviewed -> promoted/rejected``,
  ``promoted``/``rejected -> superseded`` exactly, and which
  :class:`ClassificationPromotionLedger` is the sanctioned caller of.
* **#6 — LLM extraction runs only on policy-approved content.**
  :meth:`ClassificationClaim.propose` requires ``policy_approved=True`` for
  ``method="generated"`` specifically (never for ``derived``, which is a
  deterministic computation, not an LLM call) and raises ``PermissionError``
  otherwise — the gate is in the constructor, not bolted on afterward.

Persistence deliberately rides the SAME two primitives every other
lightweight claim-shaped node in this repo already uses
(``engine.add_node``/``engine.link_nodes``/``engine.query_cypher`` — see
``research/claim_flywheel.py``), not the heavier envelope/
``ingest_graph_slice`` transaction path: a classification claim is emitted
per-subject, per-category, from an ingestion loop already holding a live
engine handle, not delivered as a connector's atomic multi-entity slice.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Literal

logger = logging.getLogger(__name__)

__all__ = [
    "CLASSIFICATION_CLAIM_NODE_TYPE",
    "CLASSIFICATION_CLAIM_LIFECYCLE_EVENT_NODE_TYPE",
    "ClaimMethod",
    "ClaimStatus",
    "CLAIM_METHODS",
    "CLAIM_STATUSES",
    "DETERMINISTIC_METHODS",
    "CANDIDATE_METHODS",
    "CONTAINS",
    "IMPLEMENTS",
    "DECLARES",
    "APPLIES_TO",
    "EXCLUDES",
    "VALIDATES",
    "OWNED_BY",
    "DERIVED_FROM",
    "HAS_PROVENANCE",
    "TYPED_RELATIONS",
    "CLASSIFICATION_LEVELS",
    "claim_id_for",
    "ClassificationClaim",
    "IllegalClaimTransition",
    "ClaimLifecycleTransition",
    "ClassificationPromotionLedger",
    "propose_cross_source_identity",
    "claim_from_raw",
    "record_claim",
    "resolve_claim_evidence",
    "query_claims",
    "query_categories",
    "query_claim_history",
]

CLASSIFICATION_CLAIM_NODE_TYPE = "ClassificationClaim"
CLASSIFICATION_CLAIM_LIFECYCLE_EVENT_NODE_TYPE = "ClassificationClaimLifecycleEvent"

# ── Method / status vocabularies (closed, per the U-47 spec) ────────────────
ClaimMethod = Literal["declared", "observed", "derived", "generated"]
ClaimStatus = Literal["candidate", "reviewed", "promoted", "rejected", "superseded"]

CLAIM_METHODS: frozenset[str] = frozenset(
    {"declared", "observed", "derived", "generated"}
)
CLAIM_STATUSES: frozenset[str] = frozenset(
    {"candidate", "reviewed", "promoted", "rejected", "superseded"}
)

#: Deterministic parser output — ground truth, never held for review
#: (Authority #3). A claim with one of these methods is structurally
#: forbidden from any status other than ``promoted``/``superseded``.
DETERMINISTIC_METHODS: frozenset[str] = frozenset({"declared", "observed"})
#: Model/heuristic output — ALWAYS begins life unreviewed (Authority #4).
CANDIDATE_METHODS: frozenset[str] = frozenset({"derived", "generated"})

# ── Typed relations (the nine the U-47 spec names) ───────────────────────────
CONTAINS = "CONTAINS"
IMPLEMENTS = "IMPLEMENTS"
DECLARES = "DECLARES"
APPLIES_TO = "APPLIES_TO"
EXCLUDES = "EXCLUDES"
VALIDATES = "VALIDATES"
OWNED_BY = "OWNED_BY"
DERIVED_FROM = "DERIVED_FROM"
HAS_PROVENANCE = "HAS_PROVENANCE"

TYPED_RELATIONS: frozenset[str] = frozenset(
    {
        CONTAINS,
        IMPLEMENTS,
        DECLARES,
        APPLIES_TO,
        EXCLUDES,
        VALIDATES,
        OWNED_BY,
        DERIVED_FROM,
        HAS_PROVENANCE,
    }
)

#: Governance ordering, aligned with ``models.company_brain.DataClassification``
#: (kept as a local literal ordering rather than a concrete import — this
#: module stays duck-typed on the artifact's plain ``classification`` string,
#: exactly the way it stays duck-typed on fragment/artifact ids).
CLASSIFICATION_LEVELS: dict[str, int] = {
    "public": 0,
    "internal": 1,
    "confidential": 2,
    "restricted": 3,
}

# candidate -> reviewed -> promoted/rejected; promoted/rejected -> superseded.
# Deterministic claims (promoted at construction) only ever move on to
# superseded — never backward into the review states, which they never
# entered.
_ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "candidate": frozenset({"reviewed", "rejected"}),
    "reviewed": frozenset({"promoted", "rejected"}),
    "promoted": frozenset({"superseded"}),
    "rejected": frozenset({"superseded"}),
    "superseded": frozenset(),
}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def claim_id_for(
    subject_id: str,
    category: str,
    value: str,
    method: str,
    evidence_refs: Sequence[str],
    source_snapshot: str,
) -> str:
    """The canonical, idempotent claim key (U-47 implementation step 6).

    Content-addressed over ``subject`` + ``category`` + a normalized
    ``value`` + ``method`` + the (order-independent, deduped) evidence set +
    ``source_snapshot`` — so re-ingesting the SAME artifact through the SAME
    extractor at the SAME snapshot mints the IDENTICAL id and upserts rather
    than duplicating. This is the idempotence proof the lane requires: two
    calls with the same six logical inputs, in any evidence order, produce
    one claim, not two.

    Deliberately EXCLUDES ``confidence``/``status``/``extractor_ref``/
    ``created_at``/``reviewer`` — those may legitimately differ across a
    re-run of the SAME observation (a parser version bump refining its
    confidence estimate, a reviewer signing off) without that being a NEW
    claim; changing any of them re-ingests as an UPDATE to the existing
    claim node, not a fork.
    """
    joined_evidence = "|".join(sorted({str(e) for e in evidence_refs}))
    normalized_value = " ".join(str(value).split()).strip().lower()
    digest = hashlib.sha256(
        "\x1f".join(
            (
                str(subject_id),
                str(category),
                normalized_value,
                str(method),
                joined_evidence,
                str(source_snapshot),
            )
        ).encode("utf-8")
    ).hexdigest()[:40]
    return f"classification_claim:{digest}"


@dataclass(frozen=True)
class ClassificationClaim:
    """One typed, provenance-bearing classification of a subject.

    Multiple claims coexist per subject (a different ``category`` each) so
    one artifact is simultaneously code, a policy implementation, a skill
    resource, and security-critical without any of the four overwriting the
    others — see the module docstring's "Structural invariants" section for
    exactly what the type does and does not allow.

    Attributes:
        claim_id: The idempotent key — always ``claim_id_for(...)`` of the
            other fields; never hand-authored (build via
            :meth:`declare`/:meth:`observe`/:meth:`propose`).
        subject_id: The artifact/fragment/entity being classified (an
            ``evidence_spine.Artifact.artifact_id`` or any other stable
            entity id).
        category: The classification axis (``"code"``, ``"policy-
            implementation"``, ``"skill-resource"``, ``"security-
            critical"``, ``"ownership"``, ``"standard"``, ...) — open
            vocabulary, unlike ``method``/``status``.
        value: The classification's own value within that category (a
            category can be boolean-shaped, e.g. ``"true"`` for
            ``security-critical``, or valued, e.g. a standard's id for
            ``policy-implementation``).
        method: ``declared`` (the source itself asserted it — a manifest
            field), ``observed`` (a deterministic parser mechanically
            inferred it — a file extension), ``derived`` (computed from
            other already-known facts, not an LLM call), or ``generated``
            (an LLM's guess). See :data:`DETERMINISTIC_METHODS` /
            :data:`CANDIDATE_METHODS`.
        status: Its place in the promotion lifecycle — see
            :class:`ClassificationPromotionLedger`.
        evidence_refs: The ``Fragment`` id(s) that justify this claim. Never
            empty — a claim with no evidence is an assertion, not a claim.
        source_snapshot: The artifact revision (a content hash, a commit
            sha, a ``RepositorySnapshot`` id — anything the ingest pipeline
            considers "the version this claim was true of") this claim was
            extracted against. Part of the idempotent key, so re-extracting
            an UNCHANGED artifact never mints a duplicate, while extracting
            a CHANGED one (a new snapshot) legitimately mints a new claim
            alongside the old — see :meth:`ClassificationPromotionLedger.
            supersede` for how the old one is retired without being deleted.
        extractor_ref: The parser/model/prompt version that produced this
            claim — provenance, cited in ``HAS_PROVENANCE``-adjacent reads.
        confidence: ``None`` is an explicit abstain (no usable signal),
            never fabricated — mirrors
            ``extraction.candidate_claims.claim_confidence``'s convention.
        superseded_by: Set on a claim that has been superseded — the id of
            the claim that replaced it. The superseded claim is RETAINED
            (never deleted), so a historical read still resolves it.
        reviewer: Who/what reviewed this claim (set by
            :meth:`ClassificationPromotionLedger.promote`/``reject``);
            empty for a still-``candidate`` or deterministic claim.
        tenant: Optional tenant scope, threaded straight through to
            ``to_node()`` for ACL-aware storage backends.
    """

    claim_id: str
    subject_id: str
    category: str
    value: str
    method: ClaimMethod
    status: ClaimStatus
    evidence_refs: tuple[str, ...]
    source_snapshot: str
    extractor_ref: str = ""
    confidence: float | None = None
    superseded_by: str | None = None
    reviewer: str = ""
    tenant: str = ""
    created_at: str = field(default_factory=_now_iso)

    def __post_init__(self) -> None:
        if not self.subject_id:
            raise ValueError("ClassificationClaim.subject_id must be non-empty")
        if not self.category:
            raise ValueError("ClassificationClaim.category must be non-empty")
        if self.method not in CLAIM_METHODS:
            raise ValueError(
                f"ClassificationClaim.method must be one of {sorted(CLAIM_METHODS)}, "
                f"got {self.method!r}"
            )
        if self.status not in CLAIM_STATUSES:
            raise ValueError(
                f"ClassificationClaim.status must be one of {sorted(CLAIM_STATUSES)}, "
                f"got {self.status!r}"
            )
        if not self.evidence_refs:
            raise ValueError(
                "ClassificationClaim.evidence_refs must cite at least one Fragment — "
                "a claim with no evidence is an assertion, not a claim."
            )
        expected = claim_id_for(
            self.subject_id,
            self.category,
            self.value,
            self.method,
            self.evidence_refs,
            self.source_snapshot,
        )
        if self.claim_id != expected:
            raise ValueError(
                "ClassificationClaim.claim_id must be claim_id_for(...) of the "
                f"other fields — got {self.claim_id!r}, expected {expected!r}. "
                "Build claims with declare()/observe()/propose() so the id can "
                "never drift from its content."
            )
        # Authority #3, structural: a deterministic observation can NEVER sit
        # in an unreviewed/rejected state — those are reachable only by
        # candidate methods. This is what makes "a guess must never be
        # indistinguishable from a fact" a type-level guarantee, not a
        # convention.
        if self.method in DETERMINISTIC_METHODS and self.status not in (
            "promoted",
            "superseded",
        ):
            raise ValueError(
                f"a {self.method!r} claim is deterministic ground truth and can "
                f"only be at status='promoted' or 'superseded' — got "
                f"{self.status!r}. Deterministic claims never enter review."
            )

    # ── sanctioned constructors ──────────────────────────────────────────
    @classmethod
    def declare(
        cls,
        *,
        subject_id: str,
        category: str,
        value: str,
        evidence_refs: Sequence[str],
        source_snapshot: str,
        extractor_ref: str = "",
        confidence: float | None = None,
        tenant: str = "",
    ) -> ClassificationClaim:
        """A deterministic parser's record of an EXPLICIT source declaration
        (a manifest field, a frontmatter tag) — always ``status="promoted"``."""
        return cls._deterministic(
            "declared",
            subject_id=subject_id,
            category=category,
            value=value,
            evidence_refs=evidence_refs,
            source_snapshot=source_snapshot,
            extractor_ref=extractor_ref,
            confidence=confidence,
            tenant=tenant,
        )

    @classmethod
    def observe(
        cls,
        *,
        subject_id: str,
        category: str,
        value: str,
        evidence_refs: Sequence[str],
        source_snapshot: str,
        extractor_ref: str = "",
        confidence: float | None = None,
        tenant: str = "",
    ) -> ClassificationClaim:
        """A deterministic parser's MECHANICAL inference (a file extension ->
        language, an import graph -> a dependency) — always
        ``status="promoted"``."""
        return cls._deterministic(
            "observed",
            subject_id=subject_id,
            category=category,
            value=value,
            evidence_refs=evidence_refs,
            source_snapshot=source_snapshot,
            extractor_ref=extractor_ref,
            confidence=confidence,
            tenant=tenant,
        )

    @classmethod
    def _deterministic(
        cls,
        method: Literal["declared", "observed"],
        *,
        subject_id: str,
        category: str,
        value: str,
        evidence_refs: Sequence[str],
        source_snapshot: str,
        extractor_ref: str,
        confidence: float | None,
        tenant: str,
    ) -> ClassificationClaim:
        refs = tuple(dict.fromkeys(str(e) for e in evidence_refs))
        claim_id = claim_id_for(
            subject_id, category, value, method, refs, source_snapshot
        )
        return cls(
            claim_id=claim_id,
            subject_id=subject_id,
            category=category,
            value=value,
            method=method,
            status="promoted",
            evidence_refs=refs,
            source_snapshot=source_snapshot,
            extractor_ref=extractor_ref,
            confidence=confidence,
            tenant=tenant,
        )

    @classmethod
    def propose(
        cls,
        *,
        subject_id: str,
        category: str,
        value: str,
        evidence_refs: Sequence[str],
        source_snapshot: str,
        method: Literal["derived", "generated"] = "generated",
        extractor_ref: str = "",
        confidence: float | None = None,
        tenant: str = "",
        policy_approved: bool = False,
    ) -> ClassificationClaim:
        """A model/heuristic PROPOSAL — always begins life ``status="candidate"``.

        ``policy_approved`` gates ``method="generated"`` specifically
        (Authority #6): an LLM must never see content outside the caller's
        extraction policy, so the caller resolves that decision BEFORE
        calling ``propose`` and passes the verdict in — this constructor
        refuses to mint a generated claim otherwise. ``method="derived"`` (a
        deterministic computation, not an LLM call) is not policy-gated.
        """
        if method not in CANDIDATE_METHODS:
            raise ValueError(
                f"propose() is for candidate methods {sorted(CANDIDATE_METHODS)}, "
                f"got {method!r} — use declare()/observe() for deterministic methods."
            )
        if method == "generated" and not policy_approved:
            raise PermissionError(
                "a 'generated' claim requires policy_approved=True (Authority #6) "
                "— LLM extraction runs only on policy-approved content; resolve "
                "that decision BEFORE calling propose(), never after."
            )
        refs = tuple(dict.fromkeys(str(e) for e in evidence_refs))
        claim_id = claim_id_for(
            subject_id, category, value, method, refs, source_snapshot
        )
        return cls(
            claim_id=claim_id,
            subject_id=subject_id,
            category=category,
            value=value,
            method=method,
            status="candidate",
            evidence_refs=refs,
            source_snapshot=source_snapshot,
            extractor_ref=extractor_ref,
            confidence=confidence,
            tenant=tenant,
        )

    # ── the ONE sanctioned transition primitive ──────────────────────────
    def with_status(
        self,
        new_status: ClaimStatus,
        *,
        reviewer: str = "",
        superseded_by: str | None = None,
    ) -> ClassificationClaim:
        """Return a NEW claim instance advanced to ``new_status``.

        The only way to move a claim off its construction-time status.
        Legality is checked against the promotion FSM
        (:data:`_ALLOWED_TRANSITIONS`) — an illegal request (``candidate ->
        promoted`` directly, skipping review; anything leaving
        ``superseded``) raises :class:`IllegalClaimTransition` rather than
        silently applying. :class:`ClassificationPromotionLedger` is the
        sanctioned caller; use it directly rather than this method unless
        you are extending the ledger itself.
        """
        allowed = _ALLOWED_TRANSITIONS.get(self.status, frozenset())
        if new_status not in allowed:
            raise IllegalClaimTransition(
                f"{self.claim_id}: {self.status} -> {new_status} is not a legal "
                f"classification-claim transition (allowed from {self.status!r}: "
                f"{sorted(allowed) or 'none — terminal'})"
            )
        return replace(
            self,
            status=new_status,
            reviewer=reviewer or self.reviewer,
            superseded_by=superseded_by
            if superseded_by is not None
            else self.superseded_by,
        )

    # ── rendering ─────────────────────────────────────────────────────────
    def to_node(self) -> dict[str, Any]:
        """Render the graph-slice/``add_node`` entity row for this claim."""
        row: dict[str, Any] = {
            "id": self.claim_id,
            "node_type": CLASSIFICATION_CLAIM_NODE_TYPE,
            "subject_id": self.subject_id,
            "category": self.category,
            "value": self.value,
            "method": self.method,
            "status": self.status,
            "evidence_refs": list(self.evidence_refs),
            "source_snapshot": self.source_snapshot,
            "extractor_ref": self.extractor_ref,
            "created_at": self.created_at,
        }
        if self.confidence is not None:
            row["confidence"] = self.confidence
        if self.superseded_by:
            row["superseded_by"] = self.superseded_by
        if self.reviewer:
            row["reviewer"] = self.reviewer
        if self.tenant:
            row["tenant"] = self.tenant
        return row

    @property
    def is_deterministic(self) -> bool:
        return self.method in DETERMINISTIC_METHODS

    @property
    def is_active_fact(self) -> bool:
        """True when downstream consumers may treat this claim as a policy
        fact — ``promoted`` and nothing else (not ``candidate``, not
        ``reviewed``, not ``rejected``, not ``superseded``)."""
        return self.status == "promoted"


# ---------------------------------------------------------------------------
# Malformed-input tolerance — mirrors extraction.candidate_claims's own
# drop-silently-not-raise contract for a structurally malformed raw record,
# so a bad row from a flaky/malformed source degrades the ingest loop, never
# crashes it or writes a partial/corrupt claim.
# ---------------------------------------------------------------------------


def claim_from_raw(
    raw: dict[str, Any],
    *,
    source_snapshot: str,
    policy_approved: bool = False,
) -> ClassificationClaim | None:
    """Build a claim from an untrusted, possibly-malformed raw record.

    Returns ``None`` (never raises) when ``raw`` is missing a required field
    (``subject_id``/``category``/``value``/``evidence_refs``) or names an
    unknown ``method`` — the caller's ingest loop simply skips the row, the
    same contract ``extraction.candidate_claims._classify_and_build`` already
    uses for a structurally malformed extracted fact. A malformed source must
    degrade the ingest, never corrupt the graph with a half-built claim and
    never propagate an exception into a batch loop that would abort every
    other, well-formed row in the same batch.
    """
    try:
        subject_id = str(raw["subject_id"]).strip()
        category = str(raw["category"]).strip()
        value = str(raw["value"])
        evidence_refs = raw["evidence_refs"]
        method = str(raw.get("method", "observed"))
    except (KeyError, TypeError) as exc:
        logger.debug("claim_from_raw: malformed record dropped: %s", exc)
        return None
    if not subject_id or not category or not evidence_refs:
        logger.debug("claim_from_raw: malformed record dropped (empty required field)")
        return None
    if not isinstance(evidence_refs, (list, tuple)) or not all(
        isinstance(e, str) and e for e in evidence_refs
    ):
        logger.debug("claim_from_raw: malformed record dropped (bad evidence_refs)")
        return None
    if method not in CLAIM_METHODS:
        logger.debug(
            "claim_from_raw: malformed record dropped (unknown method %r)", method
        )
        return None
    confidence = raw.get("confidence")
    try:
        confidence = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence = None
    extractor_ref = str(raw.get("extractor_ref", ""))
    tenant = str(raw.get("tenant", ""))
    try:
        if method in DETERMINISTIC_METHODS:
            return ClassificationClaim._deterministic(
                method,  # type: ignore[arg-type]
                subject_id=subject_id,
                category=category,
                value=value,
                evidence_refs=evidence_refs,
                source_snapshot=source_snapshot,
                extractor_ref=extractor_ref,
                confidence=confidence,
                tenant=tenant,
            )
        return ClassificationClaim.propose(
            subject_id=subject_id,
            category=category,
            value=value,
            evidence_refs=evidence_refs,
            source_snapshot=source_snapshot,
            method=method,  # type: ignore[arg-type]
            extractor_ref=extractor_ref,
            confidence=confidence,
            tenant=tenant,
            policy_approved=policy_approved,
        )
    except (ValueError, PermissionError) as exc:
        logger.debug("claim_from_raw: record rejected by constructor: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Cross-source identity proposals (W06) — a proposal IS a ClassificationClaim,
# never a parallel type, so it inherits the exact same evidence-required,
# never-fabricated, reviewable machinery every other claim has.
# ---------------------------------------------------------------------------


def propose_cross_source_identity(
    *,
    artifact_a_id: str,
    artifact_b_id: str,
    evidence_refs: Sequence[str],
    source_snapshot: str,
    extractor_ref: str,
    confidence: float | None = None,
    tenant: str = "",
) -> ClassificationClaim:
    """Propose that two DIFFERENT source artifacts identify the same logical
    object — never through display-name equality (Authority #5).

    The result is an ordinary ``category="cross_source_identity"``,
    ``method="derived"`` :class:`ClassificationClaim` — ``derived`` because
    resolving identity from matching evidence (byte-identical content, a
    matching structural fingerprint, ...) is a deterministic COMPUTATION,
    not an LLM guess, but it is still, correctly, a ``candidate``: identity
    resolution is exactly as reviewable as any other candidate claim, never
    a shortcut straight to ``promoted``. ``value`` carries the id of the
    OTHER artifact, so the promoted claim reads directly as "subject_id IS
    artifact_b_id" without a second lookup.

    Raises ``ValueError`` for the two ways this would stop being
    evidence-bearing: proposing identity between an artifact and itself, or
    citing zero evidence — a bare name match is structurally impossible to
    express through this function because there is no parameter for one.
    """
    if not artifact_a_id or not artifact_b_id:
        raise ValueError("a cross-source identity proposal requires two artifact ids")
    if artifact_a_id == artifact_b_id:
        raise ValueError(
            "a cross-source identity proposal requires two DISTINCT artifacts"
        )
    if not evidence_refs:
        raise ValueError(
            "a cross-source identity proposal must cite evidence — display-name "
            "equality is not evidence and has no parameter here."
        )
    return ClassificationClaim.propose(
        subject_id=artifact_a_id,
        category="cross_source_identity",
        value=artifact_b_id,
        method="derived",
        evidence_refs=evidence_refs,
        source_snapshot=source_snapshot,
        extractor_ref=extractor_ref,
        confidence=confidence,
        tenant=tenant,
    )


# ---------------------------------------------------------------------------
# Promotion lifecycle (W05)
# ---------------------------------------------------------------------------


class IllegalClaimTransition(ValueError):
    """Raised when a requested classification-claim transition is not legal."""


@dataclass
class ClaimLifecycleTransition:
    """One recorded, append-only classification-claim lifecycle event — the
    queryable audit trail (Authority #4: "prior claims are preserved for
    audit even after promotion or rejection")."""

    claim_id: str
    from_status: str
    to_status: str
    reason: str
    actor: str = "extraction_pipeline"
    reviewer: str = ""
    timestamp: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "from_status": self.from_status,
            "to_status": self.to_status,
            "reason": self.reason,
            "actor": self.actor,
            "reviewer": self.reviewer,
            "timestamp": self.timestamp,
        }


def _run_cypher(
    engine: Any, query: str, params: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    try:
        rows = engine.query_cypher(query, params or {})
        return list(rows or [])
    except Exception as exc:  # noqa: BLE001 — a claim read is advisory
        logger.debug("classification_claims: query_cypher failed: %s", exc)
        return []


def _link(engine: Any, source: str, target: str, rel_type: str) -> None:
    link = getattr(engine, "link_nodes", None)
    if not callable(link):
        link = getattr(engine, "add_edge", None)
    if not callable(link):
        return
    try:
        link(source, target, rel_type)
    except Exception as exc:  # noqa: BLE001 — an edge is best-effort augmentation
        logger.debug(
            "classification_claims: could not link %s -[%s]-> %s: %s",
            source,
            rel_type,
            target,
            exc,
        )


def _write_claim_node(
    engine: Any, claim: ClassificationClaim, extra_relations: Sequence[tuple[str, str]]
) -> None:
    """Raw write, no idempotence guard — the SOLE place ``add_node`` for a
    claim is actually called. Two sanctioned callers: :func:`record_claim`
    (guarded, for fresh emission) and :class:`ClassificationPromotionLedger`
    (unguarded, because a governed transition's entire job is to overwrite
    the prior status)."""
    engine.add_node(
        claim.claim_id, CLASSIFICATION_CLAIM_NODE_TYPE, properties=claim.to_node()
    )
    _link(engine, claim.claim_id, claim.subject_id, APPLIES_TO)
    for fragment_id in claim.evidence_refs:
        _link(engine, claim.claim_id, fragment_id, HAS_PROVENANCE)
    if claim.method == "declared":
        _link(engine, claim.subject_id, claim.claim_id, DECLARES)
    for relation, target_id in extra_relations:
        _link(engine, claim.claim_id, target_id, relation)


def record_claim(
    engine: Any,
    claim: ClassificationClaim,
    *,
    extra_relations: Sequence[tuple[str, str]] = (),
) -> None:
    """Persist (upsert-if-absent) one claim node and its typed relations.

    Always wires ``APPLIES_TO`` (claim -> subject) and ``HAS_PROVENANCE``
    (claim -> each evidence Fragment); wires ``DECLARES`` (subject -> claim)
    additionally when the claim is ``method="declared"`` (the subject itself
    is the thing that made the declaration); wires ``DERIVED_FROM`` (claim ->
    superseded claim) when :attr:`ClassificationClaim.superseded_by` names
    THIS claim as the successor of another (see
    :meth:`ClassificationPromotionLedger.supersede`).

    ``extra_relations`` is the pluggable hook this lane's design constraint
    (abstraction-first, no opinionation) requires: a domain-specific
    extractor that KNOWS the concrete target of ``IMPLEMENTS``/``CONTAINS``/
    ``EXCLUDES``/``VALIDATES``/``OWNED_BY`` (a policy-implementation claim
    that should ``IMPLEMENTS`` a specific ``Standard`` node, an ownership
    claim that should ``OWNED_BY`` a specific team entity) passes
    ``[(relation, target_id), ...]`` rather than this module guessing a
    domain mapping it cannot know. Every relation is validated against
    :data:`TYPED_RELATIONS` — an unknown relation name raises, so a typo
    never silently ships as an untyped edge.

    **Idempotence, and the regression it guards against.** ``claim_id`` is
    already a content-addressed key over subject + category + value + method
    + evidence + snapshot (:func:`claim_id_for`), so two emissions that
    compute the SAME id are, by construction, the SAME extraction event —
    re-ingesting an unchanged artifact must be a true no-op, not a duplicate
    write. But the underlying store's write is a full-property MERGE/SET
    (the real engine and every in-repo test double alike), so a naive
    unconditional ``add_node`` on every re-emission would silently
    OVERWRITE whatever governance state a reviewer had since applied — a
    promoted candidate reset back to ``candidate``, or worse, a
    ``superseded`` deterministic claim resurrected as ``promoted`` with a
    newer ``created_at`` that would then sort ahead of the claim that
    superseded it. Guarded here: if a claim already exists under this exact
    id, the write is skipped entirely — emission is idempotent by
    definition, never a status-regression vector. Advancing a claim's
    status is EXCLUSIVELY :class:`ClassificationPromotionLedger`'s job; its
    transitions write through :func:`_write_claim_node` directly, bypassing
    this guard on purpose (a governed transition's entire point is to
    overwrite the prior status).
    """
    for relation, _target in extra_relations:
        if relation not in TYPED_RELATIONS:
            raise ValueError(
                f"record_claim: {relation!r} is not one of TYPED_RELATIONS "
                f"({sorted(TYPED_RELATIONS)})"
            )
    existing = query_claims(engine, claim.subject_id, category=claim.category)
    if any(c.claim_id == claim.claim_id for c in existing):
        logger.debug(
            "classification_claims: re-emission of already-recorded claim %s "
            "skipped — idempotent no-op",
            claim.claim_id,
        )
        return
    _write_claim_node(engine, claim, extra_relations)


class ClassificationPromotionLedger:
    """Governed ``candidate -> reviewed -> promoted/rejected/superseded``
    lifecycle over :class:`ClassificationClaim` (W05).

    Deliberately a SEPARATE state machine from
    :class:`~..research.claim_flywheel.ClaimFlywheel`
    (``proposed -> validated -> accepted -> deprecated -> retracted``), not a
    reuse of it, for two reasons named explicitly rather than left implicit:

    1. **Different subject.** ``ClaimFlywheel`` governs MINED RESEARCH
       INSIGHTS — a claim about what the *system itself* should do
       differently (route, retry, prompt). This ledger governs
       CLASSIFICATIONS of an ingested artifact/entity — a claim about what
       the *artifact* IS. Conflating them would let a routing-policy
       promotion and an artifact-classification promotion collide on one
       vocabulary that means different things in each context.
    2. **Different terminal shape.** ``ClaimFlywheel``'s ``ACCEPTED`` gates a
       policy update and its ``DEPRECATED`` is an automatic drift response to
       a bad real-world *outcome* — there is no such outcome-feedback loop
       for "is this file security-critical". This ledger's ``SUPERSEDED``
       instead models the different, purely evidentiary event: a NEWER claim
       (a re-extraction at a new snapshot, a corrected review) replacing an
       OLDER one, with the older one explicitly retained for audit, never
       deleted — see :meth:`supersede`.

    Every transition writes a
    :class:`ClaimLifecycleTransition` event AND upserts the claim node's own
    ``status`` (unlike ``ClaimFlywheel``, which deliberately never touches the
    ``Claim`` node's own fields) — because a classification claim's ``status``
    IS the field downstream consumers filter on
    (:attr:`ClassificationClaim.is_active_fact`), so it must reflect the
    latest transition directly, not require a caller to replay the event log
    on every read.
    """

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def review(
        self, claim: ClassificationClaim, *, reason: str = "under review"
    ) -> ClassificationClaim:
        """``candidate -> reviewed`` — the claim is now under active review."""
        updated = claim.with_status("reviewed")
        self._commit(claim, updated, reason)
        return updated

    def promote(
        self,
        claim: ClassificationClaim,
        *,
        reviewer: str,
        reason: str = "promoted after review",
    ) -> ClassificationClaim:
        """``reviewed -> promoted`` — the claim is now a trusted, active fact."""
        updated = claim.with_status("promoted", reviewer=reviewer)
        self._commit(claim, updated, reason)
        return updated

    def reject(
        self, claim: ClassificationClaim, *, reviewer: str, reason: str
    ) -> ClassificationClaim:
        """``candidate/reviewed -> rejected`` — retained for audit, never
        resurfaces as an active fact (:attr:`ClassificationClaim.is_active_fact`
        is ``False`` for a rejected claim, permanently, short of a fresh
        proposal minting a NEW claim id)."""
        updated = claim.with_status("rejected", reviewer=reviewer)
        self._commit(claim, updated, reason)
        return updated

    def supersede(
        self,
        old_claim: ClassificationClaim,
        new_claim: ClassificationClaim,
        *,
        reason: str = "superseded by a newer extraction",
    ) -> tuple[ClassificationClaim, ClassificationClaim]:
        """Retire ``old_claim`` in favor of ``new_claim`` — BOTH retained.

        ``old_claim`` (``promoted`` or ``rejected``) moves to ``superseded``
        with :attr:`ClassificationClaim.superseded_by` set to
        ``new_claim.claim_id``; ``new_claim`` is recorded with a
        ``DERIVED_FROM`` edge pointing back at ``old_claim`` so the lineage
        walks both directions. Never deletes ``old_claim`` — a historical
        query (the "version change... old and new versions both queryable"
        acceptance criterion) still resolves it by id.
        """
        updated_old = old_claim.with_status(
            "superseded", superseded_by=new_claim.claim_id
        )
        self._commit(old_claim, updated_old, reason)
        record_claim(self.engine, new_claim)
        _link(self.engine, new_claim.claim_id, old_claim.claim_id, DERIVED_FROM)
        return updated_old, new_claim

    def _commit(
        self, before: ClassificationClaim, after: ClassificationClaim, reason: str
    ) -> None:
        # Bypasses record_claim's idempotence guard on purpose: THIS is the
        # sanctioned write path that is SUPPOSED to overwrite the prior
        # status — that is the entire meaning of a governed transition.
        _write_claim_node(self.engine, after, ())
        event_id = f"classification_claim_lifecycle:{after.claim_id}:{after.status}:{_now_iso()}"
        try:
            self.engine.add_node(
                event_id,
                CLASSIFICATION_CLAIM_LIFECYCLE_EVENT_NODE_TYPE,
                properties=ClaimLifecycleTransition(
                    claim_id=after.claim_id,
                    from_status=before.status,
                    to_status=after.status,
                    reason=reason,
                    reviewer=after.reviewer,
                ).to_dict(),
            )
        except Exception as exc:  # noqa: BLE001 — the audit event is best-effort
            logger.debug(
                "classification_claims: could not persist lifecycle event for %s: %s",
                after.claim_id,
                exc,
            )

    def history(self, claim_id: str) -> list[dict[str, Any]]:
        """The claim's full, chronological, append-only transition history."""
        rows = _run_cypher(
            self.engine,
            f"MATCH (e:{CLASSIFICATION_CLAIM_LIFECYCLE_EVENT_NODE_TYPE}) "
            "WHERE e.claim_id = $id RETURN e.claim_id AS claim_id, "
            "e.from_status AS from_status, e.to_status AS to_status, "
            "e.reason AS reason, e.reviewer AS reviewer, e.timestamp AS timestamp",
            {"id": claim_id},
        )
        events = [dict(r) for r in rows if isinstance(r, dict) and r.get("to_status")]
        events.sort(key=lambda e: str(e.get("timestamp") or ""))
        return events


# ---------------------------------------------------------------------------
# Query surface — the "supported Graph-OS query API" this lane's acceptance
# gate requires every proof to go through (never a direct storage-dict read).
# ---------------------------------------------------------------------------


def _row_to_claim(row: dict[str, Any]) -> ClassificationClaim | None:
    try:
        evidence_refs = row.get("evidence_refs") or []
        if isinstance(evidence_refs, str):
            evidence_refs = [evidence_refs]
        return ClassificationClaim(
            claim_id=str(row["id"]),
            subject_id=str(row["subject_id"]),
            category=str(row["category"]),
            value=str(row.get("value", "")),
            method=str(row["method"]),  # type: ignore[arg-type]
            status=str(row["status"]),  # type: ignore[arg-type]
            evidence_refs=tuple(str(e) for e in evidence_refs),
            source_snapshot=str(row.get("source_snapshot", "")),
            extractor_ref=str(row.get("extractor_ref", "")),
            confidence=(
                float(row["confidence"]) if row.get("confidence") is not None else None
            ),
            superseded_by=row.get("superseded_by") or None,
            reviewer=str(row.get("reviewer", "")),
            tenant=str(row.get("tenant", "")),
            created_at=str(row.get("created_at") or _now_iso()),
        )
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning(
            "classification_claims: discarding corrupt ClassificationClaim row %r: %s",
            row.get("id"),
            exc,
        )
        return None


def query_claims(
    engine: Any,
    subject_id: str,
    *,
    status: str | None = None,
    category: str | None = None,
) -> list[ClassificationClaim]:
    """Read every claim for ``subject_id`` back through ``query_cypher``,
    optionally filtered to one ``status``/``category``. Newest first."""
    query = f"MATCH (c:{CLASSIFICATION_CLAIM_NODE_TYPE}) WHERE c.subject_id = $sid"
    params: dict[str, Any] = {"sid": subject_id}
    if status is not None:
        query += " AND c.status = $status"
        params["status"] = status
    if category is not None:
        query += " AND c.category = $category"
        params["category"] = category
    query += (
        " RETURN c.id AS id, c.subject_id AS subject_id, c.category AS category, "
        "c.value AS value, c.method AS method, c.status AS status, "
        "c.evidence_refs AS evidence_refs, c.source_snapshot AS source_snapshot, "
        "c.extractor_ref AS extractor_ref, c.confidence AS confidence, "
        "c.superseded_by AS superseded_by, c.reviewer AS reviewer, "
        "c.tenant AS tenant, c.created_at AS created_at"
    )
    rows = _run_cypher(engine, query, params)
    claims = [c for c in (_row_to_claim(r) for r in rows) if c is not None]
    claims.sort(key=lambda c: c.created_at, reverse=True)
    return claims


def query_categories(engine: Any, subject_id: str) -> set[str]:
    """The set of categories currently ``promoted`` (active facts) for
    ``subject_id`` — the read the "N simultaneous categories" acceptance
    criterion is proven through. A ``candidate``/``reviewed``/``rejected``/
    ``superseded`` claim's category is NOT counted: only an active fact
    counts as something the subject currently, actually, IS."""
    return {c.category for c in query_claims(engine, subject_id, status="promoted")}


def query_claim_history(
    engine: Any, subject_id: str, category: str
) -> list[ClassificationClaim]:
    """Every claim ever made for one ``(subject_id, category)`` pair, across
    every ``source_snapshot`` — a version-change read: both the retired
    (``superseded``) claim and its successor resolve here, newest first."""
    return query_claims(engine, subject_id, category=category)


def resolve_claim_evidence(
    engine: Any,
    claim: ClassificationClaim,
    *,
    viewer_clearance: str = "internal",
) -> list[dict[str, Any]]:
    """Resolve a claim's evidence Fragments, REDACTED when the viewer's
    clearance is below the owning artifact's classification.

    A claim never stores evidence TEXT itself (only ``Fragment`` ids), so
    redaction is enforced at the ONE place text is actually fetched — this
    function — rather than trusted to every caller that happens to hold a
    claim. Reads the fragment's ``artifact_id`` and that artifact's stored
    ``classification`` (``evidence_spine.Artifact.to_node()``'s
    ``classification`` field) through the same ``query_cypher`` surface;
    an artifact whose classification cannot be resolved is treated as
    ``restricted`` (fail closed — see the repo's "fail closed" standing
    rule: a degraded read must never grant permission).
    """
    viewer_level = CLASSIFICATION_LEVELS.get(viewer_clearance, 0)
    out: list[dict[str, Any]] = []
    for fragment_id in claim.evidence_refs:
        rows = _run_cypher(
            engine,
            "MATCH (f:Fragment)-[:FRAGMENT_OF]->(a:Artifact) WHERE f.id = $fid "
            "RETURN f.id AS id, f.text AS text, f.address AS address, "
            "a.classification AS classification",
            {"fid": fragment_id},
        )
        if not rows:
            out.append(
                {
                    "fragment_id": fragment_id,
                    "redacted": True,
                    "text": None,
                    "reason": "unresolved",
                }
            )
            continue
        row = rows[0]
        artifact_level = CLASSIFICATION_LEVELS.get(
            str(row.get("classification") or "restricted"),
            CLASSIFICATION_LEVELS["restricted"],
        )
        if viewer_level < artifact_level:
            out.append(
                {
                    "fragment_id": fragment_id,
                    "redacted": True,
                    "text": None,
                    "reason": "insufficient_clearance",
                }
            )
        else:
            out.append(
                {
                    "fragment_id": fragment_id,
                    "redacted": False,
                    "text": row.get("text"),
                    "address": row.get("address"),
                }
            )
    return out
