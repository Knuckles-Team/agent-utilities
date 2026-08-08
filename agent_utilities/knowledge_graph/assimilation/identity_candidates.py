"""Entity-resolution candidates (CONCEPT:AU-KG.identity.entity-resolution-candidates).

Universal-ingestion program, Track 5 (``reports/program/universal-ingestion.md``):
deciding whether ``payments platform``, ``payments-platform``, and a CMDB id are
the same entity — **preserving ambiguity rather than merging incorrectly**. A
wrong merge is worse than no merge, so this module's central rule is: **nothing
in ``resolve_identity_candidates`` ever produces anything but a *candidate*.**
There is no code path from evidence, however strong, straight to a collapsed
node.

Survey first (the standing rule for this program) — most of the machinery
already exists and is reused, not rebuilt:

* :mod:`.entity_resolution` — the entropy-gated exact + MinHash/LSH
  normalized-string-similarity ladder (Graphiti-derived), reused directly for
  the "weak on its own" name tier.
* :func:`~agent_utilities.knowledge_graph.extraction.fact_extractor.aggregate_confidence`
  — the product-complement "corroboration reinforces" combiner, reused to
  combine independent evidence kinds rather than reinventing the math.
* :mod:`.dedup` — the existing ``SIMILAR_TO``/``SUPERSEDES`` auto-merge pass
  for the Feature/Article/SDDFeature research corpus. **Not extended here on
  purpose**: that pass auto-applies once a threshold clears, which is exactly
  the behavior this track's own charter forbids for general entity identity
  ("a wrong merge is worse than no merge"). This module is a deliberately
  separate, ambiguity-preserving sibling — same corpus of ideas (normalized
  name matching, confidence combination), different write discipline.

The one genuine gap: no candidate-vs-merge distinction existed for general
entity identity, and no first-class evidence-typed proposal record. This adds
:class:`EntityResolutionCandidate` (never auto-confirmed) plus the explicit,
reversible :func:`confirm_merge` / :func:`revert_merge` pair that a governed
caller drives only after review — mirroring how
:class:`~agent_utilities.knowledge_graph.research.claim_flywheel.ClaimFlywheel`
only ever RECORDS a decision a caller already made, never computes governance
validity itself.

Evidence kinds
--------------
* ``exact_identifier`` — two records share a strong identifier field's exact
  value (a CMDB id, an external system id). Declared per domain via
  :class:`~agent_utilities.models.schema_pack.IdentityRule` — see "Identity
  rules live in a pack" below.
* ``normalized_name`` / ``fuzzy_name`` — :func:`.entity_resolution.resolve_entities`'s
  exact-canonical-key / MinHash-LSH tiers. Weak on its own (many distinct real
  entities share a generic normalized name), which is exactly why this module
  never merges on it alone.
* ``structural_context`` — Jaccard overlap of two entities' graph neighbor
  sets, when a caller supplies ``neighbor_fn`` (best-effort/optional — no
  neighbor function means no structural signal is even attempted, never a
  fabricated one).

Identity rules live in a pack
------------------------------
Which fields are strong identifiers and which are display names is
corpus-specific (a CMDB id means something different in a ServiceNow pack
than in a research pack), so :class:`~agent_utilities.models.schema_pack.
IdentityRule` is declared on the active :class:`~agent_utilities.models.
schema_pack.SchemaPack` (``identity_rules``/``identity_rules_for`` —
CONCEPT:AU-KG.ontology.pack-identity-rules) and threaded in here as plain
data. This module hardcodes exactly ONE generic fallback (``cmdb_id``/
``external_id``/``id``) for when no pack rule applies to a given kind — never
a per-corpus rule.

Reversibility
-------------
:func:`confirm_merge` is the ONLY place a real ``SAME_AS`` identity is ever
recorded, and it never happens as a side effect of scoring — a caller (a
human, or a Track 6 governed-promotion policy) must call it explicitly with
its own ``decided_by``/``reason``. The returned :class:`MergeDecision` carries
the FULL evidence trail (not just a reference to it), so
:func:`revert_merge` can always re-write the complete property set — original
evidence, confidence, and decision metadata all still present — with
``reverted``/``reverted_at``/``reverted_reason`` layered on top. The edge is
NEVER deleted on revert, only marked inactive: the merge decision stays
inspectable and undoable, exactly as the track's charter requires.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from agent_utilities.knowledge_graph.extraction.fact_extractor import (
    aggregate_confidence,
)
from agent_utilities.models.knowledge_graph import RegistryEdgeType
from agent_utilities.models.schema_pack import IdentityRule

from .entity_resolution import resolve_entities

__all__ = [
    "EntityRecord",
    "IdentityEvidenceKind",
    "IdentityEvidence",
    "EntityResolutionCandidate",
    "MergeDecision",
    "GENERIC_IDENTIFIER_FIELDS",
    "resolve_identity_candidates",
    "write_candidate",
    "confirm_merge",
    "revert_merge",
]

#: The ONE generic fallback identifier-field set used when no active pack
#: declares an `IdentityRule` for a record's `kind` — a last resort, not a
#: per-corpus rule (Configuration discipline: one correct default).
GENERIC_IDENTIFIER_FIELDS: frozenset[str] = frozenset({"cmdb_id", "external_id", "id"})


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass(frozen=True)
class EntityRecord:
    """One record to entity-resolve.

    ``identifiers`` carries whatever strong-identifier fields the source
    system exposes (e.g. ``{"cmdb_id": "CI00012345"}``); ``kind`` scopes which
    :class:`~agent_utilities.models.schema_pack.IdentityRule`\\ s apply.
    """

    id: str
    name: str
    kind: str = ""
    identifiers: dict[str, str] = field(default_factory=dict)


class IdentityEvidenceKind(StrEnum):
    """The typed evidence kinds this module can produce — never fabricated,
    never expanded beyond what a real signal actually computed."""

    EXACT_IDENTIFIER = "exact_identifier"
    NORMALIZED_NAME = "normalized_name"
    FUZZY_NAME = "fuzzy_name"
    STRUCTURAL_CONTEXT = "structural_context"


@dataclass(frozen=True)
class IdentityEvidence:
    """One piece of evidence toward (or against) two records being the same entity."""

    kind: IdentityEvidenceKind
    detail: str
    score: float

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "detail": self.detail, "score": self.score}


@dataclass
class EntityResolutionCandidate:
    """A proposed identity pair — a CANDIDATE, never a merge.

    ``status`` starts and stays ``"candidate"`` for the lifetime of anything
    :func:`resolve_identity_candidates` returns; only :func:`confirm_merge`
    (an explicit, separate, human/governance-driven call) ever produces a
    real identity assertion, and even then the assertion is reversible.
    """

    id: str
    entity_a: str
    entity_b: str
    evidence: list[IdentityEvidence]
    confidence: float
    status: str = "candidate"
    created_at: str = ""


@dataclass
class MergeDecision:
    """The full, self-contained record of a confirmed (or reverted) merge.

    Carries the candidate's own evidence/confidence INLINE (not by reference)
    so :func:`revert_merge` never depends on the graph engine's property-merge
    semantics to preserve them — the Python object is the source of truth for
    what gets written on every call.
    """

    candidate_id: str
    entity_a: str
    entity_b: str
    confidence: float
    evidence: list[IdentityEvidence]
    decided_by: str
    reason: str
    decided_at: str
    reverted: bool = False
    reverted_at: str | None = None
    reverted_reason: str | None = None

    def to_edge_properties(self) -> dict[str, Any]:
        return {
            "_rel": RegistryEdgeType.SAME_AS.value,
            "candidate_id": self.candidate_id,
            "confidence": round(self.confidence, 6),
            "evidence_json": json.dumps([e.as_dict() for e in self.evidence]),
            "decided_by": self.decided_by,
            "reason": self.reason,
            "decided_at": self.decided_at,
            "reverted": self.reverted,
            "reverted_at": self.reverted_at,
            "reverted_reason": self.reverted_reason,
        }


def _pair_id(a: str, b: str) -> str:
    """Content-addressed, order-independent id for the unordered pair ``{a, b}``."""
    lo, hi = sorted((a, b))
    digest = hashlib.sha256(f"{lo}:{hi}".encode()).hexdigest()[:32]
    return f"identity_candidate:{digest}"


def _identifier_fields(rules: Sequence[IdentityRule]) -> tuple[set[str], float]:
    fields: set[str] = set()
    score = 0.98
    for rule in rules:
        if rule.identifier_fields:
            fields.update(rule.identifier_fields)
            score = max(score, rule.exact_identifier_score)
    if not fields:
        fields = set(GENERIC_IDENTIFIER_FIELDS)
    return fields, score


def _exact_identifier_evidence(
    a: EntityRecord, b: EntityRecord, rules: Sequence[IdentityRule]
) -> IdentityEvidence | None:
    fields, score = _identifier_fields(rules)
    for f in sorted(fields):
        va, vb = a.identifiers.get(f), b.identifiers.get(f)
        if va and vb and va == vb:
            return IdentityEvidence(
                kind=IdentityEvidenceKind.EXACT_IDENTIFIER,
                detail=f"{f}={va}",
                score=score,
            )
    return None


def _name_evidence(a: EntityRecord, b: EntityRecord) -> IdentityEvidence | None:
    """Normalized-string-similarity evidence via the existing entropy-gated ladder.

    Reuses :func:`.entity_resolution.resolve_entities` as a two-item batch
    rather than duplicating its exact/entropy/MinHash-LSH tiers — the ladder
    is the same one :mod:`.dedup` already trusts, applied here to exactly one
    pair so it never crosses into that module's auto-merge behavior.
    """
    result = resolve_entities([(a.id, a.name), (b.id, b.name)])
    if not result.merge_pairs:
        return None
    _survivor, _dup, score, tier = result.merge_pairs[0]
    kind = (
        IdentityEvidenceKind.NORMALIZED_NAME
        if tier == "exact"
        else IdentityEvidenceKind.FUZZY_NAME
    )
    return IdentityEvidence(
        kind=kind, detail=f"{a.name!r} ~ {b.name!r} ({tier})", score=float(score)
    )


def _structural_evidence(
    a_id: str, b_id: str, neighbor_fn: Callable[[str], set[str]] | None
) -> IdentityEvidence | None:
    """Jaccard overlap of two entities' graph neighbor sets — optional, best-effort.

    ``neighbor_fn`` is the caller's own read-only accessor (e.g. ``lambda
    node_id: set(engine.graph.neighbors(node_id))``); this module never reads
    a graph itself, so no structural signal is even attempted without one.
    """
    if neighbor_fn is None:
        return None
    na, nb = neighbor_fn(a_id), neighbor_fn(b_id)
    if not na or not nb:
        return None
    inter = len(na & nb)
    union = len(na | nb)
    if union == 0 or inter == 0:
        return None
    score = inter / union
    return IdentityEvidence(
        kind=IdentityEvidenceKind.STRUCTURAL_CONTEXT,
        detail=f"{inter}/{union} shared neighbors",
        score=score,
    )


def _active_pack_rules_for(a_kind: str, b_kind: str) -> list[IdentityRule]:
    """Resolve :class:`IdentityRule`\\ s from the process-active domain pack.

    Mirrors ``EntityClaimExtractor.__init__``'s own ``get_active_pack()``
    fallback (CONCEPT:AU-KG.research.zero-llm-pack-link) so identity rules are
    read from the SAME pack every other pack-driven enrichment already
    resolves — never a second, bespoke pack-lookup path. Best-effort: an
    unavailable/unset pack yields no rules (the generic fallback applies),
    never an import-time failure.
    """
    try:
        from agent_utilities.models.schema_pack_loader import get_active_pack

        pack = get_active_pack()
    except Exception:  # noqa: BLE001 — a missing/misconfigured pack is not fatal
        return []
    if pack is None:
        return []
    rules = list(pack.identity_rules_for(a_kind))
    if b_kind != a_kind:
        for r in pack.identity_rules_for(b_kind):
            if r not in rules:
                rules.append(r)
    return rules


def resolve_identity_candidates(
    records: Sequence[EntityRecord],
    *,
    identity_rules: Sequence[IdentityRule] | None = None,
    neighbor_fn: Callable[[str], set[str]] | None = None,
    min_confidence: float = 0.5,
) -> list[EntityResolutionCandidate]:
    """Compare every pair in ``records``; return ambiguity-preserving candidates.

    Pure function — no engine, no store, no write anywhere in this call.
    Every returned :class:`EntityResolutionCandidate` has ``status ==
    "candidate"``; nothing here ever merges.

    ``identity_rules``:

    * ``None`` (the default) — resolve from the **process-active domain
      pack** per pair via ``SchemaPack.identity_rules_for(kind)``
      (CONCEPT:AU-KG.ontology.pack-identity-rules) — the real, wired path a
      caller doing generic ingestion uses with zero extra plumbing.
    * an explicit sequence — used as-is (a caller that already resolved its
      own pack, or a test that wants a specific rule set without touching
      process-global pack state).

    Either way, a kind with no matching pack rule falls back to the ONE
    generic identifier-field set (:data:`GENERIC_IDENTIFIER_FIELDS`) plus the
    name-similarity tier — never a hardcoded per-corpus rule.

    A pair with NO evidence at all is not returned as a zero-confidence
    candidate — silence, not a fabricated low score. ``min_confidence`` (or a
    stricter pack-declared ``IdentityRule.min_confidence_to_flag``) is the
    floor below which even real, individually-weak evidence stays unflagged.
    """
    candidates: list[EntityResolutionCandidate] = []
    now = _now_iso()
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            a, b = records[i], records[j]
            if a.id == b.id:
                continue
            pair_rules = (
                _active_pack_rules_for(a.kind, b.kind)
                if identity_rules is None
                else identity_rules
            )
            rules = [r for r in pair_rules if r.applies(a.kind) or r.applies(b.kind)]
            evidence: list[IdentityEvidence] = []
            exact = _exact_identifier_evidence(a, b, rules)
            if exact is not None:
                evidence.append(exact)
            name_ev = _name_evidence(a, b)
            if name_ev is not None:
                evidence.append(name_ev)
            structural = _structural_evidence(a.id, b.id, neighbor_fn)
            if structural is not None:
                evidence.append(structural)
            if not evidence:
                continue

            confidence = aggregate_confidence(e.score for e in evidence)
            scoped_thresholds = [
                r.min_confidence_to_flag
                for r in rules
                if r.identifier_fields or r.name_fields
            ]
            threshold = min(scoped_thresholds) if scoped_thresholds else min_confidence
            if confidence < threshold:
                continue

            candidates.append(
                EntityResolutionCandidate(
                    id=_pair_id(a.id, b.id),
                    entity_a=a.id,
                    entity_b=b.id,
                    evidence=evidence,
                    confidence=confidence,
                    created_at=now,
                )
            )
    return candidates


def write_candidate(engine: Any, candidate: EntityResolutionCandidate) -> None:
    """Persist ``candidate`` as a ``POSSIBLE_SAME_AS`` edge — NEVER a merge.

    Idempotent (edge MERGE via ``engine.link_nodes``, mirroring
    :mod:`.dedup`'s own ``SIMILAR_TO`` write). This is the only KG write this
    module performs without an explicit human/governance decision, and it
    writes an edge that itself says "these might be the same", never one that
    collapses identity.
    """
    engine.link_nodes(
        candidate.entity_a,
        candidate.entity_b,
        RegistryEdgeType.POSSIBLE_SAME_AS,
        properties={
            "_rel": RegistryEdgeType.POSSIBLE_SAME_AS.value,
            "candidate_id": candidate.id,
            "confidence": round(candidate.confidence, 6),
            "evidence_json": json.dumps([e.as_dict() for e in candidate.evidence]),
            "status": candidate.status,
            "created_at": candidate.created_at,
        },
    )


def confirm_merge(
    engine: Any,
    candidate: EntityResolutionCandidate,
    *,
    decided_by: str,
    reason: str,
) -> MergeDecision:
    """Record a REAL identity merge — the ONE place this ever happens.

    Never called by :func:`resolve_identity_candidates` itself. The caller
    (a human, or a Track 6 governed-promotion policy) supplies ``decided_by``
    and ``reason``; this function performs no review of its own — it only
    RECORDS a decision already made, mirroring how
    :class:`~agent_utilities.knowledge_graph.research.claim_flywheel.ClaimFlywheel`
    never recomputes governance validity itself (see the module docstring).
    """
    decision = MergeDecision(
        candidate_id=candidate.id,
        entity_a=candidate.entity_a,
        entity_b=candidate.entity_b,
        confidence=candidate.confidence,
        evidence=list(candidate.evidence),
        decided_by=decided_by,
        reason=reason,
        decided_at=_now_iso(),
    )
    engine.link_nodes(
        candidate.entity_a,
        candidate.entity_b,
        RegistryEdgeType.SAME_AS,
        properties=decision.to_edge_properties(),
    )
    return decision


def revert_merge(engine: Any, decision: MergeDecision, *, reason: str) -> MergeDecision:
    """Undo a confirmed merge WITHOUT deleting it — reversible by design.

    Re-writes the SAME_AS edge with its ORIGINAL confidence/evidence/decision
    fields still present (carried on ``decision``, not re-derived) plus
    ``reverted=True``/``reverted_at``/``reverted_reason`` layered on top — so
    the merge decision stays inspectable and undoable, with the evidence that
    drove it retained, exactly as the track's charter requires. A reader must
    filter ``reverted != true`` rather than the edge being absent.
    """
    reverted = replace(
        decision,
        reverted=True,
        reverted_at=_now_iso(),
        reverted_reason=reason,
    )
    engine.link_nodes(
        reverted.entity_a,
        reverted.entity_b,
        RegistryEdgeType.SAME_AS,
        properties=reverted.to_edge_properties(),
    )
    return reverted
