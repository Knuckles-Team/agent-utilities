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

* :mod:`epistemic_graph.name_resolution` — the entropy-gated exact + MinHash/LSH
  normalized-string-similarity ladder (Graphiti-derived), reused directly for
  the "weak on its own" name tier.
* :func:`epistemic_graph.identity_candidate_derivation.aggregate_confidence`
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
data. EG declares exactly ONE generic fallback (``cmdb_id``/
``external_id``/``id``/``wikidata_id``) for when no pack rule applies to a
given kind — never a per-corpus rule.

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

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

# EG's generic identifier set includes `wikidata_id` for the EH-362 world
# reference alignment stream; active pack rules still take precedence.
from epistemic_graph.identity_candidate_derivation import (
    GENERIC_IDENTIFIER_FIELDS,
    EntityRecord,
    EntityResolutionCandidate,
    IdentityEvidence,
    IdentityEvidenceKind,
    applicable_rules,
    derive_candidate,
    exact_identifier_evidence,
    name_evidence,
    structural_evidence,
)
from epistemic_graph.name_resolution import resolve_entities

from agent_utilities.models.knowledge_graph import RegistryEdgeType
from agent_utilities.models.schema_pack import IdentityRule

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


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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
    return name_evidence(a, b, score, tier)


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
    return structural_evidence(neighbor_fn(a_id), neighbor_fn(b_id))


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

    No engine or store write occurs in this call; the default rule path reads
    the process-active domain pack.
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
            rules = applicable_rules(a.kind, b.kind, pair_rules)
            evidence: list[IdentityEvidence] = []
            exact = exact_identifier_evidence(a, b, rules)
            if exact is not None:
                evidence.append(exact)
            name_ev = _name_evidence(a, b)
            if name_ev is not None:
                evidence.append(name_ev)
            structural = _structural_evidence(a.id, b.id, neighbor_fn)
            if structural is not None:
                evidence.append(structural)
            candidate = derive_candidate(a, b, evidence, rules, min_confidence, now)
            if candidate is not None:
                candidates.append(candidate)
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
