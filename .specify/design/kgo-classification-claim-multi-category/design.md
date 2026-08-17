# Design Document: One subject, N simultaneous typed classifications — each with its own method and its own promotion state, enforced by the type, not a convention

> `agent_utilities/knowledge_graph/ontology/classification_claims.py:1-1159`
> (module docstring + `ClassificationClaim`/`ClassificationPromotionLedger`).
> U-47 audit item.

CONCEPT:AU-KG.ontology.classification-claim-multi-category ·
CONCEPT:AU-KG.ontology.classification-claim-promotion-lifecycle ·
CONCEPT:AU-KG.ontology.cross-source-identity-proposal

## Decision 1 — a classification is a first-class, multi-valued claim per (subject, category), never a single overwritten field

`classification_claims.py:245-337` (`ClassificationClaim`).

**The gap, confirmed by survey before writing a line of this module** (the
standing rule for this program): the repo already had an artifact/evidence
layer (`ingestion/evidence_spine.Artifact`/`Fragment`) and a candidate-fact
proposer (`extraction/candidate_claims.py`), but nothing let one subject carry
**multiple simultaneous, independently true-or-false classifications** — "this
file is code AND a policy implementation AND a skill resource AND
security-critical" — each with its own evidence and its own truth value. A
single `classification` string field (the shape `evidence_spine.Artifact`
already uses for ACL level) cannot express that; it is one value, not a set of
independently-governed axes.

**The design chosen**: `category` is an open string (unlike `method`/`status`,
which are closed enums), so a subject accrues one `ClassificationClaim` node
per `(subject_id, category)` combination that is currently true, each with its
own `method`, `evidence_refs`, and `status`. `query_categories()` — the read
the "N simultaneous categories" acceptance criterion is proven through —
returns only the categories with a currently `promoted` claim, so a
`candidate`/`rejected`/`superseded` claim never counts as something the
subject currently, actually, is.

**Rejected alternative**: extending `evidence_spine.Artifact`'s single
`classification` field into a list. Rejected because a list field on the
artifact node would still be one unversioned value with no per-entry method,
evidence, or promotion state — exactly the shape (single mutable field,
overwritten in place) Decision 2 below refuses for the promotion state.

### Identity — content-addressed, not hand-assigned (`claim_id_for`, `classification_claims.py:203-242`)

`claim_id` is `sha256(subject + category + normalized value + method +
sorted evidence + source_snapshot)[:40]` — deliberately excluding
`confidence`/`status`/`extractor_ref`/`created_at`/`reviewer`, so re-ingesting
the SAME observation at the SAME snapshot mints the IDENTICAL id (upsert, not
duplicate) even when a reviewer has since signed off or a parser version
bumped its confidence estimate. This is what makes `record_claim`'s idempotence
guard (`classification_claims.py:809-870`) possible: a claim already recorded
under this exact id is skipped, so a naive re-emission can never silently reset
a promoted claim back to `candidate`.

## Decision 2 — method is a closed, four-value enum that structurally forbids a guess from ever being read as a fact

`classification_claims.py:132-148, 318-367` (`ClaimMethod`, `__post_init__`).

**The design chosen**: `method ∈ {declared, observed, derived, generated}`,
partitioned into `DETERMINISTIC_METHODS = {declared, observed}` (ground truth
— the source itself asserted it, or a deterministic parser mechanically
inferred it) and `CANDIDATE_METHODS = {derived, generated}` (a computed or
model-produced proposal). `ClassificationClaim.__post_init__` raises if a
`declared`/`observed` claim is constructed at any `status` other than
`promoted`/`superseded` — there is no code path that produces a deterministic
observation sitting in `candidate`/`reviewed`/`rejected`. A reader filtering
`WHERE method IN ('declared','observed')` can therefore never receive a claim
that has not already cleared to ground truth, and a reader filtering `WHERE
status = 'candidate'` can never receive a deterministic observation misread as
a guess. `propose()` (`classification_claims.py:456-507`) is symmetrically the
only constructor for `derived`/`generated` claims and always mints
`status="candidate"` — `status` is not even a parameter there.

**Policy gate is in the constructor, not bolted on afterward**: `propose()`
requires `policy_approved=True` for `method="generated"` specifically (never
for `derived`, a deterministic computation, not an LLM call) and raises
`PermissionError` otherwise — the caller resolves the extraction-policy
decision *before* calling `propose`, never after.

**Rejected alternative, named directly in the module docstring**: unifying
this vocabulary with `research/claim_flywheel.ClaimFlywheel`'s
`proposed -> validated -> accepted -> deprecated -> retracted` state machine,
which already exists in the repo and governs a claim-shaped node. Rejected for
two reasons stated explicitly rather than left implicit: (1) **different
subject** — `ClaimFlywheel` governs *mined research insights* (what the system
itself should do differently); this ledger governs *classifications of an
ingested artifact/entity* (what the artifact IS). Conflating them would let a
routing-policy promotion and an artifact-classification promotion collide on
one vocabulary that means different things in each context. (2) **different
terminal shape** — `ClaimFlywheel`'s `DEPRECATED` is an automatic drift
response to a bad real-world outcome, which has no equivalent for "is this
file security-critical"; this ledger's `SUPERSEDED` instead models a purely
evidentiary event (a re-extraction at a new snapshot, a corrected review)
replacing an older claim, with the older one explicitly retained for audit.

## Decision 3 — promotion is a five-state FSM (`candidate -> reviewed -> promoted/rejected -> superseded`) with ONE sanctioned transition primitive

`classification_claims.py:186-196, 510-542, 873-993`
(`_ALLOWED_TRANSITIONS`, `ClassificationClaim.with_status`,
`ClassificationPromotionLedger`).

**The design chosen**: `with_status()` is the only way to move a claim off its
construction-time status; it checks the request against
`_ALLOWED_TRANSITIONS` and raises `IllegalClaimTransition` for anything not in
`{candidate->{reviewed,rejected}, reviewed->{promoted,rejected},
promoted->{superseded}, rejected->{superseded}, superseded->{}}` — so
`candidate -> promoted` directly (skipping review) and anything leaving
`superseded` are both structurally impossible, not merely discouraged by
convention. `ClassificationPromotionLedger` is the sanctioned caller
(`review`/`promote`/`reject`/`supersede`); every transition writes a
`ClaimLifecycleTransition` event node AND upserts the claim's own `status`
field (unlike `ClaimFlywheel`, which deliberately never touches the `Claim`
node's own fields), because `is_active_fact` — the field every downstream
consumer filters on — must reflect the latest transition directly, not
require a caller to replay the event log on every read.

`supersede()` retires the old claim (`-> superseded`, `superseded_by` set to
the new claim's id) and records the new claim with a `DERIVED_FROM` edge back
to the old one — **both retained, never deleted**, so a historical read still
resolves the old claim by id (the "old and new versions both queryable"
acceptance criterion).

**Rejected alternative**: reusing `domain_packs/ingestion/promotion.py`'s
governed-claim-promotion assembly (`CONCEPT:AU-KG.ingest.governed-claim-promotion`,
`.specify/design/kgi-domain-pack-claim-promotion/design.md`). Rejected because
that path assembles SHACL validation, PII handling, dedup, and steward review
for *domain-pack-sourced* candidate facts entering the graph as new nodes —
a different write shape than promoting a claim ALREADY attached to an existing
subject through a fixed five-state FSM with structural type guarantees. The
two solve adjacent but distinct problems and are not merged, for the same
reason `ClaimFlywheel` above is not reused.

## Decision 4 — cross-source identity is a `ClassificationClaim`, never a parallel type, and structurally cannot be name-equality

`classification_claims.py:673-722` (`propose_cross_source_identity`).

**The design chosen**: proposing that two artifacts identify the same logical
object is an ordinary `category="cross_source_identity"`, `method="derived"`
claim — `derived` because resolving identity from matching evidence (content
hash, structural fingerprint) is a deterministic computation, not an LLM
guess, but it is still, correctly, minted as `candidate`: identity resolution
is exactly as reviewable as any other candidate claim, never a shortcut
straight to `promoted`. There is no `evidence_refs` default and no code path
that constructs the claim without it, so a bare display-name match is
structurally impossible to express through this function — it has no
parameter for one.

**Rejected alternative**: a boolean `same_as` edge set directly between two
artifact nodes on a name/string match. Rejected because it would bypass the
whole evidence-required, promotion-gated machinery every other claim in this
module has — exactly the shortcut Decision 1-3 exist to close off.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/ontology/classification_claims.py`
  only — additive to `ingestion/evidence_spine.py`
  (`Artifact`/`Fragment`, duck-typed on plain string ids, not concretely
  imported) and to the engine's `add_node`/`link_nodes`/`query_cypher`
  primitives (the same lightweight claim-shaped-node path
  `research/claim_flywheel.py` already uses, not the heavier
  `ingest_graph_slice` envelope path).
- **Backward Compatible**: Yes — new node types
  (`ClassificationClaim`, `ClassificationClaimLifecycleEvent`) and new typed
  relations (`APPLIES_TO`, `HAS_PROVENANCE`, `DECLARES`, `DERIVED_FROM`, plus
  the pluggable `extra_relations` hook for `IMPLEMENTS`/`CONTAINS`/`EXCLUDES`/
  `VALIDATES`/`OWNED_BY`); nothing existing is touched.
- **Known weak point**: `resolve_claim_evidence`'s clearance-gated redaction
  (`classification_claims.py:1096-1158`) fails closed on an unresolved
  fragment/artifact (`restricted`, the maximum level), but it is only as
  correct as `evidence_spine.Artifact.to_node()`'s own `classification` field
  — a claim never stores evidence text itself, so redaction is enforced at the
  one place text is actually fetched, not trusted to every caller.
- **Malformed input**: `claim_from_raw` (`classification_claims.py:590-663`)
  drops a structurally malformed or unknown-`method` raw record and returns
  `None` rather than raising, mirroring `extraction/candidate_claims.py`'s own
  drop-silently-not-raise contract, so one bad row from a flaky source
  degrades an ingest batch instead of aborting every other row in it.
