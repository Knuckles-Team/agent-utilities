"""graph_candidate_claims MCP tool — the universal-ingestion program's Track 4
(schema-constrained candidate-claim extraction,
CONCEPT:AU-KG.enrichment.candidate-claim-extraction) and Track 5
(ambiguity-preserving entity-resolution candidates,
CONCEPT:AU-KG.identity.entity-resolution-candidates), exposed as a standalone,
directly-callable MCP/REST surface (D-CE-2).

**Why this needed its OWN tool, not a new action on ``graph_claims``.**
``graph_claims`` already has an action named ``propose`` — the X-3 mining
flywheel's ``ClaimFlywheel.propose`` (a lifecycle STATE transition on an
already-identified claim id). This module's
:class:`~agent_utilities.knowledge_graph.extraction.candidate_claims.
CandidateClaimExtractor`.propose is a DIFFERENT operation entirely: a
model-backed EXTRACTION pass over raw text that produces zero or more
brand-new, richly-shaped candidates (subject/predicate/object, evidence spans,
model-confidence). Reusing the name ``propose`` for both would collide in
meaning, not just in string; giving Track 4/5 their own tool keeps each
action's parameter surface honest (this tool's ``propose`` takes ``text``/
``fragments_json``, never a bare ``claim_id``).

**Thin dispatch only, same discipline as every other action-routed tool
here** — this module never reimplements extraction or resolution logic, only
routes into the real primitives and renders their result as JSON:

* ``propose`` — :meth:`CandidateClaimExtractor.propose`. Structurally
  NO write authority (see that module's docstring): the extractor takes no
  engine/backend/store/session parameter anywhere in its public API, so
  nothing this action does can reach the graph. UNGATED — a read-only,
  proposal-only model call, mirroring how ``graph_claims(action="evaluate")``
  is also deliberately ungated (no fact is written until a separate,
  ActionPolicy-gated step).
* ``resolve_identities`` — :func:`~agent_utilities.knowledge_graph.
  assimilation.identity_candidates.resolve_identity_candidates`. A PURE
  function (no engine parameter at all) that compares records pairwise and
  returns ambiguity-preserving candidates — never a merge. Also UNGATED for
  the same reason. When ``persist=true``, additionally calls
  :func:`~agent_utilities.knowledge_graph.assimilation.identity_candidates.
  write_candidate` for each result — the ONE write this module's sibling
  performs without a governance decision (a ``POSSIBLE_SAME_AS`` hint edge,
  never a ``SAME_AS`` identity assertion; see that function's own docstring).

**Deliberately out of scope here**: :func:`confirm_merge`/:func:`revert_merge`
(the ACTUAL identity-merge decision) are governance-gated actions — Track 5's
own charter requires an explicit human/governed-promotion caller, with the
full :class:`EntityResolutionCandidate` (not just an id) as the recorded
decision's evidence trail. Wiring THOSE into an MCP surface belongs with
whichever lane drives governed identity-merge decisions end to end (mirrors
how ``graph_claims``'s own ``accept``/``retract`` hooks were added by the
governed-promotion lane, not this one) — exposing them here first would build
a surface with no real caller driving the governance side, the same
Wire-First risk this item was originally deferred over.
"""

from __future__ import annotations

import dataclasses
import json

from pydantic import Field

from agent_utilities.mcp import kg_server
from agent_utilities.security.error_surface import public_error_json


def register_candidate_claim_tools(mcp):
    """Register the ``graph_candidate_claims`` tool onto the MCP server."""

    @mcp.tool(
        name="graph_candidate_claims",
        description=(
            "Track 4/5 of the universal-ingestion program: propose model-backed "
            "candidate claims from raw text, and/or resolve entity-identity "
            "candidates across records — NEITHER action ever writes a fact or "
            "an identity assertion (structurally no write authority; see the "
            "module docstring). 'propose' (text, source_id, fragments_json, "
            "rounds, dedup) runs CandidateClaimExtractor.propose — a "
            "schema-constrained extraction pass citing EXACT evidence spans "
            "against the supplied fragments, with an honest (never fabricated) "
            "model_confidence and a review_bucket "
            "(accepted|needs_review|rejected|quarantined). 'resolve_identities' "
            "(records_json, min_confidence, persist) runs "
            "resolve_identity_candidates — compares every record pair and "
            "returns AMBIGUITY-PRESERVING candidates (never a merge); "
            "persist=true additionally writes each as a POSSIBLE_SAME_AS hint "
            "edge (write_candidate), the one ungated write this pipeline makes. "
            "A candidate proposed here has no path to becoming a real graph "
            "fact/edge through this tool — that requires a SEPARATE, "
            "governance-gated caller (e.g. graph_claims(action='evaluate') for "
            "a claim already assembled into a PromotionRequest, or an explicit "
            "confirm_merge for an identity decision)."
        ),
        tags=["graph-os", "epistemic", "claims", "extraction", "entity-resolution"],
    )
    async def graph_candidate_claims(
        action: str = Field(
            default="propose", description="propose|resolve_identities"
        ),
        text: str = Field(
            default="",
            description="'propose' only: the source text to extract candidate claims from.",
        ),
        source_id: str = Field(
            default="",
            description="'propose' only: identifies the source document/object these candidates came from.",
        ),
        fragments_json: str = Field(
            default="[]",
            description=(
                "'propose' only: JSON array of {fragment_id, text} (or legacy "
                "{id, text}) objects — the addressable evidence-spine units an "
                "extracted evidence quote must be a genuine substring of to "
                "resolve to a real EvidenceSpan."
            ),
        ),
        rounds: int = Field(
            default=1,
            description="'propose' only: number of extraction passes (each a different sampling seed).",
        ),
        dedup: bool = Field(
            default=True,
            description="'propose' only: suppress near-duplicate candidates found within this call.",
        ),
        records_json: str = Field(
            default="[]",
            description=(
                "'resolve_identities' only: JSON array of "
                "{id, name, kind, identifiers} entity records to compare pairwise."
            ),
        ),
        min_confidence: float = Field(
            default=0.5,
            description="'resolve_identities' only: floor below which a pair is not flagged as a candidate.",
        ),
        persist: bool = Field(
            default=False,
            description=(
                "'resolve_identities' only: also persist each returned candidate "
                "as a POSSIBLE_SAME_AS edge (write_candidate) — never a merge, "
                "the one ungated write this module's sibling performs."
            ),
        ),
    ) -> str:
        """Propose candidate claims from text, or resolve entity-identity
        candidates across records — both read-only/proposal-only."""
        try:
            if action == "propose":
                if not text or not source_id:
                    return json.dumps({"error": "propose requires text and source_id"})
                try:
                    fragments = json.loads(fragments_json) if fragments_json else []
                    if not isinstance(fragments, list):
                        raise ValueError("fragments_json must decode to a JSON array")
                except Exception as exc:  # noqa: BLE001 — a malformed payload is a client error
                    return json.dumps(
                        {"error": f"invalid fragments_json: {type(exc).__name__}"}
                    )

                from agent_utilities.knowledge_graph.extraction.candidate_claims import (
                    CandidateClaimExtractor,
                )

                extractor = CandidateClaimExtractor()
                batch = await extractor.propose(
                    text,
                    fragments,
                    source_id=source_id,
                    rounds=max(1, rounds),
                    dedup=dedup,
                )
                return json.dumps(
                    {
                        "action": "propose",
                        "source_id": source_id,
                        "candidates": [
                            c.model_dump(mode="json") for c in batch.candidates
                        ],
                        "counts": {
                            "accepted": batch.counts.accepted,
                            "needs_review": batch.counts.needs_review,
                            "rejected": batch.counts.rejected,
                            "quarantined": batch.counts.quarantined,
                        },
                        "unresolved_evidence": batch.unresolved_evidence,
                        "extraction_run_id": batch.run.id,
                    },
                    default=str,
                )

            if action == "resolve_identities":
                try:
                    records_raw = json.loads(records_json) if records_json else []
                    if not isinstance(records_raw, list):
                        raise ValueError("records_json must decode to a JSON array")
                except Exception as exc:  # noqa: BLE001 — a malformed payload is a client error
                    return json.dumps(
                        {"error": f"invalid records_json: {type(exc).__name__}"}
                    )

                from agent_utilities.knowledge_graph.assimilation import (
                    EntityRecord,
                    resolve_identity_candidates,
                    write_candidate,
                )

                try:
                    records = [
                        EntityRecord(
                            id=str(r["id"]),
                            name=str(r.get("name", "")),
                            kind=str(r.get("kind", "")),
                            identifiers=dict(r.get("identifiers") or {}),
                        )
                        for r in records_raw
                    ]
                except (KeyError, TypeError) as exc:
                    return json.dumps(
                        {"error": f"invalid records_json entry: {type(exc).__name__}"}
                    )

                candidates = resolve_identity_candidates(
                    records, min_confidence=min_confidence
                )
                persisted = 0
                if persist and candidates:
                    engine = kg_server._get_engine()
                    for candidate in candidates:
                        write_candidate(engine, candidate)
                        persisted += 1
                return json.dumps(
                    {
                        "action": "resolve_identities",
                        "candidates": [dataclasses.asdict(c) for c in candidates],
                        "persisted": persisted,
                    },
                    default=str,
                )

            return json.dumps({"error": f"unknown action {action!r}"})
        except Exception as e:  # noqa: BLE001
            return public_error_json(e)

    kg_server.REGISTERED_TOOLS["graph_candidate_claims"] = graph_candidate_claims
