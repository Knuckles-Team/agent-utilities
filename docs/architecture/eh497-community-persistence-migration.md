# EH-497 community persistence migration

AU's `persist_stable_communities` observes a specific graph shape: one
`CommunityNode` named `community_cluster_{index}` for each detected group of at
least three members, with `coherence_score`, `member_count`, `is_permanent`, and
member-to-community `PART_OF_COMMUNITY` edges weighted by coherence. The index
uses the detector's returned group order. Its current method writes each item
separately and returns a count of successful communities.

EG `MineCommunity(writeback=true)` is not this contract. It runs Louvain or
label propagation, includes groups of two, names nodes by a `community:`
membership hash, stores density, and writes community-to-member
`COMMUNITY_MEMBER` edges.
Replacing AU's caller with that method would change identities, algorithm,
direction, relationship type, and observable properties.

The bounded migration uses EG's already governed, atomic `BatchUpdate` through
AU `IntelligenceGraphEngine.batch_typed_mutations`. The AU caller constructs a
single ordered typed batch from the existing detector and `CommunityNode`
model, preserving its IDs and member-to-community edges. The engine seam
resolves a verified `kg:write` session, stamps node/edge ownership and
classification, normalizes relationship types, and sends one native batch.
`native_batch=True` is explicit; a rejected or unavailable batch raises and
does not retry legacy per-item writes. The default caller path stays intact.
The current `IntelligenceGraphEngine` does not expose the old
`upsert_node`/`upsert_edge` names used by that default path; the old unit test
uses a mock that does. This is a pre-existing live-path defect to resolve as
part of the default migration gate, not evidence that the old mock represents
a successful production write.

Remaining gate: a served integration test must compare stored node properties
and edge rows against the old authority on the same graph and verified actor.
EG `upsert_edge` replaces all parallel edges for an ordered endpoint pair, so a
graph carrying other relationship types on the same member/community pair
needs a relationship-scoped upsert contract before enabling this migration by
default. Repeated `community_cluster_{index}` IDs also depend on detector
ordering; a membership-hash identity would require an alias migration for
existing readers. This slice keeps the IDs produced by the legacy call.
