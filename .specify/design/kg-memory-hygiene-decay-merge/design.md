# Design Document: Memory hygiene ARCHIVES by closing bi-temporal validity and merges near-duplicates — it never hard-deletes

CONCEPT:AU-KG.memory.decay-scanner-merge

> Realised by `agent_utilities/knowledge_graph/memory/hygiene.py:60-116`
> (`classify_node`, `plan_decay`, `semantic_merge_groups`, `merge_plan`,
> `MemoryHygiene`), driven from
> `agent_utilities/knowledge_graph/memory/cli.py:173` (`cmd_hygiene`) and
> `:370` (the `hygiene` subparser). Assimilated from `memory-os`
> (`scripts/decay_scanner.py`, `scripts/semantic_dedup.py`).

## Decision — bound memory growth without losing information, by making both hygiene operations non-destructive

Unbounded memory accumulation degrades retrieval: recall gets slower, and
near-duplicate records split the evidence for a fact across several nodes so
none of them looks strong. A hygiene pass is therefore necessary. The decision
is about what it is allowed to do.

Two operations, both non-destructive:

- **Decay.** Low-decay AI-generated memory is archived by closing its
  bi-temporal `valid_to`, not by deleting the row. The record leaves the
  current-validity window — so it stops appearing in ordinary recall — while
  remaining queryable as-of any earlier time.
- **Semantic merge.** Near-duplicates above a cosine threshold (≥ 0.92) are
  merged rather than pruned, so the evidence they carried is consolidated into
  one record instead of one surviving and the rest disappearing.

**The rejected alternative is the upstream baseline this was assimilated from:
memory-os's flat-store scan, which deletes.** The code's framing is that this is
*richer than* that scan — bi-temporal `valid_to` archival in place of a flat
delete.

The reason to refuse deletion in a memory system specifically: hygiene is
automated, unattended, and driven by a heuristic (a decay score, a similarity
threshold). Every such heuristic is wrong some of the time. If wrong-and-delete
is unrecoverable, the cost of a bad threshold is permanent and silent — the
memory is simply gone, and nothing later can tell that it existed. If
wrong-and-archive is recoverable, the same bad threshold costs only recall
until someone widens the window. Making the operation reversible is what makes
running it automatically defensible.

The bi-temporal model is what makes this affordable — it already exists for
contradiction handling, so archival needs no separate tombstone mechanism,
just a `valid_to` write.

**Evidence note:** the archival-over-delete choice is attested by the code and
by the contrast with the named upstream baseline. Unlike the sibling learning
engine — where the design doc spells out Quarq's *"history lost"* /
*"hard-deletes"* explicitly — no commit message elaborating the decay/merge
policy's own trade-off was recovered; the two capabilities were introduced
together in the `7f259b34` "Synergy checkpoint" as separately-numbered concepts.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/memory/hygiene.py`,
  `agent_utilities/knowledge_graph/memory/cli.py`.
- **Backward Compatible**: Yes — archived records leave default recall, which
  is the intended behaviour change.
- **Known weak point**: non-destructive hygiene does not actually reclaim
  storage. Growth in *recall surface* is bounded; growth on disk is not, and
  the pass can be run indefinitely without ever shrinking the store. The 0.92
  merge threshold is also a single global constant — merging two records that
  were meaningfully distinct is only reversible in principle, since the merge
  consolidates them.
