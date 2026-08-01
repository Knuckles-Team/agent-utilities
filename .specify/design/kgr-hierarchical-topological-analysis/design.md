# Design Document: Retrieve in stages — global community relevance, then local entity ranking with a parent-context boost — instead of flat per-node similarity

CONCEPT:AU-KG.retrieval.hierarchical-topological-analysis

> `agent_utilities/knowledge_graph/core/hierarchical_retrieval.py`.

## Decision — a three-stage global → community → local retrieval, with a parent-context boost at the local stage

`hierarchical_retrieval.py:4-22` (distilled from Deep GraphRAG,
`.specify/specs/research-evolution-20260606/` plan b2-04) states the
replacement directly: "instead of flat per-node similarity, retrieve in
stages — (1) **global**: rank communities by aggregate query relevance of
their members; (2) **community**: drill into the top-k communities; (3)
**local**: rank entities *within* those, with a **parent-context** boost so a
node in a highly-relevant community ranks above an equally-similar node in an
irrelevant one (Deep GraphRAG F3 context-aware ranking)."

**The rejected alternative is flat per-node similarity ranking** — scoring
every candidate node against the query independently, with no notion of
which *neighborhood* of the graph it sits in. That approach cannot express
the F3 property this module names explicitly: two nodes with identical
similarity scores to the query should NOT rank equally if one sits inside a
community that is, in aggregate, much more relevant to the query than the
other's. Flat ranking treats both nodes as interchangeable; the parent-context
boost breaks the tie in favor of topological relevance, which is information
flat similarity search structurally discards.

The module is pure Python and deterministic — lexical relevance by default,
with an embedder injectable — and composes with, rather than replaces, the
existing flat community detector: "communities come from the existing flat
detector unless supplied," so this is a re-ranking/staging layer over
existing community-detection output, not a second community-detection
implementation.

## Risk Assessment

- **Blast Radius**: `hierarchical_retrieval.py`, and any caller of
  `TopologicalAnalysisEngine`'s community output.
- **Backward Compatible**: Yes — a new staged-retrieval entrypoint alongside
  flat retrieval, not a modification of it.
- **Known weak point**: the global stage's community ranking is only as good
  as the community detector's boundaries — a query whose true answer straddles
  two communities the detector split apart gets a weaker parent-context boost
  than one whose answer sits cleanly inside a single well-formed community.
