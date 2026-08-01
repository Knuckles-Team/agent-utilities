# Design Document: Relational questions are parsed with regex only — zero LLM — and are a strict no-op the moment they aren't relational

CONCEPT:AU-KG.retrieval.relational-intent-retrieval

> `agent_utilities/knowledge_graph/retrieval/relational_intent.py`,
> consumed by `agent_utilities/knowledge_graph/retrieval/hybrid_retriever.py:636,792`.

## Decision — parse "which papers support X" with regex against a pack-supplied verb vocabulary, never an LLM call, and fall through cleanly when the query isn't relational

`relational_intent.py:4-19` (mirroring gbrain's `relational-recall.ts`) states
the mechanism directly: parse a natural-language relational question —
"which papers *support* transformers," "what *contradicts* X" — with
**regex only (zero LLM)**, resolve the seed entity, and walk the typed-edge
graph to return related nodes directly. For non-relational queries the parser
returns `None` so the arm is "a strict no-op and never regresses ordinary
retrieval."

**The rejected alternative** is an LLM-based intent classifier deciding
whether a query is relational and, if so, extracting the verb/seed/direction
— which would add a model call (latency + cost + non-determinism) to every
query just to check a fast-path applicability condition that a deterministic
regex answers in microseconds. The chosen design pays that cost only when it
actually helps: a genuinely relational query gets a direct typed-edge walk
instead of falling through to full semantic retrieval; every other query is
untouched.

**The verb→edge-type vocabulary is pack-supplied, not hard-coded**:
`SchemaPack.relational_verbs` means "the same machinery serves a VC brain
(`invested_in`/`founded`) or a research brain (`supports`/`contradicts`)
without code changes." The rejected alternative here is a fixed, hard-coded
verb list — which would force a code change (not a config/pack change) every
time a new domain's relational vocabulary needed to be added, and would leave
the module unusable for any Schema Pack whose domain wasn't anticipated at
write time. Edge types are validated against `RegistryEdgeType` before they
touch a query specifically "so a pack typo can never inject Cypher" — a
security property a purely string-templated approach would not have.

## Risk Assessment

- **Blast Radius**: `relational_intent.py`, `hybrid_retriever.py` (the arm
  that dispatches to it), `models/schema_pack.py`,
  `models/schema_packs/research.py`.
- **Backward Compatible**: Yes — the parser returning `None` for any
  non-matching query is the exact pre-existing behavior; this is a pure
  fast-path addition.
- **Known weak point**: the `_INTERROGATIVE_LEAD` regex is a fixed set of
  opening words (which/what/who/…) — a relational question phrased without
  one of those leads (e.g. an imperative "list contradictions of X" phrased
  unusually, or a non-English query) silently falls through to ordinary
  semantic retrieval rather than being recognized as relational, which is the
  safe failure direction but still a coverage gap.
