# KG-2.34 — Relational-Intent Retrieval

**Pillar:** 2 — Epistemic Knowledge Graph · **Status:** live

## What

A deterministic, **zero-LLM** retrieval arm that answers relational questions —
"which papers *support* transformers", "what *contradicts* X", "what is *cited by*
Y" — by parsing the query with regex, resolving the seed entity, and walking the
typed-edge graph. Mirrors gbrain's `relational-recall.ts`. For non-relational
queries the parser returns `None`, so the arm is a strict no-op and never regresses
ordinary semantic retrieval.

## Why

Vector search cannot reliably answer "who/what is related to X by relation R" — that
is a graph traversal. Doing it deterministically (no LLM) makes it fast,
reproducible, and evaluable, and the verb vocabulary comes from the active pack so
the same machinery serves a VC brain (`invested_in`) or a research brain (`supports`).

## How / Wiring

- `models/schema_pack.py`: `relational_verbs` (NL phrase → edge-type value).
- `knowledge_graph/retrieval/relational_intent.py`: `parse_relational_intent()`
  (interrogative-lead + longest verb-phrase match; inverse direction on
  "… is <verb> by …") and `traverse()` (seed resolution + typed Cypher hop, edge
  type validated against `RegistryEdgeType` to prevent injection).
- `HybridRetriever.retrieve_hybrid` runs the arm at the top and merges hits
  additively into the base set (recall never regresses).
- Entry point: `graph_search` query string is parsed automatically.

## Tests

`tests/knowledge_graph/test_relational_intent.py` (parse, traverse, live-path arm invocation).
