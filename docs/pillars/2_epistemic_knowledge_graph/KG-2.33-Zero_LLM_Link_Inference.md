# KG-2.33 — Zero-LLM Pack-Driven Link Inference

**Pillar:** 2 — Epistemic Knowledge Graph · **Status:** live

## What

Deterministic, **zero-LLM** typed-edge extraction on write. The active Schema Pack
declares `LinkInferenceRule`s (regex → edge-type + source/target slot); on every
document write these run over the content to materialise domain edges — for the
`research-state` pack: `supports`, `weakens`, `cites`, `uses dataset`. Mirrors
gbrain's `link-inference.ts`, but our edges are first-class graph relationships that
the OWL reasoner (KG-2.36) then closes transitively.

## Why

LLM relationship extraction is slow, costly, and non-reproducible. For well-known
domain verbs a regex is deterministic, free, and bit-for-bit repeatable across sync
runs — exactly what an "always-on" ingestion daemon needs.

## How / Wiring

- `models/schema_pack.py`: `LinkInferenceRule` and the `link_inference` field.
- `knowledge_graph/kb/link_inference.py`: `infer_links(content, source_id, rules)`,
  run **ReDoS-bounded** — input capped at `MAX_INPUT_CHARS`, each rule under a
  `regex`-module `timeout` (with an `re` nested-quantifier rejection fallback), and a
  per-rule match cap.
- `knowledge_graph/kb/entity_claim_extractor.py`: `extract_and_persist` runs
  `infer_links` after the existing deterministic phase and persists via
  `engine.link_nodes`; a generic value→`RegistryEdgeType` fallback lets new pack
  verbs persist without editing the edge-type map.
- Entry point: `graph_ingest` / `graph_write` → extractor → `infer_links`.

## Safety

ReDoS is the headline risk for user-supplied regex; it is bounded by input cap +
per-rule wall-clock timeout + match cap, verified by a catastrophic-pattern test.

## Tests

`tests/knowledge_graph/test_link_inference.py` (unit, ReDoS bound, truncation, live path).
