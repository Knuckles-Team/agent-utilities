# AU-KG.research.research-state-domain-pack — Research-State Domain Pack

**Pillar:** 2 — Epistemic Knowledge Graph · **Status:** live

## What

The flagship Schema-Pack 2.0 profile (`research-state`), realising the "academic
literature state" use case from the gbrain schema-pack discussion
([garrytan/gbrain#587](https://github.com/garrytan/gbrain/issues/587)). It activates
paper/claim/method/dataset/evidence node types and wires every Schema-Pack 2.0
capability at once. Two dedicated edge types — `WEAKENS` (distinct from flat
`CONTRADICTS_BELIEF`) and `USES_DATASET` (distinct from provenance `WAS_DERIVED_FROM`)
— give the supports/weakens vocabulary the proposal calls for.

## What it wires

- **Zero-LLM link inference (AU-KG.research.zero-llm-pack-link):** `supports → supports_belief`,
  `weakens/undermines/refutes → weakens`, `uses dataset → uses_dataset`,
  `cites → cites_source`.
- **Relational verbs (AU-KG.retrieval.relational-intent-retrieval):** "which papers support X", "what weakens Y",
  "what is cited by Z".
- **Retrieval signals (KG-2.22):** recency decay (papers ~1y half-life), source
  trust (`peer_reviewed`/`arxiv`/`preprint`/`blog`), autocut on.
- **OWL closure (KG-2.36):** `supports_belief` transitive (support chains),
  `cites_source` inverse-of `cited_by_paper` (citation back-edges).

Combined with bi-temporal `as_of` (KG-2.11) retrieval, this yields a constantly-
updated, queryable literature state — including "the state as of date D".

## How / Wiring

- `models/knowledge_graph.py`: `RegistryEdgeType.WEAKENS`, `USES_DATASET` (registered
  in the OWL promotable-edge set).
- `models/schema_packs/research.py`: the populated `ResearchSchemaPack`.
- Activate via `graph_configure(action="schema_pack", config_key="research-state")`
  or `GRAPH_SCHEMA_PACK=research-state`.

## Tests

`tests/knowledge_graph/test_research_pack.py`.
