# Design Document: Engineering wisdom (rule books AND the project constitution) becomes reasoned-over KG nodes, not static prose two humans have to keep in sync by hand

> `agent_utilities/knowledge_graph/security/rule_ingestor.py` (agent-rules-books
> → `EngineeringRuleNode`), `agent_utilities/knowledge_graph/security/policy_ingestor.py`
> (SDD constitution/prompt policy → `PolicyNode`) — two sources unified into
> one rules-reasoning layer.

CONCEPT:AU-KG.ingest.engineering-rules

## Decision — parse structured rule sources into versioned, linked, embedded KG nodes

`rule_ingestor.py:1-19`, `policy_ingestor.py:1-15`.

**The rejected alternative**: leaving engineering guidance as static
markdown/JSON prose consulted manually — no versioning, no OWL reasoning
over which rules apply where, no semantic retrieval surfacing a relevant
rule at the point of need, and (for the constitution specifically) no
connection between Spec-Driven Development governance and the same
reasoning layer book-derived rules live in.

**The design chosen**: `rule_ingestor.py` parses structured markdown rule
files (mini/nano tiers) from the `agent-rules-books` repository — the
standard `mini.md` section structure (Title → When to use → Primary bias to
correct → Decision rules → Trigger rules → Final checklist) — into
`RuleBookNode` (one per book) and `EngineeringRuleNode` (one per
decision/trigger rule), wired with SKOS `broader`/`narrower` and PROV-O
`wasDerivedFrom` relationships, with embeddings generated for semantic
retrieval when a model is available. `policy_ingestor.py` parses project
constitutions (`.specify/memory/constitution.md`) and prompt JSON files into
`PolicyNode` entries — explicitly making SDD governance PART OF the same
rules-reasoning layer, alongside book-derived rules, rather than a parallel,
disconnected governance mechanism. Three policy sources are unified this
way: constitution normative statements, quality gates, and embedded
engineering guidance.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/security/rule_ingestor.py`,
  `agent_utilities/knowledge_graph/security/policy_ingestor.py`,
  `agent_utilities/knowledge_graph/core/owl_bridge.py` (declares
  `engineering_rule`/`rule_book` as valid node types).
- **Backward Compatible**: Yes — additive node types; a graph without any
  rule books/constitution ingested is unaffected.
- **Breaking Changes**: None.
- **Known weak point**: this concept id has drifted from a precise marker
  into a loosely-applied label — several unrelated node-type declarations in
  `models/knowledge_graph.py` (e.g. `SCHEMA_PACK`, `CLAIM`,
  `SUBSUMPTION_ALIGNMENT`, `IDEA_BLOCK`/`DISTILLATION_ROUND`) carry this
  SAME concept id in their inline comments despite being genuinely different
  decisions with their own concept ids elsewhere (Schema Packs, Knowledge
  Distillation — see `.specify/design/kgi-knowledge-distillation/design.md`).
  Those are almost certainly copy-paste drift, not evidence that this
  document's scope should expand to cover them; this document grounds the
  ACTUAL Engineering Rules Ingestor decision (`rule_ingestor.py` +
  `policy_ingestor.py`) and treats the stray comment reuses as pre-existing
  marker hygiene debt outside this task's scope, not as concepts requiring
  separate coverage.
