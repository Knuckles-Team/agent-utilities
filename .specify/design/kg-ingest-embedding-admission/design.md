# Design Document: A deterministic per-unit admission classifier at the ingest-time embedding chokepoint

CONCEPT:AU-KG.ingest.embedding-admission

> `agent_utilities/knowledge_graph/ingestion/embedding_admission.py` (the
> classifier), wired into
> `agent_utilities/knowledge_graph/ingestion/envelope_ingest.py`'s
> `_prepare_embedding_envelopes` — the function immediately downstream of
> that file's own documented D-EMB/D-PERF-5 chokepoint comment block.
> Ruling authority: `plans/refactor/architecture/INGESTION-ECONOMICS-DESIGN.md`
> §2 (EH-269 in the 2.28 ingestion-economics program's ledger).

## Decision — a table-driven, deterministic classifier inside the existing chokepoint, not a per-connector opt-out

`INGESTION-ECONOMICS-DESIGN.md` §0 audited the ingest-time embedding path
and found it applies embedding to essentially everything crossing the
`ChangeEnvelope` boundary by explicit prior design — the D-EMB chokepoint
exists precisely so no connector can bypass embedding policy — gated only
by four cheap, content-blind predicates (upsert+typed-payload, a
text-unchanged short-circuit, empty-derived-text, and a global on/off
flag). No content-type policy, no per-column decision, no cost budget.

The decision: add a FIFTH, deterministic gate evaluated INSIDE that same
chokepoint (never a per-connector escape hatch, which would recreate
exactly the bypass problem the chokepoint was built to prevent). It is a
table of guard-clause rules — a classifier is a table, not an if-chain
(BUILD-CONTRACT §2) — evaluated in a fixed order:

1. **Path/filename identity** (lockfile, generated, vendored, minified,
   binary — never embed).
2. **Generated-content-header scan** (`@generated`, `DO NOT EDIT`, …).
3. **Text shape**: below a minimum length (no semantic signal), or a
   narrow set of concrete machine-token shapes (UUID, hex hash, semver, bare
   URL, email) — deliberately NOT a blanket "no whitespace" test, which
   would wrongly reject ordinary hyphenated names (caught by this program's
   own TDD: an early, broader version of this rule regressed an existing
   integration test).
4. **SQL/CDC column typing**: for the one connector
   (`debezium_envelope.py`, `connector="cdc"`) with a verified 1:1
   DB-column mapping, never embed a column classified as enum, foreign key,
   timestamp, boolean, or numeric — only genuine free-text columns reach
   the embedder.
5. **Exact-duplicate collapse**: texts that hash-match within one ingest
   page share a single embed call and vector.

Everything not affirmatively rejected by one of these rules falls through
as admitted (`ContentClass.PROSE`) — the classifier is a filter, not a
blanket denial.

**The rejected alternative** is a blanket "skip anything that might be
answerable by a cheaper index" heuristic. `INGESTION-ECONOMICS-DESIGN.md`
§2 explicitly warns against this: under-embedding silently degrades
retrieval and is much harder to detect than over-embedding, so this
classifier makes exactly ONE narrow "a cheaper index already answers this"
claim (the structured-token rule, scoped to 5 concrete evidenced shapes)
and otherwise defers broader cheaper-index reasoning to a future,
telemetry-backed iteration rather than guessing.

**Explicitly out of scope for this concept** (left designed, not built,
per the same ruling document's scope-honesty framing): the six-rung
cost/provenance ladder (EH-270, tracked separately and coordinated with the
edge-provenance ladder landing in `enrichment/models.py`), and the
retrieval-telemetry feedback loop that would demote never-retrieved
`ContentClass` buckets over time.

## Risk Assessment

- **Blast radius**: `envelope_ingest.py`'s `_prepare_embedding_envelopes`
  — every typed-entity connector's embedding decision funnels through it.
  A defect here either under-embeds (silently degrades retrieval — the
  worse, harder-to-detect failure mode) or over-embeds (no correctness
  risk, only a missed savings).
- **Backward compatible**: yes for every connector except the one scoped
  SQL/CDC path — `sql_free_text_fields` is a no-op passthrough for every
  `connector` value other than `"cdc"`, so this change is a strict no-op
  for the rest of the fleet's control flow (confirmed: the full pre-existing
  D-EMB test section passes unmodified).
- **Known weak point**: `classify_by_path`'s file-path signal is silently
  unavailable for a connector that names its path field `path`/`file_path`
  (unconditionally redacted upstream by the persistence-privacy gate's
  `_LOCATION_FIELDS`, which correctly runs before this classifier) — only
  `relpath`/`filename`/`filepath` survive that redaction. Documented at
  `embedding_admission.py`'s `_PATH_FIELDS`; not a defect in this gate, a
  correct precedence a future connector author should know about.
