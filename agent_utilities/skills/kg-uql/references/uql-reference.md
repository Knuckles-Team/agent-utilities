# UQL — the authoritative reference (grounded in the real grammar/executor)

This reference is grounded directly in the epistemic-graph engine's implementation —
**not** in aspiration or prior documentation. Every clause below is cited to the
`file:line` in `epistemic-graph/crates/eg-plan/src/uql/{lexer,parser,mod}.rs` (the
parser/AST) and `epistemic-graph/crates/eg-plan/src/exec.rs` (the executor) that
implements it. Repo root for all citations below:
`/home/apps/workspace/agent-packages/epistemic-graph`. Anything that appears in a
prior doc or system prompt but is **not** in the parser/executor is filed under
[§4 Known gaps](#4-known-gaps--bug-candidates-documented-but-not-implemented-at-the-uql-text-surface) as a
bug candidate, not the main grammar.

**Correction to a common misreading, up front.** `RANK BY ~[v0, v1, v2, v3]` is
**not** a 4-axis weighted-scoring vector (there is no "axis 0 = relevance, axis 1 =
recency, …" convention anywhere in the code). `Op::Rank { query: Vec<f32> }`
(`eg-types/src/wire.rs:431`) is a **literal embedding-space query vector of
whatever dimensionality the engine's `SemanticStore` embeddings use** — the
executor re-orders the candidate RowSet by **cosine similarity** between `query`
and each candidate's stored embedding (`rank_op`, `exec.rs:611`). The `[1.0, 0.0,
0.0, 0.0]` example that appears in the parser doctest, `docs/uql.md`, and the
agent-utilities `nl_planner` system prompt is a **4-dimensional test/example
fixture vector**, not a semantic schema — the dimensionality is whatever the bound
`SemanticStore`'s vectors are (test fixtures in this codebase are commonly small
for readability). If a caller wants relevance/recency/etc. blended, that is a
**different, real mechanism**: `RERANK MMR <lambda> <k>` (relevance vs. diversity,
§1) and the epistemic `BELIEF AS OF`/`CONFIDENCE`/`SOURCE RELIABILITY` stages
(time-decayed belief weighting, §1) — composed via `|>`, not packed into one
vector's components.

---

## 1. Complete grammar — every implemented clause

UQL is a **pipeline**: `source { "|>" stage }` (`parser.rs:193` `parse_query`). A
`source` seeds an initial `RowSet` (an ordered list of `(id, optional score)`
rows); each `|>`-separated `stage` is `RowSet -> RowSet`. The **empty-set-becomes-
source rule** (CONCEPT:EG-KG.query.empty-set-commutativity, `docs/uql.md:204-229`) means several stages
below are legal as a bare leading clause with no `MATCH` at all — noted per row.

### 1.1 Source stages

| Clause | `Op` | Parser | Feature | Semantics |
|---|---|---|---|---|
| `MATCH (:Label) [WHERE pred_list]` | `Scan{label}` [+ `Filter`] | `parse_scan`, `parser.rs:204` | base `query` | Seed every node whose `type == Label`. An inline `WHERE` is sugar for a following `|> WHERE` (proven byte-identical, `mod.rs:117-135`). |
| `REASON <Class>` / `REASON "Class"` / `REASON <http://…/Class>` | `Reason{target_class, ontology}` | `parse_reason`, `parser.rs:534` (owl build) / `parser.rs:557` (non-owl: clear "not in this build" error) | `owl` | Seed every individual the native OWL 2 reasoner **infers** as a member of the class — including ones with no asserted type edge. An angle-bracketed `<iri>` lexes as one token (`lexer.rs:279` `lex_iri`) distinct from the comparison `<` (disambiguated by requiring a `:` scheme + no whitespace). |
| `FOREIGN "<name>"` | `Foreign{name}` | `parse_foreign`, `parser.rs:480` | base `query` (resolve: `federation`) | Seed from a registered external source. See §2 Federation. |
| bare `AS OF @t`, bare `TEXT "q"`, bare `RANK BY ~[…]`, etc. | (respective Op) | — | — | Per the empty-⇒-source rule, ANY stage fed an empty RowSet acts as a source (`docs/uql.md:206-212`, proven by `plan_proptest::empty_intermediate_reseeds_source_breaks_commute`) — so a query may legally start with no `MATCH`. |

### 1.2 Relational filter (real DataFusion)

| Clause | `Op`/`Pred` | Parser | Semantics |
|---|---|---|---|
| `WHERE prop > n` | `Pred::GtNum{prop,n}` | `parse_pred`, `parser.rs:903-917` | Numeric greater-than. |
| `WHERE prop < n` | `Pred::LtNum{prop,n}` | same | Numeric less-than. |
| `WHERE prop = v` / `prop == v` | `Pred::Eq{prop,value}` | same | String equality (value JSON-stringified). |
| `WHERE p1 > n1 AND p2 = v2 …` | `Filter{preds:[…]}` | `parse_pred_list`, `parser.rs:892` | Conjunction only (no `OR`/`NOT`). Executed via real DataFusion (`filter_op`, `exec.rs:607`); when NOT the first stage it is pushed down as `id IN (…)` over the current candidates. |

`Filter{preds}` lowers to `Op::Filter` (`eg-types/src/wire.rs:427`). **This is the
only place the UQL grammar reaches `Pred` — and `Pred` has 10 variants, not 3**;
see §4.1 for what `WHERE` can never express.

### 1.3 Graph traversal (petgraph BFS)

| Clause | `Op` | Parser | Semantics |
|---|---|---|---|
| `TRAVERSE -[:REL]->{min,max}` | `Traverse{rel,min,max}` | `parse_traverse`, `parser.rs:299` | Follow outgoing `REL` edges `min..=max` hops. |
| `TRAVERSE REL{min,max}` (bare-rel shorthand) | same | same | Identical lowering, no `-[: ]->` syntax required. |
| hop range `{n}` / `{a,b}` / `{a..b}` / omitted | `min,max` | `parse_hop_range`, `parser.rs:318` | `{n}`≡`{n,n}`; `{a,b}`≡`{a..b}`; omitted ≡ `{1,1}` (exactly one hop). `max < min` is a parse error. |

Executed by `traverse_op` (`exec.rs:609`), `Op::Traverse` at `eg-types/src/wire.rs:429`.

### 1.4 Vector rank (SemanticStore kNN)

| Clause | `Op` | Parser | Semantics |
|---|---|---|---|
| `RANK BY ~[v0, v1, …]` (negatives OK: `~[-0.1, 0.2]`) | `Rank{query}` | `parse_rank`→`parse_vector_ref` (Inline), `parser.rs:338`/`357` | Cosine-similarity kNN re-order against the literal vector. `Op::Rank`, `wire.rs:431`; `rank_op`, `exec.rs:611`. |
| `RANK BY ~"free text"` | `RankEmbed{text}` | `parse_vector_ref` (Text), `parser.rs:372` | Server-side NL→vector: the executor resolves `text` to a query vector via the embedder bound on `PlanCtx::with_embedder`. No embedder bound ⇒ a clear typed error (never a panic). `Op::RankEmbed`, `wire.rs:440`; `rank_embed_op`, `exec.rs:613`. |
| `RANK BY ~handle` (bare identifier) | — | `parse_vector_ref` (Handle), `parser.rs:376` | **Reserved forward seam, not implemented** — errors "a bare-ident embedding handle … is a reserved forward seam (no by-name embedding registry yet)" (`parser.rs:344-351`). See §4.2 (this one IS a documented, self-declared gap, not a silent hole). |

### 1.5 Graph-native + diversity rerankers (dependency-free)

| Clause | `Op` | Parser | Semantics |
|---|---|---|---|
| `RERANK NODE_DISTANCE FROM <id>` | `RankNodeDistance{center}` | `parse_rerank`, `parser.rs:506-521` | Score `1/(1+hops)` shortest-path distance from a focal node (unreachable → 0). `wire.rs:446`; `rank_node_distance`, `exec.rs:615`. |
| `RERANK MENTIONS` | `RankMentions{}` | `parse_rerank`, `parser.rs:496-499` | Provenance salience: incoming-edge count, normalized to the set max. `wire.rs:451`; `rank_mentions`, `exec.rs:617`. |
| `RERANK MMR <lambda> <k>` | `RankMmr{lambda,k}` | `parse_rerank`, `parser.rs:500-505` | Maximal Marginal Relevance: greedily trades relevance (a prior `Rank` score) vs. cosine similarity to already-picked items. `lambda∈[0,1]` (1=pure relevance, 0=pure diversity); `k` caps how many are re-ranked (0=all). `wire.rs:459`; `rank_mmr`, `exec.rs:619`. |

### 1.6 Lexical (BM25) rank + hybrid fusion

| Clause | `Op` | Parser | Feature | Semantics |
|---|---|---|---|---|
| `TEXT "query"` | `RankText{query}` | `parse_text`, `parser.rs:824` (gated) / `parser.rs:838` (ungated: clear error) | `text` | BM25 relevance re-order over the lexical index. No text index configured ⇒ empty result (degrade, never error). `wire.rs:466`; `rank_text`, `exec.rs:622`. |
| `FUSE [branch] [branch] …` | `FuseRrf{branches,k:0.0}` | `parse_fuse`, `parser.rs:859` (gated) / `parser.rs:883` (ungated: clear error) | `text` | Runs each bracketed sub-pipeline over the SAME seed, then reciprocal-rank-fuses (`Σ 1/(k+rank)`, `k=60` convention) the ranked id lists — fuses RANKS not raw scores, so multi-branch strength beats single-branch strength. `wire.rs:475`; `fuse_rrf`, `exec.rs:625`. Canonical tri-modal branch set: `[RANK BY ~[…]] [TEXT "q"] [RERANK NODE_DISTANCE FROM "n"]`. |

### 1.7 Bi-temporal time (dependency-free — Pi-safe)

| Clause | `Op` | Parser | Semantics |
|---|---|---|---|
| `AS OF @t` / `AS OF VALID @t` | `AsOf{ts,axis:Valid}` | `parse_asof`, `parser.rs:396` | "What was TRUE at `t`" — filters `valid_from`/`valid_until`, half-open `[from, until)`. `wire.rs:549`; `as_of_filter`, `exec.rs:651`. |
| `AS OF TX @t` / `AS OF TRANSACTION @t` | `AsOf{ts,axis:Transaction}` | same | "What we BELIEVED at `t`" — filters `tx_from`/`tx_to`. |
| `WINDOW <n> [unit]` (bare=seconds, `s`/`m`/`h`/`d`) | `Window{secs}` | `parse_window`, `parser.rs:426` | Tumbling windowed **mean** over `(ts,value)` rows (e.g. from a source that emits scored ts rows), via eg-tsdb `time_bucket`. Under a non-`timeseries` build: RowSet-preserving passthrough. `wire.rs:566`; `window_op`, `exec.rs:659`. |
| `WINDOW <n> [unit] <agg>` (`mean`/`avg`, `sum`, `min`, `max`, `count`, `first`, `last`) | `WindowAgg{secs,agg}` | `parse_window_agg`, `parser.rs:460` | Selectable-aggregate tumbling window. Unit is matched BEFORE the aggregate keyword, so `WINDOW 30 min` is 30 **minutes**, not a `min`-aggregate — write `WINDOW 30 s min` for that. `wire.rs:575`; `window_agg_op`, `exec.rs:665`. |

### 1.8 Federation

| Clause | `Op` | Parser | Semantics |
|---|---|---|---|
| `FOREIGN "<name>"` | `Foreign{name}` | `parse_foreign`, `parser.rs:480` | Names a registered external source. Executor: `foreign_named` (`exec.rs:868`) calls `ctx.foreign.resolve(name)` when a registry is bound (feature `federation`) — REAL federation, not just a marker, as long as the name was registered server-side first via `engine_query(action="register_foreign_source", …)` (`epistemic_graph/client.py:3390`). Without a bound registry (feature off, or none attached) it is pass-through (`exec.rs:878-880`). |

### 1.9 Epistemic belief/evidence (E2)

| Clause | `Op` | Parser | Semantics |
|---|---|---|---|
| `EVIDENCE FOR <id>` | `EvidenceFor{claim_id}` | `parse_evidence_for`, `parser.rs:610` | Nodes with an incoming SUPPORTS-classified edge into `<id>`. `wire.rs:704`; `evidence_for_op`, `exec.rs:719`. |
| `CONTRADICTS <id>` | `Contradicts{node_id}` | `parse_contradicts`, `parser.rs:637` | Nodes with an incoming CONTRADICTS- or ATTACKS-classified edge into `<id>` (an attack is a stronger contradiction). `wire.rs:710`; `exec.rs:721`. |
| `SUPPORTED BY <id>` | `SupportedBy{node_id}` | `parse_supported_by`, `parser.rs:659` | Mirror of `EVIDENCE FOR` — the claims `<id>` itself supports (outgoing). `wire.rs:716`; `exec.rs:723`. |
| `CONFIDENCE` (no argument) | `ConfidenceOp{}` | `parse_confidence`, `parser.rs:778` | Re-scores each row by its own propagated belief confidence, descending. `wire.rs:740`; `confidence_op`, `exec.rs:739`. |
| `SOURCE RELIABILITY <id>` | `SourceReliability{source_id}` | `parse_source_reliability`, `parser.rs:752` | Re-weights every row currently in the set by `<id>`'s propagated reliability (uniform scalar discount). `wire.rs:732`; `exec.rs:734`. |
| `BELIEF AS OF <ts>` | `BeliefAsOf{ts}` | `parse_belief_asof`, `parser.rs:686` | Pins the TRANSACTION-time axis (what the engine believed at `ts`) then re-scores by propagated confidence at that instant. `wire.rs:725`; `belief_as_of_op`, `exec.rs:729`. |
| `VALID AS OF <ts>` | `AsOf{ts,axis:Valid}` (ALIAS) | `parse_valid_asof`, `parser.rs:718` | Pure alias for bare `AS OF @ts` — proven byte-identical (`mod.rs:636-656`). Exists so the epistemic vocabulary reads symmetrically next to `BELIEF AS OF`. |
| `EXPLAIN BELIEF <id>` | `ExplainBelief{node_id}` | `parse_explain_belief`, `parser.rs:797` | Flattens the recursive justification tree (`eg_epistemic::explain_belief`), pre-order deduped, into scored rows. `wire.rs:753`; `explain_belief_op`, `exec.rs:744`. |

All eight are `#[cfg(feature = "epistemic")]`; a non-`epistemic` build still
recognizes each keyword (so the error names the RIGHT clause) but rejects with
"requires the epistemic belief substrate … not available in this build"
(`parser.rs:616-631` etc. — one such fallback per clause).

### 1.10 Terminal

| Clause | `Op` | Parser | Semantics |
|---|---|---|---|
| `LIMIT <k>` | `Limit{k}` | `parse_limit`, `parser.rs:386` | Order-respecting top-k. `wire.rs:755`; `Op::Limit => input.limit(*k)`, `exec.rs:783`. |

### 1.11 Lexical notes (from the lexer, `lexer.rs`)

- **Quote any id containing `- . : @`.** The lexer tokenizes `-` as its own
  symbol (`Tok::Dash`, `lexer.rs:162-166`), so `RERANK NODE_DISTANCE FROM kg-2.0`
  parses `kg` then a stray `-` and errors. Quote it: `"kg-2.0"`.
- **`<...>` is an IRI token only when it is whitespace-free and contains a `:`**
  (`lex_iri`, `lexer.rs:279-301`); otherwise `<` lexes as the comparison
  operator, so `year < 2022` and `REASON <http://ex/Device>` never collide.
- **`=` and `==`** both lex to `Tok::Eq` (`lexer.rs:152-161`) — UQL has no
  distinct identity vs. equality operator.
- Numbers are always `f64` (`lex_number`, `lexer.rs:304-325`); `LIMIT`/hop
  counts additionally require a non-negative integer (`expect_usize`,
  `parser.rs:986-993`).

---

## 2. Modality seams — how UQL reaches each cross-modal subsystem

| Modality | Reached from UQL text? | Clause(s) / mechanism | Not reachable from UQL text (structured-Plan / sibling-dialect only) |
|---|---|---|---|
| **Property graph** (scan/traverse) | ✅ Yes | `MATCH`, `TRAVERSE` (§1.1, §1.3) | — |
| **Relational (DataFusion)** | ✅ Partial | `WHERE` → `Pred::{Eq,GtNum,LtNum}` only (§1.2) | `Pred::JsonPath` (deep JSONPath filter) and all 7 `Pred::Spatial*` variants exist in the wire type (`eg-types/src/wire.rs:24-84`) but `parse_pred` (`parser.rs:903-917`) can never emit them — **§4.1 gap**. Full ad-hoc SQL over `nodes`/`edges` is a sibling top-level dialect: `engine_query(action="sql", …)` → `QueryClient.sql`, `epistemic_graph/client.py:2942` — not embeddable as one pipeline stage. |
| **Vector / ANN** | ✅ Yes | `RANK BY ~[…]` / `~"text"` (§1.4) | Named/by-handle stored embeddings (`~handle`) are a *declared, self-documented* reserved seam (§1.4 row 3, §4.2). |
| **Text / BM25** | ✅ Yes (feature `text`) | `TEXT "…"`, `FUSE […] […]` (§1.6) | — |
| **Graph-native / diversity rerank** | ✅ Yes | `RERANK NODE_DISTANCE\|MENTIONS\|MMR` (§1.5) | — |
| **Bi-temporal time (AS OF / windowed agg)** | ✅ Yes | `AS OF [TX\|VALID] @t`, `WINDOW <dur> [agg]` (§1.7) | These are point-in-time FILTER / windowed-AGGREGATE ops over whatever RowSet reaches them — they are NOT a time-series *source*. See next row. |
| **Native time-series (TSDB)** | ❌ **No** | — | `Op::TsScan{series,from,to}` (source, `wire.rs:677`, executed `exec.rs:775`) and `Op::SensorFuse{streams,tolerance_ns}` (multi-sensor align, `wire.rs:662`) are feature-`timeseries` wire Ops with **no UQL keyword** in `parse_stage`'s dispatch table (`parser.rs:227-296`) — **§4.1 gap**. Only reachable via `action="unified"` with a raw `{"TsScan": {...}}` op dict, or the sibling `engine_timeseries` domain (`kg-modality-timeseries` skill) for direct append/range/asof/gapfill. |
| **OWL reasoning** | ✅ Yes (feature `owl`) | `REASON <Class>` (§1.1) | — |
| **SPARQL** | ❌ **No** (as a pipeline stage) | — | `Op::SparqlBgp{query,var}` (source, `wire.rs:500`, feature `owl-plan`) has **no UQL keyword** — **§4.1 gap**. Full SPARQL SELECT/ASK/CONSTRUCT/DESCRIBE is a sibling top-level dialect: `KnowledgeGraph.sparql()` (`agent_utilities/knowledge_graph/orchestration/engine_query.py:240`) → the engine's native RDF projection — not composable inside a UQL text pipeline. |
| **Epistemic confidence / time-decay** | ✅ Yes | `EVIDENCE FOR`, `CONTRADICTS`, `SUPPORTED BY`, `BELIEF AS OF`, `VALID AS OF`, `SOURCE RELIABILITY`, `CONFIDENCE`, `EXPLAIN BELIEF` (§1.9) | The deeper Phase-1-3 epistemic surface — `ResolveConflict` (argumentation semantics), `CausalEstimate`/`CausalCounterfactual` (Pearl do-calculus), `RankByProvenance`, `ExplainEvidence`, `EpistemicStatus`/`WhatChanged` (bitemporal diff) — are separate `QueryClient` methods reached as distinct `engine_query` **actions** (`epistemic_graph/client.py:3142-3463`), never UQL pipeline stages. This is a genuine capability, just not exposed at the UQL text surface — filed as a lower-priority gap in §4.3 since it was never *documented* as a UQL clause. |
| **Federation (external sources)** | ✅ Yes | `FOREIGN "<name>"` (§1.8), after `engine_query(action="register_foreign_source", …)` | The resolved executor `Op::ForeignScan{source, join}` (`wire.rs:534`) exposes a `join: bool` (intersect-vs-replace) the UQL `Foreign` marker cannot set — **§4.1 partial gap**. |
| **GIS / spatial** | ❌ **No** | — | `Op::SpatialScan{layer,bbox}` (source, `wire.rs:593`), `Op::Reproject{to_epsg,from_epsg}` (CRS transform, `wire.rs:603`), `Op::SpatialOp{kind}` (buffer/hull/simplify/…, `wire.rs:616`), and all 7 `Pred::Spatial*` filter predicates (`wire.rs:50-83`) are feature-`geo` wire types with **zero UQL grammar surface** — **§4.1 gap, the largest one**. The curated `graph_gis` MCP tool (`kg-gis` skill) is an unrelated **RPC-style** routing/tiling surface (`route`/`tile`/`nearest`/`geo_task`) — it is not RowSet-composable and does not go through `eg-plan` at all. |
| **Tensor (N-D array)** | ❌ **No** | — | `Op::TensorScan{layer}` / `Op::TensorOp{kind}` (feature `tensor`, `wire.rs:626,633`) have no UQL keyword — **§4.1 gap**. |
| **Stream / CEP** | ❌ **No** | — | `Op::Cep{pattern}` (feature `stream`, `wire.rs:645`) has no UQL keyword — **§4.1 gap**. |
| **Probabilistic (Bayesian distributions)** | ❌ **No** | — | `Op::Probabilistic{query}` (feature `probabilistic`, `wire.rs:695`) has no UQL keyword — **§4.1 gap**. |
| **WASM UDF** | ❌ **No** | — | `Op::Udf{id}` (feature `wasm-udf`, `wire.rs:514`) has no UQL keyword — **§4.1 gap**. |
| **KV-cache** | N/A — not a RowSet modality | — | `graph_kvcache` (`kg-kvcache` skill) is a content-addressed key→block store with `get`/`put`/`contains`/`stats` — it was never designed as a query modality and has no `Op`/RowSet integration; not a gap, just out of scope for UQL. |

**Reading the table:** every ❌ row is a real wire `Op`/`Pred` variant with a real
executor arm — these are not missing features of the *engine*, they are missing
**surface syntax** in the UQL parser specifically. That asymmetry — implemented in
`eg-plan`'s executor and the `unified` structured-Plan API, invisible to the
UQL text front-end — is exactly the class of cross-seam bug the test matrix in
§5 is built to expose (e.g. an agent that reasonably assumes "if `unified` can do
it, UQL text can too" will silently get a parse error instead of a result).

---

## 3. Cross-seam / cross-modality composition patterns

UQL's algebra composes any of the above (reachable) stages left-to-right over one
`RowSet`. These examples go from single-seam to maximal multi-seam, each annotated
with the modalities it exercises.

**1. Single seam — graph scan + relational filter.**
```
MATCH (:Doc) WHERE year > 2024 AND lang = 'en'
```
Seams: graph (scan) + SQL/DataFusion (filter).

**2. Two seams — graph traversal + vector rank.**
```
MATCH (:Doc) |> TRAVERSE -[:CITES]->{1,2} |> RANK BY ~[0.1, 0.9, 0.0] |> LIMIT 10
```
Seams: graph (scan + traverse) + vector (kNN rank).

**3. Three seams — the canonical pipeline from the parser doctest itself.**
```
MATCH (:Doc) WHERE year > 2024 |> TRAVERSE -[:CITES]->{1,2} |> RANK BY ~[1.0, 0.0, 0.0, 0.0] |> LIMIT 10
```
Seams: graph + SQL + vector. (`eg-plan/src/uql/mod.rs:20-30` docstring; proven
byte-identical to a hand-built `Plan` by `pipeline_parses_to_hand_built_plan`,
`mod.rs:46-73`.)

**4. Time + graph + diversity — no `MATCH` needed (empty-set-as-source).**
```
AS OF @1700000000 |> TRAVERSE -[:CAUSED]->{1,2} |> RERANK MMR 0.5 10
```
Seams: bi-temporal (source) + graph (traversal) + diversity rerank.

**5. Federation + time + graph, composed in one plan (the parser's own
composition proof).**
```
MATCH (:Event) WHERE level > 3 |> FOREIGN "peer-west" |> AS OF @1700000000 |> WINDOW 1 h |> TRAVERSE -[:CAUSED]->{1,2} |> LIMIT 10
```
Seams: graph + SQL + federation + bi-temporal + windowed time-series + graph
traversal. (`mod.rs:390-424` `composed_time_federation_query_parses`.)

**6. Tri-modal hybrid retrieval (RRF fusion).**
```
MATCH (:Doc) |> FUSE [RANK BY ~[1.0, 0.0]] [TEXT "graph databases"] [RERANK NODE_DISTANCE FROM "n1"] |> LIMIT 5
```
Seams: graph (scan + node-distance) + vector + BM25 text, fused by reciprocal
rank. (`mod.rs:484-522` `fuse_stage_lowers_to_fuse_rrf_like_the_builder`.)

**7. OWL reasoning seeding a vector rank.**
```
REASON <http://ex/Device> |> RANK BY ~[0.2, 0.4, -0.1] |> LIMIT 5
```
Seams: OWL/SPARQL-adjacent inference (source) + vector rank. (`REASON` seeding a
downstream `Rank`/`Traverse`/`Filter`/`Limit` is explicitly the documented
composition, `docs/uql.md:86`.)

**8. Epistemic evidence discounted by belief-time confidence.**
```
MATCH (:Claim) |> EVIDENCE FOR "c1" |> BELIEF AS OF @1700000000 |> LIMIT 10
```
Seams: graph + epistemic evidence-graph walk + bi-temporal belief-confidence
re-scoring. (`docs/uql.md:201-202`, the documented canonical epistemic example.)

**9. NL→vector (server-side embed) + BM25 fused, then epistemically filtered.**
```
MATCH (:Claim) |> FUSE [RANK BY ~"climate risk disclosure"] [TEXT "climate risk disclosure"] |> CONTRADICTS "c1" |> CONFIDENCE |> LIMIT 10
```
Seams: graph + server-side NL-embed vector + BM25 (fused) + epistemic
contradiction-graph filter + belief-confidence re-score. Composes §1.4's
`RankEmbed` seam with §1.9's epistemic ops in one plan — not directly
demonstrated in the test suite, so treat as a **live probe**, not a proven-safe
pattern (see §5, test 15).

**10. Maximal — graph, relational, federation, time, vector, text, graph-native
rerank, diversity, and epistemic confidence in ONE pipeline.**
```
MATCH (:Doc) WHERE year > 2020
  |> FOREIGN "peer-east"
  |> AS OF @1700000000
  |> WINDOW 1 h
  |> TRAVERSE -[:CITES]->{1,2}
  |> FUSE [RANK BY ~[0.1, 0.8, 0.1]] [TEXT "systemic risk"] [RERANK NODE_DISTANCE FROM "n1"]
  |> RERANK MMR 0.6 20
  |> CONFIDENCE
  |> LIMIT 10
```
Seams: graph + SQL + federation + bi-temporal (AsOf+Window) + graph traversal +
vector + text + graph-native rerank (fused) + MMR diversity + epistemic
confidence. This is a **stress composition** built to the grammar's own rules
(every clause above is individually proven in §1's citations) but not
end-to-end tested as one plan in the engine's own suite — the highest-value
probe for a genuinely novel cross-seam bug (§5, test 16).

---

## 4. Known gaps & bug candidates

### 4.1 Wire `Op`/`Pred` variants with NO UQL grammar surface

These are real, executor-implemented, feature-gated operators — reachable via
`engine_query(action="unified", params_json='{"plan": [...]}')` with a raw op
dict — that **cannot be written as UQL text** because `parse_stage`
(`parser.rs:227-296`) has no dispatch arm for them, and `parse_pred`
(`parser.rs:903-917`) has no syntax for the non-comparison `Pred` variants.
Each is a genuine cross-seam bug candidate: an agent (or the NL→UQL planner,
`agent_utilities/knowledge_graph/core/nl_planner.py`) that generates UQL text
for one of these modalities will get a **parse error**, not a routing failure —
so the failure mode to test for is "did the planner/agent silently fall back to
a different, wrong dialect, or surface the real gap?"

1. **`Pred::JsonPath`** (`eg-types/src/wire.rs:42`) — deep JSONPath document
   filter. No `WHERE` syntax reaches it.
2. **`Pred::SpatialWithin`/`SpatialDWithin`/`SpatialContains`/`SpatialCovers`/
   `SpatialTouches`/`SpatialCrosses`/`SpatialOverlaps`/`SpatialEquals`/
   `SpatialDisjoint`** (`wire.rs:50-83`, 7 DE-9IM relations + within + dwithin)
   — no `WHERE` syntax reaches any of them.
3. **`Op::TsScan`** (`wire.rs:677`, native time-series SOURCE) — `WINDOW`/`AS OF`
   exist in UQL as CONTEXT/filter ops over whatever RowSet reaches them, but
   there is no UQL keyword to *source* rows from a native TSDB series.
4. **`Op::SensorFuse`** (`wire.rs:662`, multi-sensor time-alignment) — no UQL
   keyword.
5. **`Op::SpatialScan`** (`wire.rs:593`, bbox-intersect GIS source), **`Op::
   Reproject`** (`wire.rs:603`, CRS transform), **`Op::SpatialOp`**
   (`wire.rs:616`, buffer/hull/simplify/centroid/union/intersection/
   difference) — no UQL keyword for any of the three. GIS is the single
   largest unreachable modality: neither the source, the transform, nor the
   filter predicates (item 2) have any UQL surface at all.
6. **`Op::TensorScan`/`Op::TensorOp`** (`wire.rs:626,633`, N-D array modality)
   — no UQL keyword.
7. **`Op::Cep`** (`wire.rs:645`, bounded-NFA complex-event-processing) — no
   UQL keyword.
8. **`Op::Probabilistic`** (`wire.rs:695`, Bayesian-distribution scoring) — no
   UQL keyword.
9. **`Op::Udf`** (`wire.rs:514`, sandboxed WASM RowSet transform) — no UQL
   keyword.
10. **`Op::SparqlBgp`** (`wire.rs:500`, SPARQL basic-graph-pattern as a RowSet
    source) — no UQL keyword; full SPARQL only exists as the sibling top-level
    dialect (`KnowledgeGraph.sparql()`), not embeddable in a UQL pipeline.
11. **`Op::ForeignScan`'s `join: bool`** (`wire.rs:534`) — the resolved
    federation executor supports intersect-with-local (`join:true`) vs.
    replace (`join:false`); the UQL `Foreign{name}` marker (§1.8) always
    resolves through `foreign_named` (`exec.rs:868-873`), which has no way to
    request `join:true` from text — a **partial** gap (the source itself
    works; one of its two modes doesn't).

### 4.2 Self-documented reserved seam (not a silent bug, but worth probing)

- **`RANK BY ~handle`** (bare-identifier vector reference, `parser.rs:344-351`,
  `376-379`) is explicitly rejected with "a reserved forward seam (no by-name
  embedding registry yet)". This is intentional, declared-in-code, and should
  NOT be treated as a bug — but it IS worth a regression probe (test 17, §5)
  to confirm the error stays a clear typed rejection and never silently
  degrades to a different vector or a crash as the embedder-registry seam
  evolves.

### 4.3 Deeper epistemic surface — reachable, but not as a UQL stage

`ResolveConflict` (argumentation semantics), `CausalEstimate`/
`CausalCounterfactual` (Pearl do-calculus), `RankByProvenance`,
`ExplainEvidence`, `EpistemicStatus`/`WhatChanged` (bitemporal diff) —
(`epistemic_graph/client.py:3142-3463`) are real, served `QueryClient` methods,
each its own `engine_query` action, but **none** of them has ever been
documented as a UQL clause (unlike the eight in §1.9) — so this is not a
"broken promise" gap the way §4.1 is. Listed here only so the cross-seam test
matrix doesn't waste a probe assuming they're UQL-embeddable.

### 4.4 Feature-gating is real, not cosmetic — verify per build

Every gated clause (`owl`→`REASON`, `text`→`TEXT`/`FUSE`, `epistemic`→ the
eight in §1.9) parses in every build (the front-end is dependency-free,
`mod.rs:9-11`) but only **executes** when the running server was compiled with
that feature — an ungated build returns a clear "`<CLAUSE>` requires … not
available in this build" error (one fallback function per clause, e.g.
`parser.rs:556-570` for `REASON`). A cross-seam test against a slim/Pi build
should expect that message, not a crash or a silently-empty result — confirm
which features the target deployment's `epistemic-graph-server` was actually
built with before treating a "not in this build" response as a bug.

---

## 5. Cross-seam test matrix

Concrete UQL queries, single-seam → maximal cross-modality, for the orchestrator
to run and observe. Each row: the query, the seams it exercises, and the
**expected shape** — a genuine mismatch (wrong shape, wrong error class, silent
empty result where rows were expected) is the bug signal, not the presence of
an error itself (some of these — 12-14 — are EXPECTED to error).

| # | Query | Seams exercised | Expected shape |
|---|---|---|---|
| 1 | `MATCH (:Doc)` | graph scan only | `[{"id","score":null}, …]` — every `Doc` node, `score` unset (no `Rank` ran). |
| 2 | `MATCH (:Doc) WHERE year > 2024 AND lang = 'en'` | graph + SQL/DataFusion filter | Rows narrowed to the predicate; `score:null`. |
| 3 | `MATCH (:Doc) |> TRAVERSE -[:CITES]->{1,2}` | graph + graph traversal | 1-2-hop `CITES` neighbors of every `Doc`, possible duplicates depending on de-dup semantics — check for stable ids. |
| 4 | `MATCH (:Doc) |> RANK BY ~[1.0, 0.0, 0.0, 0.0] |> LIMIT 5` | graph + vector | ≤5 rows, `score` populated, descending cosine similarity. |
| 5 | `MATCH (:Doc) |> TEXT "graph databases" |> LIMIT 5` | graph + BM25 text (feature `text`) | ≤5 rows scored by BM25, OR a clear "not in this build" error on a non-`text` server. |
| 6 | `AS OF @1700000000 |> LIMIT 5` | bi-temporal source (no `MATCH`) | Confirms the empty-⇒-source rule: nodes live at `ts`, not an empty result. |
| 7 | `MATCH (:Event) |> WINDOW 1 h SUM` | graph + windowed time-series aggregate | One row per non-empty hour bucket if the rows carry `(ts,value)`; else the RowSet-preserving passthrough on a non-`timeseries` build — confirm which. |
| 8 | `MATCH (:Doc) |> TRAVERSE -[:CITES]->{1,2} |> RANK BY ~[0.1,0.9,0.0] |> LIMIT 10` | graph + traversal + vector | The 3-seam canonical pipeline (§3.3) — must match the parser doctest's proven shape. |
| 9 | `MATCH (:Doc) |> FUSE [RANK BY ~[1.0,0.0]] [TEXT "graphs"] [RERANK NODE_DISTANCE FROM "n1"] |> LIMIT 5` | graph + vector + text + graph-distance, RRF-fused | ≤5 rows; the fused rank order should NOT equal any single branch's order in general (proves fusion actually ran, not just the first branch). |
| 10 | `REASON <http://ex/Device> |> RANK BY ~[0.2,0.4,-0.1] |> LIMIT 5` | OWL reasoning source + vector | ≤5 rows from the OWL-inferred class membership set, OR a clear "OWL reasoner … not available" error on a non-`owl` build. |
| 11 | `MATCH (:Claim) |> EVIDENCE FOR "c1" |> BELIEF AS OF @1700000000 |> LIMIT 10` | graph + epistemic evidence-graph + belief-time confidence | ≤10 rows, epistemic-scored, OR a clear "epistemic … not available" error. |
| 12 | `MATCH (:Doc) |> SPATIAL_SCAN "Doc" [0,0,10,10]` *(deliberately invalid — GIS has no UQL keyword)* | attempted GIS source — §4.1 item 5 | **Expected: a UQL PARSE error** ("expected a pipeline stage…") — if this instead silently returns rows or a 500, that is a real bug (the parser is accepting/mis-lexing something it shouldn't). |
| 13 | `MATCH (:Doc) WHERE geometry WITHIN "POLYGON(...)"` *(deliberately invalid — no spatial `WHERE` syntax)* | attempted `Pred::SpatialWithin` via `WHERE` — §4.1 item 2 | **Expected: a UQL PARSE error** at the `WITHIN` token (`parse_pred` only knows `>`/`<`/`=`). Confirms the documented gap rather than a numeric-comparison mis-parse. |
| 14 | `MATCH (:Metric) |> TS_SCAN ["cpu.util"] 0 3600` *(deliberately invalid — TsScan has no UQL keyword)* | attempted native-TSDB source — §4.1 item 3 | **Expected: a UQL PARSE error.** Cross-check against `engine_query(action="unified", params_json='{"plan":[{"TsScan":{"series":["cpu.util"],"from":0,"to":3600}}]}')`, which SHOULD succeed on a `timeseries` build — the pair proves the gap is UQL-text-specific, not an engine-wide absence. |
| 15 | `MATCH (:Claim) |> FUSE [RANK BY ~"climate risk disclosure"] [TEXT "climate risk disclosure"] |> CONTRADICTS "c1" |> CONFIDENCE |> LIMIT 10` | graph + server-embedded vector + BM25 (fused) + epistemic contradiction + confidence | Not directly proven in the engine's own test suite (§3.9) — watch for a silent wrong-order result or an unbound-embedder error swallowed into an empty RowSet instead of surfacing. |
| 16 | The §3.10 maximal composition (graph+SQL+federation+bitemporal+traverse+fuse+MMR+confidence, 9 stages) | ALL reachable seams in one plan | The highest-value probe: if any two adjacent stages interact wrongly (e.g. `WINDOW` after `TRAVERSE` sees non-`(ts,value)` rows, or `CONFIDENCE` after `RERANK MMR` loses the epistemic node identity), that is exactly a cross-seam bug this matrix exists to surface. Requires `federation`+`text`+`epistemic`+`timeseries` all compiled in — expect a specific "not in this build" error naming the FIRST unsupported clause if the target server is a slim build, not a generic failure. |
| 17 | `MATCH (:Doc) |> RANK BY ~myhandle` | reserved vector-handle seam (§4.2) | **Expected: a clear typed parse-time error** ("reserved forward seam … no by-name embedding registry yet") — regression-check that this stays a declared rejection, not a silent fallback. |
| 18 | `MATCH (:Doc) |> RANK BY ~[-0.1, 0.2, -0.3] |> LIMIT 3` | vector rank with negative components | Confirms negatives parse and rank identically to the Rust builder (`mod.rs:267-286`) — a regression probe on §1.4's negative-component note. |
| 19 | `MATCH (:Event) WHERE level > 3 |> FOREIGN "peer-west" |> AS OF @1700000000 |> WINDOW 1 h |> TRAVERSE -[:CAUSED]->{1,2} |> LIMIT 10` (needs `register_foreign_source("peer-west", …)` first) | graph + SQL + federation + bitemporal + windowed + traversal | Must match `mod.rs:390-424`'s hand-built `Plan` exactly — a strong regression oracle since the engine's own test proves the byte-identical AST; a live-engine mismatch in ROW CONTENT (not just AST shape) isolates the bug to the *executor*, not the parser. |
| 20 | `engine_query(action="unified", params_json='{"plan":[{"Scan":{"label":"Doc"}},{"SpatialOp":{"kind":"Centroid"}},{"Limit":{"k":5}}]}')` (structured form, bypassing UQL text entirely) | graph + GIS transform via the BUILDER surface | Should succeed on a `geo` build (proves the GIS *executor* works) even though test 12 proves the same modality is UQL-text-unreachable — the pair (12 vs 20) is the cleanest demonstration of the §4.1 class of gap: same engine capability, one surface reaches it and the other doesn't. |

**How to use this matrix:** run 1-11 and 18-19 first (all-implemented-and-
documented — any failure here is a genuine executor/wiring bug). Run 12-14 to
confirm the §4.1 gaps fail the RIGHT way (a clear parse error, not a crash or a
silent wrong answer) — a PASS on these means "gets rejected as documented", a
FAIL means either the parser silently accepts nonsense or the gap has been
closed and this reference needs updating. Run 15-17 as deeper probes into
undertested composition + the one declared reserved seam. Run 20 paired with 12
to prove the executor/parser asymmetry directly.
