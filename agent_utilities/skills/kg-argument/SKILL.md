---
name: kg-argument
skill_type: skill
description: >-
  Argument Interchange Format (AIF) argument maps: I-nodes (claims) linked
  through S-nodes — RA (rule-of-inference application), CA (conflict
  application), PA (preference application), plus the AIF+ TA
  (transition)/YA (illocutionary) dialogue extensions. Import AIFdb-shaped
  JSON into the KG, export a previously-imported map back out, evaluate Dung
  acceptability (grounded/preferred/stable) via the engine's real
  argumentation solver, and register new named AIF Scheme templates. Use
  when representing or exchanging a structured argument — "import this AIF
  map", "is this claim acceptable given its attackers", "what would defeat
  this argument", "export this argument graph", "register this argumentation
  scheme".
license: MIT
tags: [graph-os, argumentation, aif, epistemic, dung, engine]
tier: core
wraps: [graph_argument, graph_epistemic]
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-argument

> **Condensed intent-surface note (Seam 8).** Under the default intent surface (`MCP_TOOL_MODE=intent`), `graph_argument` is held back from the default tool list (nothing removed — REST + `_execute_tool` still reach it exactly as documented below). Two ways to use this skill unchanged: (1) `load_tools(tools=["graph_argument"])` once per session (as below), then proceed exactly as documented; or (2) call the `act`/`why` intent verb with the same natural-language request — the resolver routes to `graph_argument` for you and returns the result plus a routing justification. Set `MCP_TOOL_MODE=condensed`/`verbose`/`both` to expose the granular tool eagerly instead.

## What AIF is, and how it maps onto this KG

AIF (the Argument Interchange Format — Rahwan & Reed; AIFdb/arg-tech.org) is
the community standard for representing an argument as a graph of two node
kinds: **I-nodes** (information — a claim, premise, or datum) linked through
**S-nodes** (scheme applications) — **RA** (an inference scheme: premises
support a conclusion), **CA** (a conflict scheme: premises conflict with a
conclusion), **PA** (a preference scheme: the conclusion is the preferred one
among the compared premises), plus the AIF+ dialogue extensions **TA**
(transition) and **YA** (illocutionary force).

**This is the interchange vocabulary for argumentation this KG already does
— not a second engine.** An I-node IS a `:Belief` (the same Claim/Evidence/
BeliefState confidence machinery every other belief already uses).
Importing an RA-node also writes the underlying `SUPPORTS` edge; importing a
CA-node also writes the underlying `ATTACKS` edge — so `eg-epistemic`'s
existing Dung argumentation (`Method::ResolveConflict` — grounded/preferred/
stable semantics) computes real acceptability over the SAME topology,
without any grounded/preferred/stable solver being reimplemented anywhere in
this module. See `agent_utilities/knowledge_graph/argumentation/aif.py` and
`docs/architecture/aif-argumentation.md` (with a diagram of the full path).

## Actions (`graph_argument(action=..., ...)`)

1. **`import_aif`** — `argument_map_json` (AIFdb-shaped: `{"nodes": [{"nodeID",
   "text", "type": "I"|"RA"|"CA"|"PA"|"TA"|"YA"}, ...], "edges": [{"edgeID",
   "fromID", "toID"}, ...]}`) [+ optional `map_id`]. Validates arity first
   (an RA/CA-node needs >= 1 premise + exactly 1 conclusion; a PA-node needs
   >= 2 premises + exactly 1 conclusion — the SAME rules
   `shapes/argumentation.shapes.ttl` enforces at admission) and rejects
   locally, with no engine touched, on a malformed map. A valid map writes
   through the shared ChangeEnvelope/ingest path (the SAME primitive every
   native connector uses) — every node becomes a `:Belief`-typed (I-node) or
   AIF-scheme-typed (S-node) claim, every AIF edge becomes a typed
   `aifHasPremise`/`aifHasConclusion` edge, AND every RA/CA-node also mints
   the derived `SUPPORTS`/`ATTACKS` edge the engine's argumentation reads.
2. **`export_aif`** — `map_id` -> read a previously-imported map back out by
   its tag and render it as AIFdb-shaped JSON (best-effort; degrades to an
   empty map rather than raising if the backend can't be reached — the same
   "degraded, not broken" contract every other engine-neighborhood reader in
   this codebase follows).
3. **`evaluate`** — `argument_map_json` and/or `node_ids` [+ optional
   `semantics='grounded'|'preferred'|'stable'`]. Projects the map's CA/PA
   structure (`aif.to_dung`) and hands the I-node ids to the REAL engine
   solver via `graph_epistemic`'s own `resolve_conflict` dispatcher —
   returns `{"engine_result": {"surviving": [...], "defeated": [...],
   "undecided": [...]}, "dung_projection": {"attacks", "preferences",
   "dropped_attacks_by_preference", ...}}`. `dropped_attacks_by_preference`
   is where a PA-node's stated preference discounted one side of a mutual
   (symmetric) CA-node conflict before the engine ran — see "Honest
   limitations" below for exactly what that does and does not change.
4. **`add_scheme`** — `scheme_name` + `scheme_kind='inference'|'conflict'|
   'preference'` [+ optional `description`/`scheme_id`] -> register a new
   named AIF Scheme template (e.g. a Waltonian scheme such as "Argument from
   Expert Opinion") so a future RA/CA/PA-node can `:aifFulfills` it.

## Invoke

- **MCP:** `load_tools(tools=["graph_argument"])`.
- Import a small map:
  ```jsonc
  graph_argument(action="import_aif", argument_map_json='{
    "nodes": [
      {"nodeID": "i1", "text": "It is raining", "type": "I"},
      {"nodeID": "i2", "text": "The ground is wet", "type": "I"},
      {"nodeID": "i3", "text": "The sprinkler was on", "type": "I"},
      {"nodeID": "ra1", "text": "Default Inference", "type": "RA"},
      {"nodeID": "ca1", "text": "Default Conflict", "type": "CA"}
    ],
    "edges": [
      {"edgeID": "e1", "fromID": "i1", "toID": "ra1"},
      {"edgeID": "e2", "fromID": "ra1", "toID": "i2"},
      {"edgeID": "e3", "fromID": "i3", "toID": "ca1"},
      {"edgeID": "e4", "fromID": "ca1", "toID": "i1"}
    ]
  }', map_id="rain-map")
  ```
- Evaluate acceptability over the SAME map (re-supply `argument_map_json`, or
  target already-imported ids directly with `node_ids`):
  `graph_argument(action="evaluate", argument_map_json="<same JSON as above>", semantics="grounded")`.
- Export it back out: `graph_argument(action="export_aif", map_id="rain-map")`.
- Register a reusable scheme: `graph_argument(action="add_scheme", scheme_name="Argument from Expert Opinion", scheme_kind="inference")`.
- **REST twin:** `POST /graph/argument` with the same fields, e.g.
  `{"action": "evaluate", "node_ids": "[\"aif:rain-map:i1\", \"aif:rain-map:i3\"]", "semantics": "grounded"}`.

## Example

```jsonc
// evaluate: i3 ("sprinkler was on") attacks i1 ("it is raining") via ca1,
// with no counter-evidence for i3 itself.
graph_argument(action="evaluate", node_ids='["aif:rain-map:i1", "aif:rain-map:i3"]')
// -> {"surface": "argument", "action": "evaluate", "result": {
//      "semantics": "grounded",
//      "node_ids": ["aif:rain-map:i1", "aif:rain-map:i3"],
//      "engine_result": {"result": {"semantics": "grounded",
//        "surviving": ["aif:rain-map:i3"], "defeated": ["aif:rain-map:i1"],
//        "undecided": [], "extension_sets": [["aif:rain-map:i3", ...]]}}}}
// i3 is unattacked, so it survives (IN); i1 is attacked by a surviving
// argument, so it is defeated (OUT) — the standard grounded (skeptical)
// verdict, computed by eg-epistemic, not by this skill.
```

## Honest limitations

- `to_dung()`/`evaluate` are a PROJECTION + dispatch convenience — the
  grounded/preferred/stable computation itself is `eg-epistemic`'s
  `Method::ResolveConflict` (opt-in `epistemic-tms` engine feature; a build
  without it degrades to a clean `{"error": ...}`, same as `graph_epistemic`).
- RA-node support is written as a `SUPPORTS` edge and reported in
  `dung_projection.supports` for provenance, but classical Dung semantics
  are attack-only — support does not feed the grounded/preferred/stable
  computation itself (it DOES feed the separate, always-on confidence
  propagation `kg-epistemic-answer`'s `explain_belief` reads).
- PA-node preference filtering (`dropped_attacks_by_preference`) only ever
  discounts one side of a MUTUAL (symmetric) CA-node conflict — a
  one-directional attack with no counter-attack is left exactly as declared,
  since there is nothing for a preference to resolve there.
- `export_aif` is a full node+edge scan filtered by tag (mirrors
  `ops_causal_graph.load_ops_causal_neighborhood`'s established idiom) —
  fine for an occasional export, not a hot path on a large shared graph.
- Import is fail-loud (an engine failure raises/surfaces as an error, never
  a silently-dropped write); export is fail-soft (degrades to an empty map)
  — the same asymmetry every other native-connector write/read pair in this
  codebase follows.

## Delegation

If graph-os is reachable, prefer `graph_orchestrate(action="execute_agent")`
for a multi-step "ingest this paper's argument, then tell me which claims
survive" workflow rather than hand-chaining `import_aif` + `evaluate` calls —
use this skill directly for a single map's import/export/evaluate/add_scheme.
