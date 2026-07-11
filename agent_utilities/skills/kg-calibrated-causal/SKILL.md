---
name: kg-calibrated-causal
skill_type: skill
description: >-
  Genuine Pearl do-calculus over a small structural causal model you supply, plus
  provenance-aware retrieval ranking that scores candidates on evidence quality (source
  reliability, corroboration, calibration precision, freshness) as well as similarity.
  Use for "what happens if I set X to this value" (a true do(X=x) intervention — graph
  surgery, not conditioning), or "rank these candidates by how well-evidenced they are,
  not just how similar." Distinct from `kg-ops-causal`, which reasons over the REAL
  ingested ops entity graph (traces/services/deploys/incidents); this skill is a pure
  function over a caller-supplied model/candidate list — no graph node is read.
license: MIT
tags: [graph-os, epistemic, causal, do-calculus, ranking, engine, calibration]
tier: core
wraps: [engine_query]
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-calibrated-causal

Two `engine_query` actions, both pure functions over the request (no graph read),
both requiring the opt-in `epistemic-causal` engine feature (not in the default `full`
build):

- **`causal_estimate`** — `P(· | do(X₁=x₁, X₂=x₂, …))` over a linear-Gaussian
  structural causal model. `variables` defines the DAG in topological order, one dict
  per variable: `{"id": "z", "parents": [], "bias": 0.0, "noise_var": 1.0}` (each
  `parents` entry is `[parent_id, weight]`, referencing an EARLIER entry). `do_values`
  fixes named variables via graph surgery — incoming edges to the `do`-fixed variable
  are CUT, not conditioned on, which is the operationally meaningful difference between
  `do(X=x)` and `observe(X=x)`. Returns `{"estimates": [[var_id, {"mean", "variance",
  "interval": [lo, hi], "level"}], ...]}`, one calibrated estimate per variable in the
  same order as `variables`.
- **`rank_by_provenance`** — order caller-supplied `candidates` by a weighted blend of
  similarity AND evidence quality, so a well-sourced/well-corroborated/fresh result
  isn't outranked by a merely-more-similar unsourced one. Each candidate:
  `{"id": "doc-1", "similarity": 0.7, "source_reliability": 0.95, "freshness": 0.9,
  "calibration": {"interval": [0.85, 0.95], "level": 0.95, "evidence_count": 5}}`
  (`calibration` optional — omit for a candidate with no evidence-graph backing).
  `weights` is `{"similarity": w1, "evidence_quality": w2}`, defaulting to `{0.5, 0.5}`.
  Returns `{"ranked": [{"id", "score", "similarity", "evidence_quality"}, ...]}`,
  highest first.

## Invoke

- **MCP:** `load_tools(tools=["engine_query"])`.
- **REST twin:** `POST /engine/query` with `{"action": "causal_estimate", "params_json": "..."}`.

## Example — do-intervention

```jsonc
engine_query(action="causal_estimate", params_json='{
  "variables": [
    {"id": "z", "parents": [], "bias": 0.0, "noise_var": 1.0},
    {"id": "x", "parents": [["z", 1.0]], "bias": 0.0, "noise_var": 0.25},
    {"id": "y", "parents": [["z", 1.0], ["x", 0.5]], "bias": 0.0, "noise_var": 0.25}
  ],
  "do_values": {"x": 2.0}
}')
// -> {"estimates": [["z", {"mean": 0.0, "variance": 1.0, "interval": [-1.96, 1.96], "level": 0.95}],
//                    ["x", {"mean": 2.0, "variance": 0.0, "interval": [2.0, 2.0], "level": 0.95}],
//                    ["y", {"mean": 1.0, "variance": 0.25, "interval": [0.02, 1.98], "level": 0.95}]]}
// z's own variance is UNCHANGED by the do(x=2.0) intervention — that's graph surgery
// working correctly: fixing x does not retroactively explain z (no backward info flow).
```

## Example — provenance-aware ranking

```jsonc
engine_query(action="rank_by_provenance", params_json='{
  "candidates": [
    {"id": "doc-1", "similarity": 0.92, "source_reliability": 0.4, "freshness": 0.2},
    {"id": "doc-2", "similarity": 0.81, "source_reliability": 0.95, "freshness": 0.9,
     "calibration": {"interval": [0.85, 0.95], "level": 0.95, "evidence_count": 5}}
  ],
  "weights": {"similarity": 0.4, "evidence_quality": 0.6}
}')
// -> {"ranked": [{"id": "doc-2", "score": 0.83, ...}, {"id": "doc-1", "score": 0.51, ...}]}
// doc-2 outranks the more-similar-but-unsourced doc-1.
```

## Honest limitations

- Requires the opt-in `epistemic-causal` engine feature (gates BOTH `CausalEstimate`
  and `RankByProvenance` — neither is in `full`). Degrades to `{"error": ...}` on a
  build without it.
- `causal_estimate` only wires the do-calculus **intervention** op. The engine's
  `observe`/`counterfactual` methods (conditional inference and Pearl's full
  three-step counterfactual recipe) exist in the same Rust module but are
  **crate-internal only** — no wire `Method` exposes them, so neither this skill nor
  any other AU/MCP surface can reach them today. If you need a genuine counterfactual
  ("what WOULD have happened if"), that is not yet reachable from outside the engine.
- Both actions are pure functions over the request body — they do not read any graph
  node. To reason over the REAL ops entity graph (a live incident, a real deploy),
  use `kg-ops-causal` (`graph_ops_causal`) instead — the two are complementary, not
  overlapping.
