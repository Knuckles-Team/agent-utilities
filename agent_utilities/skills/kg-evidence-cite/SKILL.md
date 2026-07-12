---
name: kg-evidence-cite
skill_type: skill
description: >-
  Resolves a claim/belief node to its EXACT multimodal source loci — a PDF page+box, an
  audio/video interval, a SQL row version, a code line range, a distributed-trace span
  — not just "here's a citation" but "here is precisely where in the source this
  evidence came from." Use when you need to verify or display the underlying evidence
  behind a claim rather than take its confidence score on faith — "where did this come
  from", "cite the exact source", "show me the evidence span".
license: MIT
tags: [graph-os, epistemic, evidence, citation, multimodal, engine]
tier: core
wraps: [engine_query]
metadata:
  author: Genius
  version: '0.1.0'
---

# kg-evidence-cite

`engine_query(action="explain_evidence", params_json='{"node_id": "..."}')` walks the
SAME support/contradiction/attack topology `explain_belief` (see `kg-epistemic-answer`)
walks, but instead of the justification tree it returns every transitively-reachable
node that carries a **located evidence locus** — one of 11 `EvidenceSpan` kinds
(`DocumentSpan`, `TableCellRange`, `ImageRegion`, `PageBox`, `AudioSegment`,
`VideoShot`, `VideoFrameRange`, `MetricWindow`, `RowVersion`, `CodeSymbol`,
`TraceSpan`) — plus the `AssetOccurrence`/`Blob` identity chain it was extracted from.

Returns `{"citations": [{"evidence_id", "kind", "locus", "occurrence_id", "blob_ref"},
...]}` — `kind` is `"Supports"` / `"Contradicts"` / `"Attacks"`; `locus` is the
externally-tagged span (e.g. `{"PageBox": {"document_id", "page", "x", "y", "width",
"height"}}`, or `{"CodeSymbol": {"file", "symbol", "start_line", "end_line"}}`) or
`null` when that node in the chain carries no located evidence of its own.

## Invoke

- **MCP:** `load_tools(tools=["engine_query"])`, then
  `engine_query(action="explain_evidence", params_json='{"node_id": "claim:1"}')`.
- **REST twin:** `POST /engine/query` with
  `{"action": "explain_evidence", "params_json": "{\"node_id\": \"claim:1\"}"}`.

## Example

```jsonc
engine_query(action="explain_evidence", params_json='{"node_id": "claim:mine:abc123"}')
// -> {"citations": [
//      {"evidence_id": "ev:1", "kind": "Supports",
//       "locus": {"PageBox": {"document_id": "doc:report-q3", "page": 4,
//                              "x": 72, "y": 540, "width": 400, "height": 60}},
//       "occurrence_id": "occ:9f2a", "blob_ref": "blob:sha256:..."},
//      {"evidence_id": "ev:2", "kind": "Supports",
//       "locus": {"CodeSymbol": {"file": "orchestration/agent_dispatch_worker.py",
//                                  "symbol": "_fence_still_valid",
//                                  "start_line": 210, "end_line": 245}},
//       "occurrence_id": "occ:1b77", "blob_ref": null}
//    ]}
```

## Honest limitations

- Requires the opt-in `evidence-graph` engine feature (`ExplainEvidence` is always a
  valid wire method, but the handler that actually answers it needs `evidence-graph` —
  **not folded into the default `full` build**). Without it the call falls through to
  the engine's not-built catch-all — check for `{"error": ...}` before trusting an
  empty `citations` list as "no evidence" rather than "feature not built."
- A **second, independent** resolver for the same `EvidenceSpan` shape exists
  engine-side under the separate `alignment` feature (`CasEvidenceResolver`) — it can
  return a real UTF-8 excerpt for `DocumentSpan`/`TableCellRange` loci read straight off
  the blob CAS, where `explain_evidence`'s resolver returns only the located span, not
  the excerpt text. There is **no AU tool or client method reaching `alignment`'s
  resolver today** — the two resolvers are not unified, and this skill only reaches the
  `evidence-graph` one. If you need the actual excerpt bytes for a `DocumentSpan`/
  `TableCellRange` locus, you would resolve `blob_ref` yourself via `graph_write
  action=recall_media` / the blob domain (`engine_blob`), not through this call.
- Every other locus kind (image/audio/video/code/trace) resolves to a real CAS-digest
  **reference**, never a fabricated excerpt — there is no in-tree codec to crop/slice
  pixels or audio samples.

## See also

`kg-epistemic-answer` for the belief/confidence side of the same node;
`kg-context-compile` (`graph_search mode="compiled"`) if what you actually want is a
whole cited-and-budgeted context bundle rather than one node's evidence chain.
