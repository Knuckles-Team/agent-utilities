# AU MCP action parity — bounded source-contract audit (2026-09-25)

This is an evidence checkpoint for EH-624 / MCPI-29. The canonical action table is `au-mcp-parity.tsv`: 810 unique legacy names from 126 AU tools. Its dispositions remain **810 pending, 0 verified operations, 0 approved drops**. A declaration candidate does not prove a served operation or equivalent request, result and authority behavior.

The 32 declaration candidates were checked against GraphOS's 131-op `DECLARED-OPS.json` snapshot (SHA-256 `8ee7d78b46bcd26d75101c97ba1e240bee14186e6163849ad76954d5383a5d58`) and EG T5's generated `methods.json` (SHA-256 `353cb83b1f6b7fc85a937f438f678fe2a4707d095d7befa79ab29d3b0ffa61dc`) and `scopes.json` (SHA-256 `ccf2f005b1383b9bb2c73f67b85b80e470a79e7450087ae7ded19f0e409da99c`). The per-action `reason` field records the exact finding:

| Finding | Candidate actions | Interpretation |
|---|---:|---|
| EG method wire-callable, scope exact and class compatible | 17 | Contract-level candidate; runtime/behavior proof pending |
| EG method scope differs from curated GraphOS op | 5 | Composition blocker; backup/restore and shard operations add `ops:read/admin` |
| EG scope is `service-only`, GraphOS op permits `any` | 2 | Composition blocker; derived series define/drop |
| GraphOS composite handler declared | 8 | Runtime/behavior proof pending |

With the AU environment and the current GraphOS and EG T5 source paths, calling GraphOS `get_registry()` fails closed at `analytics.series.define: service-only EG method exposed to users`. The registry is therefore not a served-parity oracle yet. The GraphOS operation owner is repairing the exact authority mismatches. After composition succeeds, the serving owner must provide a canonical registry snapshot and a served-binding list from the bound six-verb runtime; then run `scripts/check_au_mcp_parity.py` with `--served-ops` and audit request/result/authority behavior before converting pending rows to `op`.

`au-skill-operation-requirements.tsv` reconciles the deployment skill's 105 distinct legacy-shaped names: 101 are AU tool names covering 745 of the 810 action rows; four (`graph_finance`, `graph_mine`, `graph_mine_deep`, `graph_rlm`) are outside that AU source inventory and need GraphOS/external ownership decisions. None of these 105 skill names is yet a verified operation mapping. Keep their executable examples marked as EH-624 pending.

The legacy `engine_*` wrapper parses `params_json`, injects the resolved graph when the client method requires it, and pre-embeds quoted-text `RANK BY ~"..."` for UQL. EG `invoke_method` treats its `graph` argument as the request `target_graph`; GraphOS `dispatch_public_eg_method` currently sets that argument to `context.caller.tenant`. Named-graph behavior therefore needs explicit target and RLS proof before direct EG candidates can replace legacy calls. In particular, `engine_query_uql` is not behavior-equivalent to the direct `query.uql` EG binding until quoted-rank handling is proven at the new surface. This is recorded in its candidate row.

No AU caller was migrated and no legacy host or tool-spec files were deleted at this checkpoint. These changes depend on the corrected composed registry and action-by-action served proof.
