# AU-RETIRE-001 — Design

## Inventory and migration

Start with `agent_utilities/knowledge_graph/core/`, `backends/`, `ingestion/`, `retrieval/`, `enrichment/`, `maintenance/`, `id_management/`, `argumentation/` and `actions/`. Use AST/import and runtime caller inventories, not directory names alone. `agent_utilities/knowledge_graph/backends/epistemic_graph_backend.py` is an existing adapter; `agent_utilities.api` is the retained agent application boundary. Record source symbol, caller, existing EG method, schema digest, owner decision and parity test for each cut. EG's native implementation must be verified in its public repository; absence is an open dependency, not an invitation to create another AU implementation.

## Interfaces and data flow

GraphOS request → AU application API → verified actor/tenant context → generated EG client/port → EG graph operation → typed result/receipt → AU agent response. Keep AU orchestration and LLM proposal semantics at the application layer. For reads, carry graph, tenant and snapshot identity into the client call and preserve citation/provenance in the return. For writes, carry an idempotency key and await EG's authoritative receipt before reporting success. Generated types and exact contract digest replace hand-copied DTOs.

## Cutover

For each operation, implement and test the EG call, rewire a live caller, then remove the AU duplicate and its tests/imports/dependencies in the same cohesive change. Use the existing [cutover matrix](../au-boundary-deconstruction/cutover-matrix.md) for AU-BOUNDARY-R012, AU-BOUNDARY-R018–AU-BOUNDARY-R035 overlap. The AU-RETIRE-R001 audit closes only after every relevant module has an explicit owner outcome and the static owner check covers new additions.

## Failure and security

Treat caller identity as server-verified. A payload cannot select another tenant or grant graph authority. Digest mismatch, unsupported operation, timeout, authorization denial and EG outage fail closed; no local fallback engine, dual write or guessed result. Compare error code, retryability, trace and receipt fields at the public entry point.

## Quality and release

Reuse existing AU API, generated EG client, adapter and owner gates. KISS review rejects a new routing framework. Use `python3 scripts/check_dupehound.py`, `python3 scripts/check_duplication.py diff` for differential jscpd, and `python3 scripts/check_complexity_staged.py` (CCCC rejects new cyclomatic >10 or cognitive >15 functions and regressions). Versions and detailed scanner settings are pinned in `pyproject.toml`; do not invent pass results. Run focused tests then `python3 scripts/uv_workspace.py run --all-extras pytest -q`, `python3 scripts/check_current_only_contract.py` and repository privacy/version gates. Publish exact merged SHA, command, result, fixture and consumer proof before marking LANDED or ACCEPTED.
