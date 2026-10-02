# Epistemic Operations Protocol (retired)

The Epistemic Operations Protocol was a second, Python-side projection of
durable graph state: twelve strict JSON Schemas plus generated Python/Rust
DTOs, shipped from `agent_utilities.protocols.epistemic_operations` alongside
the engine's own generated client.

AU-BOUNDARY-R012 deleted the catalog, the generator
(`scripts/check_epistemic_operations_protocol.py`), and every generated DTO
that had a direct equivalent in the engine's own generated client
(`epistemic_graph.generated.models`): `RequestContext`, `MutationBatch`,
`ChangeEnvelope`, `WorkItem`, `Artifact`, `KnowledgeBatch`, `AnalyticsJob`,
`TraceOutcome`, `ClaimWorkItem`, `EvidenceBundle`, the
`ResourceReservation*`/`ResourceHostUpdate*` families, and the
`DevelopmentLane*` operation request/result types. Every caller that used one
of these now imports `epistemic_graph.generated.models` directly, so the
engine client is the sole Python-side graph projection.

## What remains

`agent_utilities.protocols.epistemic_operations` still exports a handful of
hand-written types that are not a second graph projection — each adds
behavior the engine client does not provide, or deliberately narrows a wire
shape the engine client exposes more broadly, and each still has a caller in
retained Agent Utilities code:

| Type | Why it stays | Caller |
| --- | --- | --- |
| `ProtocolModel` | A reusable fail-closed Pydantic base (`extra="forbid", frozen=True, strict=True`); the engine client inlines its own `model_config` per class instead of exporting a shared base. | `data_prep/**`, `orchestration/resource_pool_authority.py`, `orchestration/service_scale_units.py` |
| `OperationResult` / `OperationError` / `OperationRedirect` | AU's own generic success/failure/placement-redirect envelope for public REST/MCP/streaming surfaces; the engine client returns a typed result per method instead of one generic envelope. | `security/error_surface.py` |
| `PlacementRoute` | A narrower, `extra="forbid"` view that deliberately rejects the engine client's additive `endpoints` field (see `knowledge_graph/core/placement_catalog.py`'s `_validate_answer` docstring). | `knowledge_graph/core/placement_catalog.py` |
| `DevelopmentLaneCleanupIntent` | A small internal intent value with no matching request/result shape in the engine client. | `orchestration/repository_work_item.py` |

None of these carry a network call, a schema catalog, or a generator; they are
plain, hand-maintained types. A follow-up requirement can retire each
individually once its caller moves or the engine client grows a direct
equivalent.
