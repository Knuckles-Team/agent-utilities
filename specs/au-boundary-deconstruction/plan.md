# Implementation plan

1. Freeze the public owner map and an old-operation inventory; classify unique gateway, MCP and deployment behavior.
2. Ship EG and SDK generated contracts first, then graph-os served routes, then AU API adapters. Pin exact compatible versions before cutting callers.
3. Cut served shell, connector lifecycle, graph/semantic state, and cross-cutting state in dependency order. Each PR identifies the requirement IDs it closes and proves a complete vertical path.
4. Remove old AU imports, modules, scripts, dependencies and stale tests in the same change that activates replacement wiring.
5. Add a generated owner-manifest rule plus import/script census to prevent reversal. Run full gates and record exact merged-head evidence.

The published specification is independently implementable. Cross-repository owners may keep their own specs, but all AU requirements, decisions and acceptance criteria needed for this cut are stated here.

## AU-BOUNDARY-R049: governed write-back execution

The new module `knowledge_graph/enrichment/writeback/governed.py` adapts AU sinks to the SDK write-back port. `SinkGrant` carries the grant that `run_writeback` already resolved: target, enable flag, risk tier and approval. `build_change_set` projects the sink operations to canonical JSON in one `ops` field. It drops private keys and injected clients. The authorization mode is `proposal_approval` for an approved replay and `standing_policy` otherwise. `SinkWriteBackTransport` runs the sink on a worker thread for `preview` and `apply`. A sink exception during `apply` becomes `OutcomeUncertainError`, and the caller receives the original error. `reconcile` reports `outcome_uncertain`, because a sink has no read-back.

A dry run registers its change set in a temporary `FileWriteBackLedger`, because it records no effect. A live apply uses one `FileWriteBackLedger` directory per change set under `runtime_dir()/writeback-ledger`. `FileAuditReservation` writes the audit reservation in the same directory before the sink call. The EG-backed `EpistemicGraphWriteBackLedger` replaces the file ledger when graph-os serves the EG `WriteBack` operation. graph-os needs no code change: it serves `graph_writeback` from the AU tool registry. GRAPHOS-FLEET-R003 remains the graph-os obligation for a typed, previewed write-back operation.

Known limits: each invocation is a new change set, so the SDK idempotency key does not deduplicate a repeated call. Sink-level external-id stamps still deduplicate creations. The source version comes from the caller signals `expected_source_version` and `current_source_version`, or it is `unversioned`.
