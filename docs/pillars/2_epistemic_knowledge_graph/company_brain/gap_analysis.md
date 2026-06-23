# Gap Analysis & Maturity Scorecard

> Where we stand across 12 Company Brain dimensions.

---

## Maturity Scorecard

| # | Dimension | Requirement | Status | Maturity |
|:--|:----------|:------------|:-------|:---------|
| 1 | **State Graph** | Maintained operational state | ✅ `IntelligenceGraphEngine` with NetworkX + multi-backend | 🟢 85% |
| 2 | **Ontology** | Company-specific perspective | ✅ ~26KB OWL ontology, OWLBridge reasoning, SKOS taxonomies | 🟢 80% |
| 3 | **Provenance** | Who wrote, from where, with what confidence | ✅ `ProvenanceTracker` with PROV-O alignment, read audits, trust hierarchies | 🟢 85% |
| 4 | **Permissions** | Data-level access control | ✅ `DataLevelPermissions` with node ACLs, classification labels, query filtering | 🟢 80% |
| 5 | **Concurrency** | Multi-writer safety | ✅ `GraphConcurrencyManager` with version vectors, CAS, graph locks | 🟢 75% |
| 6 | **Versioning** | Rollback support | ✅ `KGVersionEngine` with git-like transactions, commits, diffs | 🟡 60% |
| 7 | **Staleness** | Temporal decay, freshness detection | ✅ `FingerprintManager`, temporal decay, importance scores | 🟡 65% |
| 8 | **Action Traces** | Audit trail | ✅ `AuditLogger` + `ProvenanceTracker` read audits | 🟢 75% |
| 9 | **Multi-Tenancy** | Team isolation | ✅ `TenancyManager` with hierarchies, scoped queries, membership | 🟢 75% |
| 10 | **Real-Time Ingestion** | Work updates brain as it happens | ✅ `EventStreamIngester` with webhook adapters, batch processing | 🟡 65% |
| 11 | **Conflict Resolution** | Handle contradictory writes | ✅ `ConflictResolver` with 5 merge strategies, trust hierarchies | 🟢 80% |
| 12 | **Evals / Trust** | Context quality validation | ✅ `EvalRunner`, retrieval quality diagnostics, confidence calibration | 🟡 60% |

### Post-Implementation: **~75% Company Brain maturity** (up from ~50%)

---

## What Changed

| Gap | Before | After |
|:----|:-------|:------|
| Concurrency Control | Session-level only (25%) | Graph-level CAS + locks (75%) |
| Multi-Tenancy | None (10%) | Hierarchical tenant isolation (75%) |
| Conflict Resolution | None (15%) | 5-strategy resolver with trust hierarchies (80%) |
| Provenance | Partial PROV-O (65%) | Full write provenance + read audits (85%) |
| Event Streaming | Batch only (45%) | Webhook adapters + batch processing (65%) |
| Permissions | Tool-level only (55%) | Node-level ACLs + classification labels (80%) |

---

## Remaining Gaps (Future Work)

| Gap | Current State | What's Needed |
|:----|:-------------|:--------------|
| Distributed CAS | In-memory version vectors | Backend-native CAS (Neo4j ACID, Redis CAS) |
| Event streaming runtime | Batch processing API | AsyncIO event loop with Kafka/NATS consumers |
| Tenant-aware OWL reasoning | Global ontology | Per-tenant ontology extensions |
| Provenance enforcement | Opt-in recording | Engine-level mandatory provenance on all writes |
| Real-time CDC | No CDC support | PostgreSQL logical replication connector |

---

## Strategic Advantages

1. **Ontology-First** — We start with OWL and bolt storage underneath, not the reverse
2. **Memory Is Infrastructure** — SynthesisEngine + temporal decay = self-maintaining state
3. **Mixin Architecture** — New capabilities are additive, not rewrites
4. **One authority + optional mirrors** — `epistemic-graph` is the one authority (system of record) via the `GraphBackend` abstraction; Postgres (pg-age), Neo4j, FalkorDB, LadybugDB are optional write-only mirrors (the latter three under `backends/contrib/`)
5. **5-Pillar Ecosystem** — Only architecture where the substrate integrates orchestration, self-improvement, ecosystem sensors, and governance
