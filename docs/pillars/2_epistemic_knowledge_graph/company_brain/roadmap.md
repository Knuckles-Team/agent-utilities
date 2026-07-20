# Roadmap

> From 75% to 100% Company Brain maturity.

---

## Phase 1: Backend-Native Concurrency (Q3 2026)

**Goal**: Push concurrency control from in-memory into the graph backend for true distributed multi-writer safety.

| Task | Description | Complexity |
|:-----|:------------|:-----------|
| Neo4j ACID integration | Use Neo4j native transactions for CAS | Medium |
| Redis distributed locks | Replace in-memory locks with Redis-backed SETNX | Medium |
| Backend version vectors | Persist version vectors as node properties in the graph | Low |
| Conflict-aware KGVersionEngine | Wire ConflictResolver into the commit path | Medium |

---

## Phase 2: AsyncIO Event Streaming Runtime (Q3 2026)

**Goal**: Replace batch-only event processing with a persistent async event loop.

| Task | Description | Complexity |
|:-----|:------------|:-----------|
| AsyncIO event loop | Persistent consumer with backpressure handling | High |
| Kafka consumer adapter | aiokafka-based topic consumer | Medium |
| NATS JetStream adapter | nats-py based stream consumer | Medium |
| Webhook HTTP server | FastAPI endpoint for receiving webhook callbacks | Low |
| CDC connector | PostgreSQL logical replication via psycopg3 | High |

---

## Phase 3: Engine-Level Provenance Enforcement (Q4 2026)

**Goal**: Make provenance mandatory at the engine level, not opt-in at the application level.

| Task | Description | Complexity |
|:-----|:------------|:-----------|
| Engine write wrapper | All `add_node`/`update_node` calls require provenance metadata | Medium |
| Automatic read auditing | Query methods automatically log read audit entries | Medium |
| Provenance-gated query API | New query methods that accept `min_trust` parameter | Low |
| Trust hierarchy persistence | Store trust entries in the graph as first-class nodes | Low |

---

## Phase 4: Tenant-Aware Reasoning (Q4 2026)

**Goal**: OWL reasoning respects tenant boundaries and supports per-tenant ontology extensions.

| Task | Description | Complexity |
|:-----|:------------|:-----------|
| Tenant-scoped promotion | OWLBridge promotes only nodes within tenant scope | Medium |
| Per-tenant ontology imports | Each tenant can import additional OWL ontologies | High |
| Cross-tenant reasoning rules | Configurable rules for when reasoning crosses tenant boundaries | High |
| Tenant-aware synthesis | SynthesisEngine operates within tenant scope | Medium |

---

## Phase 5: Production Hardening (Q1 2027)

**Goal**: Enterprise deployment readiness with full observability, compliance, and scalability.

| Task | Description | Complexity |
|:-----|:------------|:-----------|
| OTEL tracing integration | Trace all Company Brain operations | Medium |
| Prometheus metrics | Expose concurrency, conflict, permission metrics | Low |
| Compliance reporting | Automated SOX/GDPR compliance reports from audit trail | High |
| Load testing | 10,000+ concurrent actors stress test | Medium |
| Horizontal scaling | Multi-node Company Brain with partition tolerance | High |

---

## Success Criteria

The Company Brain is **complete** when:

1. ✅ 50 AI agents and 200 humans can write to the same graph simultaneously without data loss
2. ✅ Every mutation has mandatory provenance (who, what, why, from where, confidence)
3. ✅ Tenant isolation prevents cross-team data leakage with zero configuration
4. ✅ Contradictions are detected and resolved within the write path, not after the fact
5. ✅ Real-time events from Slack, Jira, GitHub update the graph within 1 second
6. ✅ Data classification labels automatically enforce access control at query time
7. ✅ OWL reasoning discovers new facts across tenant boundaries (when authorized)
8. ✅ The system self-maintains: stale data decays, similar concepts merge, episodes consolidate
