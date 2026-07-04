# Concurrency Control

> **The Question**: Who wins when two actors — whether human analysts, AI agents, or hybrid teams — write to the same state simultaneously?

---

## Why Graph-Level Concurrency Matters

Without concurrency control, the Knowledge Graph is a **single-writer system**. Two agents writing to the same customer node simultaneously cause silent data loss — the last write overwrites the first with no conflict detection, no audit trail, and no notification.

The Company Brain closes this gap with three mechanisms:

1. **Version Vectors** — Every tracked node maintains a monotonically increasing version counter per actor
2. **Compare-And-Swap (CAS)** — Mutations specify their expected base version; stale writes are rejected
3. **Graph Locks** — Optional pessimistic or advisory locks for high-contention nodes

---

## Version Vectors

A `VersionVector` tracks the write history of a single node:

```python
from agent_utilities.models.company_brain import VersionVector

vv = VersionVector(node_id="customer:001")
vv.increment("agent:risk-v1")    # → version 1
vv.increment("analyst:jane")     # → version 2
vv.increment("agent:risk-v2")    # → version 3

# Check if a mutation based on version 1 is stale
vv.is_stale(1)  # True — version is now 3
vv.is_stale(3)  # False — this is the current version
```

### Fields

| Field | Type | Description |
|:------|:-----|:------------|
| `node_id` | str | The graph node being tracked |
| `versions` | dict[str, int] | Mapping of actor_id → their last version |
| `current_version` | int | Global monotonic version counter |
| `last_writer` | str | Actor who performed the most recent write |
| `last_written_at` | str | ISO timestamp of the most recent write |

---

## Compare-And-Swap (CAS)

The `GraphConcurrencyManager` provides CAS semantics:

```python
from agent_utilities.knowledge_graph.core.company_brain import GraphConcurrencyManager
from agent_utilities.models.company_brain import ActorType

gcm = GraphConcurrencyManager()
gcm.track_node("customer:001")

# Agent reads version 0, then attempts to write
result = gcm.compare_and_swap(
    node_id="customer:001",
    expected_version=0,
    actor_id="agent:risk-analyzer",
    actor_type=ActorType.AI_AGENT,
)
# result.success == True, result.new_version == 1

# Meanwhile, a human also read version 0 and tries to write
result2 = gcm.compare_and_swap(
    node_id="customer:001",
    expected_version=0,  # Stale!
    actor_id="analyst:jane",
    actor_type=ActorType.HUMAN,
)
# result2.success == False, result2.conflict_detected == True
```

### CAS Flow

```
Actor A reads node (version=5)
Actor B reads node (version=5)
Actor A writes (expected=5) → SUCCESS → version becomes 6
Actor B writes (expected=5) → FAIL → conflict detected
    → Actor B must re-read (version=6) and retry
```

---

## Graph Locks

For high-contention nodes, CAS may cause excessive retries. Graph locks provide stronger guarantees:

```python
# Acquire a pessimistic lock
lock = gcm.acquire_lock(
    target_id="customer:001",
    holder_id="agent:risk-analyzer",
    holder_type=ActorType.AI_AGENT,
    mode=LockMode.PESSIMISTIC,
    ttl_seconds=60,
)

# Perform mutations while holding the lock...

# Release when done
gcm.release_lock("customer:001", "agent:risk-analyzer")
```

### Lock Modes

| Mode | Behavior | Use Case |
|:-----|:---------|:---------|
| `OPTIMISTIC` | No lock upfront; version check at commit | Low-contention reads with occasional writes |
| `PESSIMISTIC` | Exclusive lock before any mutation | High-contention shared state |
| `ADVISORY` | Lock is informational; conflicts detected but not prevented | Read-heavy workloads with status visibility |

---

## Actor-Agnostic Design

Concurrency control applies identically to all actor types:

- A **human analyst** updating a customer risk score gets the same CAS check as an **AI agent**
- A **hybrid team** (human + AI pair) holding a lock blocks other actors just like a single agent would
- An **automated service** (CI/CD pipeline) writing build status gets version tracking just like a human developer

The `ActorType` is recorded for provenance but does not affect concurrency behavior.

---

## Integration with KGVersionEngine

The `GraphConcurrencyManager` extends the existing `KGVersionEngine` (CONCEPT:AU-KG.query.object-graph-mapper):

| KGVersionEngine | GraphConcurrencyManager |
|:----------------|:-----------------------|
| Transaction → Commit → Rollback | + Version vectors on individual nodes |
| Dict-based graph state snapshots | + Per-node CAS with conflict detection |
| In-process operation | + Distributed lock support (Redis-backed) |
| Single-writer assumption | + Multi-writer safety |

The two systems are complementary: `KGVersionEngine` provides batch transaction semantics, while `GraphConcurrencyManager` provides per-node write safety.
