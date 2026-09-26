# Enterprise Ingestion Architecture (Hub-and-Spoke)

## Overview

To support enterprise scale (100,000+ employees), the agent-utilities Knowledge Graph architecture shifts from a localized storage layer into a **federated hub-and-spoke intelligence ecosystem**.

In this architecture, `agent-utilities` serves as the core OS Kernel and semantic reasoning engine. However, the heavy lifting of raw data ingestion, transformation, and normalization is externalized to dedicated peripheral agents.

## The Spoke Agents

1. **`microsoft-agent`**: Connects to Entra ID (Azure AD), extracting Employee hierarchies, Security Groups, and Department alignments. Maps these to BFO/PROV-O taxonomy nodes before batching them into the core.
2. **`servicenow-api`**: Acts as an ITSM hook, extracting Incidents, CMDB records (Hardware/Software CIs), and infrastructure dependencies (`dependsOn`).
3. **`workday-agent`** *(planned — not yet implemented as a peripheral package)*: Synchronizes Human Capital data, translating organizational skill profiles and roles into standard ontological structures.

## Core `agent-utilities` Ingestion API

All peripheral agents transmit data into the core via the `IntelligenceGraphEngine.ingest_external_batch()` API.

```python
kg = IntelligenceGraphEngine()
kg.ingest_external_batch(domain="servicenow", entities=batch_of_nodes)
```

By pushing this logic down to high-throughput Cypher `UNWIND` statements, the core graph avoids O(N) memory bottlenecks and synchronous Python iteration issues.

## Architectural Guarantees

- **Decoupled Workloads**: NetworkX acts strictly as an ephemeral compute scratchpad.
- **Idempotency**: `MERGE` ensures that repeated syncs from Workday or AD will update attributes instead of duplicating records.
- **RBAC Enforcement**: The core injects `SecurityClearance` logic on retrieval, meaning the spoke agents can sync freely without leaking executive data to low-clearance tasks.
