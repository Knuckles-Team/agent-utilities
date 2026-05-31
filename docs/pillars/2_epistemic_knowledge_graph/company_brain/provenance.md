# Provenance Tracking

> **The Question**: Know what happened, why it mattered, who saw it, which source is trusted, and what action followed.

---

## Why Provenance Is Non-Negotiable

Without provenance, a Company Brain is just a database with extra steps. Provenance answers:

- **Who wrote this?** — A human analyst, an AI agent, or a hybrid team?
- **What type of assertion is this?** — Raw data, agent inference, or human judgment?
- **Why was it written?** — What was the rationale for this mutation?
- **From where was it derived?** — What source nodes or systems informed this?
- **How confident is the writer?** — Self-assessed reliability score
- **Who has read this?** — Read audit trail for compliance and debugging

---

## Write Provenance

Every mutation to the Company Brain generates a `ProvenanceRecord`:

```python
from agent_utilities.knowledge_graph.core.company_brain import CompanyBrain
from agent_utilities.models.company_brain import ActorType, AssertionType

brain = CompanyBrain(enforce_provenance=True)

# AI agent writes a risk assessment
brain.provenance.record_write(
    node_id="customer:001",
    actor_id="agent:risk-analyzer",
    actor_type=ActorType.AI_AGENT,
    action="update",
    assertion_type=AssertionType.AGENT_INFERENCE,
    confidence=0.87,
    source_system="internal_model",
    derived_from=["transaction:batch-2024-Q4", "customer:001:history"],
    rationale="Updated risk score based on Q4 transaction pattern analysis",
    tenant_id="trading-desk",
)

# Human analyst overrides
brain.provenance.record_write(
    node_id="customer:001",
    actor_id="analyst:jane",
    actor_type=ActorType.HUMAN,
    action="update",
    assertion_type=AssertionType.HUMAN_JUDGMENT,
    confidence=0.95,
    source_system="manual_review",
    derived_from=["customer:001:interview-notes", "regulatory:aml-guidelines"],
    rationale="Customer interview revealed legitimate business justification for unusual patterns",
    tenant_id="compliance",
)
```

---

## Read Audits

The Company Brain tracks **who queried what**, closing the "who saw it" gap:

```python
brain.provenance.record_read(
    actor_id="agent:report-generator",
    actor_type=ActorType.AI_AGENT,
    nodes_accessed=["customer:001", "customer:002", "customer:003"],
    query_summary="Generated quarterly risk report for compliance review",
    tenant_id="compliance",
)
```

This enables:
- **Compliance auditing** — Prove who accessed PII and when
- **Debugging** — Trace how an agent formed a conclusion by seeing what it read
- **Security** — Detect unusual access patterns (an agent reading nodes it normally doesn't)

---

## Trust Hierarchies

Formalize source authority so the conflict resolver knows which sources to trust:

```python
from agent_utilities.models.company_brain import TrustHierarchyEntry

# Define source trust levels
brain.provenance.add_trust_entry(TrustHierarchyEntry(
    source_system="crm",
    data_domain="customer",
    authority_level=0.95,
    rationale="CRM is the system of record for customer master data",
    overrides=["slack", "email"],
))

brain.provenance.add_trust_entry(TrustHierarchyEntry(
    source_system="human_review",
    data_domain="compliance",
    authority_level=0.99,
    rationale="Regulatory decisions require human sign-off per SOX",
    overrides=["crm", "internal_model"],
))

# Query trust level
trust = brain.provenance.get_trust_level("crm", "customer")  # → 0.95
```

---

## Provenance-Gated Retrieval

Filter query results to only include nodes from trusted sources:

```python
all_nodes = ["customer:001", "customer:002", "customer:003"]

# Only return nodes with high-confidence provenance
trusted = brain.provenance.filter_by_trust(all_nodes, min_trust=0.8)
```

---

## ProvenanceRecord Fields

| Field | Type | Description |
|:------|:-----|:------------|
| `record_id` | str | Unique provenance record identifier |
| `node_id` | str | The node this provenance applies to |
| `actor_id` | str | Who performed the write |
| `actor_type` | ActorType | Human, AI, hybrid team, service, system |
| `action` | str | create, update, delete, merge |
| `assertion_type` | AssertionType | raw_data, agent_inference, human_judgment, synthesized, external_import |
| `confidence` | float | Self-assessed confidence (0.0–1.0) |
| `source_system` | str | System of record (CRM, Slack, git, manual) |
| `derived_from` | list[str] | Node IDs this was derived from |
| `attributed_to` | str | Actor or system this is attributed to |
| `rationale` | str | Free-text explanation |
| `timestamp` | str | ISO timestamp |
| `session_id` | str | Session context for grouping |
| `tenant_id` | str | Tenant scope |

---

## PROV-O Alignment

Provenance records map directly to W3C PROV-O concepts:

| ProvenanceRecord Field | PROV-O Edge |
|:-----------------------|:------------|
| `derived_from` | `prov:wasDerivedFrom` |
| `attributed_to` | `prov:wasAttributedTo` |
| `actor_id` + `action` | `prov:wasGeneratedBy` |
| `timestamp` | `prov:generatedAtTime` |
