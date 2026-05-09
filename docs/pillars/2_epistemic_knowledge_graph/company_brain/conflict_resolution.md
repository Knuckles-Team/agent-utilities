# Conflict Resolution

> **The Question**: What happens when Agent A says "this customer is low risk" and Analyst Jane says "this customer is high risk"?

---

## The Problem of Contradictions

In a multi-writer brain, contradictions are inevitable. An AI agent analyzing transaction patterns may reach a different conclusion than a human analyst reviewing customer interviews. Neither is necessarily wrong — they have different information and different perspectives.

Without conflict resolution, the graph silently stores both values or the last write overwrites the first. Either way, downstream consumers get unreliable state.

---

## How Conflict Detection Works

The `ConflictResolver` detects when two actors write different values to the same field on the same node:

```python
from agent_utilities.knowledge_graph.core.company_brain import CompanyBrain
from agent_utilities.models.company_brain import ActorType, AssertionType, MergeStrategy

brain = CompanyBrain(default_merge_strategy=MergeStrategy.HIGHEST_CONFIDENCE_WINS)

# Detect a conflict
conflict = brain.conflicts.detect_conflict(
    node_id="customer:001",
    field_name="risk_level",
    value_a="low",
    value_b="high",
    actor_a="agent:risk-analyzer-v2",
    actor_a_type=ActorType.AI_AGENT,
    actor_b="analyst:jane",
    actor_b_type=ActorType.HUMAN,
    assertion_type_a=AssertionType.AGENT_INFERENCE,
    assertion_type_b=AssertionType.HUMAN_JUDGMENT,
    confidence_a=0.72,
    confidence_b=0.91,
)
```

A `ConflictNode` is created preserving both values and their full provenance.

---

## Merge Strategies

Five configurable strategies for resolving conflicts:

| Strategy | Behavior | Best For |
|:---------|:---------|:---------|
| `LAST_WRITE_WINS` | Most recent write takes precedence | Low-stakes, fast-moving state |
| `HIGHEST_CONFIDENCE_WINS` | Higher confidence score wins | AI-vs-AI conflicts |
| `SOURCE_AUTHORITY_WINS` | Higher-authority source wins (per trust hierarchy) | Cross-system conflicts |
| `REQUIRE_HUMAN_ARBITRATION` | Both values preserved; human must choose | High-stakes decisions |
| `MERGE_APPEND` | Both values kept as a list | Accumulating observations |

### Applying a Strategy

```python
# Automatic resolution
resolved_value = brain.conflicts.resolve(conflict)
# With confidence 0.91 > 0.72, Jane's "high" wins

# Override strategy for this specific conflict
resolved_value = brain.conflicts.resolve(
    conflict,
    strategy=MergeStrategy.REQUIRE_HUMAN_ARBITRATION,
)
# Returns None — conflict escalated for human review
```

### Per-Type Strategy Overrides

```python
# All customer nodes use source authority
brain.conflicts.set_strategy_for_type(
    "customer", MergeStrategy.SOURCE_AUTHORITY_WINS
)

# All trading signals use highest confidence
brain.conflicts.set_strategy_for_type(
    "trading_signal", MergeStrategy.HIGHEST_CONFIDENCE_WINS
)
```

---

## Trust Hierarchies

For `SOURCE_AUTHORITY_WINS`, the resolver consults a trust hierarchy:

```python
from agent_utilities.models.company_brain import TrustHierarchyEntry

# CRM is authoritative for customer data
brain.conflicts.add_trust_entry(TrustHierarchyEntry(
    source_system="crm",
    data_domain="customer",
    authority_level=0.95,
    rationale="CRM is the system of record for customer data",
))

# Slack messages are supplementary
brain.conflicts.add_trust_entry(TrustHierarchyEntry(
    source_system="slack",
    data_domain="*",
    authority_level=0.3,
    rationale="Slack is informal communication, not authoritative",
))

# Human judgment overrides AI inference for compliance
brain.conflicts.add_trust_entry(TrustHierarchyEntry(
    source_system="human_review",
    data_domain="compliance",
    authority_level=0.99,
    rationale="Regulatory decisions require human sign-off",
))
```

---

## Assertion Types

Every write carries an `AssertionType` that the resolver uses for context:

| Type | Description | Example |
|:-----|:------------|:--------|
| `RAW_DATA` | Direct observation from a system of record | CRM field value, git commit hash |
| `AGENT_INFERENCE` | AI agent's derived conclusion | Risk score from ML model |
| `HUMAN_JUDGMENT` | Human's explicit assessment | Analyst's risk rating |
| `CONSOLIDATED` | Result of memory consolidation | Distilled preference from episodes |
| `EXTERNAL_IMPORT` | Imported from external system without verification | API response, scraped data |

---

## Conflict Lifecycle

```
1. OPEN        — Conflict detected, not yet resolved
2. RESOLVED_AUTO — Resolved by merge strategy
3. RESOLVED_HUMAN — Resolved by human arbitration
4. ESCALATED   — Escalated to higher authority
5. STALE       — Older than retention window, auto-archived
```

### Querying Conflicts

```python
# Get all unresolved conflicts
open_conflicts = brain.conflicts.open_conflicts

# Get full conflict history
all_conflicts = brain.conflicts.all_conflicts
```
