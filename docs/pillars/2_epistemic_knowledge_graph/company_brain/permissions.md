# Data-Level Permissions

> **The Question**: What if a user can see the Jira ticket but not the customer call that explains it?

---

## Beyond Tool-Level Permissions

The existing `PermissionsKernel` (CONCEPT:AU-OS.config.secrets-authentication) controls **which tools agents can call**. This is necessary but insufficient for a Company Brain — we also need to control **which data actors can see**.

Data-level permissions provide:
- **Node-level ACLs** — Per-node access control lists
- **Data classification labels** — PUBLIC, INTERNAL, CONFIDENTIAL, RESTRICTED
- **Query-time filtering** — Automatic removal of unauthorized nodes from results
- **Actor-agnostic enforcement** — Humans and AIs subject to the same rules

---

## Data Classification

Every node can carry a classification label:

| Level | Visibility | Audit | Example |
|:------|:-----------|:------|:--------|
| `PUBLIC` | All authenticated actors | No | Company blog posts, public docs |
| `INTERNAL` | All actors within the tenant | No | Internal wikis, project plans |
| `CONFIDENTIAL` | Actors with explicit grant only | Optional | Customer PII, financial data |
| `RESTRICTED` | Data owners and admins only | **Mandatory** | Compensation data, legal holds |

```python
from agent_utilities.knowledge_graph.core.company_brain import CompanyBrain
from agent_utilities.models.company_brain import ActorType, DataClassification

brain = CompanyBrain()

# Classify a node
brain.permissions.classify_node(
    "customer:001:ssn",
    DataClassification.RESTRICTED,
    data_owner="compliance:officer-smith"
)

# Classify with default INTERNAL
brain.permissions.classify_node(
    "project:roadmap-2025",
    DataClassification.INTERNAL,
)
```

---

## Node ACLs

Fine-grained access control on individual nodes:

```python
from agent_utilities.models.company_brain import NodeACL

acl = NodeACL(
    node_id="customer:001",
    classification=DataClassification.CONFIDENTIAL,
    data_owner="analyst:jane",
    data_owner_type=ActorType.HUMAN,
    read_actors=["analyst:jane", "agent:risk-analyzer", "manager:bob"],
    write_actors=["analyst:jane", "agent:risk-analyzer"],
    admin_actors=["analyst:jane"],
    read_roles=["compliance_officer", "risk_manager"],
    write_roles=["risk_analyst"],
    tenant_id="trading-desk",
    inherit_from_parent=True,
    audit_on_access=True,
)
brain.permissions.set_acl(acl)
```

---

## Permission Checks

```python
# AI agent tries to read a confidential node
result = brain.permissions.check_permission(
    node_id="customer:001",
    actor_id="agent:risk-analyzer",
    actor_type=ActorType.AI_AGENT,
    action="read",
)
print(result.allowed)  # True — agent is in read_actors

# Unknown agent tries to read
result = brain.permissions.check_permission(
    node_id="customer:001",
    actor_id="agent:random-bot",
    actor_type=ActorType.AI_AGENT,
    action="read",
)
print(result.allowed)  # False
print(result.reason)   # "Read access denied"
```

---

## Query-Time Filtering

Automatically filter query results to respect permissions:

```python
all_results = ["customer:001", "customer:002", "customer:003", "customer:004"]

# Filter to only nodes this agent can see
visible = brain.permissions.filter_nodes(
    node_ids=all_results,
    actor_id="agent:report-generator",
    actor_type=ActorType.AI_AGENT,
    action="read",
    actor_roles=["risk_manager"],
)
# Returns only the nodes with matching ACLs
```

---

## Actor-Agnostic Enforcement

Permission checks apply identically regardless of actor type:
- A **human analyst** is denied access to RESTRICTED nodes if not in the ACL
- An **AI agent** gets the same denial
- A **hybrid team** is evaluated by their combined actor_id
- No actor type receives implicit elevated access
