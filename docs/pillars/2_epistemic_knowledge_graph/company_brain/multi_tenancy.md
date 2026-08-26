# Multi-Tenancy

> **The Question**: How do you let 50 teams share a single Company Brain without any team seeing another team's confidential data?

---

## Why Multi-Tenancy Matters

An organizational brain serves many teams, departments, and stakeholders. Without tenant isolation:
- Engineering can see HR's compensation data
- The trading desk can see compliance's investigation notes
- A contractor AI agent can access internal strategy documents
- A team's experimental models pollute another team's production state

Multi-tenancy provides **namespace isolation** within the same graph, with **hierarchical inheritance** so parent organizations can see across their children.

---

## Tenant Hierarchies

Tenants form a tree structure:

```
Company (root tenant)
├── Engineering
│   ├── Backend Team
│   ├── Frontend Team
│   └── ML/AI Team
├── Finance
│   ├── Trading Desk
│   └── Compliance
└── HR
    ├── Recruiting
    └── People Analytics
```

**Visibility rules:**
- A tenant can see **its own data** and **its children's data**
- A tenant **cannot** see its siblings' or parents' data
- The root tenant can see everything (organizational admin)

---

## Creating Tenants

```python
from agent_utilities.knowledge_graph.core.company_brain import CompanyBrain
from agent_utilities.models.company_brain import ActorType

brain = CompanyBrain()

# Root tenant (created by a human admin)
company = brain.tenancy.create_tenant(
    "Acme Corp",
    created_by="ceo:alice",
    created_by_type=ActorType.HUMAN,
)

# Child tenants
engineering = brain.tenancy.create_tenant(
    "Engineering",
    parent_tenant_id=company.tenant_id,
    created_by="vp:bob",
    created_by_type=ActorType.HUMAN,
)

# AI-created sub-tenant (yes, AIs can create tenants too)
ml_team = brain.tenancy.create_tenant(
    "ML/AI Team",
    parent_tenant_id=engineering.tenant_id,
    created_by="agent:team-organizer",
    created_by_type=ActorType.AI_AGENT,
)
```

---

## Durable Hierarchy (what `accessible_graphs()` actually reads)

`TenancyManager`'s tree above is **in-memory and resets with the process**. The
hierarchy that the request path actually consults lives in
`agent_utilities.knowledge_graph.core.tenant_registry` — one `:TenantHierarchy`
node per tenant in the tenant-shared `__control__` graph, so it survives a
restart and is identical for every replica reading the same engine.

```python
from agent_utilities.knowledge_graph.core import tenant_registry

# kg:admin only. Refuses a self-parent, a cycle, and anything past MAX_TENANT_DEPTH.
tenant_registry.set_parent("engineering", "acme")
tenant_registry.set_parent("acme", "holdings")

tenant_registry.ancestor_chain("engineering")   # ['acme', 'holdings']
```

Or over MCP/REST (`graph_share` / `POST /graph/share`):

```
graph_share(action="set_parent", tenant_id="engineering", parent_tenant_id="acme")
graph_share(action="hierarchy", tenant_id="engineering")
```

`tenant_sharing.accessible_graphs()` then returns the real precedence chain —
most-specific tenant first, ancestors nearest-first, **commons always last**:

```
['tenant__engineering__kg', 'tenant__acme__kg', 'tenant__holdings__kg', 'kg']
```

`read_union()` resolves a duplicate node id first-in-chain-wins over exactly
that list, so a child tenant's row overrides its parent's, which overrides
commons. Depth is capped at `MAX_TENANT_DEPTH` (4 — a tenant plus 3 ancestors)
because every accessible graph costs another per-graph query on the read hot
path; 5 graphs still fits one `read_union` fan-out wave.

Cross-graph edges are structurally impossible in the engine, so this hierarchy
is a **read-time projection**, never a storage property.

---

## Adding Members

Both humans and AIs join tenants with specific roles:

```python
# Human members
brain.tenancy.add_member("engineer:carol", ActorType.HUMAN,
                          engineering.tenant_id, role="admin")
brain.tenancy.add_member("intern:dave", ActorType.HUMAN,
                          engineering.tenant_id, role="viewer")

# AI agent members
brain.tenancy.add_member("agent:code-reviewer", ActorType.AI_AGENT,
                          engineering.tenant_id, role="member")
brain.tenancy.add_member("agent:security-scanner", ActorType.AI_AGENT,
                          engineering.tenant_id, role="member")

# Hybrid team (human+AI pair working as one unit)
brain.tenancy.add_member("team:carol+code-reviewer", ActorType.HYBRID_TEAM,
                          engineering.tenant_id, role="member")
```

### Roles

| Role | Capabilities |
|:-----|:-------------|
| `admin` | Full read/write/delete, can add/remove members, manage child tenants |
| `member` | Read/write within tenant scope |
| `viewer` | Read-only within tenant scope |
| `service` | Automated read/write (for CI/CD, monitoring agents) |

---

## Query Scoping

The `TenancyManager` automatically injects tenant filtering into Cypher queries:

```python
# Original query
query = "MATCH (n:Entity) RETURN n"

# Scoped to engineering tenant
scoped = brain.tenancy.scope_cypher_query(query, engineering.tenant_id)
# → "MATCH (n:Entity) WHERE n.tenant_id = 'tenant:abc123' RETURN n"
```

This ensures that every query automatically respects tenant boundaries without the caller needing to remember to add filtering.

---

## Membership Queries

```python
# What tenants does an actor belong to?
tenants = brain.tenancy.get_actor_tenants("engineer:carol")
# Returns: [engineering.tenant_id, company.tenant_id]
# (includes parent chain for hierarchical access)

# Is an actor a member of a specific tenant?
brain.tenancy.is_member("engineer:carol", engineering.tenant_id)  # True
brain.tenancy.is_member("engineer:carol", "finance_tenant_id")    # False
```
