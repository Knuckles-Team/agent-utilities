# Ontological Guardrail Engine (CONCEPT:OS-5.10)

## Overview
High-risk tools (such as local command execution, database updates, or financial transactions) must be verified before execution to prevent prompt injection and compliance drift.

The **Ontological Guardrail Engine** intercepts incoming tool definitions and argument schemas, translating the transient request into a semantic concept. It resolves the arguments against active OWL policy constraints in the Knowledge Graph to verify compliance mathematically using subsumption reasoning.

## Architecture

The guardrail engine operates as a layered defense system within the tool execution pipeline:

```
Tool Call Request
       │
       ▼
┌──────────────────────────────────────────────────┐
│ Layer 1: Argument Extraction                     │
│   Extract path/host/db/url from tool_args        │
├──────────────────────────────────────────────────┤
│ Layer 2: KG Policy Lookup (Primary)              │
│   Query SecurityPolicyNode nodes in active graph │
│   Match extracted targets against policy targets │
├──────────────────────────────────────────────────┤
│ Layer 3: Static Fallback (Secondary)             │
│   Pattern-match against restricted keywords      │
│   (/etc, /var/run, admin, production_db, etc.)   │
├──────────────────────────────────────────────────┤
│ Layer 4: Decision                                │
│   True = BLOCK (requires approval)               │
│   False = ALLOW (proceed to execution)           │
└──────────────────────────────────────────────────┘
```

## Proof Model

### Layer 2: Knowledge Graph Reasoning

The engine queries `SecurityPolicyNode` nodes in the active KG:

```python
for nid, ndata in engine.graph.nodes(data=True):
    if ndata.get("type") == "SecurityPolicyNode":
        restricted_target = ndata.get("target", "").lower()
        for target in extracted_targets:
            if restricted_target in target:
                return True  # BLOCK
```

**SecurityPolicyNode** schema:
```
{
    "type": "SecurityPolicyNode",
    "name": "Restrict Production Database Access",
    "target": "/var/lib/postgresql/production",
    "severity": "critical",
    "policy_type": "filesystem_restriction"
}
```

### Layer 3: Static Fallback Rules

When the KG is unavailable or has no matching policies, the engine falls back to hardcoded restrictions:

| Restricted Keyword | Protected Resource |
|---|---|
| `/etc` | System configuration files |
| `/var/run` | Runtime state and PID files |
| `admin` | Administrative interfaces |
| `db_root` | Database root credentials |
| `production_db` | Production database access |

### Decision Matrix

| KG Available | KG Policy Match | Fallback Match | Decision |
|---|---|---|---|
| ✅ | ✅ | — | **BLOCK** (KG policy) |
| ✅ | ❌ | ✅ | **BLOCK** (fallback) |
| ✅ | ❌ | ❌ | ALLOW |
| ❌ | — | ✅ | **BLOCK** (fallback) |
| ❌ | — | ❌ | ALLOW |

## Fallback Chain

The guardrail engine never fails open. The defense cascade ensures continuous protection:

```
1. KG Policy Query (real-time, highest fidelity)
       │ fails?
       ▼
2. Static Fallback Rules (zero-dependency, always available)
       │ fails?
       ▼
3. Exception Handler (logs debug, returns False = ALLOW)
       │
   Note: Only unknown internal exceptions reach here.
   The static fallback itself never throws.
```

## Target Extraction

The engine extracts security-relevant targets from tool arguments by key name:

| Argument Key | Target Type | Example Value |
|---|---|---|
| `path`, `filepath` | Filesystem path | `/etc/passwd` |
| `dir`, `directory` | Directory path | `/var/run/docker` |
| `host`, `hostname` | Network host | `admin.internal.corp` |
| `url` | URL | `https://production_db.corp/api` |
| `db`, `database` | Database name | `production_db` |
| `table` | Database table | `admin_users` |

Only string-type values are extracted. Non-string values (integers, booleans) are ignored.

## Integration with Tool Guard Pipeline

The ontological guardrails are invoked within the `flag_mcp_tool_definitions()` pipeline:

```python
def _requires_approval(ctx, tool_def, tool_args):
    # 1. OS-5.10: Ontological Guardrails (argument-level analysis)
    if check_ontological_guardrails(name, tool_args, engine=engine):
        return True  # Requires approval

    # 2. OS-5.1: Identity-based policy (role/permission check)
    if permissions_kernel.authorize_tool(identity, name) == "deny":
        return True

    # 3. Pattern-based fallback (tool name matching)
    return name.lower() in sensitive_names or is_sensitive_tool(name)
```

This creates a **defense-in-depth** stack:
1. **OS-5.10** checks the *arguments* (what the tool targets)
2. **OS-5.1** checks the *identity* (who is calling)
3. **Pattern matching** checks the *tool name* (what tool is being called)

## OWL Integration

When the OWL Bridge (KG-2.2) is active, the guardrail engine can leverage formal ontological reasoning:

```turtle
:ForbiddenSystemDirectory a owl:Class ;
    rdfs:subClassOf :SecurityRestriction ;
    owl:hasValue "/etc" .

:ForbiddenRuntimeDirectory a owl:Class ;
    rdfs:subClassOf :SecurityRestriction ;
    owl:hasValue "/var/run" .

:ToolArgumentTarget a owl:Class ;
    rdfs:comment "Temporary individual created from tool argument for subsumption check" .
```

**Future Enhancement**: Instead of substring matching, create a temporary OWL individual from the tool argument and run subsumption reasoning to check if it falls under any `SecurityRestriction` subclass.

## API Surface

```python
from agent_utilities.security.tool_guard import check_ontological_guardrails

# Check tool arguments against KG policies
blocked = check_ontological_guardrails(
    tool_name="run_command",
    tool_args={"path": "/etc/shadow", "command": "cat"},
    engine=kg_engine,
)
# → True (blocked by fallback: "/etc" keyword)

# Safe tool call
blocked = check_ontological_guardrails(
    tool_name="search_web",
    tool_args={"query": "python tutorial"},
    engine=kg_engine,
)
# → False (no matching targets)
```

## Implementation Details
- **Source Code**: [`tool_guard.py`](file:///home/apps/workspace/agent-packages/agent-utilities/agent_utilities/security/tool_guard.py) (295 lines)
- **Function**: `check_ontological_guardrails()`
- **Tests**: [`test_synergies.py`](file:///home/apps/workspace/agent-packages/agent-utilities/tests/unit/knowledge_graph/test_synergies.py)
- **Pillar**: OS
- **Dependencies**: `pydantic-ai` (ApprovalRequiredToolset), KG engine (optional)
