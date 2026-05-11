# First Principles Architecture

> **Concepts:** CONCEPT:ORCH-1.2, CONCEPT:AHE-3.3, CONCEPT:ORCH-1.2, CONCEPT:ECO-4.1

This document describes the **First Principles Architecture** layer — a set of four foundational concepts that rewire the routing, dispatch, and feedback loops of `agent-utilities` from basic primitives. These concepts were designed to solve specific scalability, performance, and intelligence bottlenecks that emerge when the system manages dozens of specialists and hundreds of tools.

## Problems Solved

| Problem | Root Cause | Solution |
|---------|-----------|----------|
| **Prompt bloat** | Every routing call serialized the full specialist registry into the LLM prompt | CONCEPT:ORCH-1.2: Hot cache filters to top-7 relevant specialists per query |
| **Redundant team discovery** | LLM re-discovers the same specialist combinations for recurring query patterns | CONCEPT:AHE-3.3: TeamConfig promotes proven coalitions as reusable templates |
| **Static tool binding** | Specialists had fixed tool sets; capabilities like RLM or critic were never auto-attached | CONCEPT:ORCH-1.2: AgentCapability nodes auto-activate based on input constraints |
| **LLM orchestration overhead** | A2A requests required a full LLM planning round-trip even when the graph planner could handle them | CONCEPT:ECO-4.1: PlannerGraphSkill provides a direct graph-backed A2A entry point |
| **No feedback loop** | Execution outcomes were never fed back to improve future routing | CONCEPT:AHE-3.3 + CONCEPT:KG-2.1: Verification outcomes update Self-Model and TeamConfig rewards |

## Architecture Overview

```mermaid
graph LR
    subgraph Ingress ["Protocol Ingress"]
        A2A[A2A] --> PGS["PlannerGraphSkill\n(CONCEPT:ECO-4.1)"]
        ACP[ACP] --> Router
        AGUI[AG-UI] --> Router
    end

    subgraph Routing ["3-Stage Hybrid Routing"]
        PGS --> Router
        Router --> TC{"TeamConfig\nMatch?\n(CONCEPT:AHE-3.3)"}
        TC -- "Hit" --> Dispatch
        TC -- "Miss" --> SM{"Self-Model\nBias?\n(CONCEPT:KG-2.1)"}
        SM --> LLM["LLM Planner\n(Filtered Prompt)"]
        LLM --> Dispatch
    end

    subgraph Execution ["Dispatch & Execute"]
        Dispatch --> Cache["Registry Cache\n(CONCEPT:ORCH-1.2)"]
        Cache --> Specs["Top-7 Specialists"]
        Specs --> Cap{"Capability\nAuto-Activate?\n(CONCEPT:ORCH-1.2)"}
        Cap --> Exec["Parallel Execution"]
    end

    subgraph Feedback ["Post-Execution Feedback"]
        Exec --> Verify["Verifier"]
        Verify --> SMUpdate["Self-Model\nUpdate"]
        Verify --> TCReward["TeamConfig\nReward"]
        SMUpdate --> CacheInv["Cache\nInvalidation"]
        TCReward --> CacheInv
    end
```

---

## CONCEPT:ORCH-1.2 — Registry Hot Cache

**Module:** `agent_utilities/graph/config_helpers.py`

### Problem

Every call to the router required a full registry scan — iterating over all registered specialists (potentially 50+) to serialize their descriptions into the LLM prompt. This created two issues:

1. **Latency**: O(N) scan on every routing call
2. **Prompt bloat**: Injecting 50+ specialist descriptions consumed thousands of tokens, reducing the LLM's effective reasoning window

### Solution

A session-scoped `_RegistryCache` singleton that caches the full specialist registry and provides filtered, query-relevant subsets:

```python
from agent_utilities.graph.config_helpers import (
    get_discovery_registry,        # Full cached registry
    get_relevant_specialists,      # Filtered top-K for a query
    invalidate_registry_cache,     # Event-driven invalidation
)

# Get only the specialists relevant to "deploy to staging"
relevant = get_relevant_specialists(
    query="deploy the app to staging",
    engine=knowledge_engine,
    top_k=7,
)
# Returns: ["DevOps", "Cloud", "Container Manager", ...]
```

### Cache Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Cold: Server Start
    Cold --> Warm: First get_discovery_registry()
    Warm --> Warm: Subsequent calls (O(1))
    Warm --> Cold: invalidate_registry_cache()

    note right of Warm
        Invalidation triggers:
        - /mcp/reload endpoint
        - Pipeline completion
        - SelfModel update
        - TeamConfig promotion
    end note
```

### Invalidation Triggers

The cache is invalidated by 4 event sources, ensuring it stays in sync:

| Trigger | Location | Why |
|---------|----------|-----|
| MCP Reload | `server/app.py` → `POST /mcp/reload` | New tools may create new specialists |
| Pipeline Completion | `pipeline/runner.py` → `run_pipeline()` | Code graph changes may affect routing |
| Self-Model Update | `knowledge_graph/self_model.py` → `update_after_session()` | New proficiency data should influence specialist ranking |
| TeamConfig Promotion | `knowledge_graph/engine_registry.py` → `promote_coalition_to_template()` | New team templates change routing priorities |

---

## CONCEPT:AHE-3.3 — TeamConfig Promotion & Proven Team Reuse

**Module:** `agent_utilities/knowledge_graph/engine_registry.py`

### Problem

The LLM planner would rediscover the same specialist combinations for recurring query patterns. A user who frequently asks "deploy to staging" would see the LLM re-derive the `[DevOps, Cloud, Container Manager]` coalition every time — wasting inference tokens and adding latency.

### Solution

**TeamConfig** nodes persist proven specialist coalitions as reusable templates in the Knowledge Graph. When a similar query arrives, the router checks for a matching TeamConfig *before* invoking the LLM planner.

### TeamConfig Lifecycle

```mermaid
graph TD
    subgraph Discovery ["1. First Encounter"]
        Q1["Query: 'deploy to staging'"] --> LLM["LLM Planner"]
        LLM --> Coalition["Coalition: DevOps + Cloud + Container"]
    end

    subgraph Promotion ["2. Promotion (on success)"]
        Coalition --> Verify["Verifier Score ≥ 0.7"]
        Verify --> Promote["promote_coalition_to_template()"]
        Promote --> TC["TeamConfigNode\n(domain_pattern='deploy*staging*')"]
    end

    subgraph Reuse ["3. Future Queries"]
        Q2["Query: 'deploy app to staging env'"] --> Match["find_matching_team_config()"]
        Match --> TC
        TC --> Bypass["Skip LLM planning\nDirect dispatch"]
    end

    subgraph Learning ["4. Continuous Learning"]
        Bypass --> Outcome["Execution Outcome"]
        Outcome --> Reward["record_team_outcome()\nsuccess_rate += EMA"]
    end
```

### Data Model

```python
class TeamConfigNode(RegistryNode):
    """CONCEPT:AHE-3.3 — Proven Team Reuse"""
    node_type: str = "TEAM_CONFIG"
    domain_pattern: str           # e.g., "deploy*staging*"
    specialist_ids: list[str]     # Ordered specialist node IDs
    success_rate: float = 0.5     # EMA-updated after each use
    uses_count: int = 0
    capability_overrides: dict    # e.g., {"rlm": True} for large inputs
```

### RLM + TeamConfig Synergy

When a TeamConfig is selected and the input exceeds a size threshold, the system auto-attaches the RLM capability to specialists:

```python
# In routing.py — when TeamConfig match is found:
if len(query) > 5000 and "rlm" not in team_config.capability_overrides:
    team_config.capability_overrides["rlm"] = True
    # RLM becomes part of the proven team template
```

### Key Functions

| Function | Purpose |
|----------|---------|
| `find_matching_team_config(domain, query)` | Search for a reusable TeamConfig matching the query |
| `promote_coalition_to_template(specialists, domain, query)` | Create a new TeamConfig from a successful coalition |
| `record_team_outcome(config_id, success)` | Update TeamConfig success_rate via EMA |
| `link_prompt_to_agent(agent_id, prompt_id)` | Create USES_PROMPT edges for traceability |

---

## CONCEPT:ORCH-1.2 — AgentCapability Type System

**Module:** `agent_utilities/models/knowledge_graph.py`, `agent_utilities/graph/executor.py`

### Problem

Specialist agents had static, fixed tool bindings defined at registration time. Cross-cutting capabilities like RLM (recursive decomposition), critic (code review), or summarizer (context compression) were never dynamically attached based on the actual task characteristics.

### Solution

`AgentCapabilityNode` is a first-class Knowledge Graph node that models capabilities with trigger conditions, handler modules, and auto-activation flags. During execution, the system queries the KG for capabilities associated with the active specialist and activates them when trigger conditions are met.

### Data Model

```python
class AgentCapabilityNode(RegistryNode):
    """CONCEPT:ORCH-1.2 — Agent Capability Type System"""
    node_type: str = "AGENT_CAPABILITY"
    capability_type: str          # e.g., "rlm", "critic", "summarizer"
    auto_activate: bool = False   # If True, system checks triggers automatically
    trigger_conditions: dict      # e.g., {"input_size_gt": 5000, "domain": "code"}
    handler_module: str           # e.g., "agent_utilities.rlm.executor"
    priority: int = 0             # Higher = checked first
```

### Auto-Activation in Executor

The executor loop in `executor.py` checks for auto-activatable capabilities before each specialist run:

```python
# Simplified from executor.py
for specialist in specialists:
    capabilities = engine.query(
        "MATCH (s)-[:HAS_CAPABILITY]->(c:AgentCapability) "
        "WHERE s.id = $sid AND c.auto_activate = true "
        "RETURN c",
        sid=specialist.id,
    )
    for cap in capabilities:
        if _check_trigger(cap.trigger_conditions, input_text):
            logger.info("[CONCEPT:ORCH-1.2] Auto-activating %s for %s", cap.capability_type, specialist.name)
            # Activate the capability handler before execution
```

### Trigger Condition Evaluation

| Condition Key | Example Value | Meaning |
|---------------|---------------|---------|
| `input_size_gt` | `5000` | Input exceeds 5000 characters |
| `domain` | `"code"` | Task is code-related |
| `has_images` | `true` | Input contains image data |
| `tool_count_gt` | `20` | Specialist has >20 tools |

---

## CONCEPT:ECO-4.1 — PlannerGraphSkill (A2A-Native Routing)

**Module:** `agent_utilities/protocols/a2a_graph_skill.py`, `agent_utilities/server/app.py`

### Problem

A2A requests always went through the full LLM-mediated pipeline: parse request → LLM decides routing → dispatch to graph. This added an unnecessary inference round-trip when the graph planner already had enough information to handle the request directly.

### Solution

`PlannerGraphSkill` is an A2A-native skill that routes requests directly through the graph planner, bypassing LLM orchestration overhead:

```python
class PlannerGraphSkill:
    """CONCEPT:ECO-4.1 — A2A-Native PlannerAgent"""

    def __init__(self, graph_bundle):
        self.graph_bundle = graph_bundle

    async def execute(self, request):
        """Direct graph-backed planning — no LLM round-trip."""
        state = GraphState(user_query=request.query)
        result = await run_graph_flow(self.graph_bundle, state)
        return result
```

### Registration

The skill is automatically registered in `server/app.py` when a `graph_bundle` is available:

```python
# In build_agent_app():
if graph_bundle:
    planner_skill = PlannerGraphSkill(graph_bundle)
    a2a_skills.append(planner_skill)  # Registered before generic LLM skill
```

### Routing Priority

| Priority | Entry Point | When Used |
|----------|------------|-----------|
| 1 (highest) | `PlannerGraphSkill` | A2A requests when `graph_bundle` is present |
| 2 | Direct Graph Execution | AG-UI/ACP when `GRAPH_DIRECT_EXECUTION=true` |
| 3 (fallback) | LLM-Mediated | When no graph is available or A2A negotiation needed |

---

## KG Schema Additions

### Node Types

| Type | Concept | Description |
|------|---------|-------------|
| `TEAM_CONFIG` | CONCEPT:AHE-3.3 | Proven specialist coalition template |
| `AGENT_CAPABILITY` | CONCEPT:ORCH-1.2 | Dynamic capability with trigger conditions |

### Edge Types

| Type | Concept | Description |
|------|---------|-------------|
| `HAS_CAPABILITY` | CONCEPT:ORCH-1.2 | Links specialist → capability |
| `REUSED_TEAM` | CONCEPT:AHE-3.3 | Links session → TeamConfig (tracks reuse) |
| `USES_PROMPT` | CONCEPT:AHE-3.3 | Links specialist → JSON prompt template |

---

## Testing

```bash
# Registry cache tests
python -m pytest tests/unit/graph/test_config_helpers.py -v

# TeamConfig promotion and reward tracking
python -m pytest tests/unit/knowledge_graph/test_team_config.py -v

# AgentCapability node and auto-activation
python -m pytest tests/unit/knowledge_graph/test_capability_nodes.py -v

# All first-principles tests
python -m pytest tests/unit/graph/test_config_helpers.py \
    tests/unit/knowledge_graph/test_team_config.py \
    tests/unit/knowledge_graph/test_capability_nodes.py -v
```

---

## Related Documentation

- [Registry Cache Deep-Dive](../1_graph_orchestration/registry-cache.md) — Focused cache architecture and performance analysis
- [Process Lifecycle Management](../5_agent_os_infrastructure/process-lifecycle.md) — Sidecar cleanup and signal handling
- [Emergent Architecture](../2_epistemic_knowledge_graph/emergent-architecture.md) — CONCEPT:KG-2.0 through CONCEPT:ORCH-1.2 (OGM, Swarm, Self-Model, Attention)
- [Architecture](../1_graph_orchestration/architecture.md) — Full system architecture with routing diagrams
