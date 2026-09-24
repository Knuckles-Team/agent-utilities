# First Principles Architecture

> **Concepts:** CONCEPT:AU-ORCH.adapter.hot-cache-invalidation, CONCEPT:AU-AHE.evaluation.interpretability-tests, CONCEPT:AU-ORCH.adapter.hot-cache-invalidation, CONCEPT:AU-ECO.messaging.native-backend-abstraction

This document describes the **First Principles Architecture** layer — a set of four foundational concepts that rewire the routing, dispatch, and feedback loops of `agent-utilities` from basic primitives. These concepts were designed to solve specific scalability, performance, and intelligence bottlenecks that emerge when the system manages dozens of specialists and hundreds of tools.

## Problems Solved

| Problem | Root Cause | Solution |
|---------|-----------|----------|
| **Prompt bloat** | Every routing call serialized the full specialist registry into the LLM prompt | CONCEPT:AU-ORCH.adapter.hot-cache-invalidation: Hot cache filters to top-7 relevant specialists per query |
| **Redundant team discovery** | LLM re-discovers the same specialist combinations for recurring query patterns | CONCEPT:AU-AHE.evaluation.interpretability-tests: TeamConfig promotes proven coalitions as reusable templates |
| **Static tool binding** | Specialists had fixed tool sets; capabilities like RLM or critic were never auto-attached | CONCEPT:AU-ORCH.adapter.hot-cache-invalidation: AgentCapability nodes auto-activate based on input constraints |
| **LLM orchestration overhead** | A2A requests required a full LLM planning round-trip even when the graph planner could handle them | CONCEPT:AU-ECO.messaging.native-backend-abstraction: PlannerGraphSkill provides a direct graph-backed A2A entry point |
| **No feedback loop** | Execution outcomes were never fed back to improve future routing | CONCEPT:AU-AHE.evaluation.interpretability-tests + CONCEPT:AU-KG.memory.tiered-memory-caching: Verification outcomes update Self-Model and TeamConfig rewards |

## Architecture Overview

<div class="admonition architecture" markdown>
<p class="admonition-title">Ingress, hybrid routing, execution, and feedback — one loop</p>

**Protocol ingress.** A2A reaches `PlannerGraphSkill`; ACP and AG-UI both
reach the router directly. **3-stage hybrid routing.** `PlannerGraphSkill`
also reaches the router, which checks for a `TeamConfig` match: a hit
dispatches immediately; a miss checks Self-Model bias, then runs the LLM
planner (filtered prompt) before dispatching. **Dispatch & execute.**
Dispatch reads the Registry Cache for the top-7 specialists, checks
capability auto-activation, then executes in parallel. **Post-execution
feedback.** Execution feeds a verifier, which updates both the Self-Model
and the TeamConfig's reward; both updates trigger cache invalidation,
closing the loop.
</div>

---

## CONCEPT:AU-ORCH.adapter.hot-cache-invalidation — Registry Hot Cache

**Module:** `agent_utilities/core/config.py`

### Problem

Every call to the router required a full registry scan — iterating over all registered specialists (potentially 50+) to serialize their descriptions into the LLM prompt. This created two issues:

1. **Latency**: O(N) scan on every routing call
2. **Prompt bloat**: Injecting 50+ specialist descriptions consumed thousands of tokens, reducing the LLM's effective reasoning window

### Solution

A session-scoped `_RegistryCache` singleton that caches the full specialist registry and provides filtered, query-relevant subsets:

```python
from agent_utilities.core.config import (
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

<div class="admonition architecture" markdown>
<p class="admonition-title">Cold, then warm until invalidated</p>

The cache starts `Cold` at server start. The first
`get_discovery_registry()` call warms it; every subsequent call while warm
is O(1). `invalidate_registry_cache()` returns it to `Cold`. Four events
trigger invalidation: MCP agent sync, pipeline completion, a Self-Model
update, or a TeamConfig promotion.
</div>

### Invalidation Triggers

The cache is invalidated by 4 event sources, ensuring it stays in sync:

| Trigger | Location | Why |
|---------|----------|-----|
| MCP Sync | `mcp/agent_manager.py` → `sync_mcp_agents()` | New tools may create new specialists |
| Pipeline Completion | `knowledge_graph/pipeline/runner.py` → `PipelineRunner.run()` | Code graph changes may affect routing |
| Self-Model Update | `knowledge_graph/retrieval/memory_retriever.py` | New proficiency data should influence specialist ranking |
| TeamConfig Promotion | `core/registry/kg_adapter.py` → `promote_coalition_to_template()` | A new reusable composition is available to reference |

---

## CONCEPT:AU-AHE.evaluation.interpretability-tests — TeamConfig Promotion & Proven Team Reuse

**Module:** `agent_utilities/core/registry/kg_adapter.py`

### Problem

The LLM planner would rediscover the same specialist combinations for recurring query patterns. A user who frequently asks "deploy to staging" would see the LLM re-derive the `[DevOps, Cloud, Container Manager]` coalition every time — wasting inference tokens and adding latency.

### Solution

**TeamConfig** nodes persist proven specialist coalitions as reusable compositions in the Knowledge Graph. They are *referenced*, never *selected by a success rate*: the router has no TeamConfig reuse step, and no AU component stores or reads a TeamConfig success rate (SWARM-TOPOLOGY-DECIDE-DESIGN ST-7, invariant T5). Which topology a task runs is EG's certified decision (`AgentAssemble` with `requirements.topology`); learning which topology works is EG's calibrated rung over independent, slate-credited evaluations.

### TeamConfig Lifecycle

<div class="admonition architecture" markdown>
<p class="admonition-title">Discover once, promote on success, reuse and keep learning</p>

**1. First encounter.** A query ("deploy to staging") reaches the LLM
planner, which derives a coalition (DevOps + Cloud + Container). **2.
Promotion (on success).** A verifier score ≥ 0.7 triggers
`promote_coalition_to_template()`, creating a `TeamConfigNode` — a reusable
composition. **3. Future queries.** A typed task goes to EG's
`AgentAssemble`, which returns a certified topology plan; a TeamConfig
carries no success rate and nothing selects it by one (ST-7).
</div>

### Data Model

```python
class TeamConfigNode(RegistryNode):
    """CONCEPT:AU-AHE.evaluation.interpretability-tests — reusable composition"""
    node_type: str = "TEAM_CONFIG"
    task_pattern: str             # what the team solves
    specialist_ids: list[str]     # Ordered specialist node IDs
    capability_overrides: dict    # e.g., {"rlm": True} for large inputs
    # No success_rate / usage_count / reuse_threshold: nothing selects on them.
```

### Key Functions

| Function | Purpose |
|----------|---------|
| `promote_coalition_to_template(coalition_id, task_pattern)` | Create a reusable TeamConfig from a successful coalition |
| `export_team_config(team_id)` / `import_team_config(bundle)` | Share a composition (imported outcome counters are dropped) |
| `link_prompt_to_agent(agent_id, prompt_id)` | Create USES_PROMPT edges for traceability |

---

## CONCEPT:AU-ORCH.adapter.hot-cache-invalidation — AgentCapability Type System

**Module:** `agent_utilities/models/knowledge_graph.py`, `agent_utilities/graph/executor.py`

### Problem

Specialist agents had static, fixed tool bindings defined at registration time. Cross-cutting capabilities like RLM (recursive decomposition), critic (code review), or summarizer (context compression) were never dynamically attached based on the actual task characteristics.

### Solution

`AgentCapabilityNode` is a first-class Knowledge Graph node that models capabilities with trigger conditions, handler modules, and auto-activation flags. During execution, the system queries the KG for capabilities associated with the active specialist and activates them when trigger conditions are met.

### Data Model

```python
class AgentCapabilityNode(RegistryNode):
    """CONCEPT:AU-ORCH.adapter.hot-cache-invalidation — Agent Capability Type System"""
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
            logger.info("[CONCEPT:AU-ORCH.adapter.hot-cache-invalidation] Auto-activating %s for %s", cap.capability_type, specialist.name)
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

## CONCEPT:AU-ECO.messaging.native-backend-abstraction — PlannerGraphSkill (A2A-Native Routing)

**Module:** `agent_utilities/protocols/a2a_graph_skill.py`, `agent_utilities/server/app.py`

### Problem

A2A requests always went through the full LLM-mediated pipeline: parse request → LLM decides routing → dispatch to graph. This added an unnecessary inference round-trip when the graph planner already had enough information to handle the request directly.

### Solution

`PlannerGraphSkill` is an A2A-native skill that routes requests directly through the graph planner, bypassing LLM orchestration overhead:

```python
class PlannerGraphSkill:
    """CONCEPT:AU-ECO.messaging.native-backend-abstraction — A2A-Native PlannerAgent"""

    def __init__(self, graph_bundle):
        self.graph_bundle = graph_bundle

    async def execute(self, request):
        """Direct graph-backed planning — no LLM round-trip."""
        state = GraphState(user_query=request.query)
        result = await execute_graph(self.graph_bundle, state)
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
| 2 | Direct Graph Execution | AG-UI/ACP through the single execution authority |
| 3 (fallback) | LLM-Mediated | When no graph is available or A2A negotiation needed |

---

## KG Schema Additions

### Node Types

| Type | Concept | Description |
|------|---------|-------------|
| `TEAM_CONFIG` | CONCEPT:AU-AHE.evaluation.interpretability-tests | Proven specialist coalition template |
| `AGENT_CAPABILITY` | CONCEPT:AU-ORCH.adapter.hot-cache-invalidation | Dynamic capability with trigger conditions |

### Edge Types

| Type | Concept | Description |
|------|---------|-------------|
| `HAS_CAPABILITY` | CONCEPT:AU-ORCH.adapter.hot-cache-invalidation | Links specialist → capability |
| `REUSED_TEAM` | CONCEPT:AU-AHE.evaluation.interpretability-tests | Links session → TeamConfig (tracks reuse) |
| `USES_PROMPT` | CONCEPT:AU-AHE.evaluation.interpretability-tests | Links specialist → JSON prompt template |

---

## Testing

```bash
# Registry cache tests
python -m pytest tests/unit/core/test_config_helpers.py -v

# TeamConfig promotion and reward tracking
python -m pytest tests/test_team_config.py -v

# AgentCapability node and auto-activation
python -m pytest tests/unit/core/test_capabilities_core.py tests/unit/core/test_capabilities_advanced.py \
    tests/unit/graph/test_capability_designation.py -v

# All first-principles tests
python -m pytest tests/unit/core/test_config_helpers.py \
    tests/test_team_config.py \
    tests/unit/core/test_capabilities_core.py tests/unit/core/test_capabilities_advanced.py -v
```

---

## Related Documentation

- [Registry Cache Deep-Dive](registry-cache.md) — Focused cache architecture and performance analysis
- [Process Lifecycle Management](process-lifecycle.md) — Sidecar cleanup and signal handling
- [Emergent Architecture](emergent-architecture.md) — CONCEPT:AU-KG.query.object-graph-mapper through CONCEPT:AU-ORCH.adapter.hot-cache-invalidation (OGM, Swarm, Self-Model, Attention)
- [Architecture](architecture.md) — Full system architecture with routing diagrams
