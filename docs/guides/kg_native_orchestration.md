# KG-Native Orchestration Architecture

> **CONCEPT:ORCH-1.1 through ORCH-1.19** — Dynamic, Knowledge-Graph-Driven Agent Orchestration

## Overview

KG-Native Orchestration transforms agent-utilities from a **KG-aware** system (where the Knowledge Graph is optionally consulted) into a **KG-driven** system where the KG is the primary control surface for all orchestration decisions.

```
Before:  Query → Static Graph Topology → (optionally consult KG) → Execute
After:   Query → KG resolves topology → Dynamic Graph Materialization → Execute → KG learns
```

## Architecture

```mermaid
flowchart TD
    Q[ORCH-1.0: User Query] --> TC[ORCH-1.0: KGTeamComposer]
    TC -->|"1. Search proven teams"| KG[(KG-2.0: Knowledge Graph)]
    TC -->|"2. Select topology"| KG
    TC -->|"3. Populate specialists"| KG
    TC --> TEAM[AHE-3.3: TeamComposition]
    TEAM --> TENG[ORCH-1.4: TopologyEngine]
    TENG -->|"Materialize"| PLAN[ORCH-1.21: Execution Plan]
    PLAN --> SEQ[ORCH-1.21: Sequential Steps]
    PLAN --> PAR[ORCH-1.21: Parallel Groups]
    PLAN --> MIX[ORCH-1.21: Mixed DAG]
    SEQ & PAR & MIX --> EXEC[ORCH-1.2: Execute Specialists]
    EXEC --> CP[ORCH-1.3: StateCheckpointer]
    CP -->|"Checkpoint"| KG
    EXEC -->|"Success?"| TC2[ORCH-1.2: Promote to TeamConfig]
    TC2 --> KG
```

## Core Components

### 1. KG-Driven Team Composer (`graph/team_composer.py`)

**CONCEPT:ORCH-1.1** — Replaces static `discover_agents()` registration with dynamic KG-topology-driven team assembly.

**Composition Flow:**
1. **Reuse**: Search KG for proven `TeamConfigNode` matching the query (AHE-3.3)
2. **Select**: Choose best `TopologyTemplateNode` by domain + complexity
3. **Populate**: Walk KG edges (`PROVIDES`, `HAS_CAPABILITY`) to assign tools
4. **Promote**: On success, save the team as a new `TeamConfigNode`

**Default Topologies:**

| Template | Mode | Complexity | Specialists |
|----------|------|-----------|-------------|
| Single Agent | Sequential | 1 | 1 executor |
| Simple Q&A | Sequential | 1-2 | router → expert → verifier |
| Multi-Source Research | Mixed | 3-4 | router → planner → [researchers] → synthesizer → verifier |
| Expert Team | Mixed | 4-5 | router → planner → architect → [implementer, reviewer] → synthesizer |
| Finance Pipeline | Sequential | 3-5 | router → alpha → risk → execution → attribution |

### 2. Dynamic Topology Engine (`graph/topology_engine.py`)

**CONCEPT:ORCH-1.2** — Materializes KG-stored topology templates into executable graphs.

**Supported Execution Modes:**

- **Sequential**: `A → B → C` — Simple pipeline
- **Parallel**: `[A, B, C]` — All execute concurrently
- **Fan-out**: `A → [B₁, B₂, ..., Bₙ]` — Scatter
- **Fan-in**: `[B₁, B₂, ..., Bₙ] → C` — Gather
- **Mixed**: `A → [B, C] → D → [E, F] → G` — Arbitrary DAG

Each materialized specialist gets:
- **System Prompt**: Role-specific or KG-loaded via `PromptNode`
- **MCP Tools**: Only the tools needed for that role
- **Model**: Per-specialist model selection
- **Memory Channels**: Shared KG channels for P2P communication

### 3. Execution State Checkpointing (`core/checkpoint/manager.py`)

**CONCEPT:ORCH-1.1** — Bridges ephemeral `GraphState` with persistent KG. The
former `graph/state_checkpoint.StateCheckpointer` was consolidated into the
`core/checkpoint/` package (`KGBackend` + `CheckpointManager`).

```python
from agent_utilities.core.checkpoint.manager import KGBackend

backend = KGBackend(engine)
checkpoint_id = backend.checkpoint(state, session_id="sess:abc")
restored = backend.restore("sess:abc")
```

**Capabilities:**
- Checkpoint at HSM transition boundaries
- Session resume after crashes
- Cross-session learning
- Multi-agent coordination (other agents can query active state)

### 4. Topological Routing Policy (`graph/routing/strategies/policy.py`)

**CONCEPT:ORCH-1.4** — Routes using KG-derived topological signals instead of keyword TF-IDF.

**Scoring Dimensions:**
1. **PageRank centrality** — Highly-connected specialists preferred
2. **Historical success rate** — Weighted by outcome evaluations
3. **Tool affinity** — Specialists with relevant `PROVIDES` edges score higher

Falls back to `RuleBasedPolicy` when no KG is available (cold start).

### 5. Persistent Background Agents (`graph/persistent_agents.py`)

**CONCEPT:ORCH-1.4** — Long-running KG-coordinated agents.

```python
mgr = PersistentAgentManager(engine)
mgr.register_agent("bg:monitor", "System Monitor",
                     subscriptions=["system.alert"],
                     schedule_cron="*/5 * * * *")
```

**Lifecycle:** `registered → idle → running → idle → ... → terminated`

**Agent Types:**
- **Monitor**: Watches KG for conditions
- **Scheduler**: Runs periodic tasks
- **Rebalancer**: Continuously adjusts configurations
- **Background**: General-purpose

### 6. Shareable Team Compositions

**CONCEPT:ORCH-1.1 Extension** — Export/import proven team configurations.

```python
# Export
bundle = engine.export_team_config("tc:proven-team")

# Import on another deployment
new_id = engine.import_team_config(bundle)
```

## Pydantic Models

| Model | Type | Purpose |
|-------|------|---------|
| `TopologyTemplateNode` | `RegistryNode` | KG-stored execution topology template |
| `SessionCheckpointNode` | `RegistryNode` | Persisted execution state |
| `PersistentAgentNode` | `RegistryNode` | Long-running background agent |
| `TeamComposition` | `BaseModel` | Result of team composition (not persisted) |

## KG Node/Edge Types

**New Node Types:**
- `TOPOLOGY_TEMPLATE` — Execution topology templates
- `SESSION_CHECKPOINT` — Execution state checkpoints
- `PERSISTENT_AGENT` — Background agent registrations
- `TOPOLOGY_TRANSITION` — Transition records

**New Edge Types:**
- `TRANSITIONS_TO` — Topology transitions between roles
- `CHECKPOINTED_STATE` — Links sessions to checkpoints
- `SUBSCRIBED_TO` — Agent event subscriptions
- `MATERIALIZED_FROM` — Links executions to templates
- `COMPOSED_TEAM` — Links compositions to team configs

## Integration with Existing Systems

- **SubagentPatternRouter** (ORCH-1.5): Now uses KG backend for O(1) historical lookups instead of O(N) NX scans; persists decisions via tiered architecture
- **CognitiveScheduler** (OS-5.2): Unified scheduler for both ephemeral and persistent agents
- **EventStreamIngester** (Company Brain): Routes events to persistent agent subscribers
- **TeamConfigNode** (AHE-3.3): Extended with export/import for cross-deployment sharing
