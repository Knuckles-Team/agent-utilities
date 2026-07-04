# Semantic Compactor & Refactorer (CONCEPT:AU-KG.query.vendor-agnostic-traversal)

## Overview
The **Semantic Compactor** resolves the exponential database bloat caused by active agents generating millions of step-by-step reasoning traces.

It periodically executes an offline compilation and refactoring sweep. The compactor aggregates raw execution traces, interaction logs, and tool parameters, replacing thousands of individual `AgentProcess` nodes with synthesized high-level declarative state triples (e.g., summarizing total execution steps, token usage, and outcomes).

## Problem Statement

At 1M+ agent scale, each agent execution generates 10–100 `AgentProcess` trace nodes. After 10K executions:
- **1M+ trace nodes** accumulate in the graph
- Query latency degrades from sub-ms to 100ms+
- Memory consumption grows linearly without bound
- Graph traversal algorithms (centrality, PageRank) become O(n²)

The Semantic Compactor reduces the trace node count by **95–99%** while preserving aggregate semantics.

## Lifecycle

```
1. QUERY: Find AgentProcess nodes for a given agent_id
       ↓
2. THRESHOLD: If count < threshold (default 10), skip
       ↓
3. AGGREGATE: Sum token usage, count state distributions
       ↓
4. CREATE: Merge a SemanticSummary node with aggregated stats
       ↓
5. LINK: Connect agent → SemanticSummary via HAS_COMPACTED_HISTORY
       ↓
6. PRUNE: DETACH DELETE all original AgentProcess nodes
       ↓
7. LOG: Report compaction count and summary node ID
```

## API Surface

```python
from agent_utilities.knowledge_graph.memory import SemanticCompactor

compactor = SemanticCompactor(engine=kg_engine)

# Compact traces for a specific agent (threshold: 10 traces)
deleted = compactor.compact_traces("agent:planner", threshold=10)
# Returns: number of trace nodes deleted (e.g., 47)

# Custom threshold for high-frequency agents
deleted = compactor.compact_traces("agent:scraper", threshold=5)
```

## Compaction Thresholds

| Agent Type | Recommended Threshold | Rationale |
|---|---|---|
| Background scrapers | 5 | High volume, low value per trace |
| Orchestration routers | 10 (default) | Medium volume, moderate diagnostic value |
| Critical planners | 25 | Lower volume, high forensic value |
| Evaluation runners | 50 | Traces needed for statistical analysis |

## Graph Schema

### Before Compaction
```
(Agent {id: "agent:planner"})
  -[:HAS_PROCESS]→ (AgentProcess {id: "proc:1", state: "completed", tokens_used: 1500})
  -[:HAS_PROCESS]→ (AgentProcess {id: "proc:2", state: "completed", tokens_used: 2300})
  -[:HAS_PROCESS]→ (AgentProcess {id: "proc:3", state: "failed", tokens_used: 800})
  ... (47 more)
```

### After Compaction
```
(Agent {id: "agent:planner"})
  -[:HAS_COMPACTED_HISTORY]→ (SemanticSummary {
      id: "summary:agent:planner:50_compacted",
      compacted_count: 50,
      total_tokens_consumed: 85000,
      agent_id: "agent:planner"
  })
```

## OWL Alignment

The `SemanticSummary` node type should be declared in the SDD ontology as:

```turtle
:SemanticSummary a owl:Class ;
    rdfs:subClassOf :KnowledgeNode ;
    rdfs:comment "Synthesized execution trace summary produced by KG-2.7 compaction" ;
    :hasProperty :compacted_count, :total_tokens_consumed, :agent_id .

:HAS_COMPACTED_HISTORY a owl:ObjectProperty ;
    rdfs:domain :Agent ;
    rdfs:range :SemanticSummary ;
    rdfs:comment "Links an agent to its compacted trace history" .
```

This ensures SHACL validators can verify that every `Agent` node with more than `threshold` historical processes has a corresponding `SemanticSummary`.

## Error Handling

| Condition | Behavior |
|---|---|
| No KG engine provided | Returns `0` (no-op) |
| Query fails | Catches exception, logs error, returns `0` |
| Partial deletion failure | Logs individual failures, returns count of successful deletes |
| Agent has no traces | Returns `0` (below threshold) |
| Backend returns unexpected row format | Handles `dict`, `list/tuple`, and object formats defensively |

## Integration Points

- **Cognitive Scheduler (OS-5.2)**: After preempting and completing many processes, schedule periodic compaction
- **Telemetry Engine (OS-5.6)**: Compaction events are logged for observability dashboards
- **Memory Tiers (KG-2.6)**: Compaction operates on the episodic tier, preserving semantic and procedural memories

## Implementation Details
- **Source Code**: [`agent_context.py`](file:///home/apps/workspace/agent-packages/agent-utilities/agent_utilities/knowledge_graph/memory/agent_context.py) (`SemanticCompactor`)
- **Classes**: `SemanticCompactor`
- **Tests**: [`test_synergies.py`](file:///home/apps/workspace/agent-packages/agent-utilities/tests/unit/knowledge_graph/test_synergies.py)
- **Pillar**: KG
- **Package Export**: `agent_utilities.knowledge_graph.memory.SemanticCompactor`
