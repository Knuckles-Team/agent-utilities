# Capabilities (Self-Healing Patterns)

> CONCEPT:ORCH-1.2 — Resilient Agent Capabilities

## Overview

The `capabilities/` module provides self-healing and resilience patterns that the agent uses to maintain stability during long-running sessions. These are cross-cutting concerns that augment any agent's behavior.

## Components

### Checkpointing (`capabilities/checkpointing.py`)

Provides state persistence during multi-step graph execution. If a step fails, execution can resume from the last checkpoint rather than restarting. **(HSM Concept: State Snapshot)**

```python
from agent_utilities.capabilities.checkpointing import CheckpointManager

mgr = CheckpointManager(workspace_path="/tmp/agent-session")
await mgr.save(step_id="step_3", state=current_state)

# On failure recovery:
restored = await mgr.load(step_id="step_3")
```

### Context Window Warnings (`capabilities/context_warnings.py`)

Monitors token usage and emits warnings when the context window is approaching its limit. Helps agents decide when to summarize or evict older messages.

- **Token tracking**: Estimates token count per message
- **Warning thresholds**: Configurable warning/critical levels
- **Auto-summarization triggers**: Fires when threshold is breached

### Eviction (`capabilities/eviction.py`)

Implements message eviction strategies for long conversations:

| Strategy | Description |
|---|---|
| `oldest_first` | Remove oldest messages when context limit is reached |
| `relevance_scored` | Keep messages with highest relevance to current task |
| `summarize_and_drop` | Summarize a block of messages, then drop originals |

### Hooks (`capabilities/hooks.py`)

Lifecycle hooks that fire at key points in agent execution:

- `on_turn_start(turn_number)`: Beginning of each turn
- `on_turn_end(turn_number, result)`: End of each turn
- `on_tool_call(tool_name, args)`: Before tool execution
- `on_tool_result(tool_name, result)`: After tool execution
- `on_error(error, context)`: When an error occurs

### Stuck-Loop Detection (`capabilities/stuck_loop.py`)

Detects when an agent is repeating the same actions without progress:

- **Repetition detection**: Identifies duplicate tool calls or responses
- **Escalation**: After N repeated patterns, forces a strategy change
- **Circuit breaker**: Terminates the loop after configurable max retries **(HSM Concept: Guard Condition)**

### Teams (`capabilities/teams.py`)

Multi-agent coordination patterns:

- **Parallel dispatch**: Send tasks to multiple agents simultaneously
- **Sequential pipeline**: Chain agent outputs as inputs to the next
- **Voting/consensus**: Aggregate multiple agent responses
- **TeamConfig Promotion (CONCEPT:AHE-3.3)**: Successful coalitions are persisted as reusable `TeamConfigNode` templates in the Knowledge Graph. See [first-principles.md](first-principles.md) for details on proven team reuse and reward tracking.

---

## AgentCapability Type System (CONCEPT:ORCH-1.2)

> See also: [First Principles Architecture](first-principles.md) for the complete CONCEPT:ORCH-1.2 deep-dive.

The AgentCapability system extends the static tool-binding model with dynamic, condition-based capability activation. Capabilities are modeled as first-class Knowledge Graph nodes (`AgentCapabilityNode`) with trigger conditions that are evaluated at execution time.

### How It Works

1. Capabilities are registered in the KG with `auto_activate=true` and `trigger_conditions`
2. Before each specialist execution, the executor queries for capabilities linked via `HAS_CAPABILITY` edges
3. If trigger conditions are met (e.g., input > 5000 chars), the capability handler is activated

### Example Capabilities

| Capability | Trigger | Effect |
|-----------|---------|--------|
| RLM (Recursive LM) | `input_size_gt: 5000` | Decomposes large inputs into recursive sub-problems |
| Critic | `domain: code` | Adds code review step before final output |
| Summarizer | `tool_count_gt: 20` | Compresses specialist context before LLM call |

---

## Registry Hot Cache (CONCEPT:ORCH-1.2)

> See also: [Registry Cache Deep-Dive](registry-cache.md) for the complete architecture.

The Registry Hot Cache provides session-scoped O(1) specialist lookups, replacing the previous O(N) full-registry scan on every routing call. Key features:

- **Filtered specialist injection**: Only the top-7 relevant specialists are injected into the LLM prompt, reducing token consumption by ~7x
- **Event-driven invalidation**: Cache invalidates on MCP reload, pipeline completion, Self-Model updates, and TeamConfig promotions
- **Zero TTL risk**: No time-based expiry — invalidation only fires when underlying data changes

### Integration with Routing

The router uses `get_relevant_specialists(query, engine)` instead of the full registry, ensuring:

1. The LLM sees fewer, more relevant specialist descriptions
2. Routing accuracy improves due to reduced prompt noise
3. Latency decreases from eliminated registry scans
