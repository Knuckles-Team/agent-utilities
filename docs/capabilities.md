# Capabilities (Self-Healing Patterns)

> CONCEPT:AU-008 — Resilient Agent Capabilities

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
