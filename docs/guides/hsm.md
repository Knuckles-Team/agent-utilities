# Hierarchical State Machine (HSM) Infrastructure

The `agent-utilities` graph orchestrator implements a **Hierarchical State Machine (HSM)** and **Behavior Tree (BT)** hybrid model via `agent_utilities.graph.hsm`.

This module provides the necessary primitives for running complex parallel workflows, tracking agent lifecycles, and guarding state invariants during graph transitions.

## Core Features

### 1. Orthogonal Regions (Concurrency)

Orthogonal regions allow an agent to run multiple independent sub-state-machines in parallel within a single superstate. This is implemented via `run_orthogonal_regions`.

When a task requires multiple sub-tasks (e.g. scanning multiple directories, fetching data from multiple APIs), the orchestrator can fan-out queries to the same specialist agent concurrently and merge the results:

```python
from agent_utilities.graph.hsm import run_orthogonal_regions

queries = [
    "Scan frontend/src for React components",
    "Scan backend/app for FastAPI routes"
]

# Runs both queries concurrently against the same agent
results = await run_orthogonal_regions(
    agent=my_specialist_agent,
    queries=queries,
    deps=ctx.deps,
    timeout=120.0,
    event_queue=ctx.deps.event_queue,
    agent_name="CodeScanner"
)

# Results is a dict mapping query -> output
```

### 2. Entry & Exit Hooks

The HSM module allows registering global plugin hooks that fire whenever the graph enters or exits a specialist node. This is heavily used for streaming events, telemetry, and duration tracking.

```python
from agent_utilities.graph.hsm import register_on_enter_hook, register_on_exit_hook

async def my_entry_plugin(deps, state, agent_name, server_name):
    print(f"Entering specialist: {agent_name}")

async def my_exit_plugin(deps, state, agent_name, success, server_name, duration):
    print(f"Exited specialist: {agent_name} in {duration}s. Success: {success}")

register_on_enter_hook(my_entry_plugin)
register_on_exit_hook(my_exit_plugin)
```

### 3. State Invariant Guards

To prevent the graph from entering infinite loops, corrupting its cursor, or losing its primary objective, `assert_state_valid` checks invariants at every transition boundary.

```python
from agent_utilities.graph.hsm import assert_state_valid, StateInvariantError

try:
    assert_state_valid(ctx.state, transition="Planner -> Dispatcher")
except StateInvariantError as e:
    # State is corrupt! Triggers automatic recovery/fallback.
    logger.error(f"Graph corruption: {e}")
```

### 4. Static Routing (Junction Pseudostates)

Before falling back to the heavy LLM Router, the HSM attempts to use `static_route_query`. This function scans the query for exact keyword matches against known specialists.

If a user explicitly types "list gitlab projects", the static router bypasses the LLM layer entirely and routes directly to the GitLab specialist, saving latency and token costs.

```python
from agent_utilities.graph.hsm import static_route_query

available_specialists = {
    "gitlab_api": "GitLab API interactions",
    "python_coder": "Writes python code"
}

# Returns "gitlab_api" based on exact keyword match
target = static_route_query("Please list gitlab projects", available_specialists)
```

### 5. Behavior Tree Results

Nodes return specific states to indicate their execution status, mirroring Behavior Tree paradigms:

- `NodeResult.SUCCESS`: The specialist completed its task successfully. Move to the next plan step.
- `NodeResult.FAILURE`: The specialist failed or circuit broke. Trigger error recovery or re-plan.
- `NodeResult.RUNNING`: The node is suspended (e.g. waiting for human-in-the-loop approval).
