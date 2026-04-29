# Recursive Language Models (RLM)

> CONCEPT:AU-007 — Recursive Language Model Execution

## Overview

The RLM subsystem provides a **persistent Python REPL** that enables agents to perform multi-step reasoning through recursive code execution. An LLM generates Python code, executes it, observes the output, and iterates — spawning sub-RLMs at deeper recursion levels when needed.

## Architecture

```
┌─────────────────────────────────────────┐
│  RLMEnvironment                         │
│                                         │
│  ┌─────────────────────────────┐        │
│  │  Persistent Globals Dict    │        │
│  │  - context, depth           │        │
│  │  - rlm_query()              │        │
│  │  - magma_view()             │        │
│  │  - graph_query()            │        │
│  │  - sub_agent_call()         │        │
│  │  - FINAL_VAR()              │        │
│  │  - run_parallel_sub_calls() │        │
│  └─────────────────────────────┘        │
│                                         │
│  ┌──────────────┐  ┌──────────────────┐ │
│  │ _execute_     │  │ _execute_        │ │
│  │ local()       │  │ container()      │ │
│  │ (exec-based)  │  │ (Docker sandbox) │ │
│  └──────────────┘  └──────────────────┘ │
└─────────────────────────────────────────┘
```

## Key Components

### `RLMEnvironment` (`rlm/repl.py`)

The core execution environment. Initializes with a context variable and a set of approved helper functions exposed to the LLM-generated code.

```python
env = RLMEnvironment(
    context={"data": large_dataset},
    depth=0,
    config=RLMConfig(max_depth=3, use_container=False),
    graph_deps=graph_deps,
)
result = await env.run_full_rlm("Analyze the dataset and find anomalies")
```

### `RLMConfig` (`rlm/config.py`)

Configuration for RLM behavior:

| Parameter | Default | Description |
|---|---|---|
| `max_depth` | `3` | Maximum recursion depth |
| `use_container` | `False` | Use Docker sandbox instead of local exec |
| `async_enabled` | `True` | Enable parallel sub-call execution |
| `sub_llm_model_large` | Provider default | Model for depth-0 reasoning |
| `sub_llm_model_small` | Provider default | Model for deeper recursion levels |
| `trajectory_storage` | `"process_flow"` | Where to store reasoning traces |

### Available Helpers

Functions available inside the RLM execution environment:

| Helper | Signature | Purpose |
|---|---|---|
| `rlm_query` | `await rlm_query(prompt, context)` | Spawn a recursive sub-RLM at depth+1 |
| `magma_view` | `await magma_view(query, views)` | MAGMA orthogonal memory views |
| `graph_query` | `await graph_query(cypher, params)` | Run Cypher against the knowledge graph |
| `sub_agent_call` | `await sub_agent_call(prompt, agent_id, data)` | Dispatch to specialist agent |
| `FINAL_VAR` | `FINAL_VAR("name", value)` | Output the final result |
| `run_parallel_sub_calls` | `await run_parallel_sub_calls(calls)` | Run multiple sub-calls in parallel |

### `run_full_rlm()` Loop

The main agent loop:

1. LLM generates a response (potentially containing ```python blocks)
2. Code blocks are extracted and executed via `execute()`
3. stdout is captured and fed back to the LLM
4. If `FINAL_VAR` was called, the result is returned
5. Otherwise, the loop continues (up to `max_turns=5`)

## Security Considerations

> **CWE-94 (Code Injection)**: The RLM REPL intentionally uses `exec()` to
> execute LLM-generated code. This is by design — the execution namespace is
> restricted to approved helpers only.

### Mitigations

1. **Restricted globals**: Only approved functions and modules (`json`, `asyncio`, `nx`) are exposed
2. **Container mode**: Set `use_container=True` to run code in an isolated Docker container
3. **Recursion limits**: `max_depth` prevents infinite recursion
4. **Turn limits**: Maximum 5 turns per RLM invocation
5. **Trajectory storage**: All executions are logged for audit

### `RLMHook` (`rlm/hook.py`)

Lifecycle hooks for observability:

- `on_code_generated(code)`: Called when LLM generates code
- `on_code_executed(code, stdout, vars)`: Called after execution
- `on_recursion(depth, prompt)`: Called when spawning sub-RLM

### `RLMSpecialist` (`rlm/specialist.py`)

Integration with the graph executor as a specialist node. The RLM specialist can be routed to by the graph router for tasks requiring iterative code execution.

## Example: Multi-Step Data Analysis

```python
from agent_utilities.rlm.repl import RLMEnvironment
from agent_utilities.rlm.config import RLMConfig

config = RLMConfig(max_depth=2, use_container=True)
env = RLMEnvironment(
    context={"csv_data": "..."},
    config=config,
)

# The LLM will iteratively write code to analyze the data
result = await env.run_full_rlm(
    "Parse the CSV data, identify outliers using IQR method, "
    "and produce a summary report with FINAL_VAR."
)
```
