# Recursive Language Models (RLM)

> CONCEPT:ORCH-1.1 — Recursive Language Model Execution

## Overview

The RLM subsystem provides a **persistent Python REPL** that enables agents to process arbitrarily long inputs through recursive, programmatic decomposition. Based on [Zhang et al. (2025)](https://github.com/alexzhang13/rlm), the key insight is that **long prompts should NOT be fed into the neural network directly** — they should be treated as part of the environment the LLM symbolically and recursively interacts with.

RLM enables agents to:
- Process inputs **two orders of magnitude** beyond model context windows
- Perform **unbounded semantic work** through recursive sub-calls
- Leverage **OWL reasoning** and **KG bulk analysis** within execution
- Power **AHE trace distillation** for large-scale evolution analysis (CONCEPT:AHE-3.0)

## Architecture

```
┌─────────────────────────────────────────┐
│  RLMEnvironment                         │
│                                         │
│  ┌─────────────────────────────┐        │
│  │  Persistent Globals Dict    │        │
│  │  - context, depth           │        │
│  │  - rlm_query(schema=)       │        │
│  │  - magma_view()             │        │
│  │  - graph_query()            │        │
│  │  - owl_query()      [NEW]   │        │
│  │  - kg_bulk_export() [NEW]   │        │
│  │  - sub_agent_call()         │        │
│  │  - FINAL_VAR()              │        │
│  │  - run_parallel_sub_calls() │        │
│  └─────────────────────────────┘        │
│                                         │
│  execute(code)                          │
│      │                                  │
│      ▼  SandboxRouter (ORCH-1.38)        │
│  ┌─────────────────────────────────────┐│
│  │ ast-analyze → cheapest capable tier ││
│  │ monty → wasm → docker → local        ││
│  │ (escalate on SandboxRejected)        ││
│  └─────────────────────────────────────┘│
└─────────────────────────────────────────┘
```

Code execution is no longer a hardcoded `local`/`container` switch: `execute()` routes each
snippet through the **tiered sandbox router** (CONCEPT:ORCH-1.38) — see
[ORCH-1.38 — Tiered RLM Sandbox](../pillars/1_graph_orchestration/ORCH-1.38-Tiered_RLM_Sandbox.md).

## Invocation Triggers

RLM is automatically invoked when any of the following conditions are met. No global `ENABLE_RLM=True` is required — the system uses **smart thresholds** to route intelligently.

| # | Trigger | Condition | Default Threshold |
|---|---|---|---|
| 1 | **Global Override** | `ENABLE_RLM=True` | Always |
| 2 | **Long Horizon** | `state.requires_long_horizon=True` | Always |
| 3 | **Large Output** | Tool/specialist output exceeds threshold | 50,000 chars |
| 4 | **AHE Distillation** | Trace count exceeds threshold | 500 traces |
| 5 | **KG Bulk Analysis** | KG query returns too many nodes | 1,000 nodes |

Use the unified `RLMConfig.should_trigger()` method for consistent routing:

```python
from agent_utilities.rlm.config import RLMConfig

config = RLMConfig()
if config.should_trigger(output_size=len(data)):
    # Route to RLM
    ...
```

## Whitepaper Alignment (Algorithm 1)

Our implementation aligns with the core algorithm from Zhang et al.:

1. **Metadata-Only Root Prompting** (`config.metadata_only_root=True`):
   The root LLM receives only constant-size metadata about the context:
   - `context_length` — character count
   - `context_prefix` — first 200 chars
   - `context_type` — inferred type (json, text, csv, xml)
   - Access instructions (slice, parse, split)

   This prevents context window pollution and forces the model to rely on symbolic variable access.

2. **Trimmed Stdout Feedback**: Each turn's stdout is stored in a numbered variable (`_stdout_N`) and only metadata is fed back to the root LLM.

3. **Recursive Sub-Calls**: `rlm_query()` spawns a full sub-RLM at `depth+1` with independent context.

## Structured Outputs (Subagent Contracts)

> CONCEPT:ORCH-1.12 — Structured Predict-RLM Runtime

A swarm of sub-agents only helps if the parent can *cleanly aggregate* what they return. When sub-agents reply with free-form prose, the parent has to re-read and re-classify dozens of unstructured blurbs and frequently loses the plot — it ends up hand-writing an answer instead of routing on the evidence. The fix is to force each sub-agent to return a **schema-constrained, typed value** that the parent reads directly. The booleans (or models, or lists) act as an **external attention mask** over the original context.

### Passing a schema to sub-agents

Both fan-out helpers accept a schema. The sub-RLM's `FINAL` is validated and **coerced** against it, so the parent receives a real Python value — not a string to parse:

```python
# Inside an RLM code block — chunk the context, then ask ONE boolean sub-agent
# per chunk whether it's relevant, in parallel. Filter on the typed result.
chunks = [context[i:i+5000] for i in range(0, len(context), 5000)]

flags = await run_parallel_sub_calls([
    {"prompt": "Does this chunk describe where Saltram lives?",
     "context": c, "schema": {"type": "boolean"}}
    for c in chunks
])
relevant = [c for c, keep in zip(chunks, flags) if keep]   # keep is a real bool

# A single typed sub-call:
is_relevant = await rlm_query("Relevant to his living situation?",
                              sub_context=chunk, schema=bool)
```

### Supported schema forms

`schema=` is normalized by `SchemaContract.from_spec()` (`rlm/schema.py`) and accepts:

| Form | Example | Validated via |
|---|---|---|
| Primitive type | `bool`, `int`, `str`, `float` | `pydantic.TypeAdapter` |
| Typing generic | `list[FindingModel]`, `dict[str, int]` | `pydantic.TypeAdapter` |
| Pydantic model | `class Finding(BaseModel): ...` | `model_validate` |
| Raw JSON Schema | `{"type": "boolean"}` | `jsonschema` (shallow fallback if absent) |

### Validate-on-FINAL, retry-don't-restart

When a contract is set, `run_full_rlm` validates the value the sub-agent passes to `FINAL_VAR`. On a mismatch the sub-agent is shown the **required JSON Schema plus the specific validation errors** (`path: message`) and asked to fix the value and call `FINAL_VAR` again — the REPL state is preserved, so it never restarts from scratch. The JSON Schema is also injected into the sub-REPL prompt at startup so the model knows the exact shape before it writes any code.

### Root-level contracts

The same machinery enforces a contract on the **root** agent. Use a `PredictRLM` signature (`InputField`/`OutputField`) for a multi-field contract, or pass a single typed spec to the entry point:

```python
from agent_utilities.rlm.runner import run_rlm

out = await run_rlm("Is this PR a security risk?", input_text=diff, output_type=bool)
# out["result"] is a real bool
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
| `sandbox` | `"auto"` | Sandbox selection (ORCH-1.38): `auto` routes per-snippet (monty→wasm→docker→local); or pin `local`/`monty`/`wasm`/`docker`. Env `RLM_SANDBOX`. |
| `use_monty` | `False` | Legacy override: force the monty sandbox (maps onto `sandbox`) |
| `use_wasm` | `False` | Legacy override: force the wasm sandbox |
| `use_container` | `False` | Legacy override: force the Docker sandbox |
| `async_enabled` | `True` | Enable parallel sub-call execution |
| `sub_llm_model_large` | Provider default | Model for depth-0 reasoning |
| `sub_llm_model_small` | Provider default | Model for deeper recursion levels |
| `trajectory_storage` | `"process_flow"` | Where to store reasoning traces |
| `metadata_only_root` | `True` | Send only metadata to root LLM |
| `trigger_on_large_output` | `True` | Auto-trigger on large tool outputs |
| `trigger_on_ahe_distillation` | `True` | Auto-trigger for AHE trace analysis |
| `trigger_on_kg_bulk_analysis` | `True` | Auto-trigger for KG bulk queries |
| `ahe_trace_threshold` | `500` | Trace count for AHE auto-trigger |
| `kg_bulk_threshold` | `1000` | Node count for KG auto-trigger |

### Available Helpers

Functions available inside the RLM execution environment:

| Helper | Signature | Purpose |
|---|---|---|
| `rlm_query` | `await rlm_query(prompt, context, schema=None)` | Spawn a recursive sub-RLM at depth+1; pass `schema=` to get a validated, typed return |
| `magma_view` | `await magma_view(query, views)` | MAGMA orthogonal memory views |
| `graph_query` | `await graph_query(cypher, params)` | Run Cypher against the knowledge graph |
| `owl_query` | `await owl_query(sparql)` | Run SPARQL against the OWL reasoner |
| `kg_bulk_export` | `await kg_bulk_export(node_type, limit)` | Export KG nodes as JSON for bulk analysis |
| `sub_agent_call` | `await sub_agent_call(prompt, agent_id, data)` | Dispatch to specialist agent |
| `FINAL_VAR` | `FINAL_VAR("name", value)` | Output the final result |
| `run_parallel_sub_calls` | `await run_parallel_sub_calls(calls)` | Run multiple sub-calls in parallel; each call dict may carry a per-call `"schema"` |

### `run_full_rlm()` Loop

The main agent loop:

1. LLM generates a response (potentially containing ```python blocks)
2. Code blocks are extracted and executed via `execute()`
3. stdout is captured and fed back to the LLM
4. If `FINAL_VAR` was called, the result is returned
5. Otherwise, the loop continues (up to `max_turns=5`)

## AHE Integration (CONCEPT:AHE-3.0)

RLM is deeply integrated with the [Agentic Harness Engineering](AHE_ARCHITECTURE.md) evolution loop:

### TraceDistiller × RLM

When the AHE `TraceDistiller` encounters more than `ahe_trace_threshold` (default: 500) failure traces in an evolution round, it automatically delegates clustering to an RLM sub-agent. The RLM can programmatically:

- Loop over all failure entries
- Apply semantic similarity grouping
- Cross-reference with KG data via `graph_query()` and `owl_query()`
- Produce structured `FailureCluster` objects

Falls back to keyword-based clustering when RLM is disabled or trace count is below threshold.

### EvolveAgent × RLM

When the serialized `EvidenceCorpus` exceeds the context threshold, the `EvolveAgent._deep_analyze_evidence()` method uses RLM to:

- Programmatically analyze all evidence entries
- Cross-reference failure patterns with KG provenance chains
- Produce a prioritized list of `ComponentEdit` proposals

## KG/OWL Integration

### `owl_query(sparql)`

Executes SPARQL queries against the OWL reasoner backend from within the RLM REPL. Enables transitive reasoning without loading raw triples into the context window:

```python
# Inside RLM code block
results = await owl_query("""
    PREFIX au: <http://agent-utilities.dev/ontology#>
    SELECT ?manifest ?edit WHERE {
        ?manifest a au:ChangeManifest .
        ?manifest au:hasEditFor ?edit .
    }
""")
for r in results:
    print(f"Manifest {r['manifest']} -> Edit {r['edit']}")
```

### `kg_bulk_export(node_type, limit)`

Exports KG nodes as JSON dicts for programmatic analysis. The LLM can aggregate, filter, and cross-reference nodes without context pollution:

```python
# Inside RLM code block
memories = await kg_bulk_export("memory", limit=200)
failures = [m for m in memories if "error" in m.get("name", "").lower()]
FINAL_VAR("failure_memories", json.dumps(failures))
```

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

### `rlm_large_output_hook` (`rlm/hook.py`)

A pre-call lifecycle hook (`async def rlm_large_output_hook(input: HookInput)`)
that auto-routes oversized tool/specialist outputs into an RLM pass before they
reach the root LLM context.

### `RecursiveReasonerSpecialist` (`rlm/specialist.py`)

Integration with the graph executor as a specialist node (wrapping an
`RLMEnvironment` via `run_with_context()`). The recursive reasoner can be routed
to by the graph router for tasks requiring iterative code execution. A
module-level `recursive_reasoner_tool(...)` coroutine exposes the same capability
as a callable tool.

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
