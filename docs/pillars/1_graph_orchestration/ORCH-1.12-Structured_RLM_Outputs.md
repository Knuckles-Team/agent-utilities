# Structured Predict-RLM Runtime + Subagent Contracts (CONCEPT:AU-ORCH.execution.predict-rlm-runtime)

## Overview

Makes RLM **structured I/O end-to-end**. The root agent already ran under a Pydantic signature
(`InputField`/`OutputField`) validated on `FINAL`. This extends the same guarantee to the **subagent
fan-out**: an RLM can force each sub-call to return a *schema-constrained, typed* value (a boolean
relevance flag, a Pydantic model, a `list[...]`) instead of free-form prose.

Why it matters: a swarm of sub-agents only helps if the parent can cleanly aggregate the results.
With free-text returns the parent re-reads and re-classifies dozens of unstructured blurbs and loses
the plot — it ends up hand-writing an answer rather than routing on the evidence. A typed return is an
**external attention mask** over the original context: the parent filters on `True`/`False` (or a
model field) directly. Aligns with the RLM-structured-outputs writeup; extends **ORCH-1.12** and
composes with the resilience telemetry of **[ORCH-1.29](ORCH-1.29-RLM_Resilience_And_Telemetry.md)**.

## How it works

- **`SchemaContract`** (`rlm/schema.py`) — `from_spec()` normalizes every supported schema form into
  plain JSON Schema: a Pydantic `BaseModel` (`model_json_schema()`), a primitive / typing generic
  (`pydantic.TypeAdapter`), or a raw JSON-Schema `dict`. `.validate(value)` returns
  `(ok, coerced_value, error)` with `path: message` errors. Raw-dict validation uses the optional
  `jsonschema` package, falling back to a **non-silent** shallow `type`/`required` check when absent.
- **Per-subagent `schema=`** (`rlm/repl.py`) — `rlm_query(prompt, context, schema=…)` and a per-call
  `"schema"` key in `run_parallel_sub_calls` build the sub-`RLMEnvironment` with an `output_contract`.
  The depth-floor fallback applies the contract via `pydantic_ai` `output_type`. The sub-RLM returns
  the **coerced typed value**, not a string.
- **Validate-on-FINAL, retry-don't-restart** — the existing `run_full_rlm` loop validates the
  `FINAL` value against the contract; on mismatch it shows the JSON Schema + specific errors and
  continues with REPL state intact (no restart). The schema is injected into the sub-REPL prompt at
  startup, and `schema=` is advertised in the helper docs so the model actually emits it (Wire-First).
- **Root contract generalized** — `run_rlm(..., output_type=…)` (`rlm/runner.py`) and
  `_generate_instruction_prompt` (`rlm/predict_rlm.py`) accept/show primitive/generic/model output
  specs, not just `str`.

## Key files / API

| Piece | Location |
|---|---|
| Schema normalizer | `rlm/schema.py` (`SchemaContract`, `from_spec`, `validate`) |
| Subagent fan-out + validation | `rlm/repl.py` (`rlm_query`, `run_parallel_sub_calls`, `_validate_outputs`, `run_full_rlm`) |
| Root signature + prompt | `rlm/predict_rlm.py` (`PredictRLM`, `InputField`/`OutputField`) |
| Entry point | `rlm/runner.py` (`run_rlm(..., output_type=…)`) |

## Example

```python
# Inside an RLM code block: one boolean sub-agent per chunk, in parallel.
flags = await run_parallel_sub_calls([
    {"prompt": "Relevant to where Saltram lives?", "context": c, "schema": {"type": "boolean"}}
    for c in chunks
])
relevant = [c for c, keep in zip(chunks, flags) if keep]   # keep is a real bool
```

## Wiring (≤3 hops)

`graph_orchestrate(rlm_run)` → `runner.run_rlm` → `predict_rlm`/`repl` → (per code block) `rlm_query`
/ `run_parallel_sub_calls` → `SchemaContract.validate`.

## Tests

- `tests/unit/rlm/test_schema_contract.py` — every spec form + jsonschema-absent fallback.
- `tests/unit/rlm/test_subagent_schema.py` — typed `rlm_query`, schema-violation retry, per-call schema.
- `tests/unit/rlm/test_subagent_schema_live_path.py` — structured fan-out on the live `run_full_rlm`
  path + prompt-surface assertion (the article's boolean-attention-mask pattern, end-to-end).
