# Design Document: Predict-RLM is a dependency-free Pydantic signature runtime, and subagent fan-out returns typed values instead of prose

CONCEPT:AU-ORCH.execution.predict-rlm-runtime

> `agent_utilities/rlm/predict_rlm.py` (the signature harness), `agent_utilities/rlm/runner.py`
> (the entry point and its explicit non-scope), `agent_utilities/rlm/schema.py` (the subagent
> output-contract normalizer), `agent_utilities/rlm/config.py:227` (the lossless-vs-compaction
> escalation decision), `agent_utilities/graph/_router_impl.py:588` (where the graph router
> invokes it), and the pre-existing catalog entry
> `docs/pillars/1_graph_orchestration/ORCH-1.12-Structured_RLM_Outputs.md`.

## Decision 1 — a native Pydantic signature system, not an external typed-program-signature dependency

`agent_utilities/rlm/predict_rlm.py:1-8` states the design goal plainly: "Predict-RLM:
Structured, type-safe RLM executions using Pydantic signatures... This module implements a
native Pydantic signature system to replicate the structured input/output contract of typed
program signatures **without adding external dependencies**." `InputField`/`OutputField`
(`predict_rlm.py:33-44`) are thin `pydantic.Field` wrappers that tag `is_input`/`is_output` in
`json_schema_extra`, and `PredictRLM` (`predict_rlm.py:47-60`) parses a caller's
`BaseModel` subclass into that input/output split. `_validate_purity()`
(`predict_rlm.py:22-30`) rejects any tool-function source using `global`/`nonlocal` — signature
functions mounted into the sandbox must be pure.

**The rejected alternative**, named directly by the docstring, is pulling in an external
typed-program-signature library (the phrasing "replicate... typed program signatures" is a
direct reference to that class of framework) as a dependency rather than building the
narrower Pydantic-based equivalent this codebase already depends on everywhere else. The
accepted cost is reimplementing signature parsing/validation instead of getting it for free
from an established library.

`agent_utilities/rlm/runner.py:1-7` draws the resulting scope boundary explicitly: "The RLM
runtime executes bounded recursive inference through `run_rlm`. **Program optimization is a
separate native epistemic-graph responsibility** exposed through `graph_evolution
action=optimize_component`; this module contains no prompt optimizer or alternate model
stack." `run_rlm()` (`runner.py:47-64`) builds an ad-hoc `AdHocRLMSignature` via
`_dynamic_signature()` for callers that don't want to hand-author a `BaseModel`
(`runner.py:26-44`), and `output_type` accepts "any type pydantic can validate — a primitive
(`int`, `bool`), a typing generic (`list[Model]`), or a Pydantic model — so the root contract
is not limited to a free-form string" (`runner.py:33`). GEPA-style genetic prompt optimization
exists in this codebase (`agent_utilities/rlm/gepa.py`, exercised by
`tests/unit/core/test_rlm_gepa.py`) but is deliberately kept as a *consumer* of this runtime's
structured traces, not folded into it — the runtime executes, a separate optimizer (documented
at `docs/pillars/1_graph_orchestration/ORCH-1.13-GEPA_Optimization.md`, outside this concept's
scope) proposes.

## Decision 2 — subagent fan-out returns a schema-constrained typed value, never free-form prose

`agent_utilities/rlm/schema.py:1-16` names the failure mode this fixes: "A Recursive Language
Model degrades when subagents return free-form prose: the parent has to re-read and
re-classify dozens of unstructured blurbs, losing the plot." `docs/guides/rlm.md:93-95`
(under the catalog heading `CONCEPT:AU-ORCH.execution.predict-rlm-runtime`) restates the
consequence more concretely: the parent "ends up hand-writing an answer instead of routing on
the evidence." The fix is `SchemaContract` (`schema.py`), which normalizes every schema form a
caller might supply — a Pydantic `BaseModel`, a primitive/typing generic via
`pydantic.TypeAdapter`, or a raw JSON-Schema dict — into one contract that both renders itself
for the LLM prompt and validates/coerces a returned value. Per the pre-existing catalog doc
(`docs/pillars/1_graph_orchestration/ORCH-1.12-Structured_RLM_Outputs.md`), `rlm_query(...,
schema=...)` and the per-call `"schema"` key in `run_parallel_sub_calls` build the sub-RLM with
this contract, and validation happens "validate-on-FINAL, retry-don't-restart": a mismatch
shows the model the JSON Schema plus specific errors and continues with REPL state intact,
rather than restarting the sub-call from scratch.

**The rejected alternative** is exactly what the docstring names: leave subagent returns as
free text and let the parent re-read/re-classify each one. The chosen design instead treats
the typed value as, in the catalog doc's words, "an external attention mask over the original
context" — the parent filters on `True`/`False` or a model field directly instead of
re-reasoning over prose.

`agent_utilities/graph/_router_impl.py:579-598` is a concrete call site that ties this back to
the wider RLM trigger policy: the graph router constructs an `RLMEnvironment` when
`rlm_config.enabled or len(agent_context) > rlm_config.max_context_threshold`, runs the
planning instruction through it, and parses the result as a `GraphPlan` — falling back to a
dedicated parser agent when the RLM output isn't valid `GraphPlan` JSON (`_router_impl.py:588-609`,
marked `R10` in the surrounding comment) — the same "structured output, with a safety-net
fallback on parse failure" posture as `SchemaContract`'s validate/coerce/retry loop.

A third, narrower decision shares this marker: `select_long_context_strategy()`
(`agent_utilities/rlm/config.py:217-241`) returns `"rlm_lossless"` when `should_trigger()`
fires, `"memento_compaction"` when the payload is merely large but under the RLM trigger
(falling back to lossy KG-2.20 Memento compaction), else `"none"`. This makes the
lossless-vs-lossy tradeoff an explicit, callable decision rather than leaving callers to infer
it from `should_trigger()` alone — RLM's structured contract is preferred whenever it is
triggered; compaction is only ever the fallback for payloads too large to ignore but not large
enough (or not the right shape) to warrant a full RLM run.

## Risk Assessment

- **Blast Radius**: `agent_utilities/rlm/predict_rlm.py`, `agent_utilities/rlm/runner.py`,
  `agent_utilities/rlm/schema.py`, `agent_utilities/rlm/config.py`,
  `agent_utilities/graph/_router_impl.py`.
- **Backward Compatible**: Yes — `output_type`/`schema=` are opt-in; callers that pass neither
  keep the free-form-string behavior.
- **Known weak point**: `SchemaContract`'s raw-JSON-Schema-dict validation path depends on the
  optional `jsonschema` package; without it, validation falls back to "a **non-silent** shallow
  `type`/`required` check" per the catalog doc — weaker coverage than the Pydantic-model or
  primitive/typing-generic paths get.
- **Review note**: `tests/unit/core/test_rlm_gepa.py:1-4` carries this concept's marker (with a
  `/31` suffix truncated by the marker grammar) but its content is GEPA optimizer tests, a
  neighboring but distinct concept (`ORCH-1.13`). That test file's marker placement is
  imprecise, not this decision — the decision itself (native Pydantic signatures + typed
  subagent contracts) is independently well-grounded in `predict_rlm.py`, `runner.py`,
  `schema.py`, and the pre-existing `ORCH-1.12` catalog doc, so it is documented here rather
  than left for retirement.
