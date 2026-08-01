# Design Document: One local-inference slot is always held back from background KG work so the interactive path never starves

CONCEPT:AU-ORCH.execution.reserved-inference-slots

> `agent_utilities/core/config.py` (the constant + `background_llm_concurrency()`) and
> `agent_utilities/knowledge_graph/ingestion/engine.py:1496` (the background enrichment sweep
> that is bounded by it).

## Decision — a hardcoded reservation of 1 slot, not a configurable knob

`agent_utilities/core/config.py:60-64` defines the constant directly in comment form: "local-
inference slots always kept free for the interactive path (the messaging responder +
graph-os-spawned pydantic-ai agents, which share the default model). Background KG work is
bounded to (capacity − this). **A constant, not a knob**: 1 is the correct universal default
(config discipline — no flag for a one-correct-value)." `RESERVED_INTERACTIVE_INSTANCES = 1`
is a bare module-level constant, not a `Field` with an env-var alias like every other tunable
in the same file.

`kg_llm_concurrency` (`config.py:4809-4819`, `KG_LLM_CONCURRENCY`) is, per its own docstring,
"the ONE knob for local-model parallelism," representing total parallel capacity of the local
inference endpoint (e.g. vLLM/LM Studio slots). `background_llm_concurrency()`
(`config.py:4821-4824`) is the derived ceiling: `max(1, self.kg_llm_concurrency -
RESERVED_INTERACTIVE_INSTANCES)` — "Floors at 1 so background never starves" even at
`kg_llm_concurrency=1`, verified directly by
`tests/unit/test_interactive_slot_reserve.py:27-29` (`_with_capacity(1).background_llm_concurrency()
== 1`, i.e. the reservation does not push background work to zero, it just stops shrinking the
margin below the floor). The same test module docstring restates the policy in one line:
"Background KG work is bounded to capacity − reserved" (`tests/unit/test_interactive_slot_reserve.py:1-6`).

The real consumer is `agent_utilities/knowledge_graph/ingestion/engine.py:1493-1503`: the
enrichment pipeline's per-document window fan-out is bounded by a semaphore sized off
`background_llm_concurrency()`, explicitly reasoned about as "the ceiling is local-inference
capacity (`KG_LLM_CONCURRENCY`) MINUS the reserved interactive slot, so this background sweep
can never starve the slot the messaging responder / graph-os-spawned agents need to answer" —
and the same semaphore is shared across both the concept-extraction and fact-extraction passes
("they run sequentially: concepts, then facts") specifically to bound *total* in-flight LLM
work against the one shared endpoint, not per-pass.

**The rejected alternative** is a configurable reservation — an env-settable
`RESERVED_INTERACTIVE_INSTANCES` knob, matching the pattern every other capacity setting in
the file follows (`alias="..."` + a `Field` default). The comment at `config.py:60-63` argues
against this on config-discipline grounds: there is no scenario where an operator legitimately
wants zero slots reserved for the interactive path (that would let a background sweep starve
the messaging responder outright) or more than a small fixed number reserved (which would just
be `kg_llm_concurrency` tuned lower instead) — so exposing it as a flag would only invite
misconfiguration of a value that has exactly one correct answer. The accepted cost is that
changing the reservation later requires a code change, not an env var, but that is the point:
it forces a deliberate decision rather than an accidental one.

## Risk Assessment

- **Blast Radius**: `agent_utilities/core/config.py` (`RESERVED_INTERACTIVE_INSTANCES`,
  `background_llm_concurrency()`), `agent_utilities/knowledge_graph/ingestion/engine.py`
  (the enrichment-window semaphore), and any other caller that sizes background LLM
  concurrency off `kg_llm_concurrency` directly instead of going through
  `background_llm_concurrency()`.
- **Backward Compatible**: Yes — a floor/ceiling policy over an existing capacity knob; no
  interface change for callers already using `background_llm_concurrency()`.
- **Known weak point**: the reservation is enforced only by convention — any code path that
  reads `AgentConfig.kg_llm_concurrency` directly instead of calling
  `background_llm_concurrency()` silently bypasses the reservation, with nothing in the type
  system or a lint rule to catch it.
