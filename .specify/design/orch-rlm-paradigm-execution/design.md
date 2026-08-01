# Design Document: A long prompt is an external REPL variable the model programs against, not text stuffed into its context window

CONCEPT:AU-ORCH.execution.recursive-language-model · CONCEPT:AU-ORCH.execution.rlm-execution

> `agent_utilities/rlm/config.py` (the configuration surface and trigger policy) and
> `agent_utilities/rlm/repl.py` / `agent_utilities/rlm/__init__.py` (the REPL environment
> that carries the policy out). Same paradigm, described from the config side and the
> execution side of the same module.

## Decision — `AU-ORCH.execution.recursive-language-model`

`RLMConfig` (`agent_utilities/rlm/config.py:10-19`) states the core architectural bet in its
own class docstring: "RLM provides a persistent Python REPL that enables agents to process
arbitrarily long inputs through recursive, programmatic decomposition... the key insight is
that long prompts should NOT be fed into the neural network directly but treated as part of
the environment the LLM symbolically interacts with." This is Algorithm 1 of Zhang et al.
(2025), and the config encodes it as concrete, testable policy rather than prose intent:

- `metadata_only_root` (`config.py:134-143`, default `True`) — the root LLM receives only
  `context_length`/`context_prefix`/`context_type`/access-instruction metadata, never the raw
  payload, "aligns with Algorithm 1... prevents context window pollution and forces the model
  to rely on symbolic variable access and sub-calls."
- A **five-level trigger hierarchy** (`config.py:20-27`, implemented in `should_trigger()` at
  `config.py:180-215`), evaluated top-to-bottom: global `enabled=True` override, then
  size/trace-count/node-count thresholds (`trigger_on_large_output`,
  `trigger_on_ahe_distillation`, `trigger_on_kg_bulk_analysis`), then
  `state.requires_long_horizon`. The docstring is explicit that callers must go through
  `should_trigger()` rather than checking flags ad hoc — "Call this instead of checking
  individual flags to ensure consistent routing decisions across the codebase"
  (`config.py:188-191`) — because with five independent triggers, scattered flag-checking
  would silently drift out of precedence order.
- `allow_auto_trigger` (`config.py:65-71`, default `False`) is a deliberate safety gate: the
  four size/trace/node auto-triggers are inert unless this is separately turned on, "so a
  disabled execution feature cannot auto-start" even if e.g. `trigger_on_large_output` is left
  at its own default of `True`.

**The rejected alternative**, named directly by the whitepaper-alignment comment at
`config.py:132`, is the conventional long-context approach: grow the model's context window
or chunk-and-summarize the payload before the model ever sees it. Both keep the data inside
the neural network's attention path. RLM's bet is that treating the payload as an inert
variable the model *writes code against* — slicing, filtering, recursing — scales past any
fixed context window and avoids feeding irrelevant content through attention at all. The
accepted cost is complexity: five trigger conditions to reason about instead of one context
limit, and a REPL execution model instead of a single forward pass.

### Pointer — `CONCEPT:AU-ORCH.execution.rlm-execution`

`agent_utilities/rlm/__init__.py:1-4` and `agent_utilities/rlm/repl.py:70-96` are the same
decision from the execution side: `RLMEnvironment` is, in its own docstring, "a persistent
Python REPL environment for Recursive Language Models... Implements Algorithm 1 from Zhang
et al. (2025): the user prompt is loaded as a variable inside the REPL — the root LLM
receives only constant-size metadata... and writes code to programmatically examine,
decompose, and recursively call itself over slices of the prompt" (`repl.py:70-79`). The
helper surface it exposes to that generated code —
`rlm_query`/`run_parallel_sub_calls`/`magma_view`/`graph_query`/`owl_query`/`kg_bulk_export`/
`sub_agent_call`/`FINAL_VAR` (`repl.py:81-89`) — is the concrete mechanism that makes the
config-side policy above (metadata-only root, recursive sub-calls) executable rather than
aspirational. There is no separate "should we build the REPL differently" decision here: the
`rlm-execution` marker names the implementing class for the same Algorithm-1 bet that
`recursive-language-model` names as configuration policy, and the two markers exist because
one module (`config.py`) and one class (`repl.py`'s `RLMEnvironment`) each got their own
docstring-level marker rather than one being cross-referenced from the other.

## Risk Assessment

- **Blast Radius**: `agent_utilities/rlm/config.py`, `agent_utilities/rlm/repl.py`,
  `agent_utilities/rlm/__init__.py`, and every caller that constructs `RLMConfig` or
  `RLMEnvironment` directly (bypassing `should_trigger()` reintroduces the precedence-drift
  risk the docstring warns against).
- **Backward Compatible**: Yes — this is the foundational module; no alternative path exists
  to compare against inside this repo.
- **Known weak point**: `should_trigger()` is advisory, not enforced — nothing prevents a new
  call site from reading `config.trigger_on_large_output` directly and skipping the hierarchy,
  which is exactly the drift the method's own docstring was written to prevent.
