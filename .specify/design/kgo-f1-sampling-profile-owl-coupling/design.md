# Design Document: LLM sampling knobs are SHACL-bounded ontology value types, so a profile is checked at the graph write gate before the evolution loop can promote it

CONCEPT:AU-KG.ontology.sampling-profile-coupling

> `agent_utilities/knowledge_graph/ontology/value_types.py:574-613`,
> `agent_utilities/agent/sampling_profile.py`.

## Decision — every sampling knob (temperature/top_p/top_k/min_p/repetition_penalty/max_tokens) is a named `ValueType` with real bounds, not a bare float/int on the profile object

`value_types.py:574-577` states the coupling directly: "A SamplingProfile (the
`InferenceProfile` interface, KG-2.95) is SHACL-checked at the graph write
gate before the AHE-3.38 loop can promote it. Bounds mirror the OpenAI/vLLM
accepted ranges and the SamplingProfile pydantic field constraints." Each
knob gets its own value type: `Temperature` (`[0,2]`), `TopP` (`[0,1]`),
`MinP` (`[0,1]`), `TopK` (`>=1`), etc. — each with the SAME bounds the
pydantic `SamplingProfile` model already enforces at the Python layer,
declared a second time as SHACL constraints so a write that bypasses the
pydantic model (a direct graph write, a bulk import) is still bounded at the
graph gate.

`sampling_profile.py:1-23` grounds the surrounding lifecycle this coupling
sits in: a profile is **selected** per task-class (deterministic low-temp for
code/extraction, exploratory high-temp for brainstorming), **threaded** as a
per-call `model_settings=` override built FROM the agent's static base
settings ("an unset knob keeps its default — pydantic-ai replaces, it does not
deep-merge"), later **evolved** by the AHE harness loop, and — the concept
this doc covers — **projected to OWL** so the evolution loop's promotion gate
can validate a candidate profile against the SAME bounds the runtime
pydantic model enforces.

**The rejected alternative is trusting the pydantic model as the ONLY
bounds check** — sufficient for profiles created through the normal
application code path, but not for the evolution loop, which promotes
CANDIDATE profiles that may originate from a mutation/search process outside
that path. Projecting the bounds to SHACL means the graph write gate — the
one checkpoint every promoted profile passes through regardless of how it was
generated — enforces the same numeric ranges, so an out-of-range candidate
(a mutated `temperature` of 5.0, say) is rejected at the gate rather than
silently promoted and only failing later at the OpenAI/vLLM API call.
`DEFAULT_PROFILE` (all-`None` knobs) reproduces `create_agent`'s static
defaults exactly, "guaranteeing zero behaviour change when no specific
profile is resolved" — the coupling is additive to callers that never
override any knob.

## Risk Assessment

- **Blast Radius**: `knowledge_graph/ontology/value_types.py`
  (`Temperature`/`TopP`/`MinP`/`TopK`/…), `agent/sampling_profile.py`
  (`SamplingProfile`), the AHE-3.38 evolvable-sampling-profiles promotion
  gate.
- **Backward Compatible**: Yes — `DEFAULT_PROFILE` reproduces prior static
  behavior exactly.
- **Known weak point**: the two bounds declarations (pydantic field
  constraints, SHACL value-type bounds) are maintained independently in two
  files — the comment states they "mirror" each other, but nothing mechanically
  enforces the mirror; a future change to one accepted range without the
  matching update to the other would silently reopen the exact gap this
  coupling exists to close.
