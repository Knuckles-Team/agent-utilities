# Design Document: Sampling parameters are a per-call, task-aware bundle layered over the agent's static settings — not one fixed setting per agent

CONCEPT:AU-ORCH.routing.sampling-profile-selection

> Realised by `agent_utilities/agent/sampling_profile.py:1-24` (module
> docstring) and `:52-87` (`SamplingProfile`), threaded at
> `agent_utilities/agent/factory.py:669` and `:828-836`, with the curated
> per-task-class defaults at
> `agent_utilities/models/model_registry.py:101-132` and the resolver at
> `:444-462`. Introduced by commit `c75bbbc7` ("Task-aware, evolvable,
> ontology-driven LLM sampling profiles").

## Decision — separate *which model answers* from *how we sample from it*, and make the second axis per-call

The router already had a well-developed answer to "which model should serve
this turn" (tiers, tags, roles — `CONCEPT:AU-ORCH.routing.conductor-per-step-model`).
It had no answer at all to the orthogonal question of *how* to sample from
whichever model won. The introducing commit states the split in one sentence:
*"The router already picks WHICH model answers a question; this picks HOW to
sample from it."*

A `SamplingProfile` (`sampling_profile.py:52-87`) bundles
temperature, top_p, top_k, min_p, repetition_penalty and the frequency/presence
penalties, and is resolved per task-class and role. `factory.py:828-836` wraps
the agent's run path so each call threads the resolved profile as a
`model_settings` override, *unless* the caller passed one explicitly.

**The rejected alternative is a single static `settings` object per agent,
which is still in the tree and is still what runs when no profile resolves.**
The choice is stated at the call site (`factory.py:830-833`): the static
settings created in `create_agent` are *"the base floor"*, and the module
docstring makes the relationship explicit — *"The static defaults in
`agent_utilities.agent.factory.create_agent` are not removed — they are the
*base* this profile merges over"* (`sampling_profile.py:20-23`). Static
per-agent settings were rejected not as wrong but as *too coarse*: one agent
serves many task classes, and a temperature good for creative drafting is bad
for structured extraction. Deleting the static path entirely was also rejected
— it is the merge base, so a task class with no curated profile degrades to
exactly the previous behaviour rather than to an arbitrary default.

## Why the sibling markers point here rather than each earning a document

Two other markers sit on this same decision observed at other seams, both
introduced by the same epic:

- `route-inference-parameters`
  (`agent_utilities/graph/adaptive_agent_router.py:79-101`, populated at
  `:222`, `:292`, `:616`) is the `RoutingDecision.sampling_profile` field —
  the same profile bundle, carried on the routing decision so the chosen
  parameters travel with the chosen model. It is the data-carrier for this
  decision, not a separate one.
- `depth-tiered-sampling` (`agent_utilities/rlm/repl.py:658-666`) is one
  *application* of this mechanism: the RLM REPL picks the `rlm-root` profile at
  `depth == 0` and `rlm-executor` below it. Commit `0fce51c5` (the same
  sampling-profile epic) shows the prior code passed no `model_settings` at all
  to either `agent.run()` call site. The depth split is a curated profile
  choice, which is precisely the extension point this decision created.

## Risk Assessment

- **Blast Radius**: `agent_utilities/agent/sampling_profile.py`,
  `agent_utilities/agent/factory.py`,
  `agent_utilities/models/model_registry.py`,
  `agent_utilities/graph/adaptive_agent_router.py`,
  `agent_utilities/rlm/repl.py`.
- **Backward Compatible**: Yes — an unresolved task class falls through to the
  static base settings, and an explicit caller-supplied `model_settings` always
  wins over the resolved profile.
- **Known weak point**: profiles are curated per task-class by hand
  (`model_registry.py:101-132`). Nothing measures whether a profile is
  *better* than the base it overrides, so a badly-chosen profile degrades
  output quality silently — the mechanism is evolvable in principle (profiles
  are ontology-published) but there is currently no closed feedback loop
  scoring a profile against outcomes, unlike the model-routing axis.
