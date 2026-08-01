# Design Document: Sampling profiles are tournament-evolved per task-class, not fixed hand-tuned defaults

CONCEPT:AU-AHE.harness.evolvable-sampling-profiles

> `agent_utilities/harness/variant_pool.py:409-430` (primary),
> `agent_utilities/agent/sampling_profile.py:15-19` (the static contract this
> evolves), `agent_utilities/models/model_registry.py:467` (where the winner
> lands).

## Decision — mutate a task-class's sampling knobs, score children via the existing capability-reward EMA, tournament-promote the winner into the live registry

`sampling_profile.py:15-19` states the baseline: a `SamplingProfile` is
**selected** per task-class (deterministic low-temp for code/extraction,
exploratory high-temp for brainstorming) and **threaded** as a per-call
`model_settings=` override — a static, hand-authored mapping. `variant_pool.py:409-414`
is where that stops being static: "the parametric-variant dimension this
module's docstring has always named ('mutating configuration parameters
(temperature, ...)') wired into a live tournament: mutate a task-class
profile, score each child by the capability reward EMA (`record_outcome`),
and promote the winner back into the registry's `task_class_profiles`" — read
by the router/factory on the next run.

**The rejected alternative is what the static baseline represents: sampling
knobs hand-tuned once per task-class and left fixed.** That requires a human
to notice a task-class is mis-tuned and manually adjust temperature/top_p/
top_k/min_p/repetition_penalty — a slow, easily-stale feedback loop. Reusing
the SAME capability-reward EMA mechanism the rest of the variant pool already
uses for scoring means a sampling-profile mutation is evaluated on exactly
the same terms as any other variant: it must actually win a tournament
against the current profile before it's promoted, not just get applied
because someone guessed it would help. `DEFAULT_PROFILE` (all-`None` knobs)
is a deliberate escape hatch: it reproduces the pre-profile static defaults
exactly, guaranteeing zero behavior change when no specific profile has been
resolved yet.

## Risk Assessment

- **Blast Radius**: `agent_utilities/harness/variant_pool.py`,
  `agent_utilities/agent/sampling_profile.py`,
  `agent_utilities/models/model_registry.py`,
  `agent_utilities/agent/factory.py` (reads `task_class_profiles`).
- **Backward Compatible**: Yes — `DEFAULT_PROFILE` is a no-op fallback; the
  static defaults in `agent.factory.create_agent` are the base every profile
  merges over, never removed.
- **Known weak point**: the reward EMA that scores a mutated profile is the
  same generic capability-reward signal used elsewhere in the variant pool —
  it is not sampling-specific, so a knob combination that scores well on the
  measured capability but degrades an unmeasured quality (e.g. output
  diversity) can still win the tournament.
