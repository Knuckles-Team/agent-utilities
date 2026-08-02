# Design Document: Reasoning starts at the cheapest topology rung and escalates one rung at a time on measured uncertainty — it never starts at the most expensive one

CONCEPT:AU-ORCH.routing.topology-escalation-policy

> Realised by `agent_utilities/graph/reasoning/policy.py:1-97`
> (`EscalationDecision`, `EscalationPolicy.choose`). Documented for readers at
> `docs/architecture/reasoning-graph-topologies.md:9`. The underlying topology
> set was ported by commit `947d007b` from *"Graph Engineering: A Unified
> Framework for Language Agent System Design"* (arXiv:2505.24354).

## Decision — cheapest-adequate-first, with escalation as the only direction of travel

Once the codebase had several reasoning topologies available — CoT, ToT, GoT,
ReAct, RAP — it needed a rule for picking one per run. The module docstring
states the rule and the alternative it displaces in a single sentence:
*"start with the cheapest ADEQUATE topology and escalate only on MEASURED
uncertainty or failure — never a hardcoded 'always use the most expensive
topology' choice."*

`EscalationPolicy.choose` starts every run at `cot`, the cheapest rung. It
escalates exactly one rung when prior confidence or prior reliability is
measured low. Three properties are deliberate:

- **One rung at a time.** Jumping straight to the top on the first weak signal
  would recover the cost profile the policy exists to avoid.
- **Never de-escalates.** Within a run, escalation is monotonic — a run that
  has already shown it needs a richer topology does not get demoted on a single
  better sample.
- **Escalation is evidence-driven, not shape-driven.** The trigger is measured
  confidence/reliability, not a guess about the prompt.

`needs_tools` is handled separately and does **not** participate in the ladder:
it routes straight to ReAct. This is the sharpest part of the design. Tool use
is a property of what the task *is*, not of how hard it is — a trivial question
that needs one API call is not "more uncertain" than a hard one that needs
none. Folding tool-need into the cost ladder would have made every tool-using
turn look like an escalation and inflated the ladder's apparent trigger rate.
So the policy has two orthogonal axes: an escalation ladder for difficulty, and
a direct route for task shape.

**The rejected alternative is the naive default the docstring names — always
run the most expensive topology.** It is the safest possible policy for answer
quality and is what a system does before anyone measures cost. It was rejected
because the topologies differ in cost by large multiples, and most turns are
adequately served by the cheapest rung, so the naive default pays the maximum
price on every turn to protect a minority of them.

## Risk Assessment

- **Blast Radius**: `agent_utilities/graph/reasoning/policy.py` and the
  reasoning-graph selection path.
- **Backward Compatible**: Behaviourally no — runs that previously used a
  richer topology by default now start at `cot`.
- **Known weak point**: escalation is driven by *prior* confidence and
  reliability, so the first run of a genuinely hard, novel task has no prior to
  escalate on and will attempt it at the cheapest rung. The policy discovers
  that class of task by getting it wrong once. There is also no mechanism for a
  caller who *knows* a task is hard to declare it up front and skip the ladder.
