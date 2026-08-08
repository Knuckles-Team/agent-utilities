# Design Document: A routing decision persists the models it REJECTED and why, ranked against the tier the picker actually used

CONCEPT:AU-ORCH.routing.rejected-candidate-provenance

> Realised by `agent_utilities/models/model_registry.py:135-176`
> (`CandidateScore`, `RoutingDecision`) and `:572-660`
> (`explain_pick_for_task`, with the ranking fix documented at `:606-611`);
> the graph node is `agent_utilities/models/knowledge_graph.py:1658`
> (`RoutingDecisionNode`); the adaptive caller is
> `agent_utilities/core/model_router.py:90-108`
> (`pick_adaptive_with_decision`). Introduced by commit `6b9d81bb`, corrected by
> commit `edc1761f`.

## Decision — keep the counterfactual, because a router that discards what it rejected can never be improved

`docs/architecture/model_registry_graph_resources.md:7-9` states the gap this
closes: *"the router picked a model but discarded what it rejected and why, so
model choice could never become an evolution target — there was no
counterfactual to learn from."*

This is the specific argument. A log line saying "chose model X" records an
outcome. It cannot answer the question that would improve routing — *was X the
right choice?* — because answering that requires knowing what else was
available, how each option scored, and why the losers lost. Without the
rejected set there is nothing to compare an outcome against, so no amount of
recorded outcomes accumulates into a better router.

So `explain_pick_for_task` persists a `RoutingDecision` carrying every
candidate's score, tier-rank and rejection reason, bounded at
`MAX_ROUTING_CANDIDATES = 8`, attached to the trace as a `RoutingDecisionNode`.
The bound is itself a decision: routing happens on every call, so an unbounded
candidate list would make provenance the dominant cost of routing.

**The rejected alternative is the prior behaviour — discard the losers, keep
only the winner** — rejected because it makes the routing layer permanently
unlearnable, not because it was expensive or wrong per call.

## The correction is part of the decision, not a footnote

The mechanism shipped subtly wrong and the fix (`edc1761f`, *"rank routing
provenance against the tier the picker actually used"*) is what makes the
recorded counterfactual trustworthy. Candidates were being scored against the
caller's *nominal* complexity tier, while the adaptive picker
(`CONCEPT:AU-ORCH.routing.adaptive-role-routing`) had already shifted to an
*effective* tier using the learned confidence. The two disagreed whenever
adaptive routing did anything at all, and the commit reproduces the result
numerically: the chosen `m-light` scored 0.333 while the rejected `m-medium`
scored 1.0 — *"any adaptive decision that actually shifted tier persisted a
`RoutingDecisionNode` showing the router picking the model it ranked WORST."*

That is worse than recording nothing. A provenance record that inverts the
ranking would teach any consumer — a human reviewer or an evolution loop — the
exact opposite of what the router did. Hence the invariant now enforced at
`:606-611`: candidates are ranked against the same effective tier and the same
confidence signal the picker used, never a re-derived one.

## Risk Assessment

- **Blast Radius**: `agent_utilities/models/model_registry.py`,
  `agent_utilities/models/knowledge_graph.py`,
  `agent_utilities/core/model_router.py`.
- **Backward Compatible**: Yes — additive provenance; the pick itself is
  unchanged.
- **Known weak point**: the ranking is only correct as long as every caller
  passes the *same* confidence and tier into `explain_pick_for_task` that it
  passed into the pick. `pick_adaptive_with_decision` guarantees this by
  construction for the adaptive path (it computes the confidence once and
  passes it to both), but a future caller that calls `explain_pick_for_task`
  independently can silently reintroduce exactly the `edc1761f` bug, and no
  test pins that coupling.
