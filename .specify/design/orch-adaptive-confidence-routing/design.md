# Design Document: Route each role's model pick through a *learned* per-route reward-EMA, not its statically-assigned tier

CONCEPT:AU-ORCH.routing.adaptive-role-routing

> Realised by `agent_utilities/core/model_router.py:1-131`
> (`route_confidence`, `record_model_outcome`, `pick_adaptive`,
> `pick_adaptive_with_decision`), wired at
> `agent_utilities/core/model_factory.py:189-206` (`_resolve_role_model`).
> The tier-shift primitive it drives is
> `agent_utilities/models/model_registry.py:474-570`
> (`_tier_down` / `_tier_up` / `_effective_tier` / `pick_for_task_adaptive`);
> the outcome write-back is
> `agent_utilities/knowledge_graph/adaptation/feedback.py:374-384`; the
> budget-path forwarder is
> `agent_utilities/core/resource_optimizer.py:131-181`
> (`ResourceOptimizer.select_model_for_step`).
> Introduced by commit `2703d4c9` ("feat: operator Phase 2/3
> autonomy+economics — adaptive model router").

## Decision — a role's tier is a *starting point* that observed outcomes move, not a fixed binding

`CONCEPT:AU-ORCH.routing.conductor-per-step-model` (documented in
`.specify/design/orch-1.27-role-specialized-routing/design.md`) established
that a functional role binds to a *tier + tags*, resolved through the registry,
rather than to a hardcoded model id. That decision is about **indirection**. It
leaves a second question open: once `planner` is bound to the `heavy` tier, is
it bound there forever?

This decision answers no. Every role-routed selection now flows through a
per-route reward-EMA (`_ROUTE_REWARDS`, α = 0.3, 0.5 neutral —
`model_router.py:29-31`). `pick_adaptive` resolves the role's *base* tier via
`registry.resolve_role`, reads the route's learned confidence, and hands both
to `pick_for_task_adaptive`, which shifts the effective tier one rung down when
confidence is high and one rung up when it is low
(`model_registry.py:476-492`, `_tier_down`/`_tier_up`, clamped at `light` and
`reasoning`). A role whose cheap local model keeps succeeding drifts *down* the
ladder and gets cheaper; one that keeps failing escalates *up*. The loop closes
at `feedback.py:374-384`: an action outcome whose `action_id` starts with
`model_route:` is forwarded into `record_model_outcome`, so the router is
trained by the same outcome stream that trains everything else.

**The rejected alternative is the static `pick_for_role` path, which is still
in the tree as the fallback.** The introducing commit states the choice in one
line: *"Default-on, degrades to static `pick_for_role`."* Static routing was
not rejected because it was wrong — it is correct on day one and remains the
behaviour whenever no registry is configured, only one model exists, or the
lookup raises. It was rejected as a *terminal* state: it can never get cheaper
than the tier a human guessed, and it cannot notice that the guess was wrong.
The commit message is explicit that this is a wiring decision, not a new
mechanism: the tiered ladder, the cost signal (`ModelCostRate`, zero = local)
and the confidence-gated selector already existed as *"three islands"*, and
`pick_for_task_adaptive` was *"built ... but with no live caller"*. The
alternative genuinely tried and abandoned was leaving those parts unconnected.

A second, narrower alternative was rejected inside the same design: making the
learned confidence **durable** state. It is deliberately process-local — a
routing cache, not a belief. The module docstring separates them: durable
cross-process learning *"rides on the canonical reward EMA via `graph_feedback
action_outcome target_id=model_route:<role>`"*, while `_ROUTE_REWARDS` is a
warm-start cache that a restart is allowed to discard. Persisting the EMA
directly would have made a transient endpoint outage into a permanently
recorded belief about a model's quality.

## Why the sibling markers point here rather than each earning a document

Three other markers sit on code that is this same decision observed from a
different seam, and none of them names an independent trade-off:

- `confidence-gated-routing-log` (`model_registry.py:474-570`) is the
  tier-shift primitive itself — the mechanism this decision *drives*. Its
  companion `GraphState.routing_confidence_log` field
  (`agent_utilities/graph/state.py:451-456`) is written by no production code
  path at all (only tests append to it), so the "log" half names an intent, not
  a shipped choice.
- `confidence-signal-forwarding`
  (`resource_optimizer.py:131-181`) is parameter-forwarding glue: it composes
  the budget-derived tier downgrade with the *same* `pick_for_task_adaptive`
  gate, guarded by `hasattr`. The composition is a call site, not a decision.
- `route-outcome-feedback` (`feedback.py:374-384`) is the write half of the
  very loop described above — the eight lines that make the EMA learn.

## Risk Assessment

- **Blast Radius**: `agent_utilities/core/model_router.py`,
  `agent_utilities/core/model_factory.py`,
  `agent_utilities/core/resource_optimizer.py`,
  `agent_utilities/models/model_registry.py`,
  `agent_utilities/knowledge_graph/adaptation/feedback.py`.
- **Backward Compatible**: Yes. `pick_adaptive` has an explicit never-raise
  contract (`model_router.py:84-87`) — it returns `None` on any failure and the
  factory falls back to its default model, so a deployment with no registry
  behaves exactly as it did before.
- **Known weak point**: the EMA is keyed by role/task-class only
  (`route_key`, `model_router.py:35-37`), so a route's confidence is shared
  across every model that role has ever been served by. A tier shift caused by
  one bad endpoint is therefore charged to the *route*, and the router cannot
  distinguish "this role is hard" from "the model we happened to pick for it
  was down". `_ALPHA = 0.3` bounds how long that misattribution persists but
  does not remove it.
