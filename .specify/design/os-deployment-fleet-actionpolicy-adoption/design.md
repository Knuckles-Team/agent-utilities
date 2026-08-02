# Design Document: Fleet-mutating deployment actions (scale, remediate, reconcile, merge-promote, engine-task admin) all consult the ONE shared, tiered `ActionPolicy` gate, instead of each subsystem keeping its own binary autonomy flag

CONCEPT:AU-OS.deployment.fleet-lifecycle-control

> `agent_utilities/orchestration/action_policy.py:1-40` (module docstring — the
> single autonomy decision point every site below adopts);
> `agent_utilities/knowledge_graph/research/auto_merge.py:139`,
> `agent_utilities/knowledge_graph/research/change_publisher.py:24-30`,
> `agent_utilities/orchestration/fleet_actuation.py:335`,
> `agent_utilities/orchestration/fleet_autoscaler.py:39`,
> `agent_utilities/orchestration/fleet_reconciler.py:28`,
> `agent_utilities/knowledge_graph/core/engine_tasks.py:1545,1575`.

## Decision — fleet actuation, autoscaling, reconciliation, remediation playbooks, engine-task administration, and research auto-merge's promotion step all consult the SAME `ActionPolicy.decide(ActionRequest(kind=..., target=..., source=...))` decision point — YAML/KG-rule match → per-action tier (`auto`/`auto_notify`/`approval_required`/`forbidden`) → maintenance window → rate limit → blast-radius cap — instead of each subsystem keeping its own env-flag switch

`action_policy.py` states the prior state of the world directly: "until now every
autonomy gate in the platform was a binary env flag (`KG_GOLDEN_AUTO_MERGE`,
`FLEET_RECONCILER` …) — an action was either fully autonomous or fully off"
(`action_policy.py:8-10`). This module replaces that cliff with per-action tiers,
and the deployment-domain marker (`fleet-lifecycle-control`) names the SET OF
SITES that adopted it: `GovernedAutoMerger` consults the reserved `merge_promotion`
kind before any proposal→active flip (`auto_merge.py:139`); `change_publisher`'s
`governed_publish`/`publish_proposal` consult the same kind before a
`ChangePublisher` runs, queuing an approval by default under the shipped policy
(`change_publisher.py:24-30`); `fleet_actuation`, `fleet_autoscaler`, and
`fleet_reconciler` consult it before scale/restart/reconcile actions; `engine_tasks`
consults it for admin operations. Every decision is audit-logged as an
`ActionDecision` KG node, and rate/blast-radius accounting reads those same nodes
back — durable and shared across processes, not per-subsystem in-memory state.

## Rejected alternative — leave each subsystem's autonomy behind its own independent binary flag

The rejected alternative is the platform's own prior, shipped behaviour, named
directly: one flag per subsystem (`KG_GOLDEN_AUTO_MERGE` for auto-merge,
`FLEET_RECONCILER` for reconciliation, and by implication a similar ad hoc flag for
each of autoscaling/remediation/publication) — each either fully autonomous or
fully off, with no shared notion of "ask for approval," no shared rate limiting, no
shared blast-radius cap, and no shared audit trail. That shape was rejected because
it gives an operator no single place to reason about "what can this platform do to
itself without asking," no way to apply a maintenance-window or rate-limit
policy consistently across subsystems, and no way to add governance depth (a
`forbidden` rule, a KG-sourced override) without touching every subsystem's own
flag-check code individually — the same "N places to remember" pattern rejected
elsewhere in this codebase for authority renewal and engine resolution. Fanning
every fleet-mutating action through one decision point makes a new rule (or a new
autonomous subsystem) inherit rate-limiting, blast-radius capping, and audit
logging for free, rather than needing its own bespoke gate.

## Risk Assessment

- **Blast Radius**: `agent_utilities/orchestration/action_policy.py` and every
  site listed above; `deploy/action-policy.default.yml` (the shipped conservative
  ruleset).
- **Backward Compatible**: Yes — the shipped default policy is conservative
  (`approval_required` for `merge_promotion`), preserving a human-in-the-loop
  posture equivalent to the old flags defaulting off.
- **Known weak point**: adoption is per-call-site, not structurally enforced — a
  new fleet-mutating subsystem added later that does NOT call
  `get_action_policy().decide(...)` before acting silently reverts to the
  pre-`ActionPolicy` "no gate at all" state, the same convergence-point trust
  risk this codebase's own delegation-authority design doc flags for its own
  chokepoint assumption.
