# Design Document: A coarse tier hint maps onto the EXISTING per-call `reasoning_effort` knob, and an unset model routes to the operator's configured default — not to a new registry, not to a literal

CONCEPT:AU-ORCH.routing.functional-role-resolution

> Realised by `agent_utilities/workflows/runner.py:183-194` and `:270-273`,
> `agent_utilities/models/sdd.py:131-141`, and
> `agent_utilities/core/model_factory.py:476-500`. Introduced by commits
> `699db89c` ("feat(g15): wire localized repair + model_tier hint into
> WorkflowRunner") and `c137f05a` ("model_factory: per-model TLS/headers +
> defined default routing").

## Decision 1 — express a task's coarse tier hint through `reasoning_effort`, explicitly declining to build a tier→model-id table

`Task.model_tier` carries a coarse hint ("small"/"cheap" vs
"standard"/"large"). The obvious implementation is a lookup table from tier to
a concrete model id. The code says, in as many words, that this was considered
and refused (`runner.py:183-194`):

> *"Deliberately NOT a new model registry ... maps onto the existing
> `run_agent(reasoning_effort=...)` per-call knob rather than a new tier->model-id
> table."*

The reason is that a second tier→model mapping would be a competing authority.
`ModelRegistry` already owns the question "which model serves this tier", with
tags, availability and the adaptive confidence loop attached
(`CONCEPT:AU-ORCH.routing.conductor-per-step-model`,
`CONCEPT:AU-ORCH.routing.adaptive-role-routing`). A workflow-local table would
answer the same question with less information and drift out of agreement,
producing two defensible-looking answers for one model choice. Routing the hint
onto `reasoning_effort` — a knob that already exists, per-call, and does not
name models at all — expresses the intent without claiming the authority.

## Decision 2 — an unset default model resolves to the operator's *configured* default, never to a hardcoded id

`model_factory.py:490-500` routes an unset model to *"the operator's DEFINED
default chat model ... instead of the hardcoded `'qwen/qwen3.6-27b'`
literal."* This is a straightforward correction with a verifiable before-state:
commit `c137f05a`'s diff shows the prior line was literally
`_model_id = model_id or "qwen/qwen3.6-27b"`.

**The rejected alternative here is a hardcoded fallback literal**, and its
failure mode is that it is invisible until it is wrong. A deployment that has
never provisioned that specific model gets a confusing downstream error rather
than a clear configuration error, and the literal silently overrides the
operator's stated default in exactly the case where the default matters most.

## Scope note — the role-resolution path is NOT re-decided here

`model_factory.py:476-489` (`_resolve_role_model`) resolves a functional role
through the registry. That is the decision documented in
`.specify/design/orch-1.27-role-specialized-routing/design.md`
(`CONCEPT:AU-ORCH.routing.conductor-per-step-model`), traced to the same origin
commit `2a7d6eab`. This document does not restate it; the two decisions above
are what this concept adds.

## Risk Assessment

- **Blast Radius**: `agent_utilities/workflows/runner.py`,
  `agent_utilities/models/sdd.py`, `agent_utilities/core/model_factory.py`.
- **Backward Compatible**: Decision 2 is a behaviour change by design — a
  deployment that was implicitly relying on the hardcoded literal now gets its
  own configured default instead.
- **Known weak point**: `reasoning_effort` is a coarser instrument than model
  selection. Two tiers that genuinely want different *models* can only express
  that as different effort against the same model, so the hint is lossy — the
  decision accepts a weaker mapping to avoid a competing authority, and that
  trade is not revisited anywhere.
