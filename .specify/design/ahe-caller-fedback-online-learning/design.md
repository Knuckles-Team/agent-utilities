# Design Document: Selection weights update from caller-observed outcomes, not a static prior

CONCEPT:AU-AHE.harness.callers-feed-back-per

> `agent_utilities/harness/memorydata/router_method.py` (primary — the EMA
> weight hook), `agent_utilities/harness/memorydata/bakeoff.py` (the bake-off
> that exercises it), `agent_utilities/harness/evolve_agent.py:917-929`
> (the same pattern generalised to agent-prompt hardening).

## Decision — an online-learning hook where the CALLER supplies the reward, not an offline/synthetic label

`GraphOSRouterMethod` (`router_method.py:4-11`) picks a retrieval config per
query from a family-tag prior table (`DEFAULT_FAMILY_PRIORS`), because "a
single retrieval surface is rarely best across MemoryData's heterogeneous
families." `record_outcome` (`router_method.py:113-123`) then updates a
per-config EMA weight from a caller-supplied `reward` float: `updated =
(1 - ema_alpha) * prior + ema_alpha * reward`. The docstring names the
decision directly: "callers feed back a per-query reward so the router can
bias future selection toward configs that performed well."

**The rejected alternative is a fixed prior table with no feedback loop** —
`DEFAULT_FAMILY_PRIORS` alone, never updated. That table is substring-matched
against the family tag (a preset like `membench-update` resolves without an
exact key) and is a reasonable starting point, but it is static domain
knowledge, not learned from what actually won. `run_bakeoff` recognises the
router as a meta-config (`bakeoff.py:9-15`) specifically so "route per family"
can be pit against every single fixed config in the same bake-off — the
router only earns its place if the caller-fed EMA weights actually beat a
static single-config choice, not by assumption.

### Pointer — the same caller-fed-outcome pattern reused for agent-prompt hardening

`evolve_agent.py:917-935`. `harden_agent_prompt` explicitly reuses this
concept id for a structurally similar but separate cycle: **attribute** (pool
an agent's `action_outcome` cases into a trainset — the outcomes ARE the
caller-observed feedback, same as the router's `reward`), **optimize** (run
prompt optimization against that trainset), **evaluate** (score baseline vs.
candidate on a held-out eval slice), **decide + apply** (gated promotion,
writing the winner only under `KG_AGENT_AUTO_APPLY`, otherwise held as
queryable). The rejected alternative is identical in spirit to the router's:
optimizing a prompt against a synthetic/offline metric instead of the agent's
own accumulated caller-observed `action_outcome` history — the same "let
real outcomes drive the next selection" discipline applied to prompts instead
of retrieval configs.

## Risk Assessment

- **Blast Radius**: `agent_utilities/harness/memorydata/router_method.py`,
  `agent_utilities/harness/memorydata/bakeoff.py`,
  `agent_utilities/harness/evolve_agent.py`.
- **Backward Compatible**: Yes — the router degrades to
  `DEFAULT_FAMILY_PRIORS`/`_DEFAULT_CONFIG` when no outcomes have been
  recorded yet; `harden_agent_prompt`'s auto-apply is gated behind an env flag.
- **Known weak point**: the EMA weight has no decay/reset mechanism for a
  config whose underlying quality changes (e.g. a retrieval backend gets
  reconfigured) — a stale high weight from before the change keeps biasing
  routing toward it until enough fresh outcomes accumulate to correct it.
