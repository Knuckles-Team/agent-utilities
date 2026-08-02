# Design Document: Decompose-depth, worker choice and inference budget are decided JOINTLY as one routing decision; the fallback chain is derived from the KG, not a static list

CONCEPT:AU-ORCH.routing.uno-orchestra-derived

> Realised by `agent_utilities/graph/adaptive_agent_router.py:24-40` (module
> docstring), `:51-125` (`RoutingPrimitive`, `RoutingCandidate`,
> `RoutingDecision`), `:195-441` (`RoutingPolicy`, `RuleBasedPolicy`,
> `TraceLearnedPolicy`), `:444-503` (`CostAwareRouter`) and `:506-548`
> (`OntologicalFallbackChain`). Derived from Uno-Orchestra
> (arXiv:2605.05007v1).

## Decision — adopt joint optimisation as the framing for routing, and express a fallback chain as a KG query

Routing in this system involves three coupled choices: how deeply to decompose
a task, which worker to hand each piece to, and how much inference budget to
spend. Deciding them separately — pick a decomposition, then pick workers for
it, then set a budget — is the natural order and produces locally sensible,
jointly poor results: a deep decomposition commits to many worker calls before
anyone has priced them, and a budget set last can only truncate a plan already
made.

The Uno-Orchestra framing this module adopts treats the three as one decision.
`RoutingDecision` (`:79-125`) carries all three, `CostAwareRouter`
(`:444-503`) filters candidates on a Pareto cost/accuracy frontier rather than
optimising either alone, and the `RoutingPolicy` hierarchy (`:195-441`) lets
the *strategy* for making that joint choice vary — `RuleBasedPolicy` for cold
start, `TraceLearnedPolicy` (TF-IDF + softmax over historical traces) once
there is history.

**The concrete rejected alternative, stated in the code, is in the fallback
chain.** `OntologicalFallbackChain`'s docstring says it directly:

> *"Instead of a hardcoded CSV list of fallback models, this class queries the
> Knowledge Graph (KG) for nearest `ModelCapabilityNode` neighbors..."*

A static CSV of fallback models is the standard implementation and it fails in
a specific way here: it encodes, at authoring time, a similarity judgement
about which model can stand in for which. That judgement goes stale the moment
a model is added, retired or reconfigured, and nothing detects the staleness —
the chain keeps falling back to a model that may no longer exist or may no
longer be comparable. Deriving neighbours from `ModelCapabilityNode` adjacency
makes the fallback chain a *query over current capability facts* instead of a
frozen opinion, so it tracks the fleet automatically.

## Naming note

The concept id is an artifact of the OKF-CIS rename — the module was originally
tagged as an *enhancement of* the specialist-routing concept (pre-migration
`ORCH-1.2`), and the slug was derived from the paper attribution rather than
chosen. The decision it names is nonetheless the module's own.

## Risk Assessment

- **Blast Radius**: `agent_utilities/graph/adaptive_agent_router.py`.
- **Backward Compatible**: Yes — `RuleBasedPolicy` is the cold-start behaviour.
- **Known weak point**: the KG-derived fallback chain is only as good as
  `ModelCapabilityNode` coverage. A model whose capability node is missing or
  thin has no neighbours, so it silently has *no* fallback chain — which is a
  worse failure than a stale CSV entry, because a static list at least always
  returns something. Nothing currently alerts on a model with an empty derived
  chain.
