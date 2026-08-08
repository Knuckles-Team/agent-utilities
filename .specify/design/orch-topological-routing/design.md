# Design Document: Score specialist candidates from KG topological signals instead of keyword TF-IDF, falling back to rules on cold start

CONCEPT:AU-ORCH.routing.topological-routing

> Realised by `agent_utilities/graph/adaptive_agent_router.py:551-639`
> (`TopologicalRoutingPolicy`, its `route()` and `_score_candidate`).

## Decision — replace lexical similarity with graph structure as the ranking signal for specialist selection

`adaptive_agent_router.py` already contained a `TraceLearnedPolicy` that ranked
specialists by TF-IDF similarity plus a softmax over historical traces. This
decision supersedes it with a policy that ranks from Knowledge-Graph structure
instead. The docstring states the substitution in one line:

> *"Routes using KG-derived topological signals instead of keyword TF-IDF."*

`_score_candidate` combines four structural signals: PageRank centrality in the
capability graph, historical success rate read from `SubagentPatternDecision`
and `OutcomeEvaluation` nodes, domain-cluster membership, and tool affinity.

**The rejected alternative — TF-IDF over trace text — is not a strawman; it is
the sibling policy still present in the same file**, and it lost for a specific
reason. TF-IDF measures whether a specialist's description *uses the same words
as* the task. That is a proxy for relevance, and it is a bad one in this
system: specialists are described in whatever vocabulary their author chose, so
a specialist that solves exactly this task but describes itself differently
ranks low, while one that shares boilerplate phrasing ranks high. The
topological signals measure something closer to what the router actually wants
to know — has this specialist been central to work like this, and did it
succeed — which is evidence about behaviour rather than about wording.

The cold-start fallback is the counterweight and is deliberate: every one of
those signals requires accumulated graph history, so `TopologicalRoutingPolicy`
falls back to `RuleBasedPolicy` when no KG engine is available. The policy
therefore degrades to something deterministic rather than to a low-confidence
structural score computed over an empty graph, which would be worse than
keywords.

## Risk Assessment

- **Blast Radius**: `agent_utilities/graph/adaptive_agent_router.py`.
- **Backward Compatible**: Yes via the fallback — a deployment with no KG
  engine routes by rules exactly as before.
- **Known weak point**: PageRank centrality and historical success rate are
  both self-reinforcing. A specialist that gets routed to accumulates
  centrality and outcome history, which raises its score, which gets it routed
  to more; a capable specialist that starts cold has no history to earn its
  first selection. Nothing in `_score_candidate` injects exploration to
  counteract this, so the ranking can lock in an early ordering.
