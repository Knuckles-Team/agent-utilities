# Design Document: A workflow is compiled from free text by heuristics, not an LLM call, with semantic matching as an optional upgrade

CONCEPT:AU-ORCH.execution.nl-compilation-pipeline

> `agent_utilities/knowledge_graph/workflow_compiler.py` (`WorkflowCompiler`) —
> the only source file carrying this concept, 12 marker sites across the module.

## The real decision

`WorkflowCompiler.compile()` turns a free-text workflow description into an
executable `GraphPlan` through a fixed five-stage pipeline
(`workflow_compiler.py:9-33`): `parse_intent → match_agents → build_dag →
GraphPlan → store.save_workflow()`. Two sub-decisions inside that pipeline are
the actual design choices worth recording.

**1. Step extraction (`_parse_steps`, `workflow_compiler.py:308-367`) is a
cascading regex/keyword heuristic, not an LLM call.** Three strategies are
tried in order and the first that produces ≥2 steps wins:

1. Explicit numbering (`"1. ... 2. ..."`, `re.split(r"\d+\.\s+", ...)`)
2. Sequential keywords (`then`, `next`, `after that`, `finally`, `followed by`)
3. Comma + "and" splitting, filtering fragments under 10 characters

This is a deliberately cheap, deterministic, in-process parse — no model call,
no network round trip, no latency or availability dependency — for the most
structurally common shapes of a workflow description. It is explicitly
narrower than an LLM-based parser would be (a novel phrasing that hits none of
the three heuristics returns an empty step list, `workflow_compiler.py:367`),
trading generality for determinism and zero marginal cost per compilation.

**2. Agent matching degrades gracefully from semantic to structural, rather
than failing when embeddings are unavailable.** `_embed()`
(`workflow_compiler.py:376-397`) builds the embedding function once per
compiler and bounds every embed call via the shared `bounded_embed` helper
("the same helper the Loop engine uses"). When the embedding endpoint is
unavailable, the method does not raise — it returns `None` and logs *"workflow
compilation falls back to structural agent matching only"*
(`workflow_compiler.py:391-393`). `_match_agent`
(`workflow_compiler.py:400-410`) uses the KG's semantic search and direct
queries together, so a down embedding endpoint degrades match quality rather
than making compilation impossible.

## The rejected alternative

For step extraction, the rejected alternative is the more capable but
heavier design: routing free-text parsing through an LLM call. That would
handle arbitrary phrasing the three heuristics miss, but at the cost of
latency, an added external dependency in the compile path, and non-determinism
in what DAG a given description produces on repeated compiles. The chosen
heuristic ladder is deterministic and free, at the cost of only recognizing
workflows described in one of three common shapes.

For agent matching, the rejected alternative is treating the embedding
endpoint as a hard dependency — fail the whole compilation when it is down.
The chosen design instead makes semantic matching a *quality* upgrade over a
structural baseline that always works, so an embedding-endpoint outage
degrades match precision rather than taking workflow compilation offline
entirely.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/workflow_compiler.py`
  only — a single-file decision.
- **Backward Compatible**: Yes.
- **Known weak point**: the three step-extraction strategies are tried in a
  fixed order and the first one producing ≥2 fragments wins outright — a
  description that happens to contain a comma-separated list *and* explicit
  numbering will always be split on the numbering strategy (tried first),
  even if the comma-based split would have produced a more accurate
  decomposition. There is no scoring or fallback between strategies once one
  succeeds.
