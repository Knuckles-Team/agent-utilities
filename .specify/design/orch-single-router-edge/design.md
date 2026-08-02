# Design Document: The router node keeps exactly ONE outgoing edge; direct completion is handled outside the graph, never by a second edge to the end node

CONCEPT:AU-ORCH.routing.single-router-edge

> Realised by `agent_utilities/graph/builder.py:780-818` — specifically the
> edge-registration block at `:783-791`, which registers `router → dispatcher`
> and nothing else. Direct completion is short-circuited before the graph is
> built, in `agent_runner._run_direct_completion`. Introduced by commit
> `35c40ca4` ("fix(graph): direct-completion outside the graph — removes the
> router broadcast-fork").

## Decision — a structural invariant, adopted because violating it silently killed every full-graph task

pydantic-graph derives control flow from the edges declared on a node. A node
with two outgoing edges is not a branch — it is a **broadcast fork**: the
node's output is delivered to *both* successors.

The rejected alternative here is not hypothetical and was not someone else's
design. It is what this codebase shipped, by the same author, immediately
before this commit. To let a "direct completion" turn (one needing no tools and
no specialists) terminate cheaply inside the graph, a second edge
`router → end_node` was added. The commit's root-cause note describes what that
did:

> *"the earlier 'router -> end_node' edge I added ... gave the router TWO
> outgoing edges, which pydantic-graph turns into a BROADCAST FORK — every
> router output went to BOTH `__end__` (terminating) AND dispatcher ... silently
> KILLED every full-graph/tool task."*

Every routed turn was reaching `__end__` in parallel with the dispatcher, so
full-graph and tool-using tasks terminated empty. The failure produced no
exception and no warning — the graph did exactly what a fork is defined to do.

The fix does not add a guard, a flag, or a conditional edge. It restores the
single `router → dispatcher` edge and moves direct completion **entirely
outside the graph**: `agent_runner._run_direct_completion` handles that case
before the graph is constructed at all. The reasoning is structural — if the
cheap path never enters the graph, the graph never needs a second exit, and the
invariant "the router has one outgoing edge" can be stated absolutely rather
than conditionally.

## This is a must-not-regress invariant, not just history

The comment at `builder.py:786` exists because nothing else enforces this.
pydantic-graph accepts the second edge without complaint; the resulting failure
is silent, affects only the *non*-trivial paths (so trivial smoke tests still
pass), and presents as "the agent returned nothing" rather than as an error. A
future topology refactor that re-adds an edge from the router — for any
reason, including a well-intentioned one like the original — reproduces the
exact prior outage with no signal.

## Risk Assessment

- **Blast Radius**: `agent_utilities/graph/builder.py` (graph topology),
  `agent_utilities/orchestration/agent_runner.py` (the out-of-graph
  direct-completion path).
- **Backward Compatible**: Yes — direct completion still happens and is still
  cheap; it simply happens before graph construction rather than as a graph
  exit.
- **Known weak point**: the invariant is enforced by a comment and by reviewer
  attention, not by a test. A structural assertion at build time — "the router
  node resolves to exactly one successor" — would convert the silent failure
  into a startup error, and does not currently exist.
