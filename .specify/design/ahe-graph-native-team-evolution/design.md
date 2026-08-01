# Design Document: Team-mutation proposals persist as a typed node + typed edge, never a single combined Cypher write

CONCEPT:AU-AHE.harness.graph-native-team-evolution

> `agent_utilities/graph/team_evolution.py`.

## Decision — `TeamEvolutionEngine` writes a mutation proposal as `add_node` + `link_nodes`, not one `MATCH...MERGE...SET...MERGE` statement

`TeamEvolutionEngine.evaluate_and_evolve` (`team_evolution.py:25-85`) queries
recent failed/errored episodes for a team, and when repeated failures are
found, proposes a mutation (e.g. adding a specialist agent) and persists it
back into the graph. The comment at the write site
(`team_evolution.py:60-71`) names the concrete constraint and the decision it
forces: "a single `MATCH ... MERGE ... SET ... MERGE` statement exceeds the
engine's native Cypher write subset (MERGE supports only a single bare node,
never an edge pattern; `epistemic-graph/crates/eg-query/src/cypher/parser.rs:1184`)."

**The rejected alternative is the obvious single-statement Cypher write** —
match the team, merge the mutation node, set its properties, and merge the
edge, all in one query. That's what a Neo4j-style backend would support
directly, but the native engine's Cypher subset doesn't. The chosen split —
a typed node upsert (`engine.add_node(mut_id, "MutationProposal", {...})`)
followed by `engine.link_nodes(team_id, mut_id, "PROPOSED_MUTATION")` — is
"portable multi-clause Cypher" for a non-native backend and a "typed
dispatch for a native authority" for the engine's own store, so the same
code path works against either. It also happens to be tolerant of a race the
single statement wasn't guaranteed to handle any better: the team having
been retired between the read above and this write reproduces the prior
`MATCH`'s silent no-op behavior rather than raising.

## Risk Assessment

- **Blast Radius**: `agent_utilities/graph/team_evolution.py`,
  `agent_utilities/knowledge_graph/core/engine.py` (`add_node`/`link_nodes`).
- **Backward Compatible**: Yes — internal persistence mechanics only; the
  public `evaluate_and_evolve(team_id)` contract is unaffected.
- **Known weak point**: splitting the write into two calls reopens a window
  (however small) between the node upsert and the edge link where a reader
  could observe a `MutationProposal` node with no `PROPOSED_MUTATION` edge
  yet — the two-step split trades one race (single-statement backend
  incompatibility) for a narrower one (a brief inconsistent intermediate
  state).
