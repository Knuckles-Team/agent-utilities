# EH-497 formal reasoning and NL query disposition

The `formal_reasoning_core` module accepts detached `graph_primitives.PyGraph`
and `PyDiGraph` values. Those callers do not necessarily have an EG graph name,
stored node identity, or a verified server session. Per-API parity is required
before removing an operation; a similarly named graph query is insufficient.

| AU operation | EG contract and disposition |
| --- | --- |
| `chromatic_schedule`, `chromatic_number_upper_bound` | The explicit `native_coloring` adapter uses `GraphColorEphemeral` for bounded, unique-ID conflict graphs. It validates every returned row and edge constraint. Keep local PyGraph coloring for callers that have no native adapter or exceed the bound. |
| Personalized PageRank | AU's formal copy was deleted; callers use `GraphComputeEngine.personalized_pagerank`, backed by EG `PersonalizedPageRank`. |
| `count_paths_of_length` | Matrix exponentiation is already delegated to `epistemic_graph.numeric` through `xp.linalg.matrix_power`; AU constructs the detached adjacency matrix and selects the `(source, target)` cell. Removing the adapter would remove the PyDiGraph entry point. |
| `reachability_within_hops` | EG `GetBlastRadius` returns affected nodes, but does not carry each node's shortest distance. No exact replacement is available. |
| Critical path, vertex/edge connectivity, minimum vertex cut, Euler tour | No current served EG result contract preserves these detached graph inputs and output shapes. Native contracts and differential fixtures are needed before deletion. |
| `generate_math_foundation_seed` | Curated source data, rather than a graph compute duplicate. Retain with the AU content/ingestion owner. |

`nl_query.py` translates user intent with an LLM, grounds the query in live
schema, checks the candidate, dispatches to EG Cypher/SQL/SPARQL, and presents
citations. EG owns execution of those query dialects; AU owns this application
translation. AUD-17 should retain that module as an adapter, not count its
deletion as a native parity obligation.
