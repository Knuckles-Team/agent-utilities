# EH-497 / AUD-17: reasoning and analytics parity checkpoint

This is the API and caller audit for the AU reasoning slice of AUD-17, based on
AU `97dc11ad9` and the EG contract in `epistemic_graph/client.py`. A matching
name does not establish replacement parity. Only the community-detection read
boundary below has a verified native response shape in this checkpoint. The
remaining rows stay open; this document does not authorize whole-file deletion.

| AU API | Current caller or entry point | EG native and disposition |
| --- | --- | --- |
| `topological_partition.detect_communities(graph)` | `TopologicalAnalysisEngine.detect_communities`, `hierarchical_retrieval`, and direct evolutionary-memory tests | `CommunityDetection` returns `list[list[str]]` through `GraphComputeEngine.community_detection` (`graph_compute.py:5312`; EG client `client.py:6532`; `eg-compute/.../community.rs`). The native result is now adapted to `list[set[str]]`; singleton filtering preserves the AU API. Native errors propagate. The local `PyGraph` fallback remains for callers without a native method and is **not** retired. |
| `topological_partition.persist_stable_communities(engine)` | `TopologicalAnalysisEngine.persist_stable_communities` and tests | EG `CommunityDetection` covers grouping, not AU's generated `CommunityNode`/`PART_OF_COMMUNITY` mutation and coherence formula. Requires a typed EG transaction or explicit retirement of this persistence contract. |
| `topological_analysis_engine.{hierarchical_retrieve,find_analogous_subgraphs,build_spectral_clusters,cluster_to_kg_nodes,analyze_blast_radius,run_full_topology_analysis}` | Dynamic service registry and retrieval tests | EG has graph compute, `BlastRadius`, and search capabilities, but these AU methods also construct DTOs or run source-code analysis. No output and authorization parity yet; retain. |
| `inference_engine.InferenceEngine.run_inference()` | `core/engine.py` initialization and clone, `engine_tasks.py`, `orchestration/engine_query.py` | EG `OwlReason`/materialization owns logical reasoning, but AU's four Cypher rules and local transitive/phase mutations need per-rule graph and provenance equivalence. The legacy engine-removal lane must repoint its callers before deletion. |
| `formal_reasoning_core.{StructuralCausalModel,CausalVerifier,SpuriousnessDetector}` | `enrichment/ops_causal_graph.py`, service registry, MCP analysis tool, tests | EG graph reasoning does not establish the SCM intervention, d-separation, causal verification, or spuriousness API. Retain pending per-method evidence. |
| `formal_reasoning_core.{BayesianBeliefPropagator,RandomWalkExplorer}` | Dynamic registry and probabilistic-reasoning tests | EG has graph algorithms, but this belief/evidence update and seeded walk output has no verified equivalent. Retain. |
| `formal_reasoning_core.{is_reflexive,is_symmetric,is_transitive,equivalence_classes,resolve_entities}` | Direct tests; a separate `assimilation/entity_resolution.py` owns production resolution calls | No verified typed EG equivalence-class or exact relation-check contract for this `PyDiGraph` API. Do not conflate the separate entity resolver with this function. Retain pending parity or an explicit dead-API disposition. |
| `formal_reasoning_core.{FormalStateMachine,MarkovTransitionModel}` | Dynamic registry and state-machine/Markov tests | These are control and forecast APIs, not shown to be EG graph-reasoning duplicates. Retain. |
| `formal_reasoning_core.chromatic_schedule` | Direct graph-theory tests | EG `GraphColoring` returns node/color pairs; the AU API accepts an inline `PyGraph`, returns a task/color map. Needs `GraphColoring` or `CommunityDetectEphemeral` style inline contract and schedule tests before removal. |
| `semantic_subsumption.SemanticSubsumptionEngine.align_node_to_ontology` | Direct tests; no production import found | Vector-prototype cosine alignment with threshold and lineage is distinct from EG's sound OWL class subsumption. No parity. Retain until the feature is intentionally retired or a matching EG classifier exists. |
| `reasoner.{ReasoningTask,ReasoningResult,ReasonerRouter}` | `facade.py`, harness/router tests and dynamic capability routing | The router learns an AU policy and selects program synthesis, world-model, deductive, or generative paradigms; EG `Reason` is a graph-logic operation. This is AU orchestration, pending ownership split rather than blanket deletion. |
| `world_model.{LatentDynamicsModel,WorldModel}` | Harness world-model task/benchmark, MCP analysis, reasoner | Prediction, rollout, and policy evaluation have no demonstrated EG API equivalence. Retain. |
| `fingerprint.{compute_fingerprint,detect_stale_files,FingerprintManager}` | Ingestion engine and direct tests | AST file fingerprint/change detection is source-ingestion work; no matching EG native established. Route under AUD-19 source ownership review. |
| `ownership_claim.{preview_claim,apply_claim,record_claim_audit}` | MCP governance tool | Includes actor admission, dual-origin selection, and audit behavior. EG ownership and authorization parity must be shown with an adversarial write test before deletion. |

Other AUD-17 modules (`synergy_engine`, `analogy_engine`, `hypergraph`,
`blast_radius`, `nl_query`, `hydration`, `maintainer`, and the
`maintenance/id_management/argumentation/actions` packages) have live direct
or dynamic callers. They need the same method-by-method evidence. In particular,
`hypergraph.PositionalInteractionEncoder` feeds verification, retrieval, and
memory; `nl_query.nl_to_query` backs an MCP query tool; `hydration.HydrationManager`
backs scheduling and gateway paths; `maintainer.GraphMaintainer` backs governance
and engine tasks. They remain open for this lane.

## Verification for the completed slice

- EG wire contract: `CommunityDetection` is `Raw<Vec<Vec<String>>>` in
  `crates/eg-types/src/result_contract/compute.rs`; the generated Python client
  declares `list[list[str]]`.
- AU adapter: `GraphComputeEngine.community_detection()` delegates directly to
  the EG graph client. `topological_partition` now consumes the grouped shape.
- Focused tests cover the grouped response, singleton filtering, and a native
  error that must not fall back to AU's different local algorithm.

The remaining local `PyGraph` algorithm and community persistence are separate
unreplaced APIs. AUD-17 remains in progress after this slice.
