# EH-497 / AUD-17: reasoning and analytics parity checkpoint

This is the API and caller audit for the AU reasoning slice of AUD-17, based on
AU `97dc11ad9` and the EG contract in `epistemic_graph/client.py`. A matching
name does not establish replacement parity. Only the community-detection read
boundary below has a verified native response shape in this checkpoint. The
remaining rows stay open; this document does not authorize whole-file deletion.

| AU API | Current caller or entry point | EG native and disposition |
| --- | --- | --- |
| `topological_partition.detect_communities(graph)` | `TopologicalAnalysisEngine.detect_communities`, `hierarchical_retrieval`, and direct evolutionary-memory tests | `CommunityDetection` returns `list[list[str]]` through `GraphComputeEngine.community_detection` (`graph_compute.py:5312`; EG client `client.py:6532`; `eg-compute/.../community.rs`). The native result is now adapted to `list[set[str]]`; singleton filtering preserves the AU API. Native errors propagate. The local `PyGraph` fallback remains for callers without a native method and is **not** retired. |
| `topological_partition.persist_stable_communities(engine)` | `TopologicalAnalysisEngine.persist_stable_communities` and tests | EG `MineCommunity` already offers typed `:Community` writeback (`mining:write`, WAL replay) through `EpistemicGraphClient.mining.community(writeback=True)`. AU still writes `CommunityNode`/`PART_OF_COMMUNITY` with its own IDs, coherence formula, and per-node/edge upserts. The native operation is the intended owner, but the current synchronous engine facade has no admitted `mine_community` method and no verified migration of the legacy node/edge query contract. Retain until callers and result readers move together. |
| `topological_analysis_engine.{hierarchical_retrieve,find_analogous_subgraphs,build_spectral_clusters,cluster_to_kg_nodes,analyze_blast_radius,run_full_topology_analysis}` | Dynamic service registry and retrieval tests | EG has graph compute, `BlastRadius`, and search capabilities, but these AU methods also construct DTOs or run source-code analysis. No output and authorization parity yet; retain. |
| `inference_engine.InferenceEngine.run_inference()` | `core/engine.py` initialization and clone, `engine_tasks.py`, `orchestration/engine_query.py` | EG `OwlReason`/materialization owns logical reasoning, but AU's four Cypher rules and local transitive/phase mutations need per-rule graph and provenance equivalence. The legacy engine-removal lane must repoint its callers before deletion. |
| `formal_reasoning_core.{StructuralCausalModel,CausalVerifier,SpuriousnessDetector}` | `enrichment/ops_causal_graph.py`, service registry, MCP analysis tool, tests | EG graph reasoning does not establish the SCM intervention, d-separation, causal verification, or spuriousness API. Retain pending per-method evidence. |
| `formal_reasoning_core.{BayesianBeliefPropagator,RandomWalkExplorer}` | Dynamic registry and probabilistic-reasoning tests | EG has graph algorithms, but this belief/evidence update and seeded walk output has no verified equivalent. Retain. |
| `formal_reasoning_core.{is_reflexive,is_symmetric,is_transitive,equivalence_classes,resolve_entities}` | Direct tests; a separate `assimilation/entity_resolution.py` owns production resolution calls | No verified typed EG equivalence-class or exact relation-check contract for this `PyDiGraph` API. Do not conflate the separate entity resolver with this function. Retain pending parity or an explicit dead-API disposition. |
| `formal_reasoning_core.{FormalStateMachine,MarkovTransitionModel}` | Dynamic registry and state-machine/Markov tests | These are control and forecast APIs, not shown to be EG graph-reasoning duplicates. Retain. |
| `formal_reasoning_core.chromatic_schedule` | Direct graph-theory tests | EG `GraphColoring` colors persisted rows. The isolated EG branch adds `GraphColorEphemeral` for a bounded inline conflict graph, and AU now accepts an injected native callable that validates the node/color map and every conflict edge. The standalone local path, oversize graphs, and duplicate string IDs retain existing behavior. EG generated contract and Rust gate are pending; no full API deletion is authorized. |
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

The following source contracts narrow the remaining migration work. A similarly
named EG operation is a candidate, not deletion evidence:

| AU boundary | Observed caller and native contract gap |
| --- | --- |
| `graph_primitives.PyGraph/PyDiGraph` | `formal_reasoning_core`, `analogy_engine`, and the inline community fixture still use local graph mutation and algorithms. EG `CommunityDetectEphemeral` accepts an inline call graph but does not replace this entire Python graph API. |
| `spectral_navigator.SpectralClusterNavigator.cluster` | `TopologicalAnalysisEngine`, semantic retrieval, service registry, and direct tests consume spectral eigengap clustering, indexed members, coherence, and centroids. The isolated EG EH-497 worktree now implements a **candidate** `MineCluster(algorithm="spectral")` contract with a bounded read-only kernel, explicit feature rows, optional coherence, and a runtime `mining:read` policy for spectral `writeback=false` (`docs/architecture/eh497-spectral-minecluster.md` there). Both production facades inject the EG client method for 2–64-row matrices and the adapter preserves AU labels, IDs, minimum-size filtering, and result order. Standalone callers and >64-row matrices retain the existing numeric implementation. Native errors do not fall back to a different algorithm. EG generated contract, executed Rust/signed-dispatch gates, and cross-runtime numerical parity remain pending; no full navigator deletion is authorized. The retired `SpectralCluster` wire method remains retired. |
| `synergy_engine.SynergyEngine` | Top-level AU public exports and tests consume concept bridges, pillar coupling, suggestions, and Markdown reporting. No EG method with these output contracts was found; public retirement requires an explicit decision and consumer inventory. |
| `analogy_engine.TopologicalAnalogyEngine.find_analogous_subgraphs` | `TopologicalAnalysisEngine`, threat defense, and direct tests consume subgraph isomorphism plus embedding thresholds and `AnalogyMatchNode`. EG's graph algorithms do not establish this result parity. |
| `hypergraph.PositionalInteractionEncoder.encode_interaction` | Verification, hybrid retrieval, and heavy-thinking memory consume deterministic seeded EncPI vectors. `HypergraphEncodeInteraction` is explicitly retired by `eg-capabilities/tests/contract_registry.rs`; no callable EG client method or runtime handler exists under that name. A replacement must use a newly reviewed contract and prove seeded vector parity. |
| `blast_radius.BlastRadiusAnalyzer.analyze` | Service registry and direct tests consume a filesystem symbol-usage report with definition-line exclusion and impact score. EG `GetBlastRadius` traverses persisted graph node IDs; these are different inputs and outputs. Source analysis needs an SDK/code-index destination decision. |
| `graph_compute.GraphComputeEngine.get_blast_radius` | The existing EG `GetBlastRadius` returns ordered node IDs, not distances. AU had reported `depth=min(result_position, max_depth)`, which gives siblings different false depths. The facade now gets EG's induced subgraph in one additional read and computes directed shortest-path depths, rejecting a changed/incomplete subgraph rather than inventing depths. This fixes the live graph API but does not replace the filesystem `BlastRadiusAnalyzer` above. |
| `nl_query.nl_to_query` | MCP query tool consumes generated, schema-grounded read-only Cypher/SQL/SPARQL plus citations. EG owns query execution; no observed EG operation replaces the model prompt, generated query review, and citation DTO together. |
| `hydration.HydrationManager` | Scheduler, gateway, source-sync, and tests consume local hydration scheduling and state. EG storage capability alone does not prove these per-method orchestration contracts. |
| `maintainer.GraphMaintainer` | Engine tasks, governance agent, and multiple integration tests call maintenance methods. EG may own the resulting graph mutations, but each operation needs an admitted EG equivalent and caller migration. |
| `id_management.OntologicalIdentifierManager/Registry` | Document ingestion and deletion tests consume `doc_*` ID parsing and an in-memory sync registry. No generated EG method for this exact format or registry contract was found; source ingestion ownership must be settled with AUD-19. |
| `argumentation.aif` | MCP argument tools consume lossless AIF import/export and Dung projection. EG `ResolveConflict` owns argument solving but does not establish AIF interchange parity. |
| `actions.*` | Remediation, action policy, deploy watch, escalation, and tests consume notification dispatch and effect execution. EG graph reasoning is not a replacement for these external effects; SDK write-back and AU action policy need a split. |

## Verification for the completed slice

- EG wire contract: `CommunityDetection` is `Raw<Vec<Vec<String>>>` in
  `crates/eg-types/src/result_contract/compute.rs`; the generated Python client
  declares `list[list[str]]`.
- AU adapter: `GraphComputeEngine.community_detection()` delegates directly to
  the EG graph client. `topological_partition` now consumes the grouped shape
  and rejects the retired `(node, label)` and mixed-ID shapes instead of
  silently writing malformed communities.
- Focused tests cover the grouped response, singleton filtering, malformed
  responses, and a native error that must not fall back to AU's different local
  algorithm.

The remaining local `PyGraph` algorithm and community persistence are separate
unreplaced APIs. AUD-17 remains in progress after this slice.
