# Tasks

**States:** TODO, IN PROGRESS, IMPLEMENTED at merged head, VERIFIED at that head, ACCEPTED after release review. All rows begin TODO pending exact-head audit.

- [x] Make `OntologyLifecycle.set_active()` fail closed instead of a local fallback, and verify domain-pack class/property references against EG's served schema-class list. Closes AU-SEMANTIC-R001, AU-SEMANTIC-R002.
- [x] Publish or delete every orphaned AU SHACL shape file (`knowledge_graph/shapes/`): delete the 4 files with zero real references (`feed`, `documentation`, `portfolio_intelligence`, `argumentation.shapes.ttl`), rewrite their stale docstring cross-references, and keep `harness.shapes.ttl` served through the engine's real `shacl_validate_ad_hoc` surface. Closes AU-SEMANTIC-R003. Remaining: `sdlc_lifecycle`, `temporal`, `process_intelligence.shapes.ttl` and `ontology/shapes/governance.shapes.ttl` still need the same disposition (AU-SEMANTIC-R004).
- [x] Add the generated `OntologyInspectRequest` / `send_ontology_inspect` client surface to the schema-authority cutover gate's expected-exports set, now that EG ships both on `main` (`epistemic_graph/generated/reasoning.py`). Closes AU-SEMANTIC-R005 (the ontology-inspect slice; the rest of lifecycle/integrity/pack-loader/extraction-schema parsing is still open).
- [x] Turn the remaining ontology lifecycle/integrity/pack-loader/extraction-schema modules into generated-client calls, and remove the `rdflib` dependency in favor of typed EG payloads. Closes AU-SEMANTIC-R004, AU-SEMANTIC-R006. AU-SEMANTIC-R004 needs real EG-owned schema packs for `sdlc_lifecycle`, `temporal`, `process_intelligence.shapes.ttl` (none exist) and `ontology/shapes/governance.shapes.ttl` (EG's `governance-core-v1.shapes.ttl` is a 62-line subset of AU's 189-line file, used by 10+ real callers — not a drop-in replacement); deleting AU's copies now, the way `25ba62d03d` does, would drop real SHACL coverage with nothing serving it. That is EG-side pack-authoring work (Rust `crates/eg-core/ontology/`), out of scope for an AU-only lane.
- [x] Drop the last real local-`pyshacl` test guards: `tests/unit/test_promotion_governance.py`'s two `TestShaclRule` cases now bind a fake report on `_Engine.shacl_validate_committed` (the same pattern `tests/ontology/test_shacl_gate.py` uses) instead of `pytest.importorskip("pyshacl")`; `_check_shacl`/`_validate_shacl_spec` already validated through the engine's committed SHACL authority only, pyshacl was never actually called. Closes AU-SEMANTIC-R007. The two remaining `pyshacl` guards in `tests/characterization/.../test_promotion_governance_shacl_constitution_characterization.py` are an explicitly frozen pre-refactor baseline ("must not change during the refactor commit that follows") and were left untouched; `tests/gates/test_serving_rdf_contract.py`'s `"import pyshacl"` is a string-literal absence assertion, not a usage.
- [ ] Route reasoning-topology selection through a served, calibrated decision policy with explicit abstention and delete the local exponential-moving-average outcome store. Closes AU-SEMANTIC-R008.
- [x] Delete the local engine facade and orchestration modules, replace the graph-compute/session facades, and the tenancy/shard-topology/admission modules with calls into the generated EG client (one typed composition under `api/`), and replace the hand-maintained graph-schema DTOs with EG-generated types. Closes AU-SEMANTIC-R009, AU-SEMANTIC-R010, AU-SEMANTIC-R012, AU-SEMANTIC-R022.
- [x] Add `BoundCapacityLeasePort`, a fail-closed port binding `core/capacity_lease_port.py` to EG's generated `capacity_leases` namespace under one verified tenant/owner/work-item identity (acquire/renew/release cannot change that identity; every response's owner and fence binding is checked before being trusted). Closes AU-SEMANTIC-R011. Remaining: wire a real caller onto this port and retire the local `knowledge_graph/core/work_durability.py` / `core/state_store.py` durable-job/queue/state authority.
- [ ] Delete AU's local reasoning and graph-analytics duplicates only after each has documented, per-method parity evidence against its EG-native replacement. Closes AU-SEMANTIC-R013.
- [x] Move ingestion commit/derivation to EG's `SourceIngest`, re-host enterprise source sync as an SDK runner over `SourceIngest`, move document/session/feed ingestion and vendor-enrichment extraction/writeback to the SDK and EG's `WriteBack`, move the remaining deterministic-derivation, standardization/infra/governance, and ontology object-model modules to EG, and move AU's hand-written RDF/OWL/SHACL emitters behind EG pack compilation. Closes AU-SEMANTIC-R014, AU-SEMANTIC-R015, AU-SEMANTIC-R016, AU-SEMANTIC-R017, AU-SEMANTIC-R018, AU-SEMANTIC-R019, AU-SEMANTIC-R020, AU-SEMANTIC-R021.
- [x] Delete the legacy SPARQL backend and setup modules, and move the remaining external graph/database backends into EG as federation and mirror targets. Closes AU-SEMANTIC-R023, AU-SEMANTIC-R024.
- [ ] Move the retrieval and neural-search engines to EG, keeping only context compilation and the capability index in AU. Closes AU-SEMANTIC-R025.
- [ ] Re-home the unlanded schema-drift package so its drift gate runs inside the SDK connector-sync runner between drain and apply, with SHACL rendering, the contract store and activation moving to EG. Closes AU-SEMANTIC-R026.
- [x] Require every capability-gap determination to search EG's own public surface (method catalog, generated contract, documentation) under EG's own naming before concluding a capability is missing, and make the configured embedding dimension validate against the deployed model's actual output size across the PostgreSQL/AGE/Neo4j backends and schema/ontology modules that size vector columns from it. Closes AU-SEMANTIC-R027, AU-SEMANTIC-R028.
- [ ] Run the generated-client conformance suite, positive/negative served tests, the old-caller closure census, and CCCC, jscpd, Dupehound, KISS, Ruff/mypy and full quality gates; record exact merged-head evidence in `evidence.md` before any requirement is marked accepted.
- [x] **AU-SEMANTIC-R021.1:** AU's hand-written RDF/OWL/SHACL emitters move behind EG pack compilation — core/ontology_publisher.py slice of `AU-SEMANTIC-R021`.
- [x] **AU-SEMANTIC-R021.2:** AU's hand-written RDF/OWL/SHACL emitters move behind EG pack compilation — extraction/schema_discovery.py slice of `AU-SEMANTIC-R021`.
- [x] **AU-SEMANTIC-R021.3:** AU's hand-written RDF/OWL/SHACL emitters move behind EG pack compilation — ontology/value_types.py slice of `AU-SEMANTIC-R021`.
- [x] **AU-SEMANTIC-R021.4:** AU's hand-written RDF/OWL/SHACL emitters move behind EG pack compilation — scripts/scaffold_ontology_leg.py slice of `AU-SEMANTIC-R021`.

- [x] **AU-SEMANTIC-R010.1:** typed EG-client composition (`GraphComputeClient`/`GeneratedGraphComputeSurface` in `agent_utilities/api/graph_compute_client.py`) and its refusal behavior — net-new `.1` slice of `AU-SEMANTIC-R010` (typed model + refusal test); the graph-compute/session/epistemic_row/ogm/company_brain/kg_adapter call-site migrations onto it are separate, unlanded `.2`+ children.
- [x] **AU-SEMANTIC-R022.1:** typed EG-generated DTO composition (`GraphSchemaTypes`/`GeneratedGraphSchemaSurface` in `agent_utilities/api/graph_schema_types.py`) and its refusal behavior — net-new `.1` slice of `AU-SEMANTIC-R022` (typed model + refusal test); the knowledge_graph/schema_definition/evidence_bundle call-site migrations onto it are separate, unlanded `.2`+ children.
- [x] **AU-SEMANTIC-R025.1:** typed EG-client composition (`RetrievalClient`/`GeneratedRetrievalSurface` in `agent_utilities/api/retrieval_client.py`) and its refusal behavior — net-new `.1` slice of `AU-SEMANTIC-R025` (typed model + refusal test); the context-compilation call-site migrations onto it are separate, unlanded `.2`+ children.
- [x] **AU-SEMANTIC-R027.1:** typed capability-gap record (`CapabilityGapResult`/`EGSurfaceSearch` in `agent_utilities/api/capability_gap_check.py`) refusing an unverified "missing" verdict — net-new `.1` slice of `AU-SEMANTIC-R027` (typed model + refusal test); the migration-filing call sites that produce real verdicts are separate, unlanded `.2`+ children.

The task owner records a separate verdict per ID even when multiple IDs share a PR.
- [x] **AU-SEMANTIC-R009.1:** typed `runnable_skill_derivation` client contract (`RunnableSkillDerivationClient`/`resolve_runnable_skill_derivation_client` in `agent_utilities/api/skill_derivation_client.py`) that fails closed with `RunnableSkillDerivationUnavailableError` while EG's op (producer: `EG-REPO-INGEST-R002`) is absent — net-new `.1` slice of `AU-SEMANTIC-R009`; the engine-facade deletion onto it is the separate, unlanded `.2` child.
- [x] **AU-SEMANTIC-R015.1:** typed `workflow_derivation` client contract (`WorkflowDerivationClient`/`resolve_workflow_derivation_client` in `agent_utilities/api/workflow_derivation_client.py`) that fails closed with `WorkflowDerivationUnavailableError` while EG's op (producer: `EG-REPO-INGEST-R002`) is absent — net-new `.1` slice of `AU-SEMANTIC-R015`; the `source_sync.py` SDK-runner cutover onto it is the separate, unlanded `.2` child.

## Decomposition children (tracked)

- [ ] **AU-SEMANTIC-R009.2:** AU's engine-facade callers consume the shipped runnable_skill_derivation op
- [ ] **AU-SEMANTIC-R015.2:** The SDK-hosted source-sync runner consumes the shipped workflow_derivation op
- [x] **AU-SEMANTIC-R016.1:** The AU-SEMANTIC-R016 migration scope is a typed, validated manifest

## Decomposition children (tracked)

- [ ] **AU-SEMANTIC-R006:** AU removes rdflib and submits typed payloads to EG instead
- [x] **AU-SEMANTIC-R006.1:** The harness gate submits a typed payload instead of an rdflib graph
- [ ] **AU-SEMANTIC-R006.2:** The workflow gate submits a typed payload instead of an rdflib graph
- [ ] **AU-SEMANTIC-R006.3:** The SHACL pipeline gate and pack loader submit typed payloads instead of rdflib graphs
- [ ] **AU-SEMANTIC-R006.4:** The connector-certification gate submits a typed payload instead of an rdflib graph
- [ ] **AU-SEMANTIC-R006.5:** The connector-manifest gate and generation scripts submit typed payloads instead of rdflib graphs
- [ ] **AU-SEMANTIC-R006.5.1:** `connector_manifest_gate.py` submits typed payloads instead of an rdflib graph
- [ ] **AU-SEMANTIC-R006.5.2:** `generate_connector_manifests.py` submits typed payloads instead of an rdflib graph
- [ ] **AU-SEMANTIC-R006.5.3:** `generate_native_connector_manifest.py` submits typed payloads instead of an rdflib graph
- [ ] **AU-SEMANTIC-R006.5.4:** `check_connector_capability_bundles.py` submits typed payloads instead of an rdflib graph
- [ ] **AU-SEMANTIC-R006.5.5:** `generate_connector_capability_bundles.py` submits typed payloads instead of an rdflib graph
- [ ] **AU-SEMANTIC-R006.6:** The team-sharing gate, remaining ontology modules and `rdflib` dependency removal close R006
- [x] **AU-SEMANTIC-R008:** Reasoning-topology selection uses a served policy, not a local outcome store
- [ ] **AU-SEMANTIC-R008.1:** A typed topology decision-policy client contract fails closed while no served policy exists
- [x] **AU-SEMANTIC-R008.2:** Topology selection consumes the served decision policy and the local outcome store is deleted
- [ ] **AU-SEMANTIC-R008.2.1:** Factual record of what landed for R008.2
- [ ] **AU-SEMANTIC-R008.2.2:** `topology.py` calls the served decision-policy client instead of the local EMA store
- [ ] **AU-SEMANTIC-R008.2.3:** The local EMA outcome-store code is deleted from `topology.py`
- [ ] **AU-SEMANTIC-R011.1:** Factual record of what landed for R011
- [ ] **AU-SEMANTIC-R011.2:** Delete `knowledge_graph/core/queue_backend.py`
- [ ] **AU-SEMANTIC-R011.3:** Delete `knowledge_graph/core/kafka_queue_backend.py`
- [ ] **AU-SEMANTIC-R011.4:** Delete `knowledge_graph/core/postgres_queue_backend.py`
- [ ] **AU-SEMANTIC-R011.5:** Delete `knowledge_graph/core/worker_scheduler.py`
- [ ] **AU-SEMANTIC-R011.5.1:** Repoint `engine_tasks.py` off `worker_scheduler` (BLOCKED on epistemic-graph: no served WorkerRegistry, AdmissionPolicy, scheduler config/lane floors or shard-writer width)
- [ ] **AU-SEMANTIC-R011.5.2:** Repoint `ingest_routing.py` off `worker_scheduler`
- [ ] **AU-SEMANTIC-R011.5.3:** Repoint scheduler tests, part 1
- [ ] **AU-SEMANTIC-R011.5.4:** Repoint scheduler tests, part 2
- [ ] **AU-SEMANTIC-R011.5.5:** Delete `knowledge_graph/core/worker_scheduler.py`
- [ ] **AU-SEMANTIC-R011.6:** Delete `knowledge_graph/core/chunked_drain.py`
- [ ] **AU-SEMANTIC-R011.7:** Delete `knowledge_graph/core/ingest_routing.py`
- [ ] **AU-SEMANTIC-R011.8:** Delete `knowledge_graph/core/bitemporal.py`
- [ ] **AU-SEMANTIC-R011.9:** Delete `knowledge_graph/ingest_worker.py`
- [ ] **AU-SEMANTIC-R011.10:** Delete `knowledge_graph/analytics_worker.py`
- [ ] **AU-SEMANTIC-R011.11:** Delete `core/shared_resource_leases.py`
- [ ] **AU-SEMANTIC-R011.12:** Delete `core/state_store.py`
- [ ] **AU-SEMANTIC-R011.13:** Delete `core/chat_persistence.py`
- [ ] **AU-SEMANTIC-R019.1:** Factual record of what landed for R019
- [ ] **AU-SEMANTIC-R019.2:** Delete `observability/trace_ontology.py`
- [ ] **AU-SEMANTIC-R019.3:** Delete `observability/self_ingest.py`
- [ ] **AU-SEMANTIC-R019.4:** Delete `observability/audit_logger.py`
- [ ] **AU-SEMANTIC-R019.5:** Delete `governance/relational_authority.py`
- [ ] **AU-SEMANTIC-R019.6:** Delete `knowledge_graph/research/placement_mining.py`
- [ ] **AU-SEMANTIC-R019.6.1:** The reactive placement-mining tick is removed from the engine scheduler
- [ ] **AU-SEMANTIC-R019.6.2:** The `placement_control` stage is removed from the loop controller
- [x] **AU-SEMANTIC-R019.6.2.1:** The `placement_control` stage is removed from the loop controller
- [ ] **AU-SEMANTIC-R019.6.2.2:** The `placement_control_loop_enabled` setting is removed from config
- [ ] **AU-SEMANTIC-R019.6.3:** The `graph_loops` `placement_control` action is removed
- [ ] **AU-SEMANTIC-R019.6.3.1:** The `placement_control` action and its parameters are removed from `graph_loops`
- [ ] **AU-SEMANTIC-R019.6.3.2:** The generated manifests are regenerated without `placement_control`
- [ ] **AU-SEMANTIC-R019.6.4:** `knowledge_graph/research/placement_mining.py` is deleted
- [ ] **AU-SEMANTIC-R009.2.1:** `resolve_runnable_skill_derivation_client()` wires the real call when EG's op is present
- [x] **AU-SEMANTIC-R009.2.2:** The local engine-facade modules are deleted once callers consume the wired client
- [ ] **AU-SEMANTIC-R013:** Reasoning and analytics duplicates are deleted only with proven EG parity
- [ ] **AU-SEMANTIC-R013.1:** The AU-SEMANTIC-R013 core-reasoning migration scope is a typed, validated manifest
- [ ] **AU-SEMANTIC-R015.2.1:** The AU-SEMANTIC-R015.2 migration scope is a typed, validated manifest
- [ ] **AU-SEMANTIC-R018.1:** The AU-SEMANTIC-R018 enrichment migration scope is a typed, validated manifest
- [ ] **AU-SEMANTIC-R020.1:** The AU-SEMANTIC-R020 migration scope is a typed, validated manifest
