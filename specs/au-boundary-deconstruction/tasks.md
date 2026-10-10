# Tasks and status

**Legend:** TODO = no accepted implementation proof; IN PROGRESS = linked change under review; IMPLEMENTED = code at merged head with source proof; VERIFIED = acceptance tests at that head; ACCEPTED = owner sign-off and release proof. A task is not completed by a draft branch.

| Task | IDs | State | Done when |
|---|---|---|---|
| Served shell and caller cut | AU-BOUNDARY-R001, AU-BOUNDARY-R002, AU-BOUNDARY-R003, AU-BOUNDARY-R004, AU-BOUNDARY-R013, AU-BOUNDARY-R014, AU-BOUNDARY-R015, AU-BOUNDARY-R016, AU-BOUNDARY-R017, AU-BOUNDARY-R039 | TODO | graph-os parity plus AU deletion and script/import census |
| Connector and source cut | AU-BOUNDARY-R005, AU-BOUNDARY-R006, AU-BOUNDARY-R007, AU-BOUNDARY-R008, AU-BOUNDARY-R009, AU-BOUNDARY-R010, AU-BOUNDARY-R011, AU-BOUNDARY-R024, AU-BOUNDARY-R025, AU-BOUNDARY-R026 | TODO | SDK transport/pack certification plus all AU caller deletions |
| Graph, schema, retrieval, memory cut | AU-BOUNDARY-R012, AU-BOUNDARY-R018, AU-BOUNDARY-R019, AU-BOUNDARY-R020, AU-BOUNDARY-R021, AU-BOUNDARY-R022, AU-BOUNDARY-R023, AU-BOUNDARY-R027, AU-BOUNDARY-R028, AU-BOUNDARY-R029, AU-BOUNDARY-R030, AU-BOUNDARY-R031, AU-BOUNDARY-R032, AU-BOUNDARY-R033, AU-BOUNDARY-R034, AU-BOUNDARY-R035, AU-BOUNDARY-R040 | TODO | generated EG method parity, trusted migration and AU adapter-only shape |
| AU-BOUNDARY-R028 split by code root (EG-REPO-INGEST-R003) | AU-BOUNDARY-R028.1, AU-BOUNDARY-R028.2, AU-BOUNDARY-R028.3, AU-BOUNDARY-R028.4, AU-BOUNDARY-R028.5, AU-BOUNDARY-R028.6, AU-BOUNDARY-R028.7, AU-BOUNDARY-R028.8 | TODO | each child's own files deleted (or reported blocked pending an EG client) and the import census passes |
| Usage and governance cut | AU-BOUNDARY-R036, AU-BOUNDARY-R037 | TODO | EG durable usage and repository-manager governance parity |
| Fabricated-data finance module removal | AU-BOUNDARY-R041 | TODO | import census shows no listed module and no caller |
| Inventory completeness check | AU-BOUNDARY-R042 | TODO | tree-versus-inventory test fails on an unlisted directory, an undecided row or a stale path |
| Retained agent surface | AU-BOUNDARY-R043 | TODO | import-boundary test over the kept packages |
| Fleet, scaling and deployment modules | AU-BOUNDARY-R044 | TODO | caller census and graph-os consumer tests |
| Knowledge-graph file-level owners | AU-BOUNDARY-R045 | TODO | every tracked file under `knowledge_graph/` maps to one owner; parity tests per move |
| File-level owners for partly covered packages | AU-BOUNDARY-R046 | TODO | no tracked file without an owner; no requirement by description only |
| Engine-facing adapters and caches | AU-BOUNDARY-R047 | TODO | differential tests against the engine; no second store or kernel |
| Domain models, shapes, assets and skill content | AU-BOUNDARY-R048 | TODO | each directory placed; pack or skill validation in the receiving repository |
| Governed write-back execution | AU-BOUNDARY-R049 | IN PROGRESS | `run_writeback` calls every sink through the SDK `DurableWritableConnector`; the governed and existing write-back suites pass at the merged head |
| Governed write-back on the EG ledger | AU-BOUNDARY-R049 | TODO | live applies use `EpistemicGraphWriteBackLedger` once graph-os serves the EG `WriteBack` operation; a restart test proves receipt replay |
| Permanent owner guard | AU-BOUNDARY-R038 | TODO | generated manifest, CI gate and exact-head check |
| Quality gates and evidence | all of the above | TODO | CCCC, KISS, Dupehound, jscpd, language linters and the full test suite pass at the merged head and the result is recorded in [evidence.md](evidence.md) |

Each row expands into a PR checklist with changed paths, owner operation, positive and negative test IDs, scanner delta, review, merged commit and acceptance artifact. Do not move a row to VERIFIED solely because another row in its range passed. Every ID is defined in [requirements.md](requirements.md); the directories each requirement removes or relocates are listed in the [deletion and relocation inventory](coverage.md#deletion-and-relocation-inventory).

## Progress notes

- `AU-BOUNDARY-R030`: the SDK half landed as `agent_connector_sdk.manifest.ontology_pack.compile_manifest_ontology` (Knuckles-Team/agent-connector-sdk#36). AU's compile-before-sync gate (`connector_manifest_gate._compiled_manifest_graph`) now calls it instead of rendering Turtle locally. `agent_utilities/knowledge_graph/ontology/manifest_compiler.py` is not yet deleted: `scripts/generate_connector_manifests.py`, `scripts/update_ontology_lock.py`, `scripts/generate_native_connector_manifest.py`, and `agent_utilities/knowledge_graph/domain_packs/pack_loader.py` still import it directly and need their own migration before the local copy is removed.
- `AU-BOUNDARY-R035`: the SHACL-removal half (candidate.py sends EG a typed `RecordContract`, never Turtle) landed at main commit `9cc9e08c1a0e0c273223b5cc84ebe9e2fcb2f946` under `AU-SEC-R005`, ahead of this spec recording it. The remaining half, moving `contract_store` and `activation` into the epistemic graph and the SDK connector-sync runner's gate/shape/classify/report steps, is open.

## Decomposition children (tracked)

- [x] **AU-BOUNDARY-R028.2.1:** Pin the kg/infra production-importer set.
- [x] **AU-BOUNDARY-R028.6.1:** Pin the observability/{trace_ontology,self_ingest,audit_logger} production-importer set.
- [x] **AU-BOUNDARY-R028.7.1:** Pin the governance/relational_authority importer set.
- [x] **AU-BOUNDARY-R028.8.1:** Pin the kg/research/placement_mining production-importer set.
- [x] **AU-BOUNDARY-R036.1:** Typed inventory of usage-store paths pending move to EG
- [x] **AU-BOUNDARY-R037.1:** Typed inventory of governance modules pending move to repository-manager
- [x] **AU-BOUNDARY-R039.1:** Typed inventory of AU's public front-end import surface
- [x] **AU-BOUNDARY-R040.1:** Typed inventory of memory/learning-engine paths pending move to EG
- [x] **AU-BOUNDARY-R044.1:** Typed inventory of orchestration paths pending move to graph-os
- [x] **AU-BOUNDARY-R045.1:** Typed inventory of unplaced knowledge_graph packages pending move
- [x] **AU-BOUNDARY-R047.1:** Typed inventory of engine-facing adapters pending thin-client cutover
- [x] **AU-BOUNDARY-R030.1:** Record what has already landed for the emitter cutover (SDK pack compilation + gate call site landed; 4 callers and 3 new modules remain).
- [ ] **AU-BOUNDARY-R030.2:** Wire `scripts/generate_connector_manifests.py` off the local manifest compiler onto the SDK pack compiler.
- [ ] **AU-BOUNDARY-R030.3:** Wire `scripts/update_ontology_lock.py` off the local manifest compiler onto the SDK pack compiler.
- [ ] **AU-BOUNDARY-R030.4:** Wire `scripts/generate_native_connector_manifest.py` off the local manifest compiler onto the SDK pack compiler.
- [ ] **AU-BOUNDARY-R030.5:** Wire `agent_utilities/knowledge_graph/domain_packs/pack_loader.py` off the local manifest compiler onto the SDK pack compiler.
- [ ] **AU-BOUNDARY-R030.6:** Delete `agent_utilities/knowledge_graph/ontology/manifest_compiler.py` once it has zero importers.
- [ ] **AU-BOUNDARY-R030.7:** Add `agent_utilities/knowledge_graph/core/ontology_publisher.py`, the typed AU declaration publisher.
- [ ] **AU-BOUNDARY-R030.8:** Add `agent_utilities/knowledge_graph/extraction/schema_discovery.py`, the EG-pack-compilation-backed schema discovery module.
- [ ] **AU-BOUNDARY-R030.9:** Add `scripts/scaffold_ontology_leg.py`, the scaffold script for a new ontology leg via EG pack compilation.
  - [ ] **AU-BOUNDARY-R030.9.1:** Remove the `scripts/scaffold_ontology_leg.py` tripwire from `check_current_only_contract.py` `RETIRED_PATHS`.
  - [x] **AU-BOUNDARY-R030.9.2:** Add `scripts/scaffold_ontology_leg.py` compiling via the SDK pack compiler (needs R030.9.1).
- [x] **AU-BOUNDARY-R048.1:** `agent_utilities/images/` already deleted/relocated; no remaining work on that directory
- [ ] **AU-BOUNDARY-R048.2:** Relocate `domains/government/` to its epistemic-graph domain pack and delete from AU
- [ ] **AU-BOUNDARY-R048.3:** Relocate `domains/hr/` to its epistemic-graph domain pack and delete from AU
- [ ] **AU-BOUNDARY-R048.4:** Relocate `domains/law/` to its epistemic-graph domain pack and delete from AU
- [ ] **AU-BOUNDARY-R048.5:** Relocate `domains/medical/` to its epistemic-graph domain pack and delete from AU
- [ ] **AU-BOUNDARY-R048.6:** Relocate `models/domains/` to its epistemic-graph domain pack and delete from AU
- [ ] **AU-BOUNDARY-R048.7:** Move `mcp/tools/analysis_tools.py` (HR workforce MCP tool) to graph-os and delete from AU
- [ ] **AU-BOUNDARY-R048.8:** Assign and act on one disposition for `ontology/shapes/`
- [ ] **AU-BOUNDARY-R048.9:** Assign and act on one disposition for `data/`
- [ ] **AU-BOUNDARY-R048.10:** Assign and act on one disposition for `protocols/voice_supply_chain/`
- [ ] **AU-BOUNDARY-R048.11:** Assign and act on one disposition for each skill-content directory under `skills/`
- [ ] **AU-BOUNDARY-R005.1:** A-E connector import census (PR #150, `79dc21bab`) that moved R005 to LANDED; `status.json`'s `landed_in` SHA is wrong (points at the unrelated `AU-HARNESS-R005.1` merge, not an A-E connector change).
- [ ] **AU-BOUNDARY-R005.2:** Migrate `ansible-tower-mcp`, `audiobookshelf-mcp`, `ciso-assistant-api`, `data-science-mcp` off the banned AU imports the census missed.
- [ ] **AU-BOUNDARY-R007.1:** K-O connector import census (PR #150, `79dc21bab`) that moved R007 to LANDED; `status.json`'s `landed_in` SHA is wrong (points at the unrelated `AU-HARNESS-R005.1` merge, not a K-O connector change).
- [ ] **AU-BOUNDARY-R007.2:** Migrate `keycloak-agent`, `langfuse-agent`, `leanix-agent`, `microsoft-agent`, `opensearch-mcp` off the banned AU imports the census missed.
- [ ] **AU-BOUNDARY-R008.1:** P-S connector import census (PR #150, `79dc21bab`) that moved R008 to LANDED; `status.json`'s `landed_in` SHA is wrong (points at the unrelated `AU-HARNESS-R005.1` merge, not a P-S connector change).
- [ ] **AU-BOUNDARY-R008.2:** Migrate `paperless-ngx-mcp`, `pulselink-mcp`, `repository-manager`, `rom-manager`, `systems-manager` off the banned AU imports the census missed.
- [ ] **AU-BOUNDARY-R001.1:** Record that PR #23 (`aa5eba73e0f7`) shipped only boundary gate scripts, not the gateway deletion; `status.json`'s LANDED is wrong.
- [ ] **AU-BOUNDARY-R001.2:** Delete `agent_utilities/gateway/api.py`.
- [ ] **AU-BOUNDARY-R001.3:** Delete `agent_utilities/gateway/artifacts_api.py`.
- [ ] **AU-BOUNDARY-R001.4:** Delete `agent_utilities/gateway/registry_api.py` (rollup of .4.1 to .4.3).
- [ ] **AU-BOUNDARY-R001.4.1:** Stop mounting registry_api routes from graph_api.py.
- [ ] **AU-BOUNDARY-R001.4.2:** ROLLUP: repoint the /api/tools catalog read in kg_server.py off registry_api (children .4.2.1 to .4.2.4).
- [ ] **AU-BOUNDARY-R001.4.2.1:** Record the owner of the servers catalog read for GET /api/tools.
- [ ] **AU-BOUNDARY-R001.4.2.2:** Add one typed servers-kind catalog reader that does not import registry_api.
- [ ] **AU-BOUNDARY-R001.4.2.3:** Call the new reader from _read_catalog_kind_sync in kg_server.py.
- [ ] **AU-BOUNDARY-R001.4.2.4:** Update test_collapse_tool_endpoints.py to patch the new reader.
- [ ] **AU-BOUNDARY-R001.4.3:** Delete the registry_api module and its tests.
- [ ] **AU-BOUNDARY-R001.5:** Delete `agent_utilities/gateway/registry.py`.
- [ ] **AU-BOUNDARY-R001.6:** Delete `agent_utilities/gateway/schemas/graph_analyze.py`.
- [ ] **AU-BOUNDARY-R001.7:** Delete `agent_utilities/gateway/widgets/genius_agent.py`.
- [ ] **AU-BOUNDARY-R001.8:** Delete `agent_utilities/gateway/widgets/_optional_client.py`.
- [ ] **AU-BOUNDARY-R002.1:** Record that PR #26 (`fca0c5ee42b3`) did not delete `backends.py`/`certification_oidc.py`; `status.json`'s LANDED is wrong.
- [ ] **AU-BOUNDARY-R002.2:** Delete `agent_utilities/deployment/backends.py`.
- [ ] **AU-BOUNDARY-R002.3:** ROLLUP: delete `agent_utilities/deployment/certification_oidc.py` via children R002.3.1 to R002.3.5.
- [ ] **AU-BOUNDARY-R002.3.1:** Record the graph-os owner of the ephemeral loopback OIDC certification authority.
- [ ] **AU-BOUNDARY-R002.3.2:** Repoint or remove the `certification_oidc` import in `skill_validation.py` and `skill_validation_assets.py`.
- [ ] **AU-BOUNDARY-R002.3.3:** Delete `scripts/certification/loopback_oidc.py` and `tests/unit/scripts/test_loopback_oidc.py`.
- [ ] **AU-BOUNDARY-R002.3.4:** Remove `certification_oidc.py` from the egress-boundary and release-wheel script path lists.
- [ ] **AU-BOUNDARY-R002.3.5:** Delete `agent_utilities/deployment/certification_oidc.py`.
- [ ] **AU-BOUNDARY-R003.1:** Record that commit `2456745c1e4d` shipped only gate scripts, not the MCP host deletion; `status.json`'s LANDED is wrong.
- [ ] **AU-BOUNDARY-R003.2:** Delete `agent_utilities/mcp/kg_server.py`.
- [ ] **AU-BOUNDARY-R003.3:** Delete `agent_utilities/mcp/_graphos_action_manifest.py`.
- [ ] **AU-BOUNDARY-R003.4:** Remove `_ingest_capabilities` from `agent_utilities/sdd/watcher.py`.
- [ ] **AU-BOUNDARY-R003.2:** RETIRED: superseded by AU-BOUNDARY-R003.5...AU-BOUNDARY-R003.29.
- [ ] **AU-BOUNDARY-R003.5:** see requirements.md.
- [ ] **AU-BOUNDARY-R003.6:** see requirements.md.
- [ ] **AU-BOUNDARY-R003.7:** see requirements.md.
- [ ] **AU-BOUNDARY-R003.8:** see requirements.md.
- [ ] **AU-BOUNDARY-R003.9:** see requirements.md.
- [ ] **AU-BOUNDARY-R003.10:** see requirements.md.
- [ ] **AU-BOUNDARY-R003.11:** see requirements.md.
- [ ] **AU-BOUNDARY-R003.12:** see requirements.md.
- [ ] **AU-BOUNDARY-R003.13:** see requirements.md.
- [ ] **AU-BOUNDARY-R003.14:** see requirements.md.
- [ ] **AU-BOUNDARY-R003.15:** see requirements.md.
- [ ] **AU-BOUNDARY-R003.16:** see requirements.md.
- [ ] **AU-BOUNDARY-R003.17:** see requirements.md.
- [ ] **AU-BOUNDARY-R003.18:** see requirements.md.
- [ ] **AU-BOUNDARY-R003.19:** see requirements.md.
- [ ] **AU-BOUNDARY-R003.20:** see requirements.md.
- [ ] **AU-BOUNDARY-R003.21:** see requirements.md.
- [ ] **AU-BOUNDARY-R003.22:** see requirements.md.
- [ ] **AU-BOUNDARY-R003.23:** see requirements.md.
- [ ] **AU-BOUNDARY-R003.24:** see requirements.md.
- [ ] **AU-BOUNDARY-R003.25:** see requirements.md.
- [ ] **AU-BOUNDARY-R003.26:** see requirements.md.
- [ ] **AU-BOUNDARY-R003.27:** see requirements.md.
- [ ] **AU-BOUNDARY-R003.28:** see requirements.md.
- [ ] **AU-BOUNDARY-R003.29:** see requirements.md.
- [ ] **AU-BOUNDARY-R004.1:** Record that commit `2456745c1e4d` shipped only gate scripts, not the multiplexer deletion; `status.json`'s LANDED is wrong.
- [ ] **AU-BOUNDARY-R004.2:** RETIRED: superseded by AU-BOUNDARY-R004.3...AU-BOUNDARY-R004.10.
- [ ] **AU-BOUNDARY-R004.3:** see requirements.md.
- [ ] **AU-BOUNDARY-R004.4:** see requirements.md.
- [ ] **AU-BOUNDARY-R004.5:** see requirements.md.
- [ ] **AU-BOUNDARY-R004.6:** see requirements.md.
- [ ] **AU-BOUNDARY-R004.7:** see requirements.md.
- [ ] **AU-BOUNDARY-R004.8:** see requirements.md.
- [ ] **AU-BOUNDARY-R004.9:** see requirements.md.
- [ ] **AU-BOUNDARY-R004.10:** see requirements.md.
- [ ] **AU-BOUNDARY-R010.1:** Record that `status.json`'s `landed_in` (`2456745c1e4d`) did not delete the connector toolkit; LANDED is wrong.
- [ ] **AU-BOUNDARY-R010.2:** RETIRED: superseded by AU-BOUNDARY-R010.3...AU-BOUNDARY-R010.25.
- [ ] **AU-BOUNDARY-R010.3:** see requirements.md.
- [ ] **AU-BOUNDARY-R010.4:** see requirements.md.
- [ ] **AU-BOUNDARY-R010.5:** see requirements.md.
- [ ] **AU-BOUNDARY-R010.6:** see requirements.md.
- [ ] **AU-BOUNDARY-R010.7:** see requirements.md.
- [ ] **AU-BOUNDARY-R010.8:** see requirements.md.
- [ ] **AU-BOUNDARY-R010.9:** see requirements.md.
- [ ] **AU-BOUNDARY-R010.10:** see requirements.md.
- [ ] **AU-BOUNDARY-R010.11:** see requirements.md.
- [ ] **AU-BOUNDARY-R010.12:** see requirements.md.
- [ ] **AU-BOUNDARY-R010.13:** see requirements.md.
- [ ] **AU-BOUNDARY-R010.14:** see requirements.md.
- [ ] **AU-BOUNDARY-R010.15:** see requirements.md.
- [ ] **AU-BOUNDARY-R010.16:** see requirements.md.
- [ ] **AU-BOUNDARY-R010.17:** see requirements.md.
- [ ] **AU-BOUNDARY-R010.18:** see requirements.md.
- [ ] **AU-BOUNDARY-R010.19:** see requirements.md.
- [ ] **AU-BOUNDARY-R010.20:** see requirements.md.
- [ ] **AU-BOUNDARY-R010.21:** see requirements.md.
- [ ] **AU-BOUNDARY-R010.22:** see requirements.md.
- [ ] **AU-BOUNDARY-R010.23:** see requirements.md.
- [ ] **AU-BOUNDARY-R010.24:** see requirements.md.
- [ ] **AU-BOUNDARY-R010.25:** see requirements.md.

## Decomposition children (tracked)

- [ ] **AU-BOUNDARY-R006.1:** Ship AU's F-J connector import-census gate
- [ ] **AU-BOUNDARY-R009.1:** Ship AU's T-Z connector import-census gate
- [ ] **AU-BOUNDARY-R016.1:** Gate AU's retained config.py against duplicate environment readers
- [x] **AU-BOUNDARY-R018.1:** Ship a typed inventory of the AU-BOUNDARY-R018 facade modules
- [ ] **AU-BOUNDARY-R018.2:** Delete the AU-BOUNDARY-R018 facade modules
- [x] **AU-BOUNDARY-R019.1:** Ship a typed inventory of the AU-BOUNDARY-R019 facade modules
- [ ] **AU-BOUNDARY-R019.2:** Delete the AU-BOUNDARY-R019 facade modules
- [x] **AU-BOUNDARY-R020.1:** Ship a typed inventory of the AU-BOUNDARY-R020 durable-work modules
- [ ] **AU-BOUNDARY-R020.2:** Move the AU-BOUNDARY-R020 durable-work modules and drop their scripts
- [x] **AU-BOUNDARY-R021.1:** Ship a typed inventory of the AU-BOUNDARY-R021 tenancy/admission modules
- [ ] **AU-BOUNDARY-R021.2:** Delete the AU-BOUNDARY-R021 tenancy/admission modules
- [x] **AU-BOUNDARY-R022.1:** Ship a typed inventory of the AU-BOUNDARY-R022 reasoning/analytics modules
- [ ] **AU-BOUNDARY-R022.2:** Delete the AU-BOUNDARY-R022 reasoning/analytics modules
- [ ] **AU-BOUNDARY-R011.1:** Record that R011's deletions (universal_connector, source_connectors, certification, manifest) have not started.
- [ ] **AU-BOUNDARY-R011.2:** ROLLUP: superseded by AU-BOUNDARY-R011.2.1...AU-BOUNDARY-R011.2.5.
- [ ] **AU-BOUNDARY-R011.2.1:** Add the read-only multi-database connector to `agent-connector-sdk` with conformance tests.
- [ ] **AU-BOUNDARY-R011.2.2:** Repoint `protocols/source_connectors/connectors/database.py` to the SDK connector.
- [ ] **AU-BOUNDARY-R011.2.3:** Repoint `tools/db_tools.py` to the SDK connector.
- [ ] **AU-BOUNDARY-R011.2.4:** Repoint/delete the universal_connector tests and docs references.
- [ ] **AU-BOUNDARY-R011.2.5:** Delete `protocols/universal_connector.py` once it has zero importers.
- [ ] **AU-BOUNDARY-R011.3:** Delete the `protocols/source_connectors/**` package and fix its importers.
- [ ] **AU-BOUNDARY-R011.4:** Delete `knowledge_graph/integrations/connector_certification.py`, `connector_certification_cli.py`, `connector_source_attestation.py` and fix their importers.
- [ ] **AU-BOUNDARY-R011.5:** Delete `knowledge_graph/ontology/connector_manifest.py`, `connector_manifest_gate.py`, `connector_manifests/**`, and drop `graph-os-certify-connector`.
- [ ] **AU-BOUNDARY-R014.1:** Record that R014's moves, including `knowledge_graph/readiness.py`, have not started.
- [ ] **AU-BOUNDARY-R014.2:** Move `knowledge_graph/readiness.py` to graph-os and delete the AU copy.
- [ ] **AU-BOUNDARY-R015.1:** Record that R015's moves, including `protocols/a2a_epistemic.py`, have not started.
- [ ] **AU-BOUNDARY-R015.2:** Delete `protocols/a2a_epistemic.py` and drop the `agent-utilities-acp` script.
- [ ] **AU-BOUNDARY-R024.1:** Record that R024's moves, including `source_sync.py` and `governance_import.py`, have not started.
- [ ] **AU-BOUNDARY-R024.2:** Move `knowledge_graph/core/source_sync.py` to the agent connector SDK and delete the AU copy.
- [ ] **AU-BOUNDARY-R024.3:** Move `knowledge_graph/governance_import.py` to the SDK and delete the AU copy.
