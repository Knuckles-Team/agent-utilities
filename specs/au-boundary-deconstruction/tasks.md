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
