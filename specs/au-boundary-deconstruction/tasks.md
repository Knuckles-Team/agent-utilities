# Tasks and status

**Legend:** TODO = no accepted implementation proof. IN PROGRESS = linked change under review. IMPLEMENTED = code at merged head with source proof. VERIFIED = acceptance tests at that head. ACCEPTED = owner sign-off and release proof. A task is not completed by a draft branch.

| Task | IDs | State | Done when |
|---|---|---|---|
| Served shell and caller cut | MERISTEM-BOUNDARY-R002, MERISTEM-BOUNDARY-R014, MERISTEM-BOUNDARY-R015, MERISTEM-BOUNDARY-R016, MERISTEM-BOUNDARY-R017 | TODO | arbora parity plus AU deletion and script/import census |
| Connector and source cut | MERISTEM-BOUNDARY-R005, MERISTEM-BOUNDARY-R006, MERISTEM-BOUNDARY-R007, MERISTEM-BOUNDARY-R008, MERISTEM-BOUNDARY-R009, MERISTEM-BOUNDARY-R010, MERISTEM-BOUNDARY-R011, MERISTEM-BOUNDARY-R024, MERISTEM-BOUNDARY-R025, MERISTEM-BOUNDARY-R026 | TODO | SDK transport/pack certification plus all AU caller deletions |
| Graph, schema, retrieval, memory cut | MERISTEM-BOUNDARY-R012, MERISTEM-BOUNDARY-R018, MERISTEM-BOUNDARY-R019, MERISTEM-BOUNDARY-R020, MERISTEM-BOUNDARY-R021, MERISTEM-BOUNDARY-R022, MERISTEM-BOUNDARY-R023, MERISTEM-BOUNDARY-R027, MERISTEM-BOUNDARY-R028, MERISTEM-BOUNDARY-R029, MERISTEM-BOUNDARY-R030, MERISTEM-BOUNDARY-R031, MERISTEM-BOUNDARY-R032, MERISTEM-BOUNDARY-R033, MERISTEM-BOUNDARY-R034, MERISTEM-BOUNDARY-R035, MERISTEM-BOUNDARY-R040 | TODO | generated EG method parity, trusted migration and AU adapter-only shape |
| MERISTEM-BOUNDARY-R028 split by code root (MEDULLA-REPO-INGEST-R003) | MERISTEM-BOUNDARY-R028.1, MERISTEM-BOUNDARY-R028.2, MERISTEM-BOUNDARY-R028.3, MERISTEM-BOUNDARY-R028.4, MERISTEM-BOUNDARY-R028.5, MERISTEM-BOUNDARY-R028.6, MERISTEM-BOUNDARY-R028.7, MERISTEM-BOUNDARY-R028.8 | TODO | each child's own files deleted (or reported blocked pending an EG client) and the import census passes |
| Usage and governance cut | MERISTEM-BOUNDARY-R036, MERISTEM-BOUNDARY-R037 | TODO | EG durable usage and repository-manager governance parity |
| Relocation owner proof | MERISTEM-BOUNDARY-R036.2, MERISTEM-BOUNDARY-R037.2, MERISTEM-BOUNDARY-R040.2, MERISTEM-BOUNDARY-R044.2, MERISTEM-BOUNDARY-R047.2 | TODO | each parent row's real owner, recovery, parity and census evidence |
| Fabricated-data finance module removal | MERISTEM-BOUNDARY-R041 | TODO | import census shows no listed module and no caller |
| Inventory completeness check | MERISTEM-BOUNDARY-R042 | TODO | tree-versus-inventory test fails on an unlisted directory, an undecided row or a stale path |
| Agent run loop | MERISTEM-BOUNDARY-R043 | TODO | run-loop and recursive-loop behaviour tests |
| Fleet, scaling and deployment modules | MERISTEM-BOUNDARY-R044 | TODO | caller census and arbora consumer tests |
| Knowledge-graph file-level owners | MERISTEM-BOUNDARY-R045 | TODO | every tracked file under `knowledge_graph/` maps to one owner; parity tests per move |
| Engine-facing adapters and caches | MERISTEM-BOUNDARY-R047 | TODO | differential tests against the engine; no second store or kernel |
| Domain models, shapes, assets and skill content | MERISTEM-BOUNDARY-R048 | TODO | each directory placed; pack or skill validation in the receiving repository |
| Governed write-back execution | MERISTEM-BOUNDARY-R049 | IN PROGRESS | `run_writeback` calls every sink through the SDK `DurableWritableConnector`. The governed and existing write-back suites pass at the merged head |
| Governed write-back on the EG ledger | MERISTEM-BOUNDARY-R049 | TODO | live applies use `EpistemicGraphWriteBackLedger` once arbora serves the EG `WriteBack` operation. A restart test proves receipt replay |
| Render-receipt check before model transport | MERISTEM-BOUNDARY-R050 | TODO | the run loop validates the MEDULLA-TYPED-PACKS-R002.10 receipt before any model request; invalid receipts cause zero requests |
| Permanent owner guard | MERISTEM-BOUNDARY-R038 | TODO | generated manifest, CI gate and exact-head check |
| Quality gates and evidence | all of the above | TODO | CCCC, KISS, Dupehound, jscpd, language linters and the full test suite pass at the merged head and the result is recorded in [evidence.md](evidence.md) |

Each row expands into a PR checklist with changed paths, owner operation, positive and negative test IDs, scanner delta, review, merged commit and acceptance artifact. Do not move a row to VERIFIED solely because another row in its range passed. Every ID is defined in [requirements.md](requirements.md); the directories each requirement removes or relocates are listed in the [deletion and relocation inventory](plan.md#deletion-and-relocation-inventory).

## Progress notes

- `MERISTEM-BOUNDARY-R030`: the SDK half landed as `agent_connector_sdk.manifest.ontology_pack.compile_manifest_ontology` (Knuckles-Team/agent-connector-sdk#36). AU's compile-before-sync gate (`connector_manifest_gate._compiled_manifest_graph`) now calls it instead of rendering Turtle locally. `meristem/knowledge_graph/ontology/manifest_compiler.py` is not yet deleted: `scripts/generate_connector_manifests.py`, `scripts/update_ontology_lock.py`, `scripts/generate_native_connector_manifest.py`, and `meristem/knowledge_graph/domain_packs/pack_loader.py` still import it directly and need their own migration before the local copy is removed.
- `MERISTEM-BOUNDARY-R035`: the SHACL-removal half (candidate.py sends EG a typed `RecordContract`, never Turtle) landed at main commit `9cc9e08c1a0e0c273223b5cc84ebe9e2fcb2f946` under `MERISTEM-SEC-R005`, ahead of this spec recording it. The remaining half, moving `contract_store` and `activation` into the epistemic graph and the SDK connector-sync runner's gate/shape/classify/report steps, is open.

## Decomposition children (tracked)

- [ ] **MERISTEM-BOUNDARY-R050.1:** Define the render-receipt consumer and its run-loop call site.
- [ ] **MERISTEM-BOUNDARY-R050.2:** Publish the consumer interface and dispatch semantics.
- [ ] **MERISTEM-BOUNDARY-R050.3:** Bind the engine receipt to the consumer before transport.
- [ ] **MERISTEM-BOUNDARY-R050.4:** Refuse invalid receipts with zero model requests.
- [ ] **MERISTEM-BOUNDARY-R050.5:** Record the consumer location against the run snapshot.
- [ ] **MERISTEM-BOUNDARY-R050.6:** Prove the served journey: one request for a valid receipt, none otherwise.
- [x] **MERISTEM-BOUNDARY-R028.2.1:** Pin the kg/infra production-importer set.
- [x] **MERISTEM-BOUNDARY-R028.6.1:** Pin the observability/{trace_ontology,self_ingest,audit_logger} production-importer set.
- [x] **MERISTEM-BOUNDARY-R028.7.1:** Pin the governance/relational_authority importer set.
- [x] **MERISTEM-BOUNDARY-R028.8.1:** Pin the kg/research/placement_mining production-importer set.
- [x] **MERISTEM-BOUNDARY-R036.1:** Typed inventory of usage-store paths pending move to EG
- [x] **MERISTEM-BOUNDARY-R037.1:** Typed inventory of governance modules pending move to repository-manager
- [x] **MERISTEM-BOUNDARY-R040.1:** Typed inventory of memory/learning-engine paths pending move to EG
- [x] **MERISTEM-BOUNDARY-R044.1:** Typed inventory of orchestration paths pending move to arbora
- [x] **MERISTEM-BOUNDARY-R045.1:** Typed inventory of unplaced knowledge_graph packages pending move
- [x] **MERISTEM-BOUNDARY-R047.1:** Typed inventory of engine-facing adapters pending thin-client cutover
- [x] **MERISTEM-BOUNDARY-R030.1:** Record what has already landed for the emitter cutover (SDK pack compilation + gate call site landed; 4 callers and 3 new modules remain).
- [ ] **MERISTEM-BOUNDARY-R030.2:** Wire `scripts/generate_connector_manifests.py` off the local manifest compiler onto the SDK pack compiler.
- [ ] **MERISTEM-BOUNDARY-R030.3:** Wire `scripts/update_ontology_lock.py` off the local manifest compiler onto the SDK pack compiler.
- [ ] **MERISTEM-BOUNDARY-R030.4:** Wire `scripts/generate_native_connector_manifest.py` off the local manifest compiler onto the SDK pack compiler.
- [ ] **MERISTEM-BOUNDARY-R030.5:** Wire `meristem/knowledge_graph/domain_packs/pack_loader.py` off the local manifest compiler onto the SDK pack compiler.
- [ ] **MERISTEM-BOUNDARY-R030.6:** Delete `meristem/knowledge_graph/ontology/manifest_compiler.py` once it has zero importers.
- [x] **MERISTEM-BOUNDARY-R030.7:** RETIRED: superseded by SCION-CONNECTOR-CONTROL-R067 (typed pack declaration authored by the SDK).
- [x] **MERISTEM-BOUNDARY-R030.8:** RETIRED: superseded by MERISTEM-SEMANTIC-R021.2/R021.4 (SDK ontology pack + `ontology_publisher.py`, MERISTEM-BOUNDARY-R030.7).
- [x] **MERISTEM-BOUNDARY-R030.9:** RETIRED: superseded by MERISTEM-SEMANTIC-R021.2/R021.4 (SDK ontology pack + `ontology_publisher.py`, MERISTEM-BOUNDARY-R030.7).
  - [x] **MERISTEM-BOUNDARY-R030.9.1:** RETIRED: superseded by MERISTEM-SEMANTIC-R021.2/R021.4 (SDK ontology pack + `ontology_publisher.py`, MERISTEM-BOUNDARY-R030.7).
  - [x] **MERISTEM-BOUNDARY-R030.9.2:** RETIRED: superseded by MERISTEM-SEMANTIC-R021.2/R021.4 (SDK ontology pack + `ontology_publisher.py`, MERISTEM-BOUNDARY-R030.7).
- [x] **MERISTEM-BOUNDARY-R048.1:** `meristem/images/` already deleted/relocated; no remaining work on that directory
- [ ] **MERISTEM-BOUNDARY-R048.2:** Relocate `domains/government/` to its medulla domain pack and delete from AU
- [ ] **MERISTEM-BOUNDARY-R048.3:** Relocate `domains/hr/` to its medulla domain pack and delete from AU
- [ ] **MERISTEM-BOUNDARY-R048.4:** Relocate `domains/law/` to its medulla domain pack and delete from AU
- [ ] **MERISTEM-BOUNDARY-R048.5:** Relocate `domains/medical/` to its medulla domain pack and delete from AU
- [ ] **MERISTEM-BOUNDARY-R048.6:** Relocate `models/domains/` to its medulla domain pack and delete from AU
- [ ] **MERISTEM-BOUNDARY-R048.8:** Assign and act on one disposition for `ontology/shapes/`
- [ ] **MERISTEM-BOUNDARY-R048.9:** Assign and act on one disposition for `data/`
- [ ] **MERISTEM-BOUNDARY-R048.10:** Assign and act on one disposition for `protocols/voice_supply_chain/`
- [ ] **MERISTEM-BOUNDARY-R048.11:** Assign and act on one disposition for each skill-content directory under `skills/`
- [ ] **MERISTEM-BOUNDARY-R005.1:** A-E connector import census (PR #150, `79dc21bab`) that moved R005 to LANDED. `status.json`'s `landed_in` SHA is wrong (points at the unrelated `MERISTEM-HARNESS-R005.1` merge, not an A-E connector change).
- [x] **MERISTEM-BOUNDARY-R005.2:** RETIRED: superseded by SCION-CONNECTOR-CONTROL-R009.
- [ ] **MERISTEM-BOUNDARY-R007.1:** K-O connector import census (PR #150, `79dc21bab`) that moved R007 to LANDED. `status.json`'s `landed_in` SHA is wrong (points at the unrelated `MERISTEM-HARNESS-R005.1` merge, not a K-O connector change).
- [x] **MERISTEM-BOUNDARY-R007.2:** RETIRED: superseded by SCION-CONNECTOR-CONTROL-R011.
- [ ] **MERISTEM-BOUNDARY-R008.1:** P-S connector import census (PR #150, `79dc21bab`) that moved R008 to LANDED. `status.json`'s `landed_in` SHA is wrong (points at the unrelated `MERISTEM-HARNESS-R005.1` merge, not a P-S connector change).
- [x] **MERISTEM-BOUNDARY-R008.2:** RETIRED: superseded by SCION-CONNECTOR-CONTROL-R012.
- [ ] **MERISTEM-BOUNDARY-R002.1:** Record that PR #26 (`fca0c5ee42b3`) did not delete `backends.py`/`certification_oidc.py`; `status.json`'s LANDED is wrong.
- [ ] **MERISTEM-BOUNDARY-R002.2:** Delete `meristem/deployment/backends.py`.
- [ ] **MERISTEM-BOUNDARY-R002.3:** ROLLUP: delete `meristem/deployment/certification_oidc.py` via children R002.3.1 to R002.3.5.
- [ ] **MERISTEM-BOUNDARY-R002.3.1:** Record the arbora owner of the ephemeral loopback OIDC certification authority.
- [ ] **MERISTEM-BOUNDARY-R002.3.2:** Repoint or remove the `certification_oidc` import in `skill_validation.py` and `skill_validation_assets.py`.
- [ ] **MERISTEM-BOUNDARY-R002.3.3:** Delete `scripts/certification/loopback_oidc.py` and `tests/unit/scripts/test_loopback_oidc.py`.
- [ ] **MERISTEM-BOUNDARY-R002.3.4:** Remove `certification_oidc.py` from the egress-boundary and release-wheel script path lists.
- [ ] **MERISTEM-BOUNDARY-R002.3.5:** Delete `meristem/deployment/certification_oidc.py`.
- [ ] **MERISTEM-BOUNDARY-R010.1:** Record that `status.json`'s `landed_in` (`2456745c1e4d`) did not delete the connector toolkit; LANDED is wrong.
- [ ] **MERISTEM-BOUNDARY-R010.2:** RETIRED: superseded by MERISTEM-BOUNDARY-R010.3...MERISTEM-BOUNDARY-R010.25.
- [ ] **MERISTEM-BOUNDARY-R010.3:** see requirements.md.
- [x] **MERISTEM-BOUNDARY-R010.4:** RETIRED: superseded by SCION-CONNECTOR-CONTROL-R780.5.
- [x] **MERISTEM-BOUNDARY-R010.5:** RETIRED: superseded by SCION-CONNECTOR-CONTROL-R780.6.
- [x] **MERISTEM-BOUNDARY-R010.6:** RETIRED: superseded by SCION-CONNECTOR-CONTROL-R780.6.
- [x] **MERISTEM-BOUNDARY-R010.7:** RETIRED: superseded by SCION-CONNECTOR-CONTROL-R780.6.
- [x] **MERISTEM-BOUNDARY-R010.8:** RETIRED: superseded by SCION-CONNECTOR-CONTROL-R780.6.
- [x] **MERISTEM-BOUNDARY-R010.9:** RETIRED: superseded by SCION-CONNECTOR-CONTROL-R780.6.
- [x] **MERISTEM-BOUNDARY-R010.10:** RETIRED: superseded by SCION-CONNECTOR-CONTROL-R780.6.
- [x] **MERISTEM-BOUNDARY-R010.11:** RETIRED: superseded by SCION-CONNECTOR-CONTROL-R780.6.
- [x] **MERISTEM-BOUNDARY-R010.12:** RETIRED: superseded by SCION-CONNECTOR-CONTROL-R780.6.
- [ ] **MERISTEM-BOUNDARY-R010.13:** see requirements.md.
- [ ] **MERISTEM-BOUNDARY-R010.14:** see requirements.md.
- [ ] **MERISTEM-BOUNDARY-R010.15:** see requirements.md.
- [ ] **MERISTEM-BOUNDARY-R010.17:** see requirements.md.
- [ ] **MERISTEM-BOUNDARY-R010.18:** see requirements.md.
- [ ] **MERISTEM-BOUNDARY-R010.19:** see requirements.md.
- [ ] **MERISTEM-BOUNDARY-R010.20:** see requirements.md.
- [ ] **MERISTEM-BOUNDARY-R010.21:** see requirements.md.
- [ ] **MERISTEM-BOUNDARY-R010.22:** see requirements.md.
- [ ] **MERISTEM-BOUNDARY-R010.24:** see requirements.md.
- [ ] **MERISTEM-BOUNDARY-R010.25:** see requirements.md.

## Decomposition children (tracked)

- [ ] **MERISTEM-BOUNDARY-R006.1:** Ship AU's F-J connector import-census gate
- [x] **MERISTEM-BOUNDARY-R006.2:** RETIRED: superseded by SCION-CONNECTOR-CONTROL-R010.
- [ ] **MERISTEM-BOUNDARY-R009.1:** Ship AU's T-Z connector import-census gate
- [x] **MERISTEM-BOUNDARY-R009.2:** RETIRED: superseded by SCION-CONNECTOR-CONTROL-R013.
- [ ] **MERISTEM-BOUNDARY-R016.1:** Gate AU's retained config.py against duplicate environment readers
- [x] **MERISTEM-BOUNDARY-R018.1:** Ship a typed inventory of the MERISTEM-BOUNDARY-R018 facade modules
- [ ] **MERISTEM-BOUNDARY-R018.2:** Delete the MERISTEM-BOUNDARY-R018 facade modules
- [x] **MERISTEM-BOUNDARY-R019.1:** Ship a typed inventory of the MERISTEM-BOUNDARY-R019 facade modules
- [ ] **MERISTEM-BOUNDARY-R019.2:** Delete the MERISTEM-BOUNDARY-R019 facade modules
- [x] **MERISTEM-BOUNDARY-R020.1:** Ship a typed inventory of the MERISTEM-BOUNDARY-R020 durable-work modules
- [ ] **MERISTEM-BOUNDARY-R020.2:** Move the MERISTEM-BOUNDARY-R020 durable-work modules and drop their scripts
- [x] **MERISTEM-BOUNDARY-R021.1:** Ship a typed inventory of the MERISTEM-BOUNDARY-R021 tenancy/admission modules
- [ ] **MERISTEM-BOUNDARY-R021.2:** Delete the MERISTEM-BOUNDARY-R021 tenancy/admission modules
- [x] **MERISTEM-BOUNDARY-R022.1:** Ship a typed inventory of the MERISTEM-BOUNDARY-R022 reasoning/analytics modules
- [ ] **MERISTEM-BOUNDARY-R022.2:** Delete the MERISTEM-BOUNDARY-R022 reasoning/analytics modules
- [ ] **MERISTEM-BOUNDARY-R011.1:** Record that R011's deletions (universal_connector, source_connectors, certification, manifest) have not started.
- [ ] **MERISTEM-BOUNDARY-R011.2:** ROLLUP: superseded by MERISTEM-BOUNDARY-R011.2.1...MERISTEM-BOUNDARY-R011.2.5.
- [x] **MERISTEM-BOUNDARY-R011.2.1:** RETIRED: superseded by SCION-SOURCE-INGEST-R900.
- [ ] **MERISTEM-BOUNDARY-R011.2.2:** Repoint `protocols/source_connectors/connectors/database.py` to the SDK connector.
- [ ] **MERISTEM-BOUNDARY-R011.2.3:** Repoint `tools/db_tools.py` to the SDK connector.
- [ ] **MERISTEM-BOUNDARY-R011.2.4:** Repoint/delete the universal_connector tests and docs references.
- [ ] **MERISTEM-BOUNDARY-R011.2.5:** Delete `protocols/universal_connector.py` once it has zero importers.
- [ ] **MERISTEM-BOUNDARY-R011.3:** Delete the `protocols/source_connectors/**` package and fix its importers.
- [ ] **MERISTEM-BOUNDARY-R011.4:** Delete `knowledge_graph/integrations/connector_certification.py`, `connector_certification_cli.py`, `connector_source_attestation.py` and fix their importers.
- [ ] **MERISTEM-BOUNDARY-R011.5:** Delete `knowledge_graph/ontology/connector_manifest.py`, `connector_manifest_gate.py`, `connector_manifests/**`, and drop `arbora-certify-connector`.
- [ ] **MERISTEM-BOUNDARY-R014.1:** Record that R014's moves, including `knowledge_graph/readiness.py`, have not started.
- [ ] **MERISTEM-BOUNDARY-R014.2:** Move `knowledge_graph/readiness.py` to arbora and delete the AU copy.
- [ ] **MERISTEM-BOUNDARY-R015.1:** Record that R015's moves, including `protocols/a2a_epistemic.py`, have not started.
- [ ] **MERISTEM-BOUNDARY-R015.2:** Delete `protocols/a2a_epistemic.py` and drop the `agent-utilities-acp` script.
- [ ] **MERISTEM-BOUNDARY-R024.1:** Record that R024's moves, including `source_sync.py` and `governance_import.py`, have not started.
- [ ] **MERISTEM-BOUNDARY-R024.2:** Move `knowledge_graph/core/source_sync.py` to the agent connector SDK and delete the AU copy.
- [ ] **MERISTEM-BOUNDARY-R024.3:** Move `knowledge_graph/governance_import.py` to the SDK and delete the AU copy.
- [x] **MERISTEM-BOUNDARY-R043.1:** Restore the `graph/` run loop over the host decision port.
- [x] **MERISTEM-BOUNDARY-R043.2:** Add the fallback plan and the fail-closed refusals.
- [x] **MERISTEM-BOUNDARY-R043.3:** Restore the `rlm/` recursive run loop.

- [x] **MERISTEM-BOUNDARY-R043.4:** Restore all five model-edited scaffold children; do not substitute execution steps.
- [x] **MERISTEM-BOUNDARY-R043.4.1:** Implement per-run whole-list scaffold state and immutable items.
- [x] **MERISTEM-BOUNDARY-R043.4.2:** Wire the validated update tool under a received scaffold decision.
- [x] **MERISTEM-BOUNDARY-R043.4.3:** Inject into each outgoing request without retaining scaffold text in history.
- [x] **MERISTEM-BOUNDARY-R043.4.4:** Prove the scaffold has no planner or selection authority.
- [x] **MERISTEM-BOUNDARY-R043.4.5:** Prove two updates through the actual run loop.
- [ ] Assign SDK snapshot and EG plan-record contracts before the host-port producer bridge; preserve the revision, scope, expiry and refusal obligations in the amendment.
- [ ] Wire Arbora's ARBORA-OPS-R021.4.5 reader to the producer and complete served inspector acceptance.

- [x] **MERISTEM-BOUNDARY-R043.5.1:** Add atomic scaffold revision/read semantics before the owner-composed snapshot bridge.

- [x] **MERISTEM-BOUNDARY-R043.5.2:** Publish actual atomic scaffold revisions through an optional run-local host binding with lifecycle, isolation and refusal tests.

- [x] **MERISTEM-BOUNDARY-R043.6:** Run the received DAG with bounded rolling slots.
- [ ] **MERISTEM-BOUNDARY-R043.6.1:** Add the tool step model and the injected ToolCaller port.
- [ ] **MERISTEM-BOUNDARY-R043.6.2:** Resolve bindings from completed outputs without a model.
- [ ] **MERISTEM-BOUNDARY-R043.6.3:** Fail a step with a typed reason and no model retry.
- [ ] **MERISTEM-BOUNDARY-R043.6.4:** Record resolved arguments and result per tool step.
- [ ] **MERISTEM-BOUNDARY-R043.6.5:** Run mixed prompt and tool plans on one scheduler.
- [ ] **MERISTEM-BOUNDARY-R043.6.6:** Wire tool steps into the dependency executor.
- [ ] **MERISTEM-BOUNDARY-R043.6.7:** Prove tool steps through the public run API.
- [x] **MERISTEM-BOUNDARY-R043.7:** Reserve aggregate limits and drain child lifecycle.
- [ ] **MERISTEM-BOUNDARY-R043.8:** Compose the host checkpoint port with dependency plans and shared budgets; qualify restart against the EG native checkpoint producer.
- [ ] **MERISTEM-BOUNDARY-R043.9:** Restore completed conversation history from a host-authorized checkpoint.
- [ ] **MERISTEM-BOUNDARY-R043.10:** Publish model and tool events through the durable host event port.
- [ ] **MERISTEM-BOUNDARY-R043.11:** Bind host-selected client tools in the real run loop and prove bounded forwarding without local effects.

- [x] Add MERISTEM-BOUNDARY-R043.12 for the existing bounded measurement producer and bind its local regression tests.
- [ ] Qualify live-provider and served-consumer behavior separately; no such acceptance is claimed here.

- [ ] **MERISTEM-BOUNDARY-R043.9.1:** Validate host-bound completed text history and hydrate sequential fresh turns without prior budgets.

- [ ] **MERISTEM-BOUNDARY-R043.9.2:** Complete checkpoint-backed history, current-run checkpoint precedence and installed EG/Arbora restart-continuation acceptance. Depends on R043.8 and the native checkpoint producer.

- [ ] **MERISTEM-BOUNDARY-R043.9.2.1:** Validate one trusted checkpoint selection before planning or harness effects. Refuse unsupported current-run restore without fallback. Preserve fresh completed-text hydration.
- [ ] **MERISTEM-BOUNDARY-R043.9.2.2:** Implement durable checkpoint writer and safe restoration with shared budgets and dependency plans. Qualify native consistent selection and served restart acceptance.

- [ ] `MERISTEM-BOUNDARY-R043.13` Deliver received text-only execution and installed acceptance.
- [ ] `MERISTEM-BOUNDARY-R043.13.1` Enforce the received constraint inside the real model loop with simulated HTTP tests.
- [ ] `MERISTEM-BOUNDARY-R043.13.2` Qualify native ownership, durable outcomes and provider execution with configured admission bounds.

## Shared child execution explorer

- [ ] `MERISTEM-BOUNDARY-R043.14` — Emit explicit admitted child lineage without converting step IDs to conversation targets.
- [ ] `MERISTEM-BOUNDARY-R043.15` — Apply host-admitted active-child input at model-turn boundaries within original limits.
- [ ] `MERISTEM-BOUNDARY-R043.16` — Consume supported selected child history for a fresh turn; keep unsupported checkpoint/dependency forms refused.
- [ ] `MERISTEM-BOUNDARY-R043.17` — Publish permitted child records through the existing durable host event seam.
