# Tasks and status

**Legend:** TODO = no accepted implementation proof; IN PROGRESS = linked change under review; IMPLEMENTED = code at merged head with source proof; VERIFIED = acceptance tests at that head; ACCEPTED = owner sign-off and release proof. A task is not completed by a draft branch.

| Task | IDs | State | Done when |
|---|---|---|---|
| Served shell and caller cut | AU-BOUNDARY-R001, AU-BOUNDARY-R002, AU-BOUNDARY-R003, AU-BOUNDARY-R004, AU-BOUNDARY-R013, AU-BOUNDARY-R014, AU-BOUNDARY-R015, AU-BOUNDARY-R016, AU-BOUNDARY-R017, AU-BOUNDARY-R039 | TODO | graph-os parity plus AU deletion and script/import census |
| Connector and source cut | AU-BOUNDARY-R005, AU-BOUNDARY-R006, AU-BOUNDARY-R007, AU-BOUNDARY-R008, AU-BOUNDARY-R009, AU-BOUNDARY-R010, AU-BOUNDARY-R011, AU-BOUNDARY-R024, AU-BOUNDARY-R025, AU-BOUNDARY-R026 | TODO | SDK transport/pack certification plus all AU caller deletions |
| Graph, schema, retrieval, memory cut | AU-BOUNDARY-R012, AU-BOUNDARY-R018, AU-BOUNDARY-R019, AU-BOUNDARY-R020, AU-BOUNDARY-R021, AU-BOUNDARY-R022, AU-BOUNDARY-R023, AU-BOUNDARY-R027, AU-BOUNDARY-R028, AU-BOUNDARY-R029, AU-BOUNDARY-R030, AU-BOUNDARY-R031, AU-BOUNDARY-R032, AU-BOUNDARY-R033, AU-BOUNDARY-R034, AU-BOUNDARY-R035, AU-BOUNDARY-R040 | TODO | generated EG method parity, trusted migration and AU adapter-only shape |
| Usage and governance cut | AU-BOUNDARY-R036, AU-BOUNDARY-R037 | TODO | EG durable usage and repository-manager governance parity |
| Fabricated-data finance module removal | AU-BOUNDARY-R041 | TODO | import census shows no listed module and no caller |
| Inventory completeness check | AU-BOUNDARY-R042 | TODO | tree-versus-inventory test fails on an unlisted directory, an undecided row or a stale path |
| Retained agent surface | AU-BOUNDARY-R043 | TODO | import-boundary test over the kept packages |
| Fleet, scaling and deployment modules | AU-BOUNDARY-R044 | TODO | caller census and graph-os consumer tests |
| Knowledge-graph file-level owners | AU-BOUNDARY-R045 | TODO | every tracked file under `knowledge_graph/` maps to one owner; parity tests per move |
| File-level owners for partly covered packages | AU-BOUNDARY-R046 | TODO | no tracked file without an owner; no requirement by description only |
| Engine-facing adapters and caches | AU-BOUNDARY-R047 | TODO | differential tests against the engine; no second store or kernel |
| Domain models, shapes, assets and skill content | AU-BOUNDARY-R048 | TODO | each directory placed; pack or skill validation in the receiving repository |
| Permanent owner guard | AU-BOUNDARY-R038 | TODO | generated manifest, CI gate and exact-head check |
| Quality gates and evidence | all of the above | TODO | CCCC, KISS, Dupehound, jscpd, language linters and the full test suite pass at the merged head and the result is recorded in [evidence.md](evidence.md) |

Each row expands into a PR checklist with changed paths, owner operation, positive and negative test IDs, scanner delta, review, merged commit and acceptance artifact. Do not move a row to VERIFIED solely because another row in its range passed. Every ID is defined in [requirements.md](requirements.md); the directories each requirement removes or relocates are listed in the [deletion and relocation inventory](coverage.md#deletion-and-relocation-inventory).
