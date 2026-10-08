# Tasks and delivery states

**Legend:** TODO means proof missing; IN PROGRESS means a change exists; IMPLEMENTED means source on merged main; VERIFIED means required gates passed on that exact head; ACCEPTED means owner release review is recorded; SUPERSEDED/REJECTED require their own decision evidence.

- [ ] Create the reserved `__control__` graph automatically (or return a typed error) before the first WorkItem runs on a fresh engine, and thread `messaging_intake_enabled` through to the backend that applies it; add positive and typed-refusal tests at a real entry point. Closes AU-INTEGRATION-R001, AU-INTEGRATION-R002.
- [ ] Audit AU's previously tracked dependency, contract-generation and test-baseline fixes against the current merged head; close any duplicate pending work that the code and its tests already cover. Closes AU-INTEGRATION-R003.
- [ ] Remove the unintended `nltk` dependency from AU's base lock, point the contract-compatibility hook at a pinned published EG artifact instead of a sibling checkout, and keep `epistemic-graph[full]` a core, always-installed dependency. Closes AU-INTEGRATION-R004, AU-INTEGRATION-R005, AU-INTEGRATION-R006.
- [ ] Confirm gold-set generation for retrieval/evaluation stays synthetic by construction with no parallel manual-labelling pipeline. Closes AU-INTEGRATION-R007.
- [ ] Keep the regression baseline (lazy config resolution, resource-lease fencing, session boundary, model-factory routing, gateway, deployment-backend threading) green with EG installed, and resolve every expired liveness deferral with recorded evidence. Closes AU-INTEGRATION-R008, AU-INTEGRATION-R009.
- [ ] Run the engine-owned Kafka enqueue-only-proof contract to completion across its build, unconstrained and constrained steps, and report the result as external dependency evidence rather than AU-authored proof. Closes AU-INTEGRATION-R010.
- [ ] Before pruning a development branch or local checkout, prove its commits and any uncommitted diff are already contained in an accepted public branch or recorded in an explicit recovery artifact. Closes AU-INTEGRATION-R011.
- [ ] Keep the external-graph contract check covering the two relocated documents with an equivalent content-marker and environment-literal check in their new home, and make the privacy gate's author-identity scanner accept the exact canonical automation identities while still rejecting unrelated bare tokens, credentials and private endpoints. Closes AU-INTEGRATION-R012, AU-INTEGRATION-R013.
- [ ] Benchmark any proposed AI predicate-grouping, prompt/KV reuse or GPU-fairness improvement against AU's existing typed query/model-serving path, with tenant and purpose authorization gates enforced, before adopting it as an optimization inside that path. Closes AU-INTEGRATION-R014.
- [ ] Remove the unused legacy ontology-publisher surface (module, Fuseki/Stardog schedule token, vendor configuration, doctor/health/widget surfaces and references) once a full module and catalog inventory shows no live caller. Closes AU-INTEGRATION-R015.
  - [x] Delete the `ontology_publisher` module and its documentation references. The module has no live importer on main. Re-cut of commit f055b24c4 (PR #13).
  - [ ] Remove the `KG_FUSEKI_PUBLISH` configuration, the `fuseki_publish` schedule token, and the profile-guard and health surfaces that reference them.
- [ ] Confirm the publicly published ontology-deletion commit's patch content matches its source change and that both the focused and expanded ontology overlay test suites pass at that commit. Closes AU-INTEGRATION-R016.
- [ ] Confirm AU's publicly published source history is fully contained in the ancestry of local main. Closes AU-INTEGRATION-R017.
- [ ] Bring the public documentation site to full parity with the canonical README/theme/home-page tense, zero orphan pages with full page-count parity, a strict MkDocs build, and passing current-only, privacy, accessibility, theme, workflow and version gates. Closes AU-INTEGRATION-R018.
- [x] Replace the literal 25 s direct budget with `MESSAGING_DIRECT_REPLY_BUDGET_S` and deliver over-budget messaging replies as a follow-up. Closes AU-INTEGRATION-R019.
- [ ] Roll out AU-INTEGRATION-R019 to the live graph-os deployment and confirm a slow Telegram chat turn receives its follow-up reply.
- [ ] Run the full test baseline, CCCC, jscpd, Dupehound, KISS review, Ruff/mypy, and the privacy/version-consistency scripts; record exact merged-head and hosted CI evidence against every requirement above before any `ACCEPTED` claim.

Four additional AU workstreams (final composition, ontology/SHACL/OWL clean cut, public application control plane, public docs) are named in `spec.md` but still await stable requirement IDs through review; track them there rather than inventing a synthetic ID here.
- [ ] Unwrap the run envelope at one chokepoint for outbound chat text. Closes AU-INTEGRATION-R020.
- [ ] Reuse `unwrap_run_envelope` in the PR #60 late-reply delivery path once both PRs land. Tracks AU-INTEGRATION-R020.
