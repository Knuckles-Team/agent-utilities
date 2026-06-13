# Tasks: Autonomous Code-Synthesis Stage for Promoted Gaps (AHE-3.22)

Wire-first: extend the existing promotion→ChangeSet hop before adding any module.

- [ ] **T1 — Surface attribution on the gap topic.** In
  `knowledge_graph/adaptation/failure_analyzer.py`, carry `component_type` + `file_path` through
  `FailurePattern` and `file_gap_topic` (lines 119, 167) onto the `failure_gap` Concept so a promoted
  proposal exposes the target the generator needs. (Attribution already exists in EvolveAgent's
  `_propose_edits_for_cluster`; this just persists it onto the proposal node.)
- [ ] **T2 — Add the generator seam in change_synthesis.** In
  `knowledge_graph/research/change_synthesis.py:synthesize_change_set` (line 384), before the prose
  branch (line 416): when `extract_embedded_files` is empty AND the proposal carries a resolvable
  `file_path`/`component_type`, call an optional `code_generator` (default an `EvolveAgent` adapter)
  to produce one `FileChange`, assign it to `files`, and let the **existing** `kind="code"` branch
  (line 402) run. No new pipeline, no second publisher.
- [ ] **T3 — Adapt EvolveAgent as that generator.** Wrap `harness/evolve_agent.py:EvolveAgent`
  (its `ComponentEdit{file_path, diff_content}` from manifest.py:48, single-file edit at
  evolve_agent.py:267/413) behind a tiny `propose_file_change(proposal) -> FileChange | None`
  adapter; constrain to one file per proposal (v1). Reuse, do not rebuild, its file read/write.
- [ ] **T4 — Keep the sandbox gate authoritative.** Confirm the generated `files` pass through the
  unchanged `validate_in_sandbox` (line 299) and `publishable` (line 100); add no bypass. A failing
  diff stays unpublished.
- [ ] **T5 — Live-path test.** Add `tests/unit/knowledge_graph/research/test_ahe_3_22_code_synthesis.py`
  (`@pytest.mark.concept(id="AHE-3.22")`): stub generator → assert `kind="code"` with the attributed
  path; un-attributed proposal → still `kind="sdd_plan"`; failing diff → not publishable.
- [ ] **T6 — Concept + docs + gates.** Tag `CONCEPT:AHE-3.22` in the seam; run
  `scripts/build_concepts_yaml.py` and `scripts/check_concepts.py`; extend
  `docs/guides/autonomous-evolution.md` with the single-file v1 constraint and the safety envelope.
- [ ] **T7 — Finish.** `pre-commit run --all-files` green (fix all, incl. pre-existing); commit in a
  worktree; merge to main locally.
