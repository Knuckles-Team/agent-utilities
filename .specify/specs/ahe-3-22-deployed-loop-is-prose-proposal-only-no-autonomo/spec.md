# Spec: Autonomous Code-Synthesis Stage for Promoted Gaps (AHE-3.22)

> Status: **proposed**.
> **Wire-First:** extend `knowledge_graph/research/change_synthesis.py:synthesize_change_set`
> (the existing promotion→ChangeSet hop) so it asks `harness/evolve_agent.py:EvolveAgent`
> to emit `files` for a component-attributed proposal BEFORE the prose fallback fires —
> reusing the unchanged `change_synthesis → validate_in_sandbox → change_publisher` pipeline.
> Do not add a parallel path; this only feeds the `kind="code"` branch that already exists.

## Pre-Flight Checklist
- [x] **Extension target identified** — `synthesize_change_set` (change_synthesis.py:384) today
  takes `kind="code"` only when `extract_embedded_files` (line 148) finds a `files`/`files_json`
  list; the deployed loop (`golden_loop.py:_synthesize_team` / auto_merge promotion) never
  populates it, so every live proposal falls to the `kind="sdd_plan"` prose skeleton (line 416).
- [x] **New CONCEPT:AU-AHE.harness.single-file-code-synthesis justified** — the *genotypic-RSI generator* is a distinct capability
  from the AHE-3.21 materializer (which assumes the diff exists) and AU-AHE.harness.failure-evolution gap-intake; it is
  the missing producer of the diff, not a new pipeline.
- [x] **Wire-First confirmed** — 1 generator hop inserted inside `synthesize_change_set`, before the
  prose branch; `EvolveAgent` already proposes `ComponentEdit{component_type, file_path, diff_content}`
  (manifest.py:48) and reads/writes the target file (evolve_agent.py:267, 324, 413).
- [x] **Success metric defined** — a component-attributed `failure_gap`/concept proposal yields a
  `kind="code"` ChangeSet with ≥1 `FileChange` whose `path` is the attributed `file_path`, and the
  existing sandbox gate runs on it (publishable iff it passes). Prose proposals stay `kind="sdd_plan"`.

## User Stories

### US-1 — A component-attributed proposal generates a real code diff
**As** the deployed golden loop, **I want** a promoted proposal whose target component is known
to be turned into an actual single-file edit, **so that** self-improvement emits code, not only a spec.
- **AC1**: `synthesize_change_set` accepts an optional code-generator seam (default: `EvolveAgent`);
  when a proposal carries a resolvable `file_path` + `component_type` (from AU-AHE.harness.failure-evolution attribution) and
  has no embedded `files`, the generator reads that file and returns one `{path, content}` edit, which
  is set on the proposal's `files` so the **existing** `kind="code"` branch is taken unchanged.
- **AC2**: the generated change is **constrained to a single, component-attributed file** (v1 limit);
  a proposal with no resolvable target, or a multi-file ask, falls through to the prose `sdd_plan`
  skeleton exactly as today (zero behavior change for prose proposals).
- **AC3**: the generated `files` flow through the **unchanged** `validate_in_sandbox` gate; a
  diff that fails syntax/import validation yields `publishable == False` and is never branched.

### US-2 — Safe, governed, and opt-in by default
**As** an operator, **I want** the generator gated and default-conservative, **so that** autonomous
code emission cannot bypass review.
- **AC4**: code generation activates only when a generator is wired and the proposal is
  component-attributed; with no generator the function behaves byte-for-byte as today (prose path).
- **AC5**: the emitted `ChangeSet` records the originating `ComponentEdit` provenance (component_type,
  edit_summary) in its `summary`/`concept_ids`, and reaches the publisher through the **same**
  `governed_publish` ActionPolicy `merge_promotion` gate — no new publish path.

## Non-Functional Requirements
- `tests/unit/knowledge_graph/research/test_ahe_3_22_code_synthesis.py`
  (`@pytest.mark.concept(id="AHE-3.22")`), ≤60s, no live engine/LLM: a stub generator returns a
  `ComponentEdit`; assert `synthesize_change_set` yields `kind="code"` with the attributed path, that
  an un-attributed proposal still yields `kind="sdd_plan"`, and that a failing diff is not publishable.
- `pre-commit run --all-files` green; `scripts/build_concepts_yaml.py` re-run so AHE-3.22 lands in
  `docs/concepts.yaml`; `scripts/check_concepts.py` passes.
- Per-concept doc authored under `docs/guides/` (extend `autonomous-evolution.md`), citing the
  single-file v1 constraint and the sandbox + ActionPolicy + full-suite gates as the safety envelope.
