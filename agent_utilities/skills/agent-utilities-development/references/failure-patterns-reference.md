# Agent-utilities failure-pattern reference

Deep reference for `agent-utilities-development`: the recorded AU/EG incidents
that were promoted into rules, each paired with what actually went wrong. These
expand the mandatory guardrail table in the parent
[`SKILL.md`](../SKILL.md); open this when a number, a gate result, a review
blocker, or a hook comment looks authoritative and you are about to act on it.
Italicised section names below (*Validate*, *Before you add anything*) are
sections of that parent `SKILL.md`.

## Failure patterns that are now rules

Each is drawn from a recorded AU/EG incident, not from principle.

| Rule | What went wrong |
|---|---|
| **A scanner counter moving down is not acceptance evidence.** | A raw-SCC reduction is explicitly *not* runtime proof (`plans/refactor/evidence/reports/AU-SCC-DECOMPOSITION.md`); the 162-edge "cut budget" is a nominal planning sum, `expected_delta: OPEN`. A predicate fix that repaired 3 failures **regressed the suite from 15 to 27** and was reverted. "A count without a command behind it is not admissible." |
| **A gate's universe is not the package's universe.** | The `mypy` hook sets `exclude: ^(tests/\|test/\|scripts/\|script/)`, `--ignore-missing-imports`, and exactly **three** `additional_dependencies` (`pydantic`, `types-PyYAML`, `types-requests`). Package-wide mypy and the hook therefore report different numbers, both correctly. **Scope work from the HOOK's output, never a package-wide count.** Same shape for `vulture` (`--exclude … tests`) and `ruff`/`bandit`/`codespell`. |
| **`uv run` is poisoned here — and it rewrites the lock.** | See *Validate* below for the interpreter trap. Additionally: `uv run`/`uv sync` re-synchronise the environment named by `UV_PROJECT_ENVIRONMENT` on **every** invocation and will rewrite `uv.lock`; the `uv-lock` hook passes `--locked` precisely so an out-of-sync lock **fails instead of being silently rewritten**. Diff `uv.lock` before and after every test run — a clean diff is part of the verification, and the AU blocker report treats "`uv.lock` untouched" as a reported result. |
| **Extraction makes latent type errors expressible.** | It does not introduce them: an inline expression carries no annotation, a signature does, and a narrowing does not survive a function boundary (`plans/complex/waves/wD11/PREAMBLE.md`). Expect new mypy errors from a pure extraction and fix them; do not `type: ignore` them. |
| **Never commit under an ambient identity.** | ★ **CORRECTED — the workspace note is stale.** `scripts/check_tracked_privacy.py` derives banned identifiers from `git config user.name`/`user.email` (the calling process's identity). The `git log -1 --format=%an` fallback **was removed on 2026-08-17** after it produced a ~450-line, ~180-file false-positive flood — HEAD's author on `main` was literally the word "claude", so every doc mentioning the tool became a manufactured leak. The rule survives in a different form: **set a real, neutral `git config user.name`/`user.email` in the lane**, and pin `AGENT_UTILITIES_PRIVACY_IDENTIFIERS` when a sandbox's ambient identity is a common word. |
| **A named reference implementation is a copy instruction.** | If a brief names an exemplar, it must also name the **shared home** — **extract the primitive before the fan-out, not after.** The incident this is drawn from (the eight near-identical scope-identity builders) is stated once, in *Before you add anything* → checklist item 3. |
| **File partitioning prevents edit collisions, not pattern collisions.** | Parallel lanes never touched each other's files and still converged on the same duplicated helper. Partition by file *and* name the shared primitive. |
| **A predicate consulted by both a validator and a producer is answering two questions.** | `is_native_scope_domain` was split for exactly this, and its replacement `requires_native_scope` immediately reproduced it one level up. Assume two callers want two different answers until proven otherwise. |
| **Verify the blocker before you work it.** | Of nine strict-review blockers on the AU candidate, **one was factually wrong on inspection** — "the fifth recorded work item in this program to prove stale or false". A "compatibility test the review wanted deleted" turned out to be the thing pinning deliberate staged work. Read the code the blocker names before you act on it. |
| **A hook's comment can outlive the gate's behaviour.** | Two `.config/pre-commit.yaml` comments still describe `scripts/openapi_coverage_baseline.json` and `scripts/wire_first_baseline.json` ratchets. **Neither file exists**; both baselines were retired 2026-08-28 under the workspace NO-RATCHETS rule and their `--update-*` flags now exit 2. Read the gate **script's** docstring, not the hook comment, before you reason about what a gate enforces. |
| **Report the denominator with every coverage result.** | Three competing fleet denominators (55 / 71 / 73) were in flight; the reconciled figure is **78** `agent-packages` repositories, digest-bound `sha256:56d9bdc5…`, out of 236 declared rows. A gate reporting coverage against the wrong universe reports coverage it does not have. |
