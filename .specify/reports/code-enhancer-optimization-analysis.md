# Code-Enhancer Skill — Optimization Analysis (for 60-repo scale + KG-native)

**Date:** 2026-06-05. **Skill:** `universal_skills/core/code-enhancer` (the Claude
`~/.claude/skills/code-enhancer` is a **symlink** to it — one physical copy, edit source only).
**Grounding:** real runs on `agent-utilities` (Python) + `epistemic-graph` (Rust), this session.

## 1. What the skill is

A 28-domain "Code Enhancement Review": **25 Python scripts** (~10k LOC) + 16 references + an 11-step
SKILL.md workflow. Language-agnostic (Python/Go/Node/Rust/Java). `run_multi_project.py` fans out
across repos with `-c` concurrency. Output: graded 0–100 report + SDD handoff, with a stated intent
to be KG-backed (`kg_query`/`kg_ingest`).

## 2. What actually works (verified)

- `detect_language.py` — correct primary language (python on AU, **rust** on EG), build systems, test
  frameworks, linters. Fast.
- `analyze_codebase.py` — real, useful output (AU scored **42/F**, correctly flagged 50 functions
  >200 lines incl. `_build_server` 1467L). ~11s on a large repo.
- `run_multi_project.py` — async orchestration with concurrency cap; correct CLI
  (`pattern -c N -o DIR`).

## 3. Defects & gaps found (the "needs love") — evidence-backed

| # | Finding | Evidence | Impact at 60-repo scale |
|---|---|---|---|
| **D1** | **Cross-language contamination in detection.** Rust `epistemic-graph` reports build systems `['pip','pyproject','cargo']` and Python linters — false Python positives. | `detect_language epistemic-graph` → `build:['pip','pyproject','cargo']` | Mis-routes analysis (runs ruff/mypy on Rust repos); pollutes 60-repo aggregate. |
| **D2** | **No `--self-test` on any script** (0/25). The sibling `comparative-analysis` skill has `--self-test` on *all* scripts. | `grep -l self-test scripts/*.py` → 0 | No way to smoke-validate the toolchain before a 60-repo batch; silent breakage. |
| **D3** | **Inconsistent CLI surface.** 6 scripts use `argparse`, 19 read `sys.argv[1]` ad-hoc (no `--help`, `--json`, `--output`). | `argparse:6 / sys.argv-only:19` | Can't drive 25 scripts uniformly from one harness; fragile automation. |
| **D4** | **No single headless per-repo driver.** All-28-domains orchestration lives *only* in the agent following 11 SKILL.md steps; there's no `enhance_repo.py <path> --json` that runs everything and emits one report. `run_multi_project` exists but per-repo coverage is implicit. | no `run_all`/`driver`/`enhance` script | 60-repo CI/cron needs one headless command per repo; today it needs an agent in the loop. |
| **D5** | **KG integration is a single fragile import, not a mode.** Only `analyze_xdg_kg.py` imports `agent_utilities`; in the live run it failed `No module named 'agent_utilities.core'` and silently skipped. Despite "evidence must come from the KG" in Best Practices, **the analysis is filesystem-only**; results are never ingested into graph-os, and nothing is queried cross-repo. | run log: `Skipping XDG KG check due to missing dependency` | The headline "KG-native" capability does not exist yet; 60-repo cross-repo insight is impossible without it. |
| **D6** | **Poor long-run progress visibility.** The multi-project run produced no incremental per-project artifact for minutes (only `orchestrator.log`). | bg run: only `orchestrator.log` after 5+ min | A 60-repo run is opaque; a stall is indistinguishable from slow. |
| **D7** | **SKILL.md drift.** SKILL.md lists ~30 scripts/steps (e.g. `analyze_tests.py`, `analyze_version_sync.py`, `analyze_architecture.py` in resources) but 25 `.py` exist; "28 domains" vs the actual script set is unverified. | `ls scripts/*.py` → 25 | Wasted agent turns invoking missing scripts; erodes trust. |

## 4. Optimization plan (prioritized for 60-repo scale)

**P0 — make it batch-safe and headless**
1. **`enhance_repo.py <path> [--json] [--domains a,b] [--out DIR]`** — one headless driver that runs
   every domain for one repo, tolerant of per-domain failure (each domain → `{score, grade,
   findings, error?}`), emits one JSON + one markdown. `run_multi_project` calls it per repo. (Fixes
   D4; enables CI/cron over 60 repos.)
2. **`--self-test` on every script** (mirror comparative-analysis): dependency-free smoke check that
   returns a known result on a tiny fixture. Add a `make selftest` that runs all 25. (Fixes D2.)
3. **Uniform CLI**: every script `argparse` with `path`, `--json`, `--out`; a shared
   `_cli(analyze_fn)` helper. (Fixes D3.)

**P1 — correctness at scale**
4. **Language gating** (Fixes D1): `detect_language` must not report Python build/linters for a repo
   whose primary is Rust unless a real `pyproject.toml`/`setup.py` is present *at root*; downstream
   scripts skip domains not applicable to the detected primary language.
5. **Incremental output + heartbeat** (Fixes D6): write each per-repo report as it completes; emit a
   progress line per domain; `--fail-fast` / `--timeout-per-domain`.
6. **SKILL.md reconciliation** (Fixes D7): regenerate the script/step list from the actual `scripts/`
   dir; assert in `--self-test` that every SKILL-referenced script exists.

**P2 — KG-native mode (the north-star, Fixes D5)** — opt-in `--kg`:
7. **Ingest**: after each repo's report, `kg_ingest` the JSON as `CodeEnhancementRun` +
   `DomainScore` + `Finding` nodes linked to `Repo`, via the **graph-os MCP** (`graph_ingest` /
   `graph_write`) — not a direct `agent_utilities.core` import (which is what broke). Talk to the
   running MCP server, so the skill has no hard dependency on the package internals.
8. **Cross-repo queries**: `graph_query` for "which of the 60 repos regressed since last run",
   "shared CWE across repos", "concept-ID collisions ecosystem-wide" (CE-029 already wants this) —
   the genuinely new value a filesystem tool can't give.
9. **Trend/Δ**: store each run bi-temporally (KG-2.11) so the report shows score deltas vs the prior
   run per repo — turns a one-shot grade into a tracked signal across 60 repos.

## 5. Already shipped this session (proof-of-direction)

- **`skill_installer --symlink`** — installs skills by symlink into `~/.claude/skills` etc. instead
  of copying: no duplicate files, auto-updates on `pip install -U`. Idempotent; falls back to copy.
  This is the same pattern `code-enhancer`/`skill-installer` already use as symlinks, now exposed to
  users. (`universal_skills/core/skill_installer/scripts/install.py`, verified: symlink + copy +
  idempotent branches.)

## 6. Recommendation

The skill is genuinely useful (the analyzers produce real, cited findings) but is **not yet
batch-safe or KG-native** — the two things your 60-repo goal needs. Do **P0 first** (headless driver
+ self-tests + uniform CLI) so a 60-repo run is one reliable command; then **P2** (graph-os MCP
ingest + cross-repo queries + bi-temporal trends) to make it the KG-native ecosystem auditor. P1 is
correctness glue in between. Each is a contained PR; suggest sequencing P0 now, then KG-native as its
own focused effort.
