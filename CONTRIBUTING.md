# Contributing to agent-utilities

Thanks for contributing! This is the Python harness (the 5-pillar platform); the
high-performance graph compute lives in the separate Rust engine
[`epistemic-graph`](https://github.com/Knuckles-Team/epistemic-graph), reached
out-of-process over MessagePack/UDS (no PyO3).

## Development setup

From a fresh clone (locally, or automatically in a Claude Code cloud session,
where `.claude/hooks/session-start.sh` runs it):

```bash
scripts/bootstrap.sh              # pinned siblings, uv >= 0.9, pinned Python, locked .venv, git hooks
scripts/bootstrap.sh --scanners   # also the pinned dupehound/jscpd clone scanners (needs cargo + npm)
scripts/bootstrap.sh --engine     # also build the native epistemic-graph engine from source (slow)
```

The bootstrap is idempotent. It checks out the sibling sources `uv.lock` needs
at the commits pinned in `scripts/siblings.lock` (the same pins CI uses), syncs
`.venv` with `uv sync --frozen` and installs the pre-commit and pre-push hooks
from `.config/pre-commit.yaml`. Hooks run under the repository `.venv` through
`scripts/hook_python.sh`, so a git hook, `uvx pre-commit run` and CI see the same
dependencies:

```bash
uvx pre-commit run --config .config/pre-commit.yaml --all-files
uv run --no-sync pytest tests/unit/<path> -q
```

A gate whose tool or sibling checkout is missing (a native scanner, the engine,
the provider fleet, the ecosystem workspace) prints `SKIPPED (<gate>): <reason>`
and passes locally; under CI (`CI` set) the same gate exits 2 (CANNOT RUN), so
CI must provision it.

The default knowledge-graph backend is zero-infra: the epistemic-graph engine is
the one authority (compute + cache + semantic + durable persistence), so most work
needs no external services. For an optional pg-age mirror set `GRAPH_BACKEND=fanout`
+ `GRAPH_MIRROR_TARGETS` and `GRAPH_DB_URI` (Postgres/pg-age).

## Branch / worktree workflow

Work on a topic branch and open a pull request against `main`:

```bash
git switch -c <topic> origin/main
# edit, run the focused tests and the hooks, commit
git push -u origin <topic>
gh pr create --draft --base main
```

Hosted CI runs the same pre-commit configuration plus the release gates on the
pull request; mark it ready for review once it is green.

On a shared host where several agents and people use one checkout, **do not
edit the canonical checkout** at `agent-packages/agent-utilities`: a background
sync can reset its working tree, and the `lane-guard` hook refuses commits
there. Take your own git worktree on your own branch:

```bash
rm_worktree add agent-utilities <your-branch>     # repository-manager MCP, or:
git worktree add ${XDG_STATE_HOME}/repository-worktrees/agent-utilities/<branch> -b <branch> main
```

Commit early and often (commits survive a working-tree reset). Push only when
asked.

## Before you push

The installed pre-push gate is deliberately bounded for a sub-10-minute
publication cycle: lint/format/type checks, lockfile consistency, public-surface
and fast contract checks, plus targeted smoke coverage. Full pytest/integration
suites, workflow replay, wheel builds, and repository-wide scans are manual or
hosted-CI validations; they are not run automatically by `git push`.

```bash
python -m pytest                              # unit suite (keep it green)
python3 scripts/safe_precommit_all_files.py   # use this instead of direct `pre-commit run --config .config/pre-commit.yaml --all-files`
```

Note: the full pytest hook is manual-only and can fail repo-wide due to an
unrelated egeria/py3.12 dependency pin — validate with the system
`python -m pytest` if so.

⚠ **Use the safe wrapper, not direct `pre-commit run --config .config/pre-commit.yaml --all-files` (D-OB-12).**
`--all-files` stashes every unstaged change before running hooks and restores it
after; a file-rewriting hook (`ruff-format`, `turtle-format`, …) touching the same
path can make that restore silently drop the unstaged edit — and
`docs/concept_reservations.yaml` is a shared, cross-session ledger deliberately
left unstaged, so a careless run can destroy another session's reservations.
`scripts/safe_precommit_all_files.py` backs up your unstaged diff first and
verifies it's still there afterward. See `AGENTS.md`'s *Quality Bar* section for
the full explanation and recovery steps.

### Guardrail ENV parity (passes-local / fails-CI)

`pre-commit run --config .config/pre-commit.yaml --all-files` runs the
guardrail gates in your **full** install.
CI's `release.yml` `gates` job runs them in a deliberately **lean** install
(`uv sync --frozen --extra test --group guardrails --no-install-package
epistemic-graph --no-install-package langfuse-agent` — no `[agent-runtime]`/`[all]`
extras). A gate that transitively imports an extra-only dependency (`pydantic_ai`,
`httpx`, `fastmcp`, …) therefore **passes locally but dies in CI**. To catch that
class locally, reproduce CI's lean env and run every gate inside it:

```bash
pre-commit run --config .config/pre-commit.yaml guardrails-lean-parity --hook-stage manual --all-files
# or directly (requires `uv`):
python scripts/run_guardrails_lean.py            # --list to preview, --keep-venv to debug
```

`scripts/run_guardrails_lean.py` builds a throwaway lean venv with the **exact**
install from `.github/workflows/release.yml`'s `gates` job and runs the gate list
**derived from the same job** in it (so it can't drift from CI). It builds a venv,
so it is manual-only locally. Heavy/extra imports on a
gate path must be lazy + guarded (see *Dependency discipline* in `AGENTS.md`) so
the package imports clean in the lean env.

## Conventions

- **No stubs.** `raise NotImplementedError` only with `# ABSTRACT-OK`.
- **Strangler-then-delete** — never "v2 beside old".
- **Name from purpose, not process** — no `wave0`/`phase2`/`v2` in identifiers;
  provenance goes in the docstring/CHANGELOG.
- **Wire-First** — a feature isn't done until a live path invokes it; ship
  primitives with a real consumer and a live-path test.
- New `CONCEPT:` ids go in `docs/concepts.yaml` (run `scripts/check_concepts.py`).

See [AGENTS.md](AGENTS.md) for the full architecture reference and guardrails.

## Spec-driven contributions

Start in [`specs/`](specs/README.md) with a stable ID and the repository-owned `spec.md`, `plan.md`,
`test-spec.md`, and `tasks.md`. Use the [universal-skills SDD
workflow](https://github.com/Knuckles-Team/universal-skills/tree/main/universal_skills/development-workflows/sdd-full-lifecycle)
and its
[spec-generator](https://github.com/Knuckles-Team/universal-skills/tree/main/universal_skills/development/spec-generator),
[spec-verifier](https://github.com/Knuckles-Team/universal-skills/tree/main/universal_skills/development/spec-verifier),
and
[task-planner](https://github.com/Knuckles-Team/universal-skills/tree/main/universal_skills/development/task-planner)
skills. The file sequence aligns with [GitHub Spec Kit
v1.0.12](https://github.com/github/spec-kit/releases/tag/v1.0.12); `test-spec.md` makes our test and
quality contract explicit. Link the PR to its spec IDs and include exact test, wiring, CCCC, jscpd,
dupehound, and KISS evidence before proposing a landed status.
