# Agent-utilities G11 replacement catalogue reference

Deep reference for `agent-utilities-development`: the full expansion of G11
(*Mandatory guardrails*, in the parent [`SKILL.md`](../SKILL.md)). Each entry
names its replacement, because a prohibition without one does not hold. Run
`repository-manager --lane doctor --lane-path .` and it will tell you which of
them you are currently violating, with the exact remedy command.

- **Never edit the canonical checkout.** Work in the lane worktree.
- **Never use the harness's worktree-isolation tool** (`Agent(isolation:"worktree")`
  / `EnterWorktree`) on this repo. It writes `core.bare = true` into the **shared**
  `$GIT_COMMON_DIR/config` and never restores it, so every one of the 26+ linked
  worktrees then fails `git status`/`git commit` with *"this operation must be run
  in a work tree"* — invisibly. Upstream defect, closed as not-planned. Use
  `repository-manager --lane start` (or a real `git worktree add`).
- **Never `update-ref` to advance a branch.** It moves the ref without the
  worktree, and the NEXT commit there silently reverts everything in between while
  `git status` reads clean and `--is-ancestor` says yes. Use `git merge --ff-only`,
  then verify by TREE: `git cat-file -e HEAD:<path>` (see *Validate* → measure the
  merged tree).
- **Never `git stash`.** `refs/stash` is ONE ref shared by every worktree here.
  To read a pristine file while yours is dirty: `git show HEAD:<path>`. To park
  work: a `wip:` commit on your branch, or `agent-utilities lane park`.
- **Never export a shared `CARGO_TARGET_DIR`** — it corrupts concurrent worktree
  builds, it does not merely serialize them. Use `--target-dir ./target-isolated`
  and prune it; `agent-utilities lane bind-cargo` makes the partition structural.
- **Never run with the shared `PRE_COMMIT_HOME`.** pre-commit writes your
  unstaged work to a patch file there and restores it in a `finally:`; a crash
  inside that window loses it. `--lane env` sets a private one.
- **Never `git branch -D`.** Only `-d` — its refusal is the safety mechanism
  telling you the work is not contained in the base.
- **Never hand-edit a generated view** (`docs/concept_reservations.yaml`,
  `reports/PROGRAM.md`, a provider's `WORKFLOW.md`/`catalog.md`). Write your
  fragment or edit the source and regenerate; `lane-guard` refuses a hand-edited
  ledger view.
- **Register writes use `--detail-file`/`--evidence-file`, never `--detail "…"`.**
  Register prose contains backticked identifiers, and inside double quotes bash
  performs command substitution on backticks — silently executing them. This has
  already truncated live entries and triggered an accidental `uv sync` against
  the shared workspace `.venv` (D-ORC-22).
- **Never `git add -A` / `git add .`.** A shared worktree routinely holds handoff
  notes, baseline markers, logs, caches, and another concern's edits. Read `git
  status --short`, then stage an explicit reviewed allowlist (`git add -- path…`,
  `git add -u -- exact/path` for deletions), then re-read `git diff --cached
  --name-status` and `git diff --cached` before committing. `*-NOTES.md`,
  scratchpads, logs, caches, test output, and branch-divergence markers are never
  product artifacts.
- **Regenerate `uv.lock` exactly once, after every `pyproject.toml` in the change
  has frozen.** Regenerating per-edit produces a lock that churns against every
  other lane and an `uv-lock --locked` failure nobody can attribute. Verify the
  lock is untouched by your test runs before you commit.
- Do not bypass failing gates or silently accept warnings.
- Do not create a second implementation for another entry point.
- Do not commit secrets, credential files, local inventories, or scratch output.
