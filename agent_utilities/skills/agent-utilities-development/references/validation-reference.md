# Validation reference

Detail for step 6 (*Validate*) of the development workflow.

★ **`uv run pytest` is poisoned in this repository.** It silently resolves the
**system** interpreter and its stale packages, and produced ~80 phantom failures
that cited this project's own guards; six lanes were burned before it was found.
Always:

```bash
python3 scripts/uv_workspace.py run --all-extras -- pytest <args>
```

Print `sys.executable` and the package count in the same run: **≈726 packages is
the correct environment, ≈44 is the stale one.** A verdict from an unproven
interpreter is not evidence. A worktree-local `.venv` has the same effect (~167
phantom failures) — `--lane doctor` refuses one.

Run the narrow tests first, then every gate touched by the change. Run the full
pre-commit suite before delivery, and take the lease for it — it is LEASE-class
because it can destroy unstaged work:

```bash
# NEVER run `pre-commit run --all-files` bare: in a shared worktree it stashes
# the WHOLE tree and can destroy your own or another session's unstaged work
# (D-OB-12). The lease and the safe wrapper are TWO separate guards; both apply.
agent-utilities lane lease --resource precommit-all-files --operation gate -- \
  python3 scripts/safe_precommit_all_files.py
```

Never `--no-verify`, and never mask a gate to force green: `noqa`, `type:
ignore`, `nosec`, `skip`, and `xfail` appearing in a delivery diff are what a
reviewer greps for first.

★ **Judge every gate DIFFERENTIALLY, against the base ref — never against
absolute green.** `main` is legitimately red, so a pre-existing failure is not
yours to clear. Compare at the granularity the gate declares (pytest **node
ids**, not counts or files), and if the baseline cannot be produced, refuse
rather than allow-all. The mechanism, the evidence, and how `governance/
merge_queue.py` implements it are stated once, above — see **"The merge gate is
DIFFERENTIAL, not absolute"** at the end of *The gates that actually enforce
this*.

★ **Measure the MERGED tree, not the branch tip** — `git merge-tree --write-tree
main HEAD`. Reasoning from `git show <branch>:<path>` misled three people in one
day; one concluded a branch had deleted a guard the merged tree in fact kept.

Inspect the final diff for generated churn, stale names, sensitive data, and
stray files. Commit with a neutral repository identity.
