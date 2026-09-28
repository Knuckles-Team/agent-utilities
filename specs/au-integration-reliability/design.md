# Architecture and evidence model

The release record is a projection of source and CI, not a second runtime authority. For every ID it stores owner repository, merged commit, changed public path, test name, gate run URL, reviewer, acceptance state and remaining risk. A generated report may render this as HTML, but the checked-in spec and evidence rows remain reviewable text. A release script refuses an `ACCEPTED` row with no reachable commit or missing required gate.

The pinned EG client digest is a dependency input. AU `agent_utilities/api` and composition import that generated client; tests inject a contract fixture. The control graph is a single EG authority. The normal flow is `graph-os authenticated request → AU API → EG generated client → durable receipt`. There is no alternate local graph or publisher. The existing `scripts/uv_workspace.py`, privacy checker, liveness checker and normal hook are reused. A fresh checkout may run pure contract/fixture gates; live service certification is separately labelled and provisioned.

For source cleanup, discover imports and registered entry points by AST/metadata, then delete a legacy path only after its last public caller uses the new owner. Use exact file allowlists for commits and compare against origin/main before worktree pruning. Maintain a machine-readable task-to-evidence list; do not infer acceptance from a branch name, plan statement or test file presence.

Quality review uses CCCC for complexity, jscpd and dupehound differential duplication, KISS for single authority and minimal glue, Ruff/mypy/Pytest, privacy checks and normal hooks. A code change cannot silence a failed test by exclusion or placeholder.
