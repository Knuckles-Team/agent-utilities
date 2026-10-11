# Plan

1. Enumerate historical item IDs and compare each against exact current main code and test history. Mark source present separately from verified behavior.
2. Fix control graph and backend propagation, dependency/contract pins, baseline tests, liveness and privacy defects at their owning seams.
3. Remove stale legacy publisher and docs gate only after caller and generated Pages navigation parity are verified.
4. Qualify a candidate composition with public fixtures and declared integration dependencies. Record artifact digests and CI URLs.
5. Prune only after reachability and dirty-tree audit. Publish the exact merged-head evidence and update task states independently.
6. AU-INTEGRATION-R019: read the direct budget from config. Shield the run at the messaging budget wall and deliver its late reply through the existing follow-up send path.

- AU-INTEGRATION-R020: add `orchestration/run_envelope.py` with the shared envelope key set and `unwrap_run_envelope`. The messaging router delegates to it.

## Design

The release record is a projection of source and CI, not a second runtime authority. For every ID it stores owner repository, merged commit, changed public path, test name, gate run URL, reviewer, acceptance state and remaining risk. A generated report may render this as HTML, but the checked-in spec and evidence rows remain reviewable text. A release script refuses an `ACCEPTED` row with no reachable commit or missing required gate.

The pinned EG client digest is a dependency input. AU `agent_utilities/api` and composition import that generated client; tests inject a contract fixture. The control graph is a single EG authority. The normal flow is `graph-os authenticated request → AU API → EG generated client → durable receipt`. There is no alternate local graph or publisher. The existing `scripts/uv_workspace.py`, privacy checker, liveness checker and normal hook are reused. A fresh checkout may run pure contract/fixture gates; live service certification is separately labelled and provisioned.

For source cleanup, discover imports and registered entry points by AST/metadata, then delete a legacy path only after its last public caller uses the new owner. Use exact file allowlists for commits and compare against origin/main before worktree pruning. Maintain a machine-readable task-to-evidence list; do not infer acceptance from a branch name, plan statement or test file presence.

Quality review uses CCCC for complexity, jscpd and dupehound differential duplication, KISS for single authority and minimal glue, Ruff/mypy/Pytest, privacy checks and normal hooks. A code change cannot silence a failed test by exclusion or placeholder.
