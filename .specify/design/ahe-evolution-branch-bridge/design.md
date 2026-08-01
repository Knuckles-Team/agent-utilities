# Design Document: Publishing a promoted proposal is a swappable protocol, never a hard dependency on fleet git tooling

CONCEPT:AU-AHE.harness.evolution-branch-bridge

> `agent_utilities/knowledge_graph/research/change_publisher.py`.

## Decision — `ChangePublisher` is a protocol; the default implementation shells out to plain `git` in a fresh worktree, never the canonical checkout

The module docstring (`change_publisher.py:4-16`) states the decision
directly: agent-utilities deliberately takes NO hard dependency on the
ecosystem's repository tooling (repository-manager's `rm_git`/`rm_worktree`
MCP tools) to turn a promoted `ChangeSet` into something reviewable.
Publication is a protocol — `ChangePublisher.publish(change_set, metadata) ->
PublishResult` — exactly like the fleet-actuation seam
(`AU-OS.config.desired-state-fleet-reconciler`). `LocalBranchPublisher` is
the DEFAULT implementation: a plain `git` subprocess (universally available,
no MCP round-trip), and critically it creates a **fresh worktree** off the
target repo's default branch under a configurable root
(`EVOLUTION_WORKTREE_ROOT`) — it never writes into the canonical checkout's
working tree.

**The rejected alternative is hard-coupling publication to the
repository-manager MCP service** — calling its `rm_git`/`rm_worktree` tools
directly as the only path. That would make every evolution-publish call
depend on the fleet service being up, and worse, risks the exact failure
mode this codebase has hit before: a repository-manager sync resetting a
canonical checkout out from under work in progress. By writing into a fresh,
isolated worktree instead of the shared canonical checkout, a proposal being
materialized and reviewed can never collide with — or be destroyed by — a
concurrent sync of the primary working tree. The change is committed with
opaque metadata, targeted regression tests run via an injected sandbox
runner, and the commit message embeds the concept id plus the gate result so
a reviewer sees pass/fail and targeted-tests state directly in `git log`.

## Risk Assessment

- **Blast Radius**: `agent_utilities/knowledge_graph/research/change_publisher.py`,
  `agent_utilities/knowledge_graph/research/change_synthesis.py`,
  `agent_utilities/knowledge_graph/research/auto_merge.py`,
  `agent_utilities/orchestration/fleet_reconciler.py`.
- **Backward Compatible**: Yes — `ChangePublisher` is a protocol; a
  repository-manager-backed implementation could be added later as an
  alternate publisher without changing the interface.
- **Known weak point**: `EVOLUTION_WORKTREE_ROOT` worktrees accumulate over
  time unless something prunes them; nothing in this module itself garbage
  collects stale evolution worktrees.
