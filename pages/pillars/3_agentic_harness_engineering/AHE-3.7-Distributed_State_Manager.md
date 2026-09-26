# Distributed Agent State Concurrency (CONCEPT:AU-AHE.harness.concept-2)

## Overview
The `BranchMergeStateLocker` handles parallel state branching, versioned optimistic staging, and three-way recursive dictionary merging to coordinate concurrent agent execution swarms without database locking bottlenecks.

## Component Architecture

<div class="admonition architecture" markdown>
<p class="admonition-title">Fork, execute, stage, merge, clean up — eight steps</p>

`expert_executor_step` (1) calls `fork_state` on
`BranchMergeStateLocker`, which (2) creates a fork (`branch_node_id`) in
the local/Redis branch cache. The step then (3) calls
`update_branch_state` into that cache and (4) executes the specialist
task on a specialist agent/node, which (5) writes its staged output back
into the branch cache. The step then (6) calls `merge_state` on
`BranchMergeStateLocker`, which (7) three-way/fast-forward merges into
the base main state and (8) cleans up the stale branch from the cache.
</div>

## Core Abstractions

### 1. State Branching & Forking
State branches are lightweight replicas keyed as `base_key:branch:branch_name`.
* **Fast & Decoupled**: Staged state avoids physical filesystem delays by utilizing local memory or high-speed Redis hash caches.
* **Traceability**: Each fork stores `base_version` (the fork origin ancestor) enabling strict convergence detection.

### 2. Convergence Merging Mechanisms
When merging a branch back to the base state:
* **Fast-Forward Merge**: If the base state version matches the branched `base_version` (i.e. no concurrent writes occurred on base), the branch simply replaces the base state and increments the version by 1.
* **Three-Way Recursive Merge**: If the base state has mutated concurrently, the locker attempts a nested key-level recursive merge:
  * If a key was changed in branch but untouched in base, the branch change is adopted.
  * If a key was changed in base but untouched in branch, the base change is kept.
  * If both changed, it resolves utilizing a custom arbitrating callback (`resolver`) or last-writer-wins (branch preference).

## Implementation Details
* **Source Code Path**: [distributed_state_manager.py](https://github.com/Knuckles-Team/agent-utilities/blob/main/agent_utilities/harness/distributed_state_manager.py)
* **Pillar**: Agentic Harness Engineering (AHE)
* **Concept ID**: `CONCEPT:AU-AHE.harness.concept-2`

## Example Usage

```python
from agent_utilities.harness.distributed_state_manager import BranchMergeStateLocker

locker = BranchMergeStateLocker()

# 1. Fork state for concurrent execution paths
locker.fork_state("workflow_state_01", "programmer_branch")

# 2. Update branch state staging
locker.update_branch_state(
    "workflow_state_01",
    "programmer_branch",
    {"staged_code": "def hello(): pass"}
)

# 3. Merge branch back to base state, resolving concurrent updates safely
success = locker.merge_state("workflow_state_01", "programmer_branch")
```
